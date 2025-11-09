#!/usr/bin/env python3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, cast
import argparse
import logging
import random
import numpy as np
import torch
import yaml
from datasets import load_dataset, DatasetDict
from transformers import (
    DataCollatorForSeq2Seq,
    EvalPrediction,
    PreTrainedTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    GenerationConfig,
)
from fast_phonemizer.config import (
    ModelConfig,
    TrainingConfig,
    InferenceConfig,
    DataConfig,
    VocabConfig,
)
from fast_phonemizer.tokenizer import build_tokenizers, charspace
from fast_phonemizer.model import FastPhonemizerModel

LOGGER = logging.getLogger("fast_phonemizer")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_yaml(yaml_path: Path) -> Dict[str, Any]:
    with yaml_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Preprocessor:
    def __init__(
        self,
        tok_in: PreTrainedTokenizerFast,
        tok_out: PreTrainedTokenizerFast,
        text_col: str,
        phon_col: str,
        max_src_len: int,
        max_tgt_len: int,
    ):
        self.tok_in = tok_in
        self.tok_out = tok_out
        self.text_col = text_col
        self.phon_col = phon_col
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __call__(self, batch: Dict[str, List[str]]) -> Dict[str, Any]:
        src = [charspace(x) for x in batch[self.text_col]]
        enc = self.tok_in(
            src,
            padding=False,
            truncation=True,
            max_length=self.max_src_len,
        )
        tgt_raw = [f"{y.strip()} <eos>" for y in batch[self.phon_col]]
        dec = self.tok_out(
            tgt_raw,
            padding=False,
            truncation=True,
            max_length=self.max_tgt_len,
        )

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc.get("attention_mask"),
            "labels": dec["input_ids"],
        }


def _edit_distance(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        ai = a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ai == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


IGNORE_INDEX = -100


def make_compute_metrics(
    tokenizer,
    edit_distance_fn: Callable[[List[str], List[str]], int] = _edit_distance,
    ignore_index: int = IGNORE_INDEX,
    normalize_space: bool = True,
    lowercase: bool = False,
) -> Callable[[EvalPrediction], Dict[str, float]]:
    """
    Compute PER (phoneme/phone error rate) for given inputs.

    Args:
        ignore_index (int): Label value to ignore (e.g., -100 from CrossEntropyLoss).
        normalize_space (bool): Whether to collapse repeated whitespace before tokenizing to words.
        lowercase (bool): Whether to lowercase strings before tokenizing (optional).
        edit_distance_fn (callable): Function that takes two token lists and returns edit distance.
                                     Defaults to `_edit_distance` if not provided.
    """
    pad_id = tokenizer.pad_token_id

    def _prep_text(s: str) -> str:
        if lowercase:
            s = s.lower()
        if normalize_space:
            s = " ".join(s.split())
        return s

    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        pred_ids, label_ids = eval_pred

        pred_ids = np.asarray(pred_ids)
        label_ids = np.asarray(label_ids)

        if ignore_index is not None:
            label_ids = np.where(label_ids == ignore_index, pad_id, label_ids)

        pred_txt = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_txt = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pers: List[float] = []
        for p, r in zip(pred_txt, label_txt):
            p = _prep_text(p)
            r = _prep_text(r)
            r_seq = r.split()
            if not r_seq:
                continue
            p_seq = p.split()
            dist = edit_distance_fn(p_seq, r_seq)
            pers.append(dist / max(1, len(r_seq)))

        return {"PER": float(np.mean(pers)) if pers else 0.0}

    return compute_metrics


def _as_list(x: Union[str, List[str]]) -> List[str]:
    return x if isinstance(x, list) else [x]


def _normalize_label_pad(v: Optional[Union[str, list[str], int]]) -> int:
    if isinstance(v, int):
        return v

    return -100


def train_command(config_path: Path) -> None:
    cfg_raw = read_yaml(config_path)

    training = TrainingConfig(**cfg_raw.get("training", {}))
    model_cfg = ModelConfig(**cfg_raw.get("model", {}))
    data_cfg = DataConfig(**cfg_raw.get("data", {}))
    vocab_cfg = VocabConfig(**cfg_raw.get("vocab", {}))
    infer_cfg = InferenceConfig(**cfg_raw.get("inference", {}))

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    LOGGER.info("Loaded config from %s", config_path)

    set_seed(42)

    data_files = {
        "train": _as_list(data_cfg.train_csv),
        "validation": _as_list(data_cfg.validation_csv),
    }

    dataset: DatasetDict = cast(
        DatasetDict,
        load_dataset(
            "csv",
            data_files=data_files,
            delimiter="\t",
            column_names=["text", "phonemes"],
        ),
    )

    tok_in, tok_out = build_tokenizers(vocab_cfg, data_cfg, dataset)

    model = FastPhonemizerModel.from_configs(model_cfg, tok_in, tok_out)

    model.generation_config = GenerationConfig.from_model_config(model.config)
    model.generation_config.max_length = training.generation_max_length
    model.generation_config.num_beams = infer_cfg.num_beams
    model.generation_config.length_penalty = infer_cfg.length_penalty

    preproc = Preprocessor(
        tok_in,
        tok_out,
        text_col=data_cfg.text_column,
        phon_col=data_cfg.phoneme_column,
        max_src_len=data_cfg.max_src_len,
        max_tgt_len=data_cfg.max_tgt_len,
    )
    dataset = cast(DatasetDict, dataset.map(preproc, batched=True))

    collator = DataCollatorForSeq2Seq(
        tokenizer=tok_in,
        model=model,
        label_pad_token_id=_normalize_label_pad(tok_out.pad_token_id),
        padding=True,
    )

    args = Seq2SeqTrainingArguments(
        output_dir=training.output_dir,
        num_train_epochs=training.num_train_epochs,
        per_device_train_batch_size=training.per_device_train_batch_size,
        per_device_eval_batch_size=training.per_device_eval_batch_size,
        learning_rate=training.learning_rate,
        weight_decay=training.weight_decay,
        warmup_ratio=training.warmup_ratio,
        logging_steps=training.logging_steps,
        eval_strategy=training.evaluation_strategy,
        eval_steps=training.eval_steps,
        save_strategy=training.save_strategy,
        save_steps=training.save_steps,
        save_total_limit=training.save_total_limit,
        predict_with_generate=training.predict_with_generate,
        fp16=training.fp16 and torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),  # type: ignore
        data_collator=collator,
        compute_metrics=make_compute_metrics(tok_out),
    )

    LOGGER.info("Starting training…")
    trainer.train()

    LOGGER.info("Evaluating…")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        LOGGER.info("%s: %s", k, v)

    out_dir = Path(training.output_dir) / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tok_in.save_pretrained(out_dir / "tok_in")
    tok_out.save_pretrained(out_dir / "tok_out")


def build_arg_parser() -> argparse.ArgumentParser:
    main_parser = argparse.ArgumentParser(
        description="Tiny G2P trainer (Transformers + Seq2SeqTrainer)"
    )
    subparser = main_parser.add_subparsers(dest="command", required=True)

    train_parser = subparser.add_parser("train", help="Train a tiny G2P model")
    train_parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML config"
    )

    return main_parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        train_command(args.config)
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
