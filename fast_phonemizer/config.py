from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class TrainingConfig:
    output_dir: str = "out_g2p"
    num_train_epochs: int = 500
    per_device_train_batch_size: int = 4096
    per_device_eval_batch_size: int = 4096
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    logging_steps: int = 500
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 2048
    save_total_limit: int = 2
    predict_with_generate: bool = True
    generation_max_length: int = 64
    fp16: bool = True


@dataclass
class ModelConfig:
    hidden_size: int = 128
    intermediate_size: int = 256
    num_attention_heads: int = 8
    num_encoder_layers: int = 8
    num_decoder_layers: int = 8
    max_position_embeddings: int = 128
    layer_norm_eps: float = 1e-5


@dataclass
class DataConfig:
    train_csv: Union[str, List[str]] = "data/train.csv"
    validation_csv: Union[str, List[str]] = "data/dev.csv"
    text_column: str = "text"
    phoneme_column: str = "phonemes"
    max_src_len: int = 64
    max_tgt_len: int = 64


@dataclass
class VocabConfig:
    grapheme_tokens: Optional[List[str]] = None
    phoneme_tokens: Optional[List[str]] = None
    grapheme_vocab_path: Optional[str] = None
    phoneme_vocab_path: Optional[str] = None


@dataclass
class InferenceConfig:
    num_beams: int = 4
    length_penalty: float = 0.6


@dataclass
class FastPhonemizerConfig:
    training: TrainingConfig
    model: ModelConfig
    data: DataConfig
    vocab: VocabConfig
    inference: InferenceConfig
