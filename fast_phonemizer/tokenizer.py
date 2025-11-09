#!/usr/bin/env python3
from typing import List, Optional, Tuple, Iterable
from transformers import PreTrainedTokenizerFast
from datasets import DatasetDict
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from .config import VocabConfig, DataConfig


def charspace(text: str) -> str:
    """Convert a raw string into space-delimited characters for WordLevel tokenization."""
    return " ".join(list(str(text).lower()))


def ensure_special_tokens(
    tokens: List[str], special_tokens: Iterable[str]
) -> List[str]:
    """
    Ensures that the given special tokens are included at the beginning of the token list.

    If a special token is not present in the token list, it is inserted at the beginning.

    Args:
        tokens: The list of existing tokens.
        special_tokens: The iterable of special tokens to ensure presence.

    Returns:
        A new list of tokens with special tokens prepended if they were missing.
    """

    existing_tokens = set(tokens)
    missing_tokens = [token for token in special_tokens if token not in existing_tokens]

    return missing_tokens + tokens


def auto_build_vocabs(
    dataset: DatasetDict,
    text_col: str,
    phon_col: str,
    base_graphemes: Optional[List[str]] = None,
    base_phonemes: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Automatically builds vocabulary lists for graphemes and phonemes from the dataset.

    Graphemes are derived from the text column, extracted as individual lowercase characters.
    Phonemes are derived from the phoneme column, split by whitespace.

    Args:
        dataset: The dataset containing the text and phoneme columns.
        text_col: The name of the column containing text data.
        phon_col: The name of the column containing phoneme data.
        base_graphemes: Optional list of base grapheme tokens to add to the vocabulary.
        base_phonemes: Optional list of base phoneme tokens to add to the vocabulary.

    Returns:
        A tuple containing two lists: the first for grapheme tokens, the second for phoneme tokens.
    """
    grapheme_set = set(base_graphemes or [])
    for text in dataset["train"][text_col]:
        grapheme_set.update(list(str(text).lower()))

    phoneme_set = set(base_phonemes or [])
    for phoneme_seq in dataset["train"][phon_col]:
        phoneme_set.update(str(phoneme_seq).split())

    grapheme_tokens = sorted(grapheme_set)
    phoneme_tokens = sorted(phoneme_set)

    grapheme_tokens = ensure_special_tokens(grapheme_tokens, ("<pad>", "<eos>"))
    phoneme_tokens = ensure_special_tokens(phoneme_tokens, ("<pad>", "<eos>"))

    return grapheme_tokens, phoneme_tokens


def build_wordlevel_tokenizer(tokens: List[str]) -> PreTrainedTokenizerFast:
    """
    Creates a Hugging Face WordLevel tokenizer with the provided vocabulary.

    The tokenizer is configured with a whitespace pre-tokenizer and includes special tokens:
    - "<pad>" for padding sequences.
    - "<eos>" for end-of-sequence markers.

    Args:
        tokens: The list of tokens to include in the vocabulary.

    Returns:
        A PreTrainedTokenizerFast instance with the specified vocabulary and special tokens.
    """
    vocab = {token: idx for idx, token in enumerate(tokens)}
    model = WordLevel(vocab=vocab, unk_token="<unk>")
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = Whitespace()  # type: ignore
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

    if "<pad>" in vocab:
        hf_tokenizer.add_special_tokens({"pad_token": "<pad>"})
    if "<eos>" in vocab:
        hf_tokenizer.add_special_tokens({"eos_token": "<eos>"})

    return hf_tokenizer


def load_tokens_from_file(file_path: Optional[str]) -> Optional[List[str]]:
    """
    Loads a list of tokens from a file.

    Args:
        file_path: The path to the file containing the tokens.

    Returns:
        A list of strings read from the file, or None if the file path is not provided.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if file_path is None:
        return None

    with Path(file_path).open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def build_tokenizers(
    vocab_cfg: VocabConfig,
    data_cfg: DataConfig,
    dataset: DatasetDict,
) -> Tuple[PreTrainedTokenizerFast, PreTrainedTokenizerFast]:
    """
    Builds and returns grapheme and phoneme tokenizers based on the provided configuration and dataset.

    Args:
        vocab_cfg: Configuration object containing paths and base tokens for vocabulary.
        data_cfg: Configuration object containing column names for text and phoneme data.
        dataset: The dataset used to build the vocabulary if not provided in the configuration.

    Returns:
        A tuple containing two tokenizers: the first for graphemes, the second for phonemes.
    """
    grapheme_tokens = vocab_cfg.grapheme_tokens or load_tokens_from_file(
        vocab_cfg.grapheme_vocab_path
    )
    phoneme_tokens = vocab_cfg.phoneme_tokens or load_tokens_from_file(
        vocab_cfg.phoneme_vocab_path
    )

    if grapheme_tokens is None or phoneme_tokens is None:
        base_graphemes = grapheme_tokens or []
        base_phonemes = phoneme_tokens or []
        grapheme_tokens, phoneme_tokens = auto_build_vocabs(
            dataset,
            data_cfg.text_column,
            data_cfg.phoneme_column,
            base_graphemes,
            base_phonemes,
        )

    tokenizer_in = build_wordlevel_tokenizer(grapheme_tokens)
    tokenizer_out = build_wordlevel_tokenizer(phoneme_tokens)

    return tokenizer_in, tokenizer_out
