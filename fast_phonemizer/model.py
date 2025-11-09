#!/usr/bin/env python3

from transformers import (
    BertConfig,
    EncoderDecoderConfig,
    EncoderDecoderModel,
    PreTrainedTokenizerBase,
    PretrainedConfig,
)
from fast_phonemizer.config import ModelConfig


class FastPhonemizerModel(EncoderDecoderModel):

    def __init__(
        self, model_config: PretrainedConfig, tokenizer_out: PreTrainedTokenizerBase
    ):
        super().__init__(model_config)
        self._set_tokenizer_settings(tokenizer_out)

    def _set_tokenizer_settings(self, tokenizer_out: PreTrainedTokenizerBase) -> None:
        eos_id = tokenizer_out.eos_token_id
        pad_id = tokenizer_out.pad_token_id

        if not isinstance(eos_id, int):
            raise TypeError(f"eos_id needs to be integer. Got {eos_id}")

        if not isinstance(pad_id, int):
            raise TypeError(f"pad_id needs to be integer. Got {pad_id}")

        self.config.decoder_start_token_id = eos_id
        self.config.eos_token_id = eos_id
        self.config.pad_token_id = pad_id
        self.config.vocab_size = len(tokenizer_out)

    @classmethod
    def from_configs(
        cls,
        model_config: ModelConfig,
        tokenizer_in: PreTrainedTokenizerBase,
        tokenizer_out: PreTrainedTokenizerBase,
    ) -> EncoderDecoderModel:
        if model_config.hidden_size % model_config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({model_config.hidden_size}) must be divisible by num_attention_heads ({model_config.num_attention_heads})"
            )

        enc_config = BertConfig(
            vocab_size=len(tokenizer_in),
            hidden_size=model_config.hidden_size,
            intermediate_size=model_config.intermediate_size,
            num_attention_heads=model_config.num_attention_heads,
            num_hidden_layers=model_config.num_encoder_layers,
            max_position_embeddings=model_config.max_position_embeddings,
            layer_norm_eps=model_config.layer_norm_eps,
        )

        dec_config = BertConfig(
            vocab_size=len(tokenizer_out),
            hidden_size=model_config.hidden_size,
            intermediate_size=model_config.intermediate_size,
            num_attention_heads=model_config.num_attention_heads,
            num_hidden_layers=model_config.num_decoder_layers,
            max_position_embeddings=model_config.max_position_embeddings,
            layer_norm_eps=model_config.layer_norm_eps,
            is_decoder=True,
            add_cross_attention=True,
        )

        ed_config = EncoderDecoderConfig.from_encoder_decoder_configs(
            enc_config, dec_config
        )
        model = cls(ed_config, tokenizer_out)

        return model
