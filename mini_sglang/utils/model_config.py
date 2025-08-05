import json
from typing import List, Optional

from transformers import AutoConfig


class ModelConfig:
    def __init__(
        self,
        model_path: str,
    ):
        self.model_path = model_path
        self.hf_config = AutoConfig.from_pretrained(model_path)
        self.hidden_size = self.hf_config.hidden_size
        self.num_hidden_layers = self.hf_config.num_hidden_layers
        self.head_dim = self.hf_config.head_dim
        self.num_heads = self.hf_config.num_attention_heads
        self.num_kv_heads = self.hf_config.num_key_value_heads

        self.max_context_len = self.hf_config.max_position_embeddings
        self.kv_cache_dtype = self.hf_config.torch_dtype
