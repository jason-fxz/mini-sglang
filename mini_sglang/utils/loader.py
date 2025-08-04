import os
from glob import glob

import torch
from safetensors import safe_open
from torch import nn

from mini_sglang.models.qwen3 import Qwen3ForCausalLM
from mini_sglang.utils.model_config import ModelConfig


def load_model(config: ModelConfig) -> nn.Module:
    # TODO(support other model types)
    model = Qwen3ForCausalLM(config.hf_config)

    for file in glob(os.path.join(config.model_path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            weights = []
            for weight_name in f.keys():
                weights.append((weight_name, f.get_tensor(weight_name)))
            model.load_weights(weights)

    return model
