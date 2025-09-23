import os
from glob import glob

import torch
from safetensors import safe_open
from torch import nn
from tqdm import tqdm

from mini_sglang.models.qwen3 import Qwen3ForCausalLM
from mini_sglang.utils.model_config import ModelConfig


def load_model(config: ModelConfig) -> nn.Module:
    # TODO(support other model types)
    model = Qwen3ForCausalLM(config.hf_config)

    files = sorted(glob(os.path.join(config.model_path, "*.safetensors")))
    for file in tqdm(files, desc="Loading Weights"):
        with safe_open(file, "pt", "cpu") as f:
            weights = []
            for weight_name in f.keys():
                weights.append((weight_name, f.get_tensor(weight_name)))
            model.load_weights(weights)

    return model


def load_dummy_model(config: ModelConfig) -> nn.Module:
    model = Qwen3ForCausalLM(config.hf_config)

    for name, param in model.named_parameters():
        if param.requires_grad:
            nn.init.normal_(param, mean=0.0, std=0.1)
    return model
