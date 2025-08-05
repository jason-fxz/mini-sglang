import os

import pytest
import torch
import torch.distributed as dist

from mini_sglang.utils.loader import load_model
from mini_sglang.utils.model_config import ModelConfig


@pytest.fixture(scope="module", autouse=True)
def setup_dist():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def test_load_model_success():
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    config = ModelConfig(model_path)
    torch.set_default_device("cuda")
    model = load_model(config)
    torch.set_default_device("cpu")
    assert model is not None
