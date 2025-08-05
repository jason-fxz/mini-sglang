"""
Server arguments manager for mini_sglang.
"""

import argparse
import dataclasses
import os


@dataclasses.dataclass
class ServerArgs:
    model: str
    page_size: int = 1
    attention_backend: str = "torch"

    max_num_reqs: int = 1024

    tp: int = 1
    device: str = "cuda"

    gpu_memory_utilization: float = 0.9

    def __post_init__(self):
        assert os.path.isdir(self.model)

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--model", type=str, required=True, help="Path to the model directory."
        )
        parser.add_argument(
            "--page_size",
            type=int,
            default=ServerArgs.page_size,
            help="Page size for KV cache.",
        )
        parser.add_argument(
            "--eos", type=int, default=ServerArgs.eos, help="End of sequence token ID."
        )
        parser.add_argument(
            "--attention_backend",
            type=str,
            default=ServerArgs.attention_backend,
            choices=["fa2", "fa3", "torch"],
            help="Attention backend to use.",
        )
        parser.add_argument(
            "--tp",
            type=int,
            default=ServerArgs.tp,
            help="Tensor parallelism degree.",
        )
        parser.add_argument(
            "--device",
            type=str,
            default=ServerArgs.device,
            help="Device to run the model on (e.g., 'cuda:0').",
        )
        parser.add_argument(
            "--gpu_memory_utilization",
            type=float,
            default=ServerArgs.gpu_memory_utilization,
            help="GPU memory utilization percentage.",
        )
        parser.add_argument(
            "--max_num_reqs",
            type=int,
            default=ServerArgs.max_num_reqs,
            help="Maximum number of requests to handle.",
        )
