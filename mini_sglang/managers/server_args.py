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
    eos: int = -1
    attention_backend: str = "fa3"

    tp: int = 1
    device: str

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
            choices=["fa2", "fa3"],
            help="Attention backend to use.",
        )
