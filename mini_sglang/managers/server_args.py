"""
Server arguments manager for mini_sglang.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import socket
import tempfile


@dataclasses.dataclass
class ServerArgs:
    model: str
    page_size: int = 1
    attention_backend: str = "torch"

    max_num_reqs: int = 1024

    scheduler_policy: str = "fcfs"

    max_running_bs: int = 128

    tp_size: int = 1
    device: str = "cuda"
    nccl_port: int = None

    gpu_memory_utilization: float = 0.9
    log_level: str = "INFO"

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
            "--tp-size",
            type=int,
            default=ServerArgs.tp_size,
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
        parser.add_argument(
            "--schedule_policy",
            type=str,
            default=ServerArgs.schedule_policy,
            choices=["fcfs", "lof", "random", "lpm", "dfs-weight"],
            help="Scheduling policy to use.",
        )
        parser.add_argument(
            "--max_running_bs",
            type=int,
            default=ServerArgs.max_running_bs,
            help="Maximum batch size for running requests.",
        )
        parser.add_argument(
            "--log_level",
            type=str,
            default=ServerArgs.log_level,
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Logging level for the server.",
        )


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # 绑定到端口0，操作系统自动分配一个空闲端口
        addr, port = s.getsockname()
        return port


@dataclasses.dataclass
class PortArgs:
    # ZMQ ipc filename for tokenizer -> scheduler
    scheduler_input_ipc: str
    # ZMQ ipc filename for scheduler -> detokenizer
    detokenizer_input_ipc: str
    # ZMQ ipc filename for detokenizer -> tokenizer
    tokenizer_input_ipc: str

    # for nccl initialization (torch.dist)
    nccl_port: int

    @staticmethod
    def init_new(server_args: ServerArgs) -> PortArgs:
        if server_args.nccl_port is not None:
            nccl_port = server_args.nccl_port
        else:
            nccl_port = find_free_port()

        return PortArgs(
            scheduler_input_ipc=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            detokenizer_input_ipc=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            tokenizer_input_ipc=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            nccl_port=nccl_port,
        )
