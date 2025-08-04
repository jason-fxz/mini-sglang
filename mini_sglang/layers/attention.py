# FA2 interface https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py

# FA3 interface https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_attn_interface.py

import torch
from torch import nn

from mini_sglang.managers.batch_info import BatchInfo


class Attention(nn.Module):

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        layer_id: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.layer_id = layer_id

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, batch: BatchInfo
    ):
        return batch.attn_backend.forward(q=q, k=k, v=v, layer=self, batch=batch)
