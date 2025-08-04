"""
Batch information
"""

from __future__ import annotations

import dataclasses
from enum import IntEnum, auto
from typing import TYPE_CHECKING, Optional, Union

import torch

if TYPE_CHECKING:
    from mini_sglang.layers.attn.attn_backend import AttentionBackend


class ForwardMode(IntEnum):
    EXTEND = auto()
    DECODE = auto()

    def is_extend(self) -> bool:
        return self == ForwardMode.EXTEND

    def is_decode(self) -> bool:
        return self == ForwardMode.DECODE


@dataclasses.dataclass
class BatchInfo:
    """Information about the batch."""

    """
    for prefill (with prefix match)

    |      prefix_ids      |      input_ids      |
    | <- prefix_seq_len -> | <- input_seq_len -> |
    | <-               seq_len                -> |

    for decode: input_seq_len = 1


    """
    ## Basic information
    # forward mode
    forward_mode: ForwardMode
    # batch size
    batch_size: int
    # input ids (flattened)   shape: [sum(input_seq_len)]
    input_ids: torch.Tensor
    # positions (flattened)   shape: [sum(input_seq_lens)]
    positions: torch.Tensor
    # sequence lengths   shape: [batch_size]
    input_seq_lens: torch.Tensor
    # full sequence lengths   shape: [batch_size]
    seq_lens: torch.Tensor
    # prefix sequence lengths   shape: [batch_size]
    prefix_seq_lens: torch.Tensor

    # output cache location in the token_to_kv_pool  shape: [batch_size]
    out_cache_loc: torch.Tensor

    # Attention Backends
    attn_backend: AttentionBackend = None

    # KV cache
    req_pool_indices: torch.Tensor
