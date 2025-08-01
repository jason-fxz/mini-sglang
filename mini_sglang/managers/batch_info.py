"""
Batch information
"""

import dataclasses
from enum import IntEnum, auto
from typing import Optional, Union

import torch

from mini_sglang.layers.attn.attn_backend import AttentionBackend


class ForwardMode(IntEnum):
    PREFILL = auto()
    DECODE = auto()

    def is_prefill(self) -> bool:
        return self == ForwardMode.PREFILL

    def is_decode(self) -> bool:
        return self == ForwardMode.DECODE


@dataclasses.dataclass
class BatchInfo:
    """Information about the batch."""

    """
    for prefill (with prefix match)

    |    prefix_ids    |      input_ids      |
    | <- prefix_len -> | <- input_seq_len -> |
    | <-             seq_len              -> |

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

    # Attention Backends
    attn_backend: AttentionBackend

    ## for FA3
    # Used in FA3 prefill mode
    # cu_seqlens_q = [0, input_seq_len_1, input_seq_len_1 + input_seq_len_2, ...]
    cu_seqlens_q: Optional[torch.Tensor] = None
    # cu_seqlens_k = [0, seq_len_1, seq_len_1 + seq_len_2, ...]
    cu_seqlens_k: Optional[torch.Tensor] = None
