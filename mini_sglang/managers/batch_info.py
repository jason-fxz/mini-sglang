"""
Batch information
"""

from __future__ import annotations

import dataclasses
from enum import IntEnum, auto
from typing import TYPE_CHECKING, List, Optional, Union

import torch

from mini_sglang.managers.req_info import Req, ReqStatus

if TYPE_CHECKING:
    from mini_sglang.layers.attn.attn_backend import AttentionBackend
    from mini_sglang.mem_cache.req2token import ReqToTokenPool
    from mini_sglang.mem_cache.token2kv import KVCachePool, PageAllocator


class ForwardMode(IntEnum):
    EXTEND = auto()
    DECODE = auto()

    def is_extend(self) -> bool:
        return self == ForwardMode.EXTEND

    def is_decode(self) -> bool:
        return self == ForwardMode.DECODE

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name


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
    forward_mode: ForwardMode = None
    # input ids (flattened)
    input_ids: torch.Tensor = None  # shape: [sum(input_seq_len)]  type: int32
    # positions (flattened)
    positions: torch.Tensor = None  # shape: [sum(input_seq_len)] type: int32
    # sequence lengths (only EXTEND mode)
    input_seq_lens: torch.Tensor = None  # shape: [batch_size] type: int32
    # full sequence lengths
    seq_lens: torch.Tensor = None  # shape: [batch_size] type: int32
    # prefix sequence lengths (only EXTEND mode)
    prefix_seq_lens: torch.Tensor = None  # shape: [batch_size] type: int32

    ## KV Cache information
    # output cache location in the token_to_kv_pool
    out_cache_loc: torch.Tensor = None  # shape: [batch_size]
    # request pool indices
    req_pool_indices: torch.Tensor = None  # shape: [batch_size]
    # request to token pool
    req_to_token_pool: ReqToTokenPool = None
    # page allocator
    page_allocator: PageAllocator = None
    # kv cache pool
    kv_cache_pool: KVCachePool = None

    # Attention Backends
    attn_backend: AttentionBackend = None

    # Reqs
    reqs: List[Req] = None

    # Device
    device: str = "cuda"

    @classmethod
    def init_new(
        cls,
        reqs: List[Req],
        req_to_token_pool: ReqToTokenPool,
        page_allocator: PageAllocator,
        kv_cache_pool: KVCachePool,
    ):
        return cls(
            reqs=reqs,
            req_to_token_pool=req_to_token_pool,
            page_allocator=page_allocator,
            kv_cache_pool=kv_cache_pool,
            device=req_to_token_pool.device,
        )

    @property
    def batch_size(self) -> int:
        return len(self.reqs)

    def alloc_token_slots(self, num_tokens: int) -> torch.Tensor:
        """
        Allocates token slots in the token pool.
        Returns a tensor of indices to the token pool.
        """
        return self.req_to_token_pool.alloc(num_tokens)

    def prepare_for_extend(self):
        # Set forward mode
        self.forward_mode = ForwardMode.EXTEND

        # alloc req slots
        bs = self.batch_size
        req_pool_indices = self.req_to_token_pool.alloc(bs)
        for i, req in enumerate(self.reqs):
            req.req_pool_idx = req_pool_indices[i]
            req.status = ReqStatus.RUNNING

        # Init tensors
        reqs = self.reqs
        input_ids = [r.token_ids[len(r.prefix_indices) :] for r in reqs]
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = [len(r.token_ids) for r in reqs]
        prefix_lens = [len(r.prefix_indices) for r in reqs]
        extend_lens = [len(r.token_ids) - len(r.prefix_indices) for r in reqs]
        positions = [list(range(prefix_lens[i], seq_lens[i])) for i in range(bs)]

        req_pool_indices_tensor = torch.tensor(req_pool_indices, dtype=torch.int32).to(
            self.device, non_blocking=True
        )
        input_ids_tensor = torch.tensor(sum(input_ids, []), dtype=torch.int32).to(
            self.device, non_blocking=True
        )
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32).to(
            self.device, non_blocking=True
        )
        prefix_lens_tensor = torch.tensor(prefix_lens, dtype=torch.int32).to(
            self.device, non_blocking=True
        )
        input_lens_tensor = seq_lens_tensor - prefix_lens_tensor
        positions_tensor = torch.tensor(sum(positions, []), dtype=torch.int32).to(
            self.device, non_blocking=True
        )

        # Alloc kv cache
        if self.page_allocator.page_size == 1:
            out_cache_loc = self.page_allocator.alloc(extend_num_tokens).to(
                self.device, non_blocking=True
            )
            # out_cache_loc = torch.tensor(out_cache_loc, dtype=torch.int32).to(self.device, non_blocking=True)
        else:
            # TODO: support page size > 1
            raise RuntimeError("Currently only support page size of 1 for kv cache")

        # Write to req2token pool
        pos = 0
        for i in range(bs):
            self.req_to_token_pool.write(
                (req_pool_indices[i], slice(prefix_lens[i], seq_lens[i])),
                out_cache_loc[pos : pos + extend_lens[i]],
            )
            pos += extend_lens[i]
        # Set attributes
        self.input_ids = input_ids_tensor
        self.req_pool_indices = req_pool_indices_tensor
        self.seq_lens = seq_lens_tensor
        self.out_cache_loc = out_cache_loc
        self.input_seq_lens = input_lens_tensor
        self.prefix_seq_lens = prefix_lens_tensor
        self.positions = positions_tensor

    def prepare_for_decode(self):
        # Set forward mode
        self.forward_mode = ForwardMode.DECODE

        bs = self.batch_size

        # Init tensors
        reqs = self.reqs
        input_ids = [r.token_ids[len(r.prefix_indices) :] for r in reqs]
        locs = self.seq_lens.clone()
        self.seq_lens.add_(1)
        self.prefix_seq_lens = None
        self.input_seq_lens = None

        input_ids_tensor = torch.tensor(sum(input_ids, []), dtype=torch.int32).to(
            self.device, non_blocking=True
        )

        # Alloc kv cache
        if self.page_allocator.page_size == 1:
            out_cache_loc = self.page_allocator.alloc(bs).to(
                self.device, non_blocking=True
            )
            # out_cache_loc = torch.tensor(out_cache_loc, dtype=torch.int32).to(self.device, non_blocking=True)
        else:
            # TODO: support page size > 1
            raise RuntimeError("Currently only support page size of 1 for kv cache")

        # Write to req2token pool
        self.out_cache_loc = out_cache_loc
        self.req_to_token_pool.write((self.req_pool_indices, locs), self.out_cache_loc)
        self.positions = locs
        self.input_ids = input_ids_tensor
