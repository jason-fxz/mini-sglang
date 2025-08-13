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

    def is_empty(self):
        return len(self.reqs) == 0

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

        if self.page_allocator.page_size == 1:
            # Alloc kv cache
            out_cache_loc = self.page_allocator.alloc(extend_num_tokens).to(
                self.device, non_blocking=True
            )

            # Write to req2token pool
            pos = 0
            for i in range(bs):
                self.req_to_token_pool.write(
                    (req_pool_indices[i], slice(prefix_lens[i], seq_lens[i])),
                    out_cache_loc[pos : pos + extend_lens[i]],
                )
                pos += extend_lens[i]
        else:
            page_size = self.page_allocator.page_size
            num_pages = [
                (tokens + page_size - 1) // page_size for tokens in extend_lens
            ]
            page_loc = self.page_allocator.alloc(sum(num_pages))
            # Expand each page id to its token locations: [pid*page_size, (pid+1)*page_size)
            expanded = page_loc.unsqueeze(1) * page_size + torch.arange(
                page_size, device=page_loc.device
            )
            token_loc = expanded.reshape(-1).to(torch.int32)

            pos_page, pos_token = 0, 0
            out_cache_loc = torch.empty(
                extend_num_tokens, dtype=torch.int32, device="cpu"
            )
            for i in range(bs):
                assert (
                    prefix_lens[i] % page_size == 0
                ), f"Prefix length {prefix_lens[i]} is not page-aligned"
                start_page = prefix_lens[i] // page_size
                self.req_to_token_pool.req_to_page[
                    req_pool_indices[i], start_page : start_page + num_pages[i]
                ] = page_loc[pos_page : pos_page + num_pages[i]]
                self.req_to_token_pool.req_to_token[
                    req_pool_indices[i],
                    prefix_lens[i] : prefix_lens[i] + extend_lens[i],
                ] = token_loc[
                    pos_page * page_size : pos_page * page_size + extend_lens[i]
                ]
                out_cache_loc[pos_token : pos_token + extend_lens[i]] = token_loc[
                    pos_page * page_size : pos_page * page_size + extend_lens[i]
                ]
                pos_page += num_pages[i]
                pos_token += extend_lens[i]
            out_cache_loc = out_cache_loc.to(self.device, non_blocking=True)

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
        seq_lens_old = locs.tolist()
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
        else:
            page_size = self.page_allocator.page_size
            num_pages = sum(1 for l in seq_lens_old if l % page_size == 0)
            page_loc = self.page_allocator.alloc(num_pages)
            # Expand each page id to its token locations: [pid*page_size, (pid+1)*page_size)
            pos_page = 0
            out_cache_loc = []
            for i in range(bs):
                if seq_lens_old[i] % page_size == 0:
                    # new page
                    self.req_to_token_pool.req_to_page[
                        self.reqs[i].req_pool_idx, seq_lens_old[i] // page_size
                    ] = page_loc[pos_page]
                    out_cache_loc.append(page_loc[pos_page] * page_size)
                    pos_page += 1
                else:
                    out_cache_loc.append(
                        self.req_to_token_pool.req_to_token[
                            self.reqs[i].req_pool_idx, seq_lens_old[i] - 1
                        ]
                        + 1
                    )
            out_cache_loc = torch.tensor(out_cache_loc, dtype=torch.int32).to(
                self.device, non_blocking=True
            )

        # Write to req2token pool
        self.out_cache_loc = out_cache_loc
        self.req_to_token_pool.write((self.req_pool_indices, locs), self.out_cache_loc)
        self.positions = locs.to(self.device, non_blocking=True)
        self.input_ids = input_ids_tensor

    def merge_batch(self, other: BatchInfo):
        """
        Merge another batch into this batch.
        """
        assert (
            self.forward_mode == other.forward_mode
        ), "Cannot merge batches with different forward modes."

        self.reqs.extend(other.reqs)
        self.input_ids = torch.cat([self.input_ids, other.input_ids], dim=0)
        self.positions = torch.cat([self.positions, other.positions], dim=0)
        self.seq_lens = torch.cat([self.seq_lens, other.seq_lens], dim=0)
        self.input_seq_lens = torch.cat(
            [self.input_seq_lens, other.input_seq_lens], dim=0
        )
        self.prefix_seq_lens = torch.cat(
            [self.prefix_seq_lens, other.prefix_seq_lens], dim=0
        )
        self.req_pool_indices = torch.cat(
            [self.req_pool_indices, other.req_pool_indices], dim=0
        )
        self.out_cache_loc = torch.cat([self.out_cache_loc, other.out_cache_loc], dim=0)

    def filter_req(self, keep_indices: List[int]):
        """
        Filter the requests in the batch based on the given indices to keep.

        call this method before prepare_for_xxx!!
        """

        self.reqs = [self.reqs[i] for i in keep_indices]

        # Tensors need to be updated
        self.seq_lens = self.seq_lens[keep_indices]
