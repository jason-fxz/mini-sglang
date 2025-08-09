from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Optional

import torch

from mini_sglang.layers.attention import Attention
from mini_sglang.layers.attn.attn_backend import AttentionBackend
from mini_sglang.managers.batch_info import BatchInfo
from mini_sglang.managers.model_runner import ModelRunner

try:
    flash_attn_version = version("flash-attn")
    assert flash_attn_version.startswith("2."), "flash-attn version must be 2.x.x"
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except PackageNotFoundError:
    raise RuntimeError("flash-attn package is required for FA2AttentionBackend")


class FlashAttn2Metadata:
    """Metadata which will be created once during model forward and reused across layers forward."""

    # cu_seqlens_q = [0, input_seq_len_1, input_seq_len_1 + input_seq_len_2, ...]
    # cu_seqlens_k = [0, seq_len_1, seq_len_1 + seq_len_2, ...]
    cache_seqlens_int32: torch.Tensor = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    page_table: torch.Tensor = None


class FlashAttn2Backend(AttentionBackend):

    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata: FlashAttn2Metadata = None
        self.device = model_runner.device
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.kv_cache_pool = model_runner.kv_cache_pool
        self.page_size = model_runner.page_size

    def init_forward_metadata(self, batch: BatchInfo):
        metadata = FlashAttn2Metadata()
        batch_size = batch.batch_size
        seqlens_in_batch = batch.seq_lens
        device = seqlens_in_batch.device

        # Get sequence lengths for key/values
        metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
        metadata.max_seqlen_k = batch.seq_lens.max().item()
        metadata.cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
        )

        # FA2 supports only page_size % 256 == 0
        if self.page_size % 256 == 0:
            max_num_page = (
                metadata.max_seqlen_k + self.page_size - 1
            ) // self.page_size
            metadata.page_table = self.req_to_token_pool.req_to_page[
                batch.req_pool_indices, :max_num_page
            ]
        else:
            raise RuntimeError(
                f"Unsupported page size {self.page_size}, must be a multiple of 256."
            )

        if batch.forward_mode.is_extend():
            # Get sequence lengths for query
            metadata.max_seqlen_q = batch.input_seq_lens.max().item()
            metadata.cu_seqlens_q = torch.nn.functional.pad(
                torch.cumsum(batch.input_seq_lens, dim=0, dtype=torch.int32), (1, 0)
            )
        elif batch.forward_mode.is_decode():
            # For decode, query length is always 1
            metadata.max_seqlen_q = 1
            metadata.cu_seqlens_q = torch.arange(
                0, batch_size + 1, dtype=torch.int32, device=device
            )

        self.forward_metadata = metadata

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: Attention,
        batch: BatchInfo,
        save_kv_cache: bool = True,
    ):
        # save kv cache
        cache_loc = batch.out_cache_loc
        if save_kv_cache:
            self.kv_cache_pool.set_kv_buffer(layer, cache_loc, k, v)

        metadata = self.forward_metadata

        k_cache, v_cache = self.kv_cache_pool.get_kv_buffer(layer.layer_id)
        o = flash_attn_varlen_func(
            q=q.contiguous().view(-1, layer.num_heads, layer.head_dim),
            k=k_cache.view(-1, self.page_size, layer.num_kv_heads, layer.head_dim),
            v=v_cache.view(-1, self.page_size, layer.num_kv_heads, layer.head_dim),
            block_table=metadata.page_table,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k=metadata.cu_seqlens_k,
            max_seqlen_q=metadata.max_seqlen_q,
            max_seqlen_k=metadata.max_seqlen_k,
            softmax_scale=layer.scale,
            causal=True,
        )

        return o.view(-1, layer.num_heads * layer.head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: Attention,
        batch: BatchInfo,
        save_kv_cache: bool = True,
    ):
        # save kv cache
        cache_loc = batch.out_cache_loc
        if save_kv_cache:
            self.kv_cache_pool.set_kv_buffer(layer, cache_loc, k, v)

        metadata = self.forward_metadata

        k_cache, v_cache = self.kv_cache_pool.get_kv_buffer(layer.layer_id)
        o = flash_attn_with_kvcache(
            q=q.contiguous().view(-1, layer.num_heads, layer.head_dim).unsqueeze(1),
            k_cache=k_cache.view(
                -1, self.page_size, layer.num_kv_heads, layer.head_dim
            ),
            v_cache=v_cache.view(
                -1, self.page_size, layer.num_kv_heads, layer.head_dim
            ),
            block_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens_int32,
            softmax_scale=layer.scale,
            causal=True,
        )

        return o.view(-1, layer.num_heads * layer.head_dim)
