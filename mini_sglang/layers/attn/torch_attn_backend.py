from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from torch.nn.functional import scaled_dot_product_attention

from mini_sglang.layers.attention import Attention
from mini_sglang.layers.attn.attn_backend import AttentionBackend
from mini_sglang.managers.batch_info import BatchInfo
from mini_sglang.managers.model_runner import ModelRunner


class TorchNativeAttnBackend(AttentionBackend):

    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.kv_cache_pool = model_runner.kv_cache_pool

    def init_forward_metadata(self, batch: BatchInfo):
        # No metadata needed for torch native attention
        pass

    # NOTE: COPY FROM https://github.com/sgl-project/sglang/blob/888cb175a6a8a24b4ffe07ee0e1ace1bda8ea850/python/sglang/srt/layers/attention/torch_native_backend.py
    def _run_sdpa_forward_extend(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        """Run the extend forward by using torch native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            extend_prefix_lens: [num_seqs]
            extend_seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.

            extend_seq_len_q = extend_seq_lens[seq_idx]
            prefill_seq_len_q = extend_prefix_lens[seq_idx]

            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + extend_seq_len_q
            end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]
            per_req_query_redudant = torch.empty(
                (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
                dtype=per_req_query.dtype,
                device=per_req_query.device,
            )

            per_req_query_redudant[:, prefill_seq_len_q:, :] = per_req_query

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            per_req_out_redudant = (
                scaled_dot_product_attention(
                    per_req_query_redudant.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=causal,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out_redudant[prefill_seq_len_q:, :, :]
            start_q, start_kv = end_q, end_kv
        return output

    # NOTE: COPY FROM https://github.com/sgl-project/sglang/blob/888cb175a6a8a24b4ffe07ee0e1ace1bda8ea850/python/sglang/srt/layers/attention/torch_native_backend.py
    def _run_sdpa_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        """Run the decode forward by using torch native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.

            seq_len_q = 1
            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + seq_len_q
            end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            per_req_out = (
                scaled_dot_product_attention(
                    per_req_query.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=causal,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out
            start_q, start_kv = end_q, end_kv

        return output

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: Attention,
        batch: BatchInfo,
        save_kv_cache: bool = True,
    ):
        o = torch.empty_like(q)

        if save_kv_cache:
            self.kv_cache_pool.set_kv_buffer(layer, batch.out_cache_loc, k, v)

        use_gqa = layer.num_heads != layer.num_kv_heads

        q_ = q.view(-1, layer.num_heads, layer.head_dim)
        o_ = o.view(-1, layer.num_heads, layer.head_dim)

        self._run_sdpa_forward_extend(
            q_,
            o_,
            self.kv_cache_pool.get_key_buffer(layer.layer_id),
            self.kv_cache_pool.get_value_buffer(layer.layer_id),
            self.req_to_token_pool.req_to_token,
            batch.req_pool_indices,
            batch.seq_lens,
            batch.prefix_seq_lens,
            batch.input_seq_lens,
            scaling=layer.scale,
            enable_gqa=use_gqa,
            causal=True,
        )

        return o

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: Attention,
        batch: BatchInfo,
        save_kv_cache: bool = True,
    ):
        o = torch.empty_like(q)

        if save_kv_cache:
            self.kv_cache_pool.set_kv_buffer(layer, batch.out_cache_loc, k, v)

        use_gqa = layer.num_heads != layer.num_kv_heads

        q_ = q.view(-1, layer.num_heads, layer.head_dim)
        o_ = o.view(-1, layer.num_heads, layer.head_dim)

        self._run_sdpa_forward_decode(
            q_,
            o_,
            self.kv_cache_pool.get_key_buffer(layer.layer_id),
            self.kv_cache_pool.get_value_buffer(layer.layer_id),
            self.req_to_token_pool.req_to_token,
            batch.req_pool_indices,
            batch.seq_lens,
            scaling=layer.scale,
            enable_gqa=use_gqa,
            causal=True,
        )

        return o
