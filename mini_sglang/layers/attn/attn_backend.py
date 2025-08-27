from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from mini_sglang.layers.attention import Attention
from mini_sglang.managers.batch_info import BatchInfo, ForwardMode


class AttentionBackend(ABC):
    """The base class of attention backends"""

    @abstractmethod
    def init_forward_metadata(self, batch: BatchInfo):
        """Init the metadata for a forward pass."""
        raise NotImplementedError()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: Attention,
        batch: BatchInfo,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Run forward on an attention layer."""
        if batch.forward_mode.is_decode():
            return self.forward_decode(
                q,
                k,
                v,
                layer,
                batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
        else:
            return self.forward_extend(
                q,
                k,
                v,
                layer,
                batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: Attention,
        batch: BatchInfo,
        save_kv_cache: bool = True,
    ):
        """Run a forward for decode."""
        raise NotImplementedError()

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: Attention,
        batch: BatchInfo,
        save_kv_cache: bool = True,
    ):
        """Run a forward for extend."""
        raise NotImplementedError()

    def init_cuda_graph_state(self, max_bs: int):
        """Init the global shared states for cuda graph."""
        raise NotImplementedError()

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        forward_mode: ForwardMode,
    ):
        """Init the metadata for a forward pass for capturing a cuda graph."""
        raise NotImplementedError()

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
    ):
        """Init the metadata for a forward pass for replaying a cuda graph."""
        raise NotImplementedError()
