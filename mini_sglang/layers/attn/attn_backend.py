from abc import ABC, abstractmethod

import torch

from mini_sglang.layers.attention import Attention
from mini_sglang.managers.batch_info import BatchInfo


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
            return self.forward_prefill(
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

    def forward_prefill(
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
