"""

"""

import logging
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch

from mini_sglang.layers.attention import Attention

logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024


class KVCachePool(ABC):
    @abstractmethod
    def __init__(
        self, size: int, page_size: int, dtype: torch.dtype, layer_num: int, device: str
    ):
        """Base class for Key-Value Cache Pool.
        Args:
            size (int): Total number of token slots in the cache.
            page_size (int): page size for the cache.
            dtype (torch.dtype): Data type for the cache tensors.
            head_num (int): Number of attention heads.
            head_dim (int): Dimension of each attention head.
            layer_num (int): Number of layers in the model.
            device (str): Device to store the tensors (e.g., 'cpu' or 'cuda').
        """
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.layer_num = layer_num
        self.device = device

        self.mem_usage = 0.0

    @abstractmethod
    def get_key_buffer(self, layer: int) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def get_value_buffer(self, layer: int) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def get_kv_buffer(self, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @abstractmethod
    def set_kv_buffer(
        self,
        layer: Attention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        """
        Set the key-value buffer for a specific layer.
        Args:
            layer (Attention): The attention layer for which the cache is being set.
            loc (torch.Tensor): The locations in the cache where the keys and values will be stored.
            cache_k (torch.Tensor): The keys to be cached.
            cache_v (torch.Tensor): The values to be cached.
        """
        raise NotImplementedError()


class MHAKVPool(KVCachePool):
    """
    Multi-Head Attention Key-Value Cache Pool.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
    ):
        super().__init__(size, page_size, dtype, layer_num, device)
        self.head_num = head_num
        self.head_dim = head_dim

        self._create_buffer()

        k_size, v_size = self.get_kv_size_bytes()
        logger.info(
            f"KV Cache is allocated. #page: {size} #token: {size * page_size}, K size: {k_size / GB:.2f} GB, V size: {v_size / GB:.2f} GB"
        )

        self.mem_usage = (k_size + v_size) / GB

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        k_size_bytes = 0
        for k_cache in self.k_buffer:
            k_size_bytes += np.prod(k_cache.shape) * k_cache.dtype.itemsize
        v_size_bytes = 0
        for v_cache in self.v_buffer:
            v_size_bytes += np.prod(v_cache.shape) * v_cache.dtype.itemsize
        return k_size_bytes, v_size_bytes

    def _create_buffer(self):
        # [size, head_num, head_dim] for each layer
        # slot 0 not used
        self.k_buffer = [
            torch.zeros(
                (self.size + self.page_size, self.head_num, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(self.layer_num)
        ]

        self.v_buffer = [
            torch.zeros(
                (self.size + self.page_size, self.head_num, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(self.layer_num)
        ]

    def set_kv_buffer(
        self,
        layer: Attention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        layer_id = layer.layer_id
        self.k_buffer[layer_id][loc] = cache_k.view(-1, self.head_num, self.head_dim)
        self.v_buffer[layer_id][loc] = cache_v.view(-1, self.head_num, self.head_dim)

    def get_value_buffer(self, layer_id: int):
        return self.v_buffer[layer_id]

    def get_key_buffer(self, layer_id: int):
        return self.k_buffer[layer_id]

    def get_kv_buffer(self, layer_id: int):
        return self.k_buffer[layer_id], self.v_buffer[layer_id]


class PageAllocator:
    """
    A simple page allocator that manages free slots in a memory pool.

    map page_id -> KVCachePool Location:
    [page_id * page_size, (page_id + 1) * page_size)


    """

    def __init__(
        self,
        page_num: int,
        page_size: int,
        device: str,
    ):
        """
        Args:
            page_num (int): Total number of pages in the memory pool.
            page_size (int): Size of each page.
            device (str): Device to store the memory pool.
        """
        self.page_num = page_num
        self.page_size = page_size
        self.device = device
        self.clear()

    def available_size(self):
        return len(self.free_page)

    def alloc(self, num: int):
        """
        Allocate a number of pages from the free pool.
        Args:
            num (int): Number of pages to allocate.
        Returns:
            torch.Tensor: The allocated page IDs.
        """
        if num > len(self.free_page):
            raise RuntimeError("Not enough free pages available.")

        select_index = self.free_page[:num]
        self.free_page = self.free_page[num:]
        return select_index

    def free(self, page_ids: torch.Tensor):
        """
        Free a number of pages back to the free pool.
        Args:
            page_ids (torch.Tensor): The page IDs to free.
        """
        self.free_page = torch.cat([self.free_page, page_ids], dim=0)

    def clear(self):
        self.free_page = torch.arange(
            1, self.page_num + 1, dtype=torch.int32, device=self.device
        )
