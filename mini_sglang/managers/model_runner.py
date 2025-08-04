from typing import Optional

import torch.distributed as dist

from mini_sglang.managers.server_args import ServerArgs
from mini_sglang.mem_cache.req2token import ReqToTokenPool
from mini_sglang.mem_cache.token2kv import KVCachePool, PageAllocator
from mini_sglang.utils.model_config import ModelConfig


class ModelRunner:
    def __init__(
        self,
        model_config: ModelConfig,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        req_to_token_pool: ReqToTokenPool,
        page_allocator: PageAllocator,
        kv_cache_pool: KVCachePool,
    ):
        self.device = server_args.device
        self.gpu_id = gpu_id

        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.page_size = server_args.page_size
        self.model_config = model_config

    def _calc_max_num_token(self, total_gpu_memory: int):
        # cell_size = ()
        pass

    def init_memory_pool(
        self,
        total_gpu_memory: int,
        max_num_reqs: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
    ):
        max_kv_cache_size = (
            self.model_config.hidden_size
            * self.model_config.num_heads
            * self.model_config.num_hidden_layers
        )

        self.page_allocator = PageAllocator(
            page_num=total_gpu_memory // self.page_size,
            page_size=self.page_size,
            device=self.device,
        )

        self.req_to_token_pool = ReqToTokenPool(
            size=max_num_reqs,
            max_tokens=max_total_tokens,
            page_size=self.page_size,
            device=self.device,
        )

        # self.kv_cache_pool = KVCachePool(
        #     size=
        # )
