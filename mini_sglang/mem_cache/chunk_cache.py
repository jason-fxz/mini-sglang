"""
Chunk Cache
"""

from mini_sglang.managers.req_info import Req
from mini_sglang.mem_cache.base_cache import BasePrefixCache
from mini_sglang.mem_cache.req2token import ReqToTokenPool
from mini_sglang.mem_cache.token2kv import KVCachePool, PageAllocator


class ChunkCache(BasePrefixCache):
    """Chunk Cache for managing memory chunks."""

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        kv_cache_pool: KVCachePool,
        page_allocator: PageAllocator,
        page_size: int,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.kv_cache_pool = kv_cache_pool
        self.page_allocator = page_allocator
        self.page_size = page_size

    def reset(self):
        pass

    def match_prefix(self, **unused_kwargs):
        return None

    def cache_finished_req(self, req: Req):
        page_ids = self.req_to_token_pool.req_to_page[
            req.req_pool_idx,
            : (req.num_tokens - 1 + self.page_size - 1) // self.page_size,
        ]
        self.req_to_token_pool.free(req.req_pool_idx)
        self.page_allocator.free(page_ids)

    def cache_unfinished_req(self, req: Req):
        req.prefix_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx][
            : len(req) - 1
        ]
