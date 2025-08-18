from __future__ import annotations

import random
from enum import Enum
from typing import List, Optional, Union

from mini_sglang.managers.req_info import Req
from mini_sglang.mem_cache.base_cache import BasePrefixCache


class CachePolicy(Enum):
    # Cache-aware policies
    LPM = "lpm"  # longest prefix match
    DFS_WEIGHT = "dfs-weight"  # depth-first search weighting
    # Non-cache-aware policies
    FCFS = "fcfs"  # first come first serve
    LOF = "lof"  # longest output first
    RANDOM = "random"


class SchedulerPolicy:

    def __init__(
        self,
        policy: str,
        tree_cache: Optional[BasePrefixCache] = None,
    ):
        self.policy = CachePolicy(policy)
        self.tree_cache = tree_cache

    def _sort_by_lof(self, waiting_queue: List[Req]) -> List[Req]:
        waiting_queue.sort(key=lambda req: -req.sampling_params.max_new_tokens)

    def _sort_by_random(self, waiting_queue: List[Req]) -> List[Req]:
        random.shuffle(waiting_queue)

    def calc_priority(self, waiting_queue: List[Req]):
        if self.policy == CachePolicy.FCFS:
            return
        if self.policy == CachePolicy.LOF:
            self._sort_by_lof(waiting_queue)
            return
        if self.policy == CachePolicy.RANDOM:
            self._sort_by_random(waiting_queue)
            return

        # Cache-aware policies
        assert (
            self.tree_cache is not None
        ), "Tree cache must be provided for cache-aware policies"
