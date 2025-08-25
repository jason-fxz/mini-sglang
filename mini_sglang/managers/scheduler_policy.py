from __future__ import annotations

import random
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional, Set

import torch

from mini_sglang.managers.req_info import Req
from mini_sglang.mem_cache.base_cache import BasePrefixCache
from mini_sglang.mem_cache.radix_cache import RadixCache, TreeNode

IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD = int(32)
IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD = int(32)


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

        # used for finding prefix matches for in-batch prefix caching
        self.waiting_queue_radix_tree = RadixCache(
            req_to_token_pool=None, kv_cache_pool=None, page_allocator=None, page_size=1
        )

    def _sort_by_lof(self, waiting_queue: List[Req]):
        waiting_queue.sort(key=lambda req: -req.sampling_params.max_new_tokens)

    def _sort_by_random(self, waiting_queue: List[Req]):
        random.shuffle(waiting_queue)

    def _sort_by_lpm(self, waiting_queue: List[Req], temporary_deprioritized: Set[str]):
        """Sort by longest prefix match, deprioritizing requests that have many in-batch prefix matches."""
        waiting_queue.sort(
            key=lambda req: (
                -len(req.prefix_indices)
                if req.rid not in temporary_deprioritized
                else 1e10 - len(req.prefix_indices)
            )
        )

    def _sort_by_dfs_weight(
        self, waiting_queue: List[Req], tree_cache: BasePrefixCache
    ):
        """Sort by depth-first search weighting."""
        node_to_reqs = defaultdict(list)
        for req in waiting_queue:
            node_to_reqs[req.prefix_node].append(req)
        node_to_weight = defaultdict(int)
        for node in node_to_reqs.keys():
            node_to_weight[node] = len(node_to_reqs[node])

        self._dfs_calc_weight(tree_cache.root, node_to_weight)

        waiting_queue.clear()
        self._dfs_sort(tree_cache.root, node_to_weight, node_to_reqs, waiting_queue)

    def _dfs_calc_weight(self, node: TreeNode, node_to_weight: Dict[TreeNode, int]):
        for child in node.children.values():
            self._dfs_calc_weight(child, node_to_weight)
            node_to_weight[node] += node_to_weight[child]

    def _dfs_sort(
        self,
        node: TreeNode,
        node_to_weight: Dict[TreeNode, int],
        node_to_reqs: Dict[TreeNode, List[Req]],
        sorted_reqs: List[Req],
    ):
        children = list(node.children.values())
        children.sort(key=lambda child: -node_to_weight[child])
        for child in children:
            self._dfs_sort(child, node_to_weight, node_to_reqs, sorted_reqs)
        sorted_reqs.extend(node_to_reqs.get(node, []))

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

        temporary_deprioritized = self._compute_prefix_matches(waiting_queue)
        if self.policy == CachePolicy.LPM:
            self._sort_by_lpm(
                waiting_queue=waiting_queue,
                temporary_deprioritized=temporary_deprioritized,
            )
            return
        if self.policy == CachePolicy.DFS_WEIGHT:
            self._sort_by_dfs_weight(
                waiting_queue=waiting_queue, tree_cache=self.tree_cache
            )
            return

    def _compute_prefix_matches(self, waiting_queue: List[Req]) -> Set[str]:
        temporary_deprioritized = set()
        self.waiting_queue_radix_tree.reset()

        for req in waiting_queue:
            req.calc_prefix(self.tree_cache)

            if len(req.prefix_indices) <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD:
                in_batch_match = self.waiting_queue_radix_tree.match_prefix(
                    key=req.token_ids
                )

                if (
                    len(in_batch_match.match_indices)
                    >= IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD
                ):
                    temporary_deprioritized.add(req.rid)
                else:
                    self.waiting_queue_radix_tree.insert(
                        key=req.token_ids,
                        value=torch.empty(len(req.token_ids), dtype=torch.bool),
                    )

        return temporary_deprioritized
