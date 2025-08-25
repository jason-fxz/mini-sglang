"""
Radix Tree Cache Implementation
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Optional, Tuple

import torch

from mini_sglang.managers.req_info import Req
from mini_sglang.mem_cache.base_cache import BasePrefixCache, MatchResult
from mini_sglang.mem_cache.req2token import ReqToTokenPool
from mini_sglang.mem_cache.token2kv import KVCachePool, PageAllocator


class TreeNode:
    id_counter = 0
    time_counter = 0

    @staticmethod
    def time_tick():
        TreeNode.time_counter += 1
        return TreeNode.time_counter

    def __init__(self, id: Optional[int] = None):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: List[int] = None
        self.value: Optional[torch.Tensor] = None
        self.lock_ref = 0
        self.last_access_time = TreeNode.time_tick()

        self.id = TreeNode.id_counter if id is None else id
        TreeNode.id_counter += 1

    def __lt__(self, other: TreeNode):
        return self.last_access_time < other.last_access_time

    def update_access_time(self):
        self.last_access_time = TreeNode.time_tick()


class RadixCache(BasePrefixCache):
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

        if self.kv_cache_pool:
            self.device = self.kv_cache_pool.device
        else:
            self.device = torch.device("cpu")

        self.reset()

    def reset(self):
        self.root = TreeNode()
        self.root.key = []
        self.root.value = []
        self.root.lock_ref = 1
        self.evictable_size = 0  # size of unlocked nodes
        self.protected_size = 0  # size of locked nodes

    def _get_child_key(self, key: List[int]) -> Tuple[int]:
        return tuple(key[: self.page_size])

    def _match_key(self, keya: List[int], keyb: List[int]) -> int:
        min_len = min(len(keya), len(keyb))
        i = 0
        while (
            i < min_len and keya[i : i + self.page_size] == keyb[i : i + self.page_size]
        ):
            i += self.page_size
        return i

    def _split_node(self, child: TreeNode, key: List[int], split_pos: int) -> TreeNode:
        """
        split node into two nodes at split_pos
        new_node -> child
        """
        new_node = TreeNode()
        new_node.children = {self._get_child_key(key[split_pos:]): child}
        new_node.parent = child.parent
        new_node.key = child.key[:split_pos]
        new_node.value = child.value[:split_pos]
        new_node.lock_ref = child.lock_ref
        child.parent = new_node
        child.key = child.key[split_pos:]
        child.value = child.value[split_pos:]
        new_node.parent.children[self._get_child_key(key)] = new_node

        return new_node

    def match_prefix(self, key: List[int], **kwargs) -> MatchResult:
        """Find the matching prefix from the radix tree.
        Args:
            key: A list of token IDs to find a matching prefix.
        Returns:
            A tuple of a tensor of matching prefix token IDs and
            the last node that contains the prefix values. Note that
            this API can modify the internal state of the Radix tree.
            The last node create a new child if the prefix is shorter
            than the last node's value.
        """

        if len(key) == 0:
            return MatchResult(
                match_indices=torch.empty((0,), dtype=torch.int32, device=self.device),
                last_node=self.root,
            )

        # cut to page size
        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        # traverse the tree
        node = self.root
        node.update_access_time()
        values = []
        child_key = self._get_child_key(key)

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.update_access_time()
            prefix_len = self._match_key(key, child.key)
            if prefix_len < len(child.key):
                # split the child node
                new_node = self._split_node(child, child.key, prefix_len)
                values.append(new_node.value)
                node = new_node
                break
            else:
                values.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key) > 0:
                    child_key = self._get_child_key(key)

        if values:
            value = torch.cat(values)
        else:
            value = torch.empty((0,), dtype=torch.int32, device=self.device)

        return MatchResult(match_indices=value, last_node=node)

    def insert(self, key: List[int], value=None):
        """
        insert a new key-value pair into the radix tree
        return matched prefix length (dividable by page_size)
        """
        if len(key) == 0:
            return 0

        if value is None:
            value = [x for x in key]

        # traverse the tree
        node = self.root
        node.update_access_time()
        child_key = self._get_child_key(key)
        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            # print("key", key, child_key)
            node = node.children[child_key]
            node.update_access_time()
            prefix_len = self._match_key(key, node.key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]
            # print("prefix_len", prefix_len, len(node.key), len(key))
            if prefix_len < len(node.key):
                new_node = self._split_node(node, node.key, prefix_len)
                node = new_node

            if len(key) > 0:
                child_key = self._get_child_key(key)

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[child_key] = new_node
            # print("node:", node.key, "add child:", new_node.key)

        assert total_prefix_length % self.page_size == 0
        return total_prefix_length

    def inc_lock_ref(self, node: TreeNode):
        """increment the lock reference count from the node to the root"""
        if node is None:
            return

        while node != self.root:
            if node.lock_ref == 0:
                self.evictable_size -= len(node.value)
                self.protected_size += len(node.value)
            node.lock_ref += 1
            node = node.parent

    def dec_lock_ref(self, node: TreeNode):
        """decrement the lock reference count from the node to the root"""
        if node is None:
            return

        while node != self.root:
            node.lock_ref -= 1
            if node.lock_ref == 0:
                self.evictable_size += len(node.value)
                self.protected_size -= len(node.value)
            node = node.parent

    def _print_tree(self, node: TreeNode, depth: int = 0):
        print("  " * depth, len(node.key), node.key[:10], f"r={node.lock_ref}")
        for key, child in node.children.items():
            self._print_tree(child, depth + 1)
            assert key == self._get_child_key(
                child.key
            ), f"{key} vs {self._get_child_key(child.key)}"

    def pretty_print(self):
        """print the radix tree structure"""
        self._print_tree(self.root)

    def cache_unfinished_req(self, req: Req):
        """Cache the unfinished request into the radix tree."""
        page_size = self.page_size
        tokens_len = len(req.token_ids) - 1  # exclude the last generated token
        page_aligned_len = tokens_len // page_size * page_size

        kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, :tokens_len]

        # get page aligned key & value
        aligned_token_ids = req.token_ids[:page_aligned_len]  # List[int]
        aligned_kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :page_aligned_len
        ].to(dtype=torch.int32, copy=True)

        # remove prefix that is already cached
        new_prefix_len = self.insert(aligned_token_ids, aligned_kv_indices)
        old_prefix_len_aligned = len(req.prefix_indices) // page_size * page_size
        self.page_allocator.free(
            self.req_to_token_pool.req_to_page[
                req.req_pool_idx,
                old_prefix_len_aligned // page_size : new_prefix_len // page_size,
            ]
        )  # len(req.prefix_indices) may not divide by page_size, we drop last incomplete page

        # update prefix indices
        new_prefix_indices, new_last_node = self.match_prefix(aligned_token_ids)
        self.req_to_token_pool.req_to_token[
            req.req_pool_idx, old_prefix_len_aligned : len(new_prefix_indices)
        ] = new_prefix_indices[old_prefix_len_aligned:]

        if page_size > 1:
            assert len(new_prefix_indices) % page_size == 0

            stride_indices = torch.arange(
                old_prefix_len_aligned,
                len(new_prefix_indices),
                page_size,
                device=new_prefix_indices.device,
            )

            self.req_to_token_pool.req_to_page[
                req.req_pool_idx,
                old_prefix_len_aligned
                // page_size : len(new_prefix_indices)
                // page_size,
            ] = (
                new_prefix_indices[stride_indices] // page_size
            )

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        if page_size == 1:
            req.prefix_indices = new_prefix_indices
        else:
            req.prefix_indices = torch.cat(
                [new_prefix_indices, kv_indices[len(new_prefix_indices) :]]
            )
        req.last_node = new_last_node

    def cache_finished_req(self, req: Req):
        """Cache the finished request into the radix tree."""
        page_size = self.page_size
        tokens_len = len(req.token_ids) - 1  # exclude the last generated token
        page_aligned_len = tokens_len // page_size * page_size

        kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, :tokens_len]

        # get page aligned key & value
        aligned_token_ids = req.token_ids[:page_aligned_len]  # List[int]
        aligned_kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :page_aligned_len
        ].to(dtype=torch.int32, copy=True)

        # remove prefix that is already cached
        new_prefix_len = self.insert(aligned_token_ids, aligned_kv_indices)
        old_prefix_len_aligned = len(req.prefix_indices) // page_size * page_size
        self.page_allocator.free(
            self.req_to_token_pool.req_to_page[
                req.req_pool_idx,
                old_prefix_len_aligned // page_size : new_prefix_len // page_size,
            ]
        )  # len(req.prefix_indices) may not divide by page_size, we drop last incomplete page

        # for page_size > 1, release the last incomplete page
        if page_size > 1 and tokens_len % page_size != 0:
            self.page_allocator.free(
                self.req_to_token_pool.req_to_page[
                    req.req_pool_idx,
                    tokens_len // page_size : (tokens_len // page_size) + 1,
                ]
            )

        # release lock & slot
        self.req_to_token_pool.free(req.req_pool_idx)
        self.dec_lock_ref(req.last_node)


if __name__ == "__main__":
    tree = RadixCache(None, None, None, 1)
    tree.insert("Hello")
    tree.insert("He")
    tree.insert("Hello World!")

    tree.insert("sglang is good")
    tree.pretty_print()
