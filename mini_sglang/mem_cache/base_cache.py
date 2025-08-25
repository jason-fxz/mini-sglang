from abc import ABC, abstractmethod
from typing import Any, List, NamedTuple

import torch

from mini_sglang.managers.req_info import Req


class MatchResult(NamedTuple):
    match_indices: torch.Tensor
    last_node: Any


class BasePrefixCache(ABC):
    """Base class for prefix cache management."""

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def match_prefix(self, key: List[int], **kwargs) -> MatchResult:
        pass

    @abstractmethod
    def cache_finished_req(self, req: Req, **kwargs):
        pass

    @abstractmethod
    def cache_unfinished_req(self, req: Req, **kwargs):
        pass
