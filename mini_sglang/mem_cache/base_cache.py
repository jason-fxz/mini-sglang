from abc import ABC, abstractmethod
from typing import List, Tuple

import torch

from mini_sglang.managers.req_info import Req


class BasePrefixCache(ABC):
    """Base class for prefix cache management."""

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def match_prefix(self, key: List[int], **kwargs):
        pass

    @abstractmethod
    def cache_finished_req(self, req: Req, **kwargs):
        pass

    @abstractmethod
    def cache_unfinished_req(self, req: Req, **kwargs):
        pass
