from copy import copy
from enum import Enum, auto
from itertools import count
from typing import List

import torch

from mini_sglang.managers.sampling_params import SamplingParams
from mini_sglang.managers.server_args import ServerArgs


class ReqStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Req:
    counter = count()

    def __init__(
        self,
        token_ids: List[int],
        sampling_params: SamplingParams,
        req_pool_idx: int = -1,
    ):
        self.req_id = next(Req.counter)
        self.status = ReqStatus.WAITING
        self.token_ids = copy(token_ids)  # upd
        self.max_tokens = sampling_params.max_tokens
        self.num_prompt_tokens = len(token_ids)
        self.ignore_eos = sampling_params.ignore_eos
        self.sampling_params = sampling_params
        self.req_pool_idx = req_pool_idx

        # prefix info
        # The indices to kv cache for shared prefix
        self.prefix_indices: torch.tensor = torch.tensor([], dtype=torch.int32)  # upd

    @property
    def last_token_id(self) -> int:
        return self.token_ids[-1] if self.token_ids else None

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, idx):
        return self.token_ids[idx]

    @property
    def is_finished(self):
        return self.status == ReqStatus.FINISHED
