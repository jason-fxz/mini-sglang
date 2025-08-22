from copy import copy
from enum import Enum, auto
from itertools import count
from typing import List, Union

import torch

from mini_sglang.managers.sampling_params import SamplingParams
from mini_sglang.managers.server_args import ServerArgs


class BaseFinishReason:
    def __init__(self, is_error: bool = False):
        self.is_error = is_error

    def to_json(self):
        raise NotImplementedError()


class FINISH_MATCHED_TOKEN(BaseFinishReason):
    def __init__(self, matched: Union[int, List[int]]):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


class FINISH_LENGTH(BaseFinishReason):
    def __init__(self, length: int):
        super().__init__()
        self.length = length

    def to_json(self):
        return {
            "type": "length",  # to match OpenAI API's return value
            "length": self.length,
        }


class FINISH_ABORT(BaseFinishReason):
    def __init__(self, message=None, status_code=None, err_type=None):
        super().__init__(is_error=True)
        self.message = message or "Aborted"
        self.status_code = status_code
        self.err_type = err_type

    def to_json(self):
        return {
            "type": "abort",
            "message": self.message,
            "status_code": self.status_code,
            "err_type": self.err_type,
        }


class ReqStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Req:
    counter = count()

    def __init__(
        self,
        rid: str,
        token_ids: List[int],
        sampling_params: SamplingParams,
        req_pool_idx: int = -1,
    ):
        self.rid = rid
        self.req_id = next(Req.counter)
        self.status = ReqStatus.WAITING
        self.token_ids = copy(token_ids)  # upd
        self.max_tokens = len(token_ids) + sampling_params.max_new_tokens
        self.num_prompt_tokens = len(token_ids)
        self.ignore_eos = sampling_params.ignore_eos
        self.sampling_params = sampling_params
        self.req_pool_idx = req_pool_idx

        self.finish_reason = None  # upd

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

    def check_finished(self):
        if self.num_tokens >= self.max_tokens:
            self.status = ReqStatus.FINISHED
            self.finish_reason = FINISH_LENGTH(self.num_tokens)
        elif (
            self.last_token_id == self.sampling_params.eos_token_id
            and not self.ignore_eos
        ):
            self.status = ReqStatus.FINISHED
            self.finish_reason = FINISH_MATCHED_TOKEN(self.last_token_id)

        return self.is_finished
