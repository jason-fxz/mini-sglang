from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mini_sglang.managers.req_info import BaseFinishReason
from mini_sglang.managers.sampling_params import SamplingParams


@dataclass
class TokenizedGenerateReqInput:
    # The request id
    rid: str
    # The input text
    input_text: str
    # The input token ids
    input_ids: List[int]
    # The sampling parameters
    sampling_params: SamplingParams
    # Whether to stream the output
    stream: bool = False


@dataclass
class FlushCacheReqInput:
    pass


@dataclass
class BatchTokenIDOut:
    # The request ids
    rids: List[str]
    # The finish reason
    finished_reasons: List[BaseFinishReason]
    # The output token ids
    output_ids: List[int]


@dataclass
class GenerateReqInput:
    # The input prompt.
    text: Optional[str] = None
    # The token ids for text; one can specify either text or input_ids
    input_ids: Optional[List[int]] = None
    # The sampling parameters for the request
    sampling_params: Optional[Dict] = None

    # The request id
    rid: Optional[str] = None
    # Whether to stream the output
    stream: bool = False


@dataclass
class BatchStrOut:
    # The request ids
    rids: List[str]
    # The finish reason
    finished_reasons: List[BaseFinishReason]
    # The output texts
    output_texts: List[str]
