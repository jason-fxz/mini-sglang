import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import zmq.asyncio
from transformers import AutoTokenizer

from mini_sglang.managers.io_struct import (
    BatchStrOut,
    GenerateReqInput,
    TokenizedGenerateReqInput,
)
from mini_sglang.managers.sampling_params import SamplingParams
from mini_sglang.managers.server_args import PortArgs, ServerArgs
from mini_sglang.utils.utils import (
    TypeBasedDispatcher,
    configure_logger,
    get_zmq_socket,
)

logger = logging.getLogger(__name__)


@dataclass
class ReqState:
    out_list: List[Dict[Any, Any]]
    finished: bool
    event: asyncio.Event
    obj: GenerateReqInput

    # For metrics
    created_time: float
    finished_time: float = 0.0

    # For steaming output
    last_output_offset: int = 0
    text: str = ""


class TokenizerManager:
    def __init__(self, server_args: ServerArgs, port_args: PortArgs):
        self.server_args = server_args
        self.port_args = port_args

        # IPC init
        context = zmq.asyncio.Context(2)
        self.recv_from_detokenizer = get_zmq_socket(
            context, zmq.PULL, port_args.tokenizer_input_ipc, True
        )
        self.send_to_scheduler = get_zmq_socket(
            context, zmq.PUSH, port_args.scheduler_input_ipc, True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(server_args.model)

        self.rid_to_state: Dict[str, ReqState] = {}

        self._req_dispatcher = TypeBasedDispatcher(
            [(BatchStrOut, self._handle_batch_output)]
        )

        self.has_create_loop = False

    async def _tokenize_one_request(
        self, obj: GenerateReqInput
    ) -> TokenizedGenerateReqInput:

        if obj.input_ids is None:
            # Tokenize the input text
            input_ids = self.tokenizer.encode(obj.text)
        else:
            input_ids = obj.input_ids

        return TokenizedGenerateReqInput(
            rid=obj.rid,
            input_text=obj.text or input_ids,
            input_ids=input_ids,
            sampling_params=(
                SamplingParams.init_from_dict(obj.sampling_params)
                if obj.sampling_params
                else SamplingParams()
            ),
            stream=obj.stream,
        )

    def _send_one_request(
        self,
        obj: GenerateReqInput,
        tokenized_obj: TokenizedGenerateReqInput,
        created_time: Optional[float] = None,
    ):
        logger.debug(f"Sending request {obj.rid} to scheduler")
        self.send_to_scheduler.send_pyobj(tokenized_obj)
        state = ReqState([], False, asyncio.Event(), obj, created_time=created_time)
        self.rid_to_state[obj.rid] = state
        return state

    async def _wait_one_response(
        self,
        obj: GenerateReqInput,
        state: ReqState,
    ):
        """wait for the response from the detokenizer"""

        while True:
            try:
                await asyncio.wait_for(state.event.wait(), timeout=4)
            except asyncio.TimeoutError:
                continue

            out = state.out_list[-1]
            state.out_list = []

            if state.finished:
                # check abort
                finish_reason = out.get("meta_info", {}).get("finish_reason", None)
                if finish_reason.get("type") == "abort":
                    raise RuntimeError(
                        f"Request {obj.rid} aborted: {finish_reason.get('message', 'Unknown error')}"
                    )

                logger.debug(f"Request {obj.rid} finished")
                yield out
                break

            state.event.clear()

            if obj.stream:
                yield out
            # else:
            #     raise RuntimeError(
            #         f"Request {obj.rid} is not finished, but no more output received. "
            #         "This might be due to a bug in the detokenizer."
            #     )

    async def event_loop(self):
        "The event loop that handles requests from detokenizer"

        while True:
            recv_obj = await self.recv_from_detokenizer.recv_pyobj()
            self._req_dispatcher(recv_obj)

    def _handle_batch_output(self, recv_obj: BatchStrOut):
        for i, rid in enumerate(recv_obj.rids):
            state = self.rid_to_state.get(rid, None)
            assert state is not None, f"State for request {rid} not found"

            # meta_info for response
            meta_info = {
                "id": rid,
                "finish_reason": (
                    recv_obj.finished_reasons[i].to_json()
                    if recv_obj.finished_reasons[i]
                    else None
                ),
            }

            # append text
            state.last_output_offset = len(state.text)
            state.text += recv_obj.output_texts[i]
            out_dict = {
                "text": state.text,
                "meta_info": meta_info,
            }

            state.finished = recv_obj.finished_reasons[i] is not None
            if state.finished:
                state.finished_time = time.time()
                del self.rid_to_state[rid]

            state.out_list.append(out_dict)
            state.event.set()

    def auto_create_event_loop(self):
        configure_logger(self.server_args.log_level, prefix=" (tokenizer)")
        if self.has_create_loop:
            return
        self.has_create_loop = True
        # Create an event loop for handling requests
        loop = asyncio.get_event_loop()
        loop.create_task(self.event_loop())

    async def generate_request(self, obj: GenerateReqInput):
        self.auto_create_event_loop()
        created_time = time.time()

        tokenized_obj = await self._tokenize_one_request(obj)
        state = self._send_one_request(obj, tokenized_obj, created_time)

        async for response in self._wait_one_response(obj, state):
            yield response
