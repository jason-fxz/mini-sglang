import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import setproctitle
import zmq
from transformers import AutoTokenizer

from mini_sglang.managers.io_struct import BatchStrOut, BatchTokenIDOut
from mini_sglang.managers.server_args import PortArgs, ServerArgs
from mini_sglang.utils.utils import (
    TypeBasedDispatcher,
    configure_logger,
    get_zmq_socket,
    is_printable_text,
)

logger = logging.getLogger(__name__)


@dataclass
class DecodeState:
    """
    Store the state of steaming decoding.
    """

    decoded_text: str
    decode_ids: List[int]

    # The offset of text that has been sent to the tokenizer.
    send_offset: int
    # The offset of decode_ids that has been decoded.
    read_offset: int

    # decode_ids | x x x x x x x x x x | x x x n
    #                                  ^
    #                             read_offset
    #  Wait until the new character is printable. => new_text = decoded_text[read_offset: ]
    #


class DetokenizerManager:

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        # Init IPC
        context = zmq.Context(2)
        self.recv_from_scheduler = get_zmq_socket(
            context, zmq.PULL, port_args.detokenizer_input_ipc, True
        )
        self.send_to_tokenizer = get_zmq_socket(
            context, zmq.PUSH, port_args.tokenizer_input_ipc, False
        )

        self.tokenizer = AutoTokenizer.from_pretrained(server_args.model)

        self._req_dispatcher = TypeBasedDispatcher(
            [(BatchTokenIDOut, self._handle_batch_token_id_out)]
        )

        self.decode_stats: Dict[str, DecodeState] = {}

    def event_loop(self):
        """
        The event loop that handles requests from the scheduler.
        """
        while True:
            recv_obj = self.recv_from_scheduler.recv_pyobj()
            output = self._req_dispatcher(recv_obj)
            self.send_to_tokenizer.send_pyobj(output)

    def _handle_batch_token_id_out(self, obj: BatchTokenIDOut) -> BatchStrOut:
        """
        Convert a batch of token IDs to strings.
        """
        # logger.debug(f"Detokenizing batch of size {len(obj.rids)}")
        bs = len(obj.rids)
        output_texts = []
        for i in range(bs):
            rid = obj.rids[i]
            if rid not in self.decode_stats:
                state = DecodeState(
                    decoded_text="", decode_ids=[], send_offset=0, read_offset=0
                )
                self.decode_stats[rid] = state
            else:
                state = self.decode_stats[rid]

            state.decode_ids.append(obj.output_ids[i])

            # Decode the token IDs to text
            new_text = self.tokenizer.decode(state.decode_ids[state.read_offset :])

            if obj.finished_reasons[i] is not None or is_printable_text(new_text):
                # If the new text is printable or the request is finished, we can decode it.
                state.decoded_text += new_text
                state.read_offset = len(state.decode_ids)

            output_texts.append(state.decoded_text[state.send_offset :])
            state.send_offset = len(state.decoded_text)

        return BatchStrOut(
            rids=obj.rids,
            finished_reasons=obj.finished_reasons,
            output_texts=output_texts,
        )


def run_detokenizer_process(server_args: ServerArgs, port_args: PortArgs):
    """
    Run the detokenizer manager.
    """
    setproctitle.setproctitle("mini-sglang::detokenizer")
    configure_logger(server_args.log_level, prefix=" (detokenizer)")

    detokenizer_manager = DetokenizerManager(server_args, port_args)
    detokenizer_manager.event_loop()
