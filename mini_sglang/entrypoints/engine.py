import asyncio
import dataclasses
import logging
import multiprocessing as mp
import uuid
from typing import AsyncIterator, Dict, Iterator, List, Optional, Union

import setproctitle

from mini_sglang.managers.detokenizer_manager import run_detokenizer_process
from mini_sglang.managers.io_struct import GenerateReqInput
from mini_sglang.managers.scheduler import run_scheduler_process
from mini_sglang.managers.server_args import PortArgs, ServerArgs
from mini_sglang.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


class Engine:

    def __init__(self, server_args: ServerArgs):
        self.server_args = server_args
        self.port_args = PortArgs.init_new(server_args)
        logger.info(f"{server_args=}")

        # Launch subprocesses
        tokenizer_manager = launch_engine_subprocess(server_args, self.port_args)
        self.tokenizer_manager = tokenizer_manager

    def generate(
        self,
        prompt: Optional[str] = None,
        sampling_params: Optional[Dict] = None,
        # either input_ids or prompt
        input_ids: Optional[List[int]] = None,
        stream: bool = False,
    ) -> Union[Dict, Iterator[Dict]]:
        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            stream=stream,
            rid=uuid.uuid4().hex,
        )
        loop = asyncio.get_event_loop()
        generator = self.tokenizer_manager.generate_request(obj)

        if stream:

            def generator_wrapper():
                while True:
                    try:
                        chunk = loop.run_until_complete(generator.__anext__())
                        yield chunk
                    except StopAsyncIteration:
                        break

            return generator_wrapper()
        else:
            ret = loop.run_until_complete(generator.__anext__())
            return ret

    async def async_generate(
        self,
        prompt: Optional[str] = None,
        sampling_params: Optional[Dict] = None,
        # either input_ids or prompt
        input_ids: Optional[List[int]] = None,
        stream: bool = False,
    ) -> Union[Dict, AsyncIterator[Dict]]:
        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            stream=stream,
            rid=uuid.uuid4().hex,
        )
        generator = self.tokenizer_manager.generate_request(obj)

        if stream:
            return generator
        else:
            ret = await generator.__anext__()
            return ret

    def flush_cache(self):
        loop = asyncio.get_event_loop()
        ret = loop.run_until_complete(self.tokenizer_manager.flush_cache())
        return ret

    def get_server_info(self):
        loop = asyncio.get_event_loop()
        internal_states = loop.run_until_complete(
            self.tokenizer_manager.get_internal_state()
        ).internal_state
        return {
            **dataclasses.asdict(self.tokenizer_manager.server_args),
            "internal_states": [internal_states],
        }


def launch_engine_subprocess(server_args: ServerArgs, port_args: PortArgs):
    """
    Launch the engine subprocess.

    TokenizerManager in the main process.
    the Scheduler in a subprocess.
    the DetokenizerManager in another subprocess.
    """

    mp.set_start_method("spawn", force=True)

    scheduler_procs = []
    scheduler_pipe_readers = []
    for tp_rank in range(server_args.tp_size):
        reader, writer = mp.Pipe(duplex=False)
        gpu_id = tp_rank
        proc = mp.Process(
            target=run_scheduler_process,
            args=(server_args, port_args, gpu_id, tp_rank, writer),
            daemon=True,
        )

        proc.start()

        scheduler_procs.append(proc)
        scheduler_pipe_readers.append(reader)

    for reader in scheduler_pipe_readers:
        data = reader.recv()
        assert (
            data["status"] == "ok"
        ), f"Scheduler process failed to start: {data['message']}"

    detokenizer_proc = mp.Process(
        target=run_detokenizer_process, args=(server_args, port_args), daemon=True
    )
    detokenizer_proc.start()

    tokenizer_manager = TokenizerManager(server_args, port_args)

    return tokenizer_manager
