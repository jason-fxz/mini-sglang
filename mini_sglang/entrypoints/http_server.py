import asyncio
import logging
import multiprocessing as mp
import uuid
from typing import AsyncIterator, Dict, Iterator, List, Optional, Union

import setproctitle

from mini_sglang.entrypoints.engine import launch_engine_subprocess
from mini_sglang.managers.detokenizer_manager import run_detokenizer_process
from mini_sglang.managers.io_struct import GenerateReqInput
from mini_sglang.managers.scheduler import run_scheduler_process
from mini_sglang.managers.server_args import PortArgs, ServerArgs
from mini_sglang.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)
import dataclasses
from http import HTTPStatus

import orjson
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, Response, StreamingResponse


@dataclasses.dataclass
class _GlobalState:
    tokenizer_manager: TokenizerManager


_global_state: Optional[_GlobalState] = None


def set_global_state(global_state: _GlobalState):
    global _global_state
    _global_state = global_state


# FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def set_uvicorn_logging_configs():
    from uvicorn.config import LOGGING_CONFIG

    LOGGING_CONFIG["formatters"]["default"][
        "fmt"
    ] = "[%(asctime)s] %(levelprefix)s %(message)s"
    LOGGING_CONFIG["formatters"]["default"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
    LOGGING_CONFIG["formatters"]["access"][
        "fmt"
    ] = '[%(asctime)s] %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
    LOGGING_CONFIG["formatters"]["access"]["datefmt"] = "%Y-%m-%d %H:%M:%S"


def launch_server(
    server_args: ServerArgs,
):
    server_args = server_args
    port_args = PortArgs.init_new(server_args)

    # Launch subprocesses
    tokenizer_manager = launch_engine_subprocess(server_args, port_args)

    set_global_state(_GlobalState(tokenizer_manager=tokenizer_manager))

    # Update uvicorn logging format
    set_uvicorn_logging_configs()
    app.server_args = server_args

    # Listen for requests
    uvicorn.run(
        app,
        host=server_args.host,
        port=server_args.port,
        log_level=server_args.log_level,
        timeout_keep_alive=5,
        loop="uvloop",
    )


def _create_error_response(e):
    return ORJSONResponse(
        {"error": {"message": str(e)}}, status_code=HTTPStatus.BAD_REQUEST
    )


@app.get("/health")
async def health() -> Response:
    """Check the health of the http server."""
    return Response(status_code=200)


@app.get("/get_model_info")
async def get_model_info():
    """Get the model information."""
    result = {"model_path": _global_state.tokenizer_manager.model_path}
    return result


@app.get("/get_server_info")
async def get_server_info():
    return {**dataclasses.asdict(_global_state.tokenizer_manager.server_args)}


@app.api_route("/generate", methods=["POST", "PUT"])
async def generate_request(obj: GenerateReqInput, request: Request):

    if obj.stream:

        async def stream_results() -> AsyncIterator[bytes]:
            try:
                async for out in _global_state.tokenizer_manager.generate_request(
                    obj, request
                ):
                    yield b"data: " + orjson.dumps(
                        out, option=orjson.OPT_NON_STR_KEYS
                    ) + b"\n\n"
            except ValueError as e:
                out = {"error": {"message": str(e)}}
                logger.error(f"[http_server] Error: {e}")
                yield b"data: " + orjson.dumps(
                    out, option=orjson.OPT_NON_STR_KEYS
                ) + b"\n\n"
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
            background=_global_state.tokenizer_manager.create_abort_task(obj),
        )
    else:
        try:
            ret = await _global_state.tokenizer_manager.generate_request(
                obj, request
            ).__anext__()
            return ret
        except ValueError as e:
            logger.error(f"[http_server] Error: {e}")
            return _create_error_response(e)
