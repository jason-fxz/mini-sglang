"""Launch the inference server."""

import os
import sys

from mini_sglang.entrypoints.http_server import launch_server
from mini_sglang.managers.server_args import prepare_server_args

if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])

    launch_server(server_args)
