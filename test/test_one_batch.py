import logging
import os
import time

import pytest
from torch import distributed as dist
from transformers import AutoTokenizer

from mini_sglang.managers.batch_info import BatchInfo
from mini_sglang.managers.model_runner import ModelRunner
from mini_sglang.managers.req_info import Req
from mini_sglang.managers.sampling_params import SamplingParams
from mini_sglang.managers.server_args import ServerArgs
from mini_sglang.utils.model_config import ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module", autouse=True)
def setup_dist():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def print_batch_info(batch: BatchInfo):
    logger.info(
        "Forward Mode: %s\n"
        "Input IDs: %s\n"
        "Positions: %s\n"
        "Input Sequence Lengths: %s\n"
        "Full Sequence Lengths: %s\n"
        "Prefix Sequence Lengths: %s\n"
        "Number of Requests: %d\n"
        "Requests:\n%s",
        batch.forward_mode,
        batch.input_ids.tolist(),
        batch.positions.tolist(),
        batch.input_seq_lens.tolist() if batch.input_seq_lens is not None else "N/A",
        batch.seq_lens.tolist() if batch.seq_lens is not None else "N/A",
        batch.prefix_seq_lens.tolist() if batch.prefix_seq_lens is not None else "N/A",
        len(batch.reqs),
        "\n".join(
            f"  Request ID: {req.req_id}, Status: {req.status}, Tokens: {req.token_ids}"  # type: ignore
            for req in batch.reqs
        ),
    )


def test_one_batch():
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    server_args = ServerArgs(model=model_path)
    assert server_args.tp == 1, "Tensor parallelism is not supported in this script."

    model_config = ModelConfig(model_path)

    # Initialize the model runner
    model_runner = ModelRunner(
        model_config=model_config,
        server_args=server_args,
        gpu_id=0,
        tp_rank=0,
        tp_size=1,
    )
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_path)

    assert model_runner is not None
    assert tokenizer is not None

    sampling_params = SamplingParams(1.0, 64, eos_token_id=tokenizer.eos_token_id)

    # Prepare a batch of input data
    input_texts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Tell me a joke.",
    ]
    reqs = []
    for input_text in input_texts:
        input_ids = tokenizer.encode(input_text)
        req = Req(input_ids, sampling_params)
        reqs.append(req)

    batch = BatchInfo.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        page_allocator=model_runner.page_allocator,
        kv_cache_pool=model_runner.kv_cache_pool,
    )

    assert batch is not None

    batch.prepare_for_extend()
    logits = model_runner.forward_extend(batch)
    model_runner.process_extend_result(batch, logits)
    print_batch_info(batch)

    for i in range(10):
        batch.prepare_for_decode()
        logits = model_runner.forward_decode(batch)
        model_runner.process_decode_result(batch, logits)
        print_batch_info(batch)
    # Detokenize the output
    output_texts = []
    for req in batch.reqs:
        output_text = tokenizer.decode(req.token_ids, skip_special_tokens=True)
        output_texts.append(output_text)

    print("Output texts:")
    for text in output_texts:
        print(text)
