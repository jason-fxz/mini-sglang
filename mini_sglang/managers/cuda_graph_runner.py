from __future__ import annotations

import bisect
from typing import TYPE_CHECKING, Callable, Dict, Optional

import torch
import torch.distributed as dist
import tqdm

from mini_sglang.layers.logits_processor import LogitsProcessorOutput
from mini_sglang.managers.batch_info import BatchInfo, ForwardMode
from mini_sglang.managers.server_args import ServerArgs
from mini_sglang.utils.model_config import ModelConfig
from mini_sglang.utils.utils import get_available_gpu_memory

if TYPE_CHECKING:
    from mini_sglang.managers.model_runner import ModelRunner

import logging

logger = logging.getLogger(__name__)

_global_graph_memory_pool = None


class CudaGraphRunner:
    ...

    def __init__(
        self,
        model_runner: ModelRunner,
    ):
        self.model_runner = model_runner
        self.capture_cudagraph()

    def capture_one_batch_size(self, bs: int, forward: Callable):
        graph = torch.cuda.CUDAGraph()

        batch = BatchInfo(
            forward_mode=ForwardMode.DECODE,
            input_ids=self.input_ids[:bs],
            positions=self.positions[:bs],
            seq_lens=self.seq_lens[:bs],
            out_cache_loc=self.out_cache_loc[:bs],
            req_pool_indices=self.req_pool_indices[:bs],
            req_to_token_pool=self.model_runner.req_to_token_pool,
            kv_cache_pool=self.model_runner.kv_cache_pool,
            page_allocator=self.model_runner.page_allocator,
            attn_backend=self.model_runner.attn_backend,
        )

        # init attn metadata for this batch size
        self.model_runner.attn_backend.init_forward_metadata_capture_cuda_graph(
            bs, batch.forward_mode
        )

        # warm up
        for _ in range(2):
            torch.cuda.synchronize()
            outputs = forward(batch.input_ids, batch.positions, batch)

        global _global_graph_memory_pool
        with torch.cuda.graph(
            graph, _global_graph_memory_pool, self.model_runner.stream
        ):
            assert self.model_runner.stream == torch.cuda.current_stream()
            outputs = forward(batch.input_ids, batch.positions, batch)
            self.next_token_logits[:bs] = outputs.next_token_logits
            outputs = LogitsProcessorOutput(self.next_token_logits[:bs])

        _global_graph_memory_pool = graph.pool()

        return graph, outputs

    def get_capture_batch_sizes(self, max_bs: int):
        if max_bs < 4:
            graph_bs = [1, 2, 3][:max_bs]
        elif max_bs < 8:
            graph_bs = [1, 2, 4]
            if max_bs != 4:
                graph_bs.append(max_bs)
        else:
            graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))

        return graph_bs

    @torch.inference_mode()
    def capture_cudagraph(self):
        max_bs = min(
            self.model_runner.server_args.max_num_reqs,
            self.model_runner.server_args.max_capture_bs,
        )
        vocab_size = self.model_runner.model_config.vocab_size
        # init graph_vars
        with torch.device("cuda"):
            # normal vars inputs
            self.input_ids = torch.zeros(max_bs, dtype=torch.int32)
            self.positions = torch.zeros(max_bs, dtype=torch.int32)
            self.out_cache_loc = torch.zeros(max_bs, dtype=torch.int32)
            self.seq_lens = torch.ones(max_bs, dtype=torch.int32)
            self.req_pool_indices = torch.zeros(max_bs, dtype=torch.int32)
            # output vars
            self.next_token_logits = torch.zeros(
                max_bs, vocab_size, dtype=torch.float32
            )

            # attn metadata vars inputs
            self.model_runner.attn_backend.init_cuda_graph_state(max_bs)

        self.graph_bs = self.get_capture_batch_sizes(max_bs)
        self.graphs = {}
        self.graph_pool = None
        self.output_buffers: Dict[int, LogitsProcessorOutput] = {}

        logger.info(f"Start capturing cuda graphs for batch sizes: {self.graph_bs}")

        capture_bs_range = (
            tqdm.tqdm(list(reversed(self.graph_bs)))
            if dist.get_rank() == 0
            else reversed(self.graph_bs)
        )

        for bs in capture_bs_range:
            if dist.get_rank() == 0:
                avail_mem = get_available_gpu_memory(self.model_runner.gpu_id, False)

                capture_bs_range.set_description(
                    f"Capture cuda graph {bs=}, {avail_mem=:.2f} GB"
                )

            graph, outputs = self.capture_one_batch_size(
                bs, self.model_runner.model.forward
            )

            self.graphs[bs] = graph
            self.output_buffers[bs] = outputs

            torch.cuda.synchronize()

    def replay(self, batch: BatchInfo) -> LogitsProcessorOutput:
        # prepare inputs
        raw_bs = batch.batch_size
        index = bisect.bisect_left(self.graph_bs, raw_bs)
        bs = self.graph_bs[index]
        if bs != raw_bs:
            self.seq_lens.fill_(1)
            self.out_cache_loc.zero_()

        # common inputs
        self.input_ids[:raw_bs].copy_(batch.input_ids)
        self.positions[:raw_bs].copy_(batch.positions)
        self.seq_lens[:raw_bs].copy_(batch.seq_lens)
        self.out_cache_loc[:raw_bs].copy_(batch.out_cache_loc)
        self.req_pool_indices[:raw_bs].copy_(batch.req_pool_indices)

        # output

        # attn metadata inputs
        self.model_runner.attn_backend.init_forward_metadata_replay_cuda_graph(
            bs,
            req_pool_indices=self.req_pool_indices,
            seq_lens=self.seq_lens,
            seq_lens_cpu=batch.seq_lens_cpu,
            forward_mode=batch.forward_mode,
        )

        # replay
        self.graphs[bs].replay()

        # trim output if needed
        output = self.output_buffers[bs]
        return LogitsProcessorOutput(
            next_token_logits=output.next_token_logits[:raw_bs]
        )

    def can_run(self, batch: BatchInfo) -> bool:
        return batch.forward_mode.is_decode() and batch.batch_size <= self.graph_bs[-1]
