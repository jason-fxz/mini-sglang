from typing import List, Optional, Tuple

import torch
import zmq

from mini_sglang.managers.batch_info import BatchInfo, ForwardMode
from mini_sglang.managers.io_struct import (
    BatchTokenIDOut,
    FlushCacheReqInput,
    TokenizedGenerateReqInput,
)
from mini_sglang.managers.model_runner import ModelRunner
from mini_sglang.managers.req_info import Req
from mini_sglang.managers.scheduler_policy import SchedulerPolicy
from mini_sglang.managers.server_args import PortArgs, ServerArgs
from mini_sglang.mem_cache.req2token import ReqToTokenPool
from mini_sglang.mem_cache.token2kv import KVCachePool, MHAKVPool, PageAllocator
from mini_sglang.utils.model_config import ModelConfig
from mini_sglang.utils.utils import TypeBasedDispatcher, get_zmq_socket


class Scheduler:
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
    ):
        self.server_args = server_args
        self.port_args = port_args
        self.tp_rank = tp_rank
        self.tp_size = server_args.tp
        self.gpu_id = gpu_id

        self.page_size = server_args.page_size
        self.device = server_args.device

        self.forward_ct = 0

        # IPC init
        context = zmq.asyncio.Context(2)
        self.recv_from_tokenizer = get_zmq_socket(
            context, zmq.PULL, port_args.scheduler_input_ipc, True
        )
        self.send_to_detokenizer = get_zmq_socket(
            context, zmq.PUSH, port_args.detokenizer_input_ipc, True
        )

        # init memory pool
        self.model_config = ModelConfig(server_args.model)
        self.init_memory_pool(
            server_args.max_num_reqs, self.model_config.max_context_len
        )

        # init model runner
        self.model_runner = ModelRunner(
            model_config=self.model_config,
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            tp_size=self.tp_size,
            req_to_token_pool=self.req_to_token_pool,
            page_allocator=self.page_allocator,
            kv_cache_pool=self.kv_cache_pool,
        )

        # init waiting queue
        self.waiting_queue: List[Req] = []
        # init running batch
        self.running_batch: BatchInfo = BatchInfo(reqs=[])
        # current forward batch
        self.cur_batch: Optional[BatchInfo] = None
        # last processed batch
        self.last_batch: Optional[BatchInfo] = None

        # init request dispatcher
        self._req_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.handle_generate_request),
                (FlushCacheReqInput, self.handle_flush_cache),
            ]
        )

        # scheduler policy
        self.scheduler_policy = server_args.scheduler_policy
        self.policy = SchedulerPolicy(self.scheduler_policy)

    def calc_max_num_token(self) -> Tuple[int, int]:
        free_size, total_size = torch.cuda.mem_get_info(
            self.gpu_id
        )  # free memory in bytes
        # used_size = total_size - free_size
        used_size = torch.cuda.memory_allocated(self.gpu_id)
        num_kv_heads = self.model_config.num_kv_heads // self.tp_size

        cell_size = (
            num_kv_heads
            * self.model_config.head_dim
            * self.model_config.num_hidden_layers
            * 2  # k + v
            * self.model_config.kv_cache_dtype.itemsize
        )
        block_size = self.page_size * cell_size

        num_kvcache_pages = (
            int(total_size * self.server_args.gpu_memory_utilization - used_size)
            // block_size
        )

        num_kvcache_tokens = num_kvcache_pages * self.page_size

        return num_kvcache_pages, num_kvcache_tokens

    def init_memory_pool(
        self,
        max_num_reqs: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
    ):
        num_pages, num_tokens = self.calc_max_num_token()
        self.num_pages = num_pages
        self.num_tokens = num_tokens

        self.page_allocator = PageAllocator(
            page_num=self.num_pages,
            page_size=self.page_size,
            device=self.device,
        )

        self.req_to_token_pool = ReqToTokenPool(
            size=max_num_reqs,
            max_tokens=max_total_tokens,
            page_size=self.page_size,
            device=self.device,
        )

        self.kv_cache_pool = MHAKVPool(
            size=self.num_tokens,
            page_size=self.page_size,
            dtype=self.model_config.kv_cache_dtype,
            head_num=self.model_config.num_kv_heads,
            head_dim=self.model_config.head_dim,
            layer_num=self.model_config.num_hidden_layers,
            device=self.device,
        )

    def handle_generate_request(self, recv_req: TokenizedGenerateReqInput):
        # handle new request
        req = Req(
            rid=recv_req.rid,
            token_ids=recv_req.input_ids,
            sampling_params=recv_req.sampling_params,
        )
        self.waiting_queue.append(req)

    def handle_flush_cache(self, req: FlushCacheReqInput):
        # TODO
        pass

    def recv_requests(self) -> List:
        """
        Receive requests from the tokenizer.
        """
        recv_reqs = []
        while True:
            try:
                recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
            except zmq.ZMQError:
                break
            recv_reqs.append(recv_req)

        return recv_reqs

    def process_input_requests(self, recv_reqs: List):
        """
        process received requests using dispatcher
        """
        for recv_req in recv_reqs:
            self._req_dispatcher(recv_req)

    def get_new_batch_prefill(self) -> BatchInfo:
        self.policy.calc_priority(self.waiting_queue)

        can_run_reqs: List[Req] = []

        for req in self.waiting_queue:
            # TODO schedule policy
            if self.running_batch >= self.server_args.max_running_bs:
                break

            can_run_reqs.append(req)

        # update waiting queue
        if len(can_run_reqs) == 0:
            return None

        self.waiting_queue = [
            x for x in self.waiting_queue if x not in set(can_run_reqs)
        ]

        new_batch = BatchInfo.init_new(
            can_run_reqs,
            self.req_to_token_pool,
            self.page_allocator,
            self.kv_cache_pool,
        )

        new_batch.prepare_for_extend()
        return new_batch

    def update_running_batch(self, batch: BatchInfo):
        batch.prepare_for_decode()
        return batch

    def get_next_batch_to_run(self):
        # merge last prefill batch with current running batch
        if self.last_batch is not None:
            if self.running_batch.is_empty():
                self.running_batch = self.last_batch
            else:
                self.running_batch.merge(self.last_batch)

        new_batch = self.get_new_batch_prefill()

        if new_batch is not None:
            # Run prefill first
            ret = new_batch
        else:
            # Run decode
            if self.running_batch.is_empty():
                ret = None
            else:
                self.running_batch = self.update_running_batch(self.running_batch)
                ret = self.running_batch if not self.running_batch.is_empty() else None

        return ret

    def run_batch(self, batch: BatchInfo):
        self.forward_ct += 1

        return self.model_runner.forward_generate(batch)

    def process_batch_result(self, batch: BatchInfo, result: any):
        logits = result

        temperatures = torch.tensor(
            [req.sampling_params.temperature for req in batch.reqs],
            dtype=torch.float32,
            device=self.device,
        )
        # sampling
        output_ids = self.sampler(logits.next_token_logits, temperatures)

        # Update the batch info with the output ids
        for i, req in enumerate(batch.reqs):
            req.token_ids.append(output_ids[i].item())
            req.prefix_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx][
                : len(req) - 1
            ]

        # check finish req
        finished_reqs_indices = []
        for i, req in enumerate(batch.reqs):
            if req.check_finished():
                finished_reqs_indices.append(i)

        # send to detokenizer
        batch_out = BatchTokenIDOut(
            rids=[req.rid for req in batch.reqs],
            finished_reasons=[req.finish_reason for req in batch.reqs],
            output_ids=[req.last_token_id for req in batch.reqs],
        )

        self.send_to_detokenizer.send_pyobj(batch_out)

        # remove finished reqs
        batch.filter_reqs(finished_reqs_indices)

    def event_loop_normal(self):
        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                pass

            self.last_batch = batch
