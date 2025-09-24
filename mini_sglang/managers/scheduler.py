import logging
import os
import time
import traceback
from types import SimpleNamespace
from typing import List, Optional, Tuple

import setproctitle
import torch
import zmq
from sgl_kernel import spatial
from torch import distributed as dist

from mini_sglang.layers.sampler import Sampler
from mini_sglang.managers.batch_info import BatchInfo, ForwardMode
from mini_sglang.managers.io_struct import (
    AbortReq,
    BatchTokenIDOut,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    GetInternalStateReqInput,
    GetInternalStateReqOutput,
    TokenizedGenerateReqInput,
)
from mini_sglang.managers.model_runner import ModelRunner
from mini_sglang.managers.req_info import Req
from mini_sglang.managers.scheduler_policy import (
    AddReqResult,
    PrefillAdder,
    SchedulerPolicy,
)
from mini_sglang.managers.server_args import PortArgs, ServerArgs
from mini_sglang.mem_cache.chunk_cache import ChunkCache
from mini_sglang.mem_cache.radix_cache import RadixCache
from mini_sglang.mem_cache.req2token import ReqToTokenPool
from mini_sglang.mem_cache.token2kv import KVCachePool, MHAKVPool, PageAllocator
from mini_sglang.utils.global_vars import global_vars
from mini_sglang.utils.model_config import ModelConfig
from mini_sglang.utils.profiler import SafeProfiler
from mini_sglang.utils.utils import (
    TypeBasedDispatcher,
    broadcast_pyobj,
    configure_logger,
    get_zmq_socket,
    set_random_seed,
)

logger = logging.getLogger(__name__)


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
        self.tp_size = server_args.tp_size
        self.gpu_id = gpu_id

        self.page_size = server_args.page_size
        self.device = server_args.device

        # Log some info
        self.forward_ct = 0
        self.decode_forward_ct = 0
        self.last_tps = 0.0
        self.total_retracted_reqs = 0

        torch.cuda.set_device(gpu_id)

        # set random seed
        set_random_seed(server_args.random_seed)

        # IPC init
        if self.tp_rank == 0:
            context = zmq.Context(2)
            self.recv_from_tokenizer = get_zmq_socket(
                context, zmq.PULL, port_args.scheduler_input_ipc, False
            )
            self.send_to_detokenizer = get_zmq_socket(
                context, zmq.PUSH, port_args.detokenizer_input_ipc, False
            )
            self.send_to_tokenizer = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_input_ipc, False
            )
        else:
            self.send_to_tokenizer = SimpleNamespace(send_pyobj=lambda x: None)
            self.send_to_detokenizer = SimpleNamespace(send_pyobj=lambda x: None)

        # init dist group
        self.init_dist_group(tp_rank, self.tp_size)

        # init green ctx
        sm_counts = spatial.get_sm_available(gpu_id)
        self.streams_group = spatial.create_greenctx_stream_by_value(
            sm_counts // 2, sm_counts - (sm_counts // 2), gpu_id
        )

        # init model runner
        self.model_config = ModelConfig(server_args.model)
        self.model_runner = ModelRunner(
            model_config=self.model_config,
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            tp_size=self.tp_size,
            stream=self.streams_group[0],
        )
        self.max_total_num_tokens = self.model_runner.num_tokens

        self.req_to_token_pool = self.model_runner.req_to_token_pool
        self.page_allocator = self.model_runner.page_allocator
        self.kv_cache_pool = self.model_runner.kv_cache_pool
        self.init_tree_cache()

        self.sampler = Sampler()

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
                (AbortReq, self.handle_abort_request),
                (GetInternalStateReqInput, self.handle_get_internal_state),
            ]
        )

        # scheduler policy
        self.init_schedule_policy()

    def init_schedule_policy(self):
        self.scheduler_policy = self.server_args.schedule_policy
        self.policy = SchedulerPolicy(self.scheduler_policy, self.tree_cache)

        # runtime constraints for schedule
        assert (
            self.server_args.schedule_conservativeness >= 0
        ), "Invalid schedule_conservativeness"
        self.init_new_token_ratio = min(
            global_vars.default_init_new_token_ratio
            * self.server_args.schedule_conservativeness,
            1.0,
        )
        self.min_new_token_ratio = min(
            self.init_new_token_ratio * global_vars.default_min_new_token_ratio_factor,
            1.0,
        )
        self.new_token_ratio_decay = (
            self.init_new_token_ratio - self.min_new_token_ratio
        ) / global_vars.default_new_token_ratio_decay_steps
        self.new_token_ratio = self.init_new_token_ratio

    def init_dist_group(self, tp_rank: int, tp_size: int):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(self.port_args.nccl_port)
        dist.init_process_group(backend="nccl", rank=tp_rank, world_size=tp_size)
        self.tp_cpu_group = dist.new_group(
            backend="gloo", ranks=[i for i in range(tp_size)]
        )

    def init_tree_cache(self):
        # init radix cache
        if self.server_args.disable_radix_cache:
            self.tree_cache = ChunkCache(
                req_to_token_pool=self.req_to_token_pool,
                page_allocator=self.page_allocator,
                kv_cache_pool=self.kv_cache_pool,
                page_size=self.page_size,
            )
        else:
            self.tree_cache = RadixCache(
                req_to_token_pool=self.req_to_token_pool,
                page_allocator=self.page_allocator,
                kv_cache_pool=self.kv_cache_pool,
                page_size=self.page_size,
            )

    def handle_get_internal_state(self, req: GetInternalStateReqInput):
        ret = {}
        ret["memory_usage"] = {
            "weight": round(self.model_runner.weight_load_mem_usage, 2),
            "kvcache": round(self.kv_cache_pool.mem_usage, 2),
            "cuda_graph": round(self.model_runner.cuda_graph_mem_usage, 2),
            "token_capacity": self.max_total_num_tokens,
        }
        ret["last_gen_throughput"] = self.last_tps
        return GetInternalStateReqOutput(ret)

    def handle_generate_request(self, recv_req: TokenizedGenerateReqInput):
        # handle new request
        req = Req(
            rid=recv_req.rid,
            token_ids=recv_req.input_ids,
            sampling_params=recv_req.sampling_params,
        )
        self.waiting_queue.append(req)

    def handle_abort_request(self, recv_req: AbortReq):
        # handle abort request
        to_del = []
        for i, req in enumerate(self.waiting_queue):
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                to_del.append(i)

        # Sort in reverse order to avoid index issues when deleting
        for i in reversed(to_del):
            # Abort method 1: directly pop from the queue
            # This only works for requests that have not started anything.
            # We still need to send something back to TokenizerManager to clean up the state.
            req = self.waiting_queue.pop(i)
            self.send_to_tokenizer.send_pyobj(AbortReq(req.rid))
            logger.debug(f"Abort queued request. {req.rid=}")

        # Delete requests in the running batch
        if self.cur_batch is self.running_batch or self.cur_batch is None:
            reqs = self.running_batch.reqs
        else:
            reqs = self.running_batch.reqs + self.cur_batch.reqs

        for req in reqs:
            if not req.is_finished and (
                recv_req.abort_all or req.rid.startswith(recv_req.rid)
            ):
                # Abort method 2: set `to_abort=True`
                # The request will still run one decode forward pass.
                # Then we reuse all existing code to clean up the KV cache allocation.
                logger.debug(f"Abort running request. {req.rid=}")
                req.to_abort = True

    def handle_flush_cache(self, req: FlushCacheReqInput):
        if len(self.waiting_queue) == 0 and self.running_batch.is_empty():
            self.cur_batch = None
            self.last_batch = None
            self.tree_cache.reset()

            self.req_to_token_pool.clear()
            self.page_allocator.clear()
            torch.cuda.empty_cache()
            self.forward_ct = 0
            self.decode_forward_ct = 0

            logger.info("Flush cache successfully.")
            if_success = True
        else:
            logger.warning(
                "Cannot flush cache when there are running or queued requests."
            )
            if_success = False
        return FlushCacheReqOutput(success=if_success)

    def check_memory(self):
        available_size = self.page_allocator.available_size()
        evictable_size = self.tree_cache.evictable_size()
        protected_size = self.tree_cache.protected_size()

        if protected_size != 0:
            msg = f"tree_cache protected_size should be 0 When there is no running req, {protected_size=}"
            raise RuntimeError(msg)

        if available_size + evictable_size != self.max_total_num_tokens:
            msg = f"page_allocator memory leak detected! {self.max_total_num_tokens=}, {available_size=}, {evictable_size=}, {protected_size=}"
            raise RuntimeError(msg)

        if len(self.req_to_token_pool.free_slots) != self.req_to_token_pool.size:
            msg = f"req_to_token_pool memory leak detected! total_size={self.req_to_token_pool.size}, available_size={len(self.req_to_token_pool.free_slots)}"
            raise RuntimeError(msg)

    def recv_requests(self) -> List:
        """
        Receive requests from the tokenizer.
        """
        recv_reqs = []
        if self.tp_rank == 0:
            while True:
                try:
                    recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                recv_reqs.append(recv_req)
        else:
            recv_reqs = None

        if self.tp_size > 1:
            recv_reqs = broadcast_pyobj(
                recv_reqs, self.tp_rank, self.tp_cpu_group, src=0
            )

        if len(recv_reqs) > 0:
            logger.debug(f"Received {len(recv_reqs)} requests from tokenizer.")
        return recv_reqs

    def process_input_requests(self, recv_reqs: List):
        """
        process received requests using dispatcher
        """
        for recv_req in recv_reqs:
            output = self._req_dispatcher(recv_req)
            if output is not None:
                self.send_to_tokenizer.send_pyobj(output)

    def get_new_batch_prefill(self) -> BatchInfo:
        self.policy.calc_priority(self.waiting_queue)

        adder = PrefillAdder(
            page_size=self.page_size,
            tree_cache=self.tree_cache,
            page_allocator=self.page_allocator,
            new_token_ratio=self.new_token_ratio,
            max_input_tokens=self.server_args.max_prefill_tokens,
            running_batch=self.running_batch,
        )
        for req in self.waiting_queue:
            if (
                len(adder.can_run_reqs) + self.running_batch.batch_size
                >= self.server_args.max_num_reqs
            ):
                break

            if not self.server_args.disable_radix_cache:
                req.calc_prefix(self.tree_cache)

            res = adder.add_one_req(req)
            if res == AddReqResult.NO_TOKEN:
                break

        # update waiting queue
        if len(adder.can_run_reqs) == 0:
            return None

        self.waiting_queue = [
            x for x in self.waiting_queue if x not in set(adder.can_run_reqs)
        ]

        new_batch = BatchInfo.init_new(
            adder.can_run_reqs,
            self.req_to_token_pool,
            self.page_allocator,
            self.kv_cache_pool,
            self.tree_cache,
        )

        for req in new_batch.reqs:
            req.num_cached_tokens = len(req.prefix_indices)

        new_batch.prepare_for_extend()
        return new_batch

    def update_running_batch(self, batch: BatchInfo):
        # check if decode OOM
        if not batch.check_decode_mem():
            old_ratio = self.new_token_ratio

            retracted_reqs, new_ratio = batch.retract_decode()
            self.new_token_ratio = new_ratio

            logger.info(
                f"KV cache is full. Retract {len(retracted_reqs)} requests. "
                f"#new_token_ratio: {old_ratio:.4f} -> {new_ratio:.4f}"
            )

            self.waiting_queue.extend(retracted_reqs)
            self.total_retracted_reqs = len(retracted_reqs)

        else:
            self.new_token_ratio = max(
                self.new_token_ratio - self.new_token_ratio_decay,
                self.min_new_token_ratio,
            )

        batch.prepare_for_decode()
        return batch

    def get_next_batch_to_run(self):
        # merge last prefill batch with current running batch
        if self.last_batch and self.last_batch.forward_mode.is_extend():
            if self.running_batch.is_empty():
                self.running_batch = self.last_batch
            else:
                self.running_batch.merge_batch(self.last_batch)

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
        logits, run_cuda_graph = result

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

        # print batch info
        self.print_batch(batch, run_cuda_graph)

        # check finish req
        finished_reqs_indices = []
        for i, req in enumerate(batch.reqs):
            if req.check_finished():
                finished_reqs_indices.append(i)
                self.tree_cache.cache_finished_req(req)
            elif batch.forward_mode.is_extend():
                # decode mode will not update tree cache
                self.tree_cache.cache_unfinished_req(req)

        # send to detokenizer
        if self.tp_rank == 0:
            batch_out = BatchTokenIDOut(
                rids=[req.rid for req in batch.reqs],
                finished_reasons=[req.finish_reason for req in batch.reqs],
                output_ids=[req.last_token_id for req in batch.reqs],
                prompt_tokens=[req.num_prompt_tokens for req in batch.reqs],
                completion_tokens=[req.num_completion_tokens for req in batch.reqs],
                cached_tokens=[req.num_cached_tokens for req in batch.reqs],
            )

            self.send_to_detokenizer.send_pyobj(batch_out)

        # remove finished reqs
        batch.filter_reqs(finished_reqs_indices)

    def _get_token_info(self):
        available_size = self.page_allocator.available_size()
        evictable_size = self.tree_cache.evictable_size()
        num_used = self.max_total_num_tokens - (available_size + evictable_size)
        token_usage = num_used / self.max_total_num_tokens
        return num_used, token_usage, available_size, evictable_size

    def print_batch(self, batch: BatchInfo, run_cuda_graph: bool = False):
        if self.tp_rank != 0:
            return

        if batch.forward_mode.is_extend():
            self.decode_forward_ct = 0
            self.decode_token_ct = 0
            self.decode_start_time = time.time()

            extend_token_lens = 0
            prefix_token_lens = 0
            for req in batch.reqs:
                extend_token_lens += len(req.token_ids) - len(req.prefix_indices)
                prefix_token_lens += len(req.prefix_indices)

            logger.info(
                f"Extend #BS: {batch.batch_size}  #Tokens: {extend_token_lens + prefix_token_lens}  #PrefixTokens: {prefix_token_lens}  #ExtendTokens: {extend_token_lens}  #waiting-queue: {len(self.waiting_queue)}"
            )
        else:
            self.decode_forward_ct += 1
            self.decode_token_ct += batch.batch_size
            if self.decode_forward_ct % 64 == 1:
                end_time = time.time()
                elapsed = end_time - self.decode_start_time
                tps = self.decode_token_ct / elapsed if elapsed > 0 else 0
                tokens, token_usage, available_size, evictable_size = (
                    self._get_token_info()
                )
                self.decode_start_time = end_time
                self.decode_token_ct = 0
                self.last_tps = tps

                logger.info(
                    f"Decode #BS: {batch.batch_size}  #Tokens: {tokens} {available_size} {evictable_size} ({token_usage:.2f})  #TPS(token/s): {tps:.2f}  #CUDA-Graph: {run_cuda_graph}  #waiting-queue: {len(self.waiting_queue)}  #new-token-ratio: {self.new_token_ratio:.3f}"
                )

        # self.tree_cache.pretty_print(use_logger=True)

    @torch.inference_mode()
    def event_loop_normal(self, prof: torch.profiler.profile = None):
        logger.info("Scheduler event loop started.")
        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch:
                with self.streams_group[0]:
                    result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                self.check_memory()

            self.last_batch = batch


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    pipe_writer,
):
    """
    Run the scheduler manager.
    """
    prefix = ""
    if server_args.tp_size > 1:
        prefix = f" TP{tp_rank}"

    setproctitle.setproctitle(f"mini-sglang::scheduler{prefix}")

    configure_logger(server_args.log_level, prefix=prefix)

    # fix random seed
    torch.manual_seed(server_args.random_seed)

    try:
        scheduler = Scheduler(
            server_args=server_args,
            port_args=port_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
        )
        pipe_writer.send(
            {
                "status": "ok",
                "message": f"Scheduler process {prefix} started successfully.",
            }
        )

        if server_args.profile:
            with SafeProfiler(
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                scheduler.event_loop_normal(prof)
        else:
            scheduler.event_loop_normal()

    except Exception as e:
        exc = traceback.format_exc()
        logger.error(f"Scheduler process {prefix} failed: {exc}")
        pipe_writer.send(
            {
                "status": "error",
                "message": f"Scheduler process {prefix} failed: {exc}",
            }
        )
