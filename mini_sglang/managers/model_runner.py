import logging
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from mini_sglang.layers.logits_processor import LogitsProcessorOutput
from mini_sglang.layers.sampler import Sampler
from mini_sglang.managers.batch_info import BatchInfo
from mini_sglang.managers.sampling_params import SamplingParams
from mini_sglang.managers.server_args import ServerArgs
from mini_sglang.mem_cache.req2token import ReqToTokenPool
from mini_sglang.mem_cache.token2kv import KVCachePool, MHAKVPool, PageAllocator
from mini_sglang.utils.loader import load_model
from mini_sglang.utils.model_config import ModelConfig

logger = logging.getLogger(__name__)


class ModelRunner:
    def __init__(
        self,
        model_config: ModelConfig,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        sampling_params: Optional[SamplingParams] = None,
        # req_to_token_pool: ReqToTokenPool,
        # page_allocator: PageAllocator,
        # kv_cache_pool: KVCachePool,
    ):
        self.device = server_args.device
        self.gpu_id = gpu_id

        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.page_size = server_args.page_size
        self.model_config = model_config
        self.server_args = server_args

        torch.set_default_dtype(model_config.hf_config.torch_dtype)
        torch.set_default_device(self.device)
        self.model = load_model(model_config)
        self.init_memory_pool(server_args.max_num_reqs, model_config.max_context_len)
        torch.set_default_device("cpu")

        self.sampler = Sampler()
        self.sampling_params = sampling_params or SamplingParams(
            eos_token_id=model_config.hf_config.eos_token_id
        )

        self.attn_backend = None
        self.init_attn_backend()

    def _calc_max_num_token(self) -> Tuple[int, int]:
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

    def init_attn_backend(self):
        if self.server_args.attention_backend == "torch":
            from mini_sglang.layers.attn.torch_attn_backend import (
                TorchNativeAttnBackend,
            )

            logger.info("Using Torch native attention backend")
            self.attn_backend = TorchNativeAttnBackend(self)
        elif self.server_args.attention_backend == "fa3":
            from mini_sglang.layers.attn.fa3_attn_backend import FlashAttn3Backend

            logger.info("Using FlashAttention 3 backend")
            self.attn_backend = FlashAttn3Backend(self)
        elif self.server_args.attention_backend == "fa2":
            from mini_sglang.layers.attn.fa2_attn_backend import FlashAttn2Backend

            logger.info("Using FlashAttention 2 backend")
            self.attn_backend = FlashAttn2Backend(self)
        else:
            raise ValueError(
                f"Unsupported attention backend: {self.server_args.attention_backend}"
            )

    def init_memory_pool(
        self,
        max_num_reqs: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
    ):
        num_pages, num_tokens = self._calc_max_num_token()
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

    def forward_extend(self, batch: BatchInfo) -> LogitsProcessorOutput:
        assert batch.forward_mode.is_extend()
        batch.attn_backend = self.attn_backend
        batch.attn_backend.init_forward_metadata(batch)

        return self.model.forward(batch.input_ids, batch.positions, batch)

    def forward_decode(self, batch: BatchInfo) -> LogitsProcessorOutput:
        assert batch.forward_mode.is_decode()
        batch.attn_backend = self.attn_backend
        batch.attn_backend.init_forward_metadata(batch)

        return self.model.forward(batch.input_ids, batch.positions, batch)

    def process_extend_result(self, batch: BatchInfo, logits: LogitsProcessorOutput):
        """
        Process the result of the extend forward pass.
        This includes writing the logits to the token pool and updating the batch info.
        """
        temperatures = torch.tensor(
            [req.sampling_params.temperature for req in batch.reqs],
            dtype=torch.float32,
            device=self.device,
        )

        # sampling
        output_ids = self.sampler(
            logits.next_token_logits, temperatures
        )  # shape [batch_size]

        # Update the batch info with the output ids
        for i, req in enumerate(batch.reqs):
            req.token_ids.append(output_ids[i].item())
            req.prefix_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx][
                : len(req) - 1
            ]

    def process_decode_result(self, batch: BatchInfo, logits: LogitsProcessorOutput):
        """
        Process the result of the decode forward pass.
        This includes writing the logits to the token pool and updating the batch info.
        """
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
