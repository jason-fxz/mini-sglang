"""
Logits processor for language model heads.
"""

import dataclasses

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from mini_sglang.layers.embed_head import ParallelLMHead
from mini_sglang.managers.batch_info import BatchInfo


@dataclasses.dataclass
class LogitsProcessorOutput:
    # The logits of the next token.   shape: [batch_size, vocab_size]
    next_token_logits: torch.Tensor


class LogitsProcessor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        lm_head: ParallelLMHead,
        batch: BatchInfo,
    ):
        """
        Process the logits for the next token prediction.

        Args:
            hidden_states: The hidden states from the model. shape: [*, hidden_size]
            lm_head: The language model head for logits computation.
            batch_info: Information about the current batch.

        Returns:
            LogitsProcessorOutput
        """

        if batch.forward_mode.is_prefill():
            # for PREFILL mode, only compute the last token's logits
            last_indices = batch.cu_seqlens_q[1:] - 1
            hidden_states = hidden_states[last_indices].contiguous()

        # now hidden_states shape: [batch_size, hidden_size]

        # compute logits
        logits = F.linear(hidden_states, lm_head.weight, lm_head.bias)

        if self.tp_size > 1:
            all_logits = (
                [torch.empty_like(logits) for _ in range(self.tp_size)]
                if self.tp_rank == 0
                else None
            )
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None

        # Return the processed logits
        return LogitsProcessorOutput(next_token_logits=logits)
