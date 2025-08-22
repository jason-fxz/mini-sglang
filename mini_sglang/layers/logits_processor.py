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
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tp_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()

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

        if batch.forward_mode.is_extend():
            # for EXTEND mode, only compute the last token's logits
            last_indices = torch.cumsum(batch.input_seq_lens, dim=0, dtype=torch.int32)
            last_indices.add_(-1)
            hidden_states = hidden_states[last_indices].contiguous()

        # now hidden_states shape: [batch_size, hidden_size]

        # compute logits
        if hasattr(lm_head, "bias") and lm_head.bias is not None:
            logits = F.linear(hidden_states, lm_head.weight, lm_head.bias)
        else:
            logits = F.linear(hidden_states, lm_head.weight)

        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)

        # Return the processed logits
        return LogitsProcessorOutput(next_token_logits=logits)
