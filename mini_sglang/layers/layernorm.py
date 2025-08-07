from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from sgl_kernel import fused_add_rmsnorm, rmsnorm
from torch import nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self, x: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            fused_add_rmsnorm(x, residual, self.weight.data, self.eps)
            return x, residual
        out = rmsnorm(x, self.weight.data, self.eps)
        return out

        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        x = (x * self.weight).to(orig_dtype)

        if residual is not None:
            return x, residual
        return x
