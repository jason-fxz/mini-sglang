import torch
import torch.nn.functional as F
from torch import nn


class SiluAndMul(nn.Module):
    """Silu activation followed by multiplication"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, dim=-1)
        return F.silu(x) * y
