from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import distributed as dist
from torch import nn
from torch.nn.parameter import Parameter, UninitializedParameter


def check_divide(x: int, y: int) -> int:
    assert x % y == 0, f"{x} is not divisible by {y}"
    return x // y


class LinearBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This method should be overridden by subclasses.")


class ReplicatedLinear(LinearBase):
    """Simple replicated linear layer.
    Y = XA + b
    X: [*, input_size]
    A: [input_size, output_size]
    b: [output_size]
    Y: [*, output_size]
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size)

        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor) -> None:
        assert param.shape == loaded_weight.shape
        param.data.copy_(loaded_weight)


class ColumnParallelLinear(LinearBase):
    """Column parallel linear layer.

    Y = XA + b => Y = X[A_1, A_2, ..., A_p] + [b_1, b_2, ..., b_p]  (p is the number of partitions)
    Y_i = X A_i + b_i

    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size)

        self.input_size_per_partition = input_size
        self.output_size_per_partition = check_divide(output_size, self.tp_size)

        self.weight = nn.Parameter(
            torch.empty(self.output_size_per_partition, input_size)
        )
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor) -> None:
        param_data = param.data
        shard_size = self.output_size_per_partition
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, start_idx + shard_size)
        assert (
            param_data.shape == loaded_weight.shape
        ), f"Shape mismatch: {param_data.shape} vs {loaded_weight.shape}"
        param_data.copy_(loaded_weight)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """Merged column parallel linear layer.

    Similar to ColumnParallelLinear, but the weight matrix is concatenated
    along the output dimension. When the weight matrix is loaded, the
    different partitions are sharded separately.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(
        self, param: Parameter, loaded_weight: torch.Tensor, loaded_shard_id
    ) -> None:
        assert loaded_shard_id < len(self.output_sizes)
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(0, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, 0)[self.tp_rank]
        assert (
            param_data.shape == loaded_weight.shape
        ), f"Shape mismatch: {param_data.shape} vs {loaded_weight.shape}"
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """Linear layers for the attention's QKV transformation.

    Linear layers for the linear transformation of the query, key, and value
    vectors in the attention layer. The weight matrix is concatenated along
    the output dimension. The layer is parallelized along the head dimension.
    When the number of key/value heads is smaller than the number of query
    heads (e.g., multi-query/grouped-query attention), the key/value head may
    be replicated while the query heads are partitioned.

    Args:
        hidden_size: input hidden state size of the transformer.
        head_size: size of each attention head.
        total_num_heads: total number of attention query heads.
        total_num_kv_heads: total number of attention key/value heads. If
                            None, assume total_num_kv_heads = total_num_heads.
        bias: If true, add bias.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
        bias: bool = False,
    ):
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = (
            total_num_kv_heads if total_num_kv_heads is not None else total_num_heads
        )
        tp_size = dist.get_world_size()
        self.num_heads = check_divide(total_num_heads, tp_size)
        self.num_kv_heads = check_divide(self.total_num_kv_heads, tp_size)
        input_size = hidden_size
        output_size = (
            self.total_num_heads + 2 * self.total_num_kv_heads
        ) * self.head_size
        super().__init__(input_size, output_size, bias)

    def weight_loader(
        self, param: Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str
    ) -> None:
        param_data = param.data
        assert loaded_shard_id in [
            "q",
            "k",
            "v",
        ], f"Invalid shard_id: {loaded_shard_id}"
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:  # loaded_shard_id == 'v'
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = (self.num_heads + self.num_kv_heads) * self.head_size
        param_data = param_data.narrow(0, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, 0)[self.tp_rank]
        assert (
            param_data.shape == loaded_weight.shape
        ), f"Shape mismatch: {param_data.shape} vs {loaded_weight.shape}"
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """Row parallel linear layer.

    Y = XA + b =>

               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -

    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size)

        self.input_size_per_partition = check_divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size

        self.weight = nn.Parameter(
            torch.empty(output_size, self.input_size_per_partition)
        )
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_rank > 1:
            dist.all_reduce(y, op=dist.ReduceOp.SUM)
        return y

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor) -> None:
        param_data = param.data
        shard_size = self.input_size_per_partition
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(1, start_idx, shard_size)
        assert (
            param_data.shape == loaded_weight.shape
        ), f"Shape mismatch: {param_data.shape} vs {loaded_weight.shape}"
        param_data.copy_(loaded_weight)
