"""

"""

from typing import List, Union

import torch


class ReqToTokenPool:
    """
    A memory pool that manages requests tokens kvcache loc.

    req_to_token[request_pos, token_pos] = token_loc
        token_loc refers to the location in the KV cache where the token is stored.

    req_to_page[request_pos, page_pos] = page_id
        req_to_page equals to req_to_token when page_size == 1.
        This is the page_table can be used in FA2/FA3

    Args:
        size (int): The total number of requests.
        max_tokens (int): The maximum number of tokens per request.
        page_size (int): The size of each page.
        device (str): The device to store the memory pool.
    """

    def __init__(
        self,
        size: int,
        max_tokens: int,
        page_size: int,
        device: str,
    ):
        assert (
            max_tokens % page_size == 0
        ), "max_tokens must be a multiple of page_size."
        self.size = size
        self.max_tokens = max_tokens
        self.page_size = page_size
        self.max_page_num = max_tokens // page_size
        self.device = device
        self.req_to_token = torch.zeros(
            (size, max_tokens), dtype=torch.int32, device=device
        )
        if page_size > 1:
            self.req_to_page = torch.zeros(
                (size, self.max_page_num), dtype=torch.int32, device=device
            )
        else:
            self.req_to_page = self.req_to_token

        self.free_slots = list(range(size))

    # def write(self, indices, values):
    #     self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, num: int) -> List[int]:
        if num > len(self.free_slots):
            raise RuntimeError("Not enough free slots available.")

        allocated = self.free_slots[:num]
        self.free_slots = self.free_slots[num:]
        return allocated

    def free(self, indices: Union[int, List[int]]):
        if isinstance(indices, int):
            indices = [indices]
        self.free_slots.extend(indices)

    def clear(self):
        self.free_slots = list(range(self.size))

    def write_tokens(self, req_idx: int, indices: slice, values: torch.Tensor):
        self.req_to_token[req_idx, indices] = values
        if self.page_size > 1:
            assert (
                indices.start % self.page_size == 0
                and indices.stop % self.page_size == 0
            ), "Token indices must be page-aligned when page_size > 1."
            start_page = indices.start // self.page_size
            end_page = indices.stop // self.page_size
            page_indices = torch.arange(
                0, end_page - start_page, self.page_size, device=self.device
            )
            self.req_to_page[req_idx, start_page:end_page] = (
                values[page_indices] // self.page_size
            )
