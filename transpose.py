import torch

import triton
import triton.language as tl
from numba import cuda
import numpy as np
from utils.benchmarks import benchmark_and_verify


DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def transpose_triton_kernel(in_ptr,
                            output_ptr,
                            num_rows,
                            num_cols,
                            block_x : tl.constexpr,
                            block_y : tl.constexpr):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)

    input_block_ptr = tl.make_block_ptr(
        base=in_ptr,
        shape=(num_rows, num_cols),
        strides=(num_rows, 1),
        offsets=(pid_x * block_x, pid_y * block_y),
        block_shape=(block_x, block_y),
        order=(0, 1)
    )

    block = tl.load(input_block_ptr)

    block = block.T

    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(num_cols, num_rows),  # Transposed shape
        strides=(num_rows, 1),
        offsets=(pid_y * block_y, pid_x * block_x),
        block_shape=(block_x, block_y),
        order=(0, 1)
    )

    tl.store(output_block_ptr, block)


def transpose_triton_wrapper(x: torch.Tensor) -> torch.Tensor:
    # We need to preallocate the output.
    output = torch.zeros(x.shape).to(device=DEVICE)

    block_size = [32, 32]
    m = x.shape[0]
    n = x.shape[1]
    
    grid = lambda meta: (triton.cdiv(x.shape[0], block_size[0]), triton.cdiv(x.shape[1], block_size[1]))

    transpose_triton_kernel[grid](x, output, m, n, block_size[0], block_size[1])

    return output


def transpose_torch_wrapper(x: torch.Tensor) -> torch.Tensor:
    return x.t()


def main():
    torch.manual_seed(0)

    m = 2048
    n = 2048
    x = torch.rand([m, n], device=DEVICE)

    print("running torch: ")
    benchmark_and_verify(transpose_torch_wrapper, x)

    print("\nrunning triton: ")
    benchmark_and_verify(transpose_triton_wrapper, x)


main()