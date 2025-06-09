import torch

import triton
import triton.language as tl
from numba import cuda
import numpy as np
from utils.benchmarks import benchmark_transpose


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


@triton.jit
def inplace_transpose(in_ptr, num_rows, num_cols,
                      block_x: tl.constexpr, block_y: tl.constexpr):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    # diagonal
    if pid_x == pid_y:
        ptr = tl.make_block_ptr(in_ptr,
                                (num_rows, num_cols),
                                (num_rows, 1),
                                (pid_x*block_x, pid_y*block_y),
                                (block_x, block_y),
                                order=(0,1))
        blk = tl.load(ptr)

        blk = blk.T

        tl.store(ptr, blk)
    else:
        # Вне диагонали: обмениваем (i,j) и (j,i) один раз
        # Загружаем блок A = (i,j)
        ptr_a = tl.make_block_ptr(in_ptr,
                                (num_rows, num_cols),
                                (num_rows, 1),
                                (pid_x*block_x, pid_y*block_y),
                                (block_x, block_y),
                                order=(0,1))
        A = tl.load(ptr_a)

        # Загружаем блок B = (j,i)
        ptr_b = tl.make_block_ptr(in_ptr,
                                (num_rows, num_cols),
                                (num_rows, 1),
                                (pid_y*block_x, pid_x*block_y),
                                (block_x, block_y),
                                order=(0,1))
        B = tl.load(ptr_b)

        # Записываем Aᵀ в положение B и Bᵀ в положение A
        tl.store(ptr_b, A.T)
        tl.store(ptr_a, B.T)



def transpose_triton_wrapper(x: torch.Tensor) -> torch.Tensor:
    # We need to preallocate the output.
    output = torch.zeros(x.shape).to(device=DEVICE)

    block_size = [32, 32]
    m = x.shape[0]
    n = x.shape[1]
    
    grid = lambda meta: (triton.cdiv(x.shape[0], block_size[0]), triton.cdiv(x.shape[1], block_size[1]))

    transpose_triton_kernel[grid](x, output, m, n, block_size[0], block_size[1])

    return output


def inplace_transpose_triton_wrapper(x: torch.Tensor) -> torch.Tensor:
    block_size = [32, 32]
    m = x.shape[0]
    n = x.shape[1]
    
    grid = lambda meta: (triton.cdiv(x.shape[0], block_size[0]), triton.cdiv(x.shape[1], block_size[1]))

    inplace_transpose[grid](x, m, n, block_size[0], block_size[1])


def transpose_torch_wrapper(x: torch.Tensor) -> torch.Tensor:
    return x.t()


def main():
    torch.manual_seed(0)

    size = 4096
    m = size
    n = size
    x = torch.rand([m, n], device=DEVICE)

    print("running torch: ")
    benchmark_transpose(transpose_torch_wrapper, x)

    print("\nrunning triton: ")
    benchmark_transpose(transpose_triton_wrapper, x)

    print("\nInplace triton: ")
    benchmark_transpose(inplace_transpose_triton_wrapper, x)


main()