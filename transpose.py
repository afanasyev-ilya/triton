import torch

import triton
import triton.language as tl


DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def transpose_kernel(in_ptr,
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

def transpose_wrapper(x: torch.Tensor):
    # We need to preallocate the output.
    output = torch.zeros(x.shape).to(device=DEVICE)

    stride_in_row = x.stride(0)
    stride_in_col = x.stride(1)
    print(stride_in_row, " @! ", stride_in_col)

    block_size = [32, 32]
    
    grid = lambda meta: (triton.cdiv(x.shape[0], block_size[0]), triton.cdiv(x.shape[1], block_size[1]))

    transpose_kernel[grid](x, output, m, n, block_size[0], block_size[1])

    return output


# %%
# We can now use the above function to compute the element-wise sum of two `torch.tensor` objects and test its correctness:

torch.manual_seed(0)

m = 72
n = 72
x = torch.rand([m, n], device=DEVICE)

print(x.shape)

output_torch = torch.transpose(x, 0, 1)
output_triton = transpose_wrapper(x)

print(output_torch.shape)

print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')