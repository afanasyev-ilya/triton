import torch
import triton
import triton.language as tl
import numpy as np


def benchmark_transpose(transpose_fn, *fn_args, iters: int = 10, **fn_kwargs):
    """
    Benchmarks a transpose function over multiple iterations and verifies correctness.

    Parameters:
      transpose_fn: a function whose first argument is a torch.Tensor on GPU
                    and that returns the transposed tensor.
      *fn_args: positional arguments to pass into transpose_fn (first one must be x).
      iters: number of timed iterations (default = 10).
      **fn_kwargs: keyword arguments to pass into transpose_fn.

    This function:
      1) Extracts x = fn_args[0] and computes the reference transpose via x.t().
      2) Performs one warmup run of transpose_fn.
      3) Runs transpose_fn iters times, measuring each execution with CUDA events.
      4) Verifies that the first measured output matches the reference exactly.
      5) Prints the minimum, average, and maximum elapsed times (in ms) across all iterations.
    """
    # 1) Assume fn_args[0] is the input tensor x on GPU
    x = fn_args[0]
    if not isinstance(x, torch.Tensor):
        raise ValueError("The first positional argument must be a torch.Tensor on GPU")

    # 1b) Compute the reference (golden) result using PyTorch's transpose
    golden = x.t()

    # 2) Warmup
    _ = transpose_fn(*fn_args, **fn_kwargs)
    torch.cuda.synchronize()

    # 3) Run timed iterations
    times = []
    first_output = None
    in_place = False

    input = x

    for i in range(iters):
      # Create new CUDA events for timing
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)

      torch.cuda.synchronize()
      start_event.record()

      out = transpose_fn(*fn_args, **fn_kwargs)

      # inplace case
      if out is None:
        in_place = True
        out = fn_args[0]

      end_event.record()
      torch.cuda.synchronize()

      elapsed_ms = start_event.elapsed_time(end_event)
      times.append(elapsed_ms)

      # Save the first output for correctness check
      if i == 0:
          first_output = out

    # 4) Verify correctness using the first iteration's output
    max_diff = torch.max(torch.abs(first_output - golden))
    if not in_place:
      assert max_diff == 0, f"Mismatch detected: max difference = {max_diff.item()}"

    # 5) Compute statistics
    min_time = min(times)
    max_time = max(times)
    avg_time = sum(times) / len(times)

    print(f"Ran {iters} iterations")
    print(f"Min time: {min_time:.3f} ms")
    print(f"Avg time: {avg_time:.3f} ms")
    print(f"Max time: {max_time:.3f} ms")