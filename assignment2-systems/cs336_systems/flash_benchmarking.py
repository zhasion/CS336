import triton 
import triton.language as tl
import triton.testing as ttesting
import torch
import timeit
import itertools

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from cs336_basics.cs336_basics.model import scaled_dot_product_attention
from cs336_systems.flash_attention import FlashAttentionTriton

WARMUP_ITER = 25
REPEATE_ITER = 100


def bench_forward(function, *args):
    def _forward():
        _ = function(*args)

    return ttesting.do_bench(_forward, warmup=WARMUP_ITER, rep=REPEATE_ITER)


def bench_backward(function, *args):
    for x in args:
        if isinstance(x, torch.Tensor) and x.dtype in [torch.float32, torch.bfloat16] :
            x.requires_grad_(True)
    out = function(*args)

    def _backward():
        out.sum().backward(retain_graph=True)
        for x in args:
            if isinstance(x, torch.Tensor) and x.dtype in [torch.float32, torch.bfloat16] :
                x.grad = None
    
    return ttesting.do_bench(_backward, warmup=WARMUP_ITER, rep=REPEATE_ITER)

def bench_total(function, *args):
    for x in args:
        if isinstance(x, torch.Tensor) and x.dtype in [torch.float32, torch.bfloat16] :
            x.requires_grad_(True)
    
    def _forward_backward():
        out = function(*args)
        out.sum().backward(retain_graph=True)
        for x in args:
            if isinstance(x, torch.Tensor) and x.dtype in [torch.float32, torch.bfloat16] :
                x.grad = None
    
    return ttesting.do_bench(_forward_backward, warmup=WARMUP_ITER, rep=REPEATE_ITER)

def synchronize(device):
    if device == 'cuda':
        torch.cuda.synchronize()

def run_benchmark(dtype: torch.dtype, context_length: int, d_model: int, device: str):

    Q = torch.randn((1, context_length, d_model), dtype=dtype, device=device)
    K = torch.randn((1, context_length, d_model), dtype=dtype, device=device)
    V = torch.randn((1, context_length, d_model), dtype=dtype, device=device)
    
    # Regular Pytorch.
    torch_fwd_time, torch_bwd_time, torch_total_time = 'N/A', 'N/A', 'N/A'

    mask = torch.tril(torch.ones((context_length, context_length), dtype=torch.bool, device=device))

    try:
        torch_fwd_time = bench_forward(scaled_dot_product_attention, Q, K, V, mask)
        synchronize(device)
        torch_bwd_time = bench_backward(scaled_dot_product_attention, Q, K, V, mask)
        synchronize(device)
        torch_total_time = bench_total(scaled_dot_product_attention, Q, K, V, mask)
        synchronize(device)
    except torch.cuda.OutOfMemoryError:
        pass
    finally:
        torch.cuda.empty_cache()

    # My Triton.
    triton_fwd_time, triton_bwd_time, triton_total_time = 'N/A', 'N/A', 'N/A'
    try:
        triton_fwd_time = bench_forward(FlashAttentionTriton.apply, Q, K, V, True)
        synchronize(device)
        triton_bwd_time = bench_backward(FlashAttentionTriton.apply, Q, K, V, True)
        synchronize(device)
        triton_total_time = bench_total(FlashAttentionTriton.apply, Q, K, V, True)
        synchronize(device)
    except torch.cuda.OutOfMemoryError:
        pass
    finally:
        torch.cuda.empty_cache()

    return torch_fwd_time, torch_bwd_time, torch_total_time, triton_fwd_time, triton_bwd_time, triton_total_time

def main():
    device = 'cuda'
    context_length_list = [2 ** x for x in range(7, 17)]
    dtype_list = [torch.float32, torch.bfloat16]
    d_model_list = [2 ** x for x in range(4, 8)]

    for dtype, context_length, d_model in itertools.product(dtype_list, context_length_list, d_model_list):
        torch_fwd_time, torch_bwd_time, torch_total_time, triton_fwd_time, triton_bwd_time, triton_total_time = run_benchmark(dtype, context_length, d_model, device)
        print(dtype, context_length, d_model, torch_fwd_time, torch_bwd_time, torch_total_time, triton_fwd_time, triton_bwd_time, triton_total_time)

if __name__ == '__main__':
    main()

