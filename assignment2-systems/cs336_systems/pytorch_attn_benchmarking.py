import os
import torch
import sys
import timeit
import itertools
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from cs336_basics.cs336_basics.model import scaled_dot_product_attention

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def main():
    batch_size = 8
    head_dims = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]
    device = 'cuda'
    WARMUP = 5
    STEPS = 100

    for head_dim, seq_len in itertools.product(head_dims, seq_lens):
        try:
            q = torch.randn((batch_size, seq_len, head_dim), device=device, requires_grad=True)
            k = torch.randn_like(q, device=device)
            v = torch.randn_like(q, device=device)

            # warm-up forward
            with torch.no_grad():
                torch.cuda.synchronize()
                for _ in range(WARMUP):
                    
                    out = scaled_dot_product_attention(q, k, v)
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

            # timed forward
            with torch.no_grad():
                torch.cuda.synchronize()
                time_start = timeit.default_timer()
                for _ in range(STEPS):
                    scaled_dot_product_attention(q, k, v)
                    torch.cuda.synchronize()
                fwd_time = (timeit.default_timer() - time_start) * 1e3 / STEPS

            # warm-up backward
            for _ in range(WARMUP):
                torch.cuda.synchronize()
                out = scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize()
                out.sum().backward()
                torch.cuda.synchronize()
                q.grad = k.grad = v.grad = None
            
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            mem_before_bwd = 0
            bwd_time = 0
            for _ in range(STEPS):
                torch.cuda.synchronize()
                out = scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize()
                mem_before_bwd += torch.cuda.memory_allocated()
                time_start = timeit.default_timer()
                out.sum().backward()
                torch.cuda.synchronize()
                bwd_time += (timeit.default_timer() - time_start) * 1e3
                q.grad = k.grad = v.grad = None
                torch.cuda.empty_cache()

            mem_mb = (mem_before_bwd / STEPS) / 1e6
            bwd_time = bwd_time / STEPS

            print(f'{head_dim} {seq_len} {fwd_time:.3f} {bwd_time:.3f} {mem_mb}')

        except torch.OutOfMemoryError:
            print(f'{head_dim} {seq_len} OOM OOM OOM')
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()