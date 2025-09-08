import timeit
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import itertools
import os
import numpy as np

# os.environ['NCCL_SHM_DISABLE'] = '1'
# os.environ['NCCL_P2P_DISABLE'] = '0'

def synchronize(device: str):
    if device == 'cuda':
        torch.cuda.synchronize()


def set_ddp(rank: int, backend: str, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '23431'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if backend == 'nccl':
        device = 'cuda'
        torch.cuda.set_device(rank)
    else:
        device = 'cpu'
    
    return device
    
def bench_dist(rank, backend, tensor_size, world_size, warm_iter, repeate_iter):
    try:
        device = set_ddp(rank, backend, world_size)
        x = torch.randn(tensor_size // 4, dtype=torch.float32, device=device)

        # Warm up iter.
        for _ in range(warm_iter):
            dist.all_reduce(x, async_op=False)
            synchronize(device=device)
        
        # Repeate iter.
        cost_time_list = []
        for i in range(repeate_iter):
            synchronize(device=device)
            start_time = timeit.default_timer()
            dist.all_reduce(x, async_op=False)
            synchronize(device=device)
            cost_time = timeit.default_timer() - start_time
            cost_time_list.append(cost_time)
        
        # Gather timings
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(cost_time_list, gathered, dst=0)

        # dist.gather_object(gathered, cost_time_list, dst=0)


        if rank == 0:
            flat = list(itertools.chain.from_iterable(gathered))
            flat = np.array(flat)
            memory = tensor_size // 10 ** 6
            print(f'{device.upper()} {backend.upper()} {memory}MB {world_size} {flat.mean() * 1000 :.3f}')
        
        if backend == 'nccl':
            dist.barrier(device_ids=[rank])
        else:
            dist.barrier()
        dist.destroy_process_group()
    except Exception as e:
        print(f"Process {rank} failed with error: {e}")
        if dist.is_initialized():
            dist.destroy_process_group()
        raise e




def main():
    warm_iter = 5
    repeate_iter = 10
    
    world_size_list = [2, 4, 6]
    backend_list = ['nccl', 'gloo']
    tensor_size_list = [10 ** x for x in [6, 7, 8, 9]]

    for backend, tensor_size, world_size in itertools.product(
        backend_list, tensor_size_list, world_size_list
    ):
        if backend == 'nccl' and torch.cuda.device_count() < world_size:
            continue

        mp.spawn(fn=bench_dist, args=(backend, tensor_size, world_size, warm_iter, repeate_iter), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()