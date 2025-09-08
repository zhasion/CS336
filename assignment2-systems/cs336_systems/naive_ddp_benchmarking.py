import torch
import os
import torch.multiprocessing as mp
import torch.distributed as dist
import timeit
import sys
import itertools
import numpy as np
from pathlib import Path
from contextlib import nullcontext

sys.path.append(str(Path(__file__).parent.parent))

from cs336_basics.cs336_basics.model import BasicsTransformerLM
from cs336_basics.cs336_basics.optimizer import AdamW
from cs336_basics.cs336_basics.nn_utils import cross_entropy
from cs336_systems.ddp_overlap_individual_parameters import DDPOverlapIndividualParameters


model_zoo = {
    'small':  {'d_model':768, 'd_ff': 3072, 'num_layers': 12, 'num_heads': 12},
    'medium': {'d_model':1024, 'd_ff': 4096, 'num_layers': 24, 'num_heads': 16}, 
    'large':  {'d_model':1280, 'd_ff': 5120, 'num_layers': 36, 'num_heads': 20},
    'xl':     {'d_model':1600, 'd_ff': 6400, 'num_layers': 48, 'num_heads': 25},
    '2.7B':   {'d_model':2560, 'd_ff': 10240, 'num_layers': 32, 'num_heads': 32}
}


def get_model(model_type: str, basic_config: dict) -> BasicsTransformerLM:
    config = model_zoo[model_type]
    config.update(basic_config)
    model = BasicsTransformerLM(**config)
    return model


def _synchronize(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def set_ddp(rank: int, backend: str, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '4568'
    dist.init_process_group(backend=backend, world_size=world_size, rank=rank)
    if backend == 'nccl':
        assert torch.cuda.is_available()
        torch.cuda.set_device(rank)
        device = 'cuda'
    else:
        device = 'cpu'
    
    return device


def run_beachmark(
    rank, 
    backend, 
    world_size, 
    warm_iter, 
    test_iter, 
    batch_size, 
    model_type, 
    basic_config,
    use_jit,
    use_mix_precision,
    use_flat,
    use_overlap_communicate
):
    device = set_ddp(rank=rank, backend=backend, world_size=world_size)

    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    
    model = get_model(model_type, basic_config).to(device=device)
    optimizer = AdamW(model.parameters())

    if use_jit is True:
        model = torch.compile(model)

    if use_overlap_communicate:
        model = DDPOverlapIndividualParameters(model)
    else:
        for p in model.parameters():
            dist.broadcast(p.data, src=0)

    ctx = torch.autocast(device_type=device, dtype=torch.float16) if use_mix_precision is True else nullcontext
    params = sum([p.numel() for p in model.parameters()])
    print(f'MODEL PARAMETERS: {params}', )

    forward_time_list = []
    backward_time_list = []
    communicate_time_list = []
    optimizer_time_list = []
    total_time_list = []
    for it in range(warm_iter + test_iter):
        optimizer.zero_grad()

        input_x = torch.randint(0, basic_config['vocab_size'], (local_batch_size, basic_config['context_length']), dtype=torch.long, device=device)
        input_y = torch.randint(0, basic_config['vocab_size'], (local_batch_size, ), dtype=torch.long, device=device)
        
        _synchronize(device=device)
        total_start_time = timeit.default_timer()

        _synchronize(device=device)
        start_time = timeit.default_timer()
        logist = model(input_x)[..., -1, :]
        _synchronize(device=device)
        forward_time = timeit.default_timer() - start_time

        loss = cross_entropy(logist, input_y)
        _synchronize(device=device)
        start_time = timeit.default_timer()
        loss.backward()
        _synchronize(device=device)
        backward_time = timeit.default_timer() - start_time


        _synchronize(device=device)
        start_time = timeit.default_timer()
        if not use_overlap_communicate:
            if use_flat:
                params_with_grads = [p for p in model.parameters() if p.grad is not None]
                grads = [p.grad for p in params_with_grads]
                flatten_grad = torch._utils._flatten_dense_tensors(grads)

                dist.all_reduce(flatten_grad)
                flatten_grad.div_(world_size)

                unflatten_grad = torch._utils._unflatten_dense_tensors(flatten_grad, grads)
                for p, update_grad in zip(params_with_grads, unflatten_grad):
                    p.grad = update_grad
            else:
                for p in model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad)
                        p.grad.div_(world_size)
        else:
            model.finish_gradient_synchronization()

        _synchronize(device=device)
        communicate_time = timeit.default_timer() - start_time

        _synchronize(device=device)
        start_time = timeit.default_timer()
        optimizer.step()
        _synchronize(device=device)
        optimizer_time = timeit.default_timer() - start_time

        total_time = timeit.default_timer() - total_start_time

        if it >= warm_iter:
            forward_time_list.append(forward_time)
            backward_time_list.append(backward_time)
            communicate_time_list.append(communicate_time)
            optimizer_time_list.append(optimizer_time)
            total_time_list.append(total_time)

    gather_forward = [None] * world_size if rank == 0 else None
    dist.gather_object(forward_time_list, gather_forward, dst=0)

    gather_backward = [None] * world_size if rank == 0 else None
    dist.gather_object(backward_time_list, gather_backward, dst=0)

    gather_optimizer = [None] * world_size if rank == 0 else None
    dist.gather_object(optimizer_time_list, gather_optimizer, dst=0)

    gather_communicate = [None] * world_size if rank == 0 else None
    dist.gather_object(communicate_time_list, gather_communicate, dst=0)

    gather_total = [None] * world_size if rank == 0 else None
    dist.gather_object(total_time_list, gather_total, dst=0)


    if rank == 0:
        gather_forward = np.array(list(itertools.chain.from_iterable(gather_forward)))
        gather_backward = np.array(list(itertools.chain.from_iterable(gather_backward)))
        gather_optimizer = np.array(list(itertools.chain.from_iterable(gather_optimizer)))
        gather_communicate = np.array(list(itertools.chain.from_iterable(gather_communicate)))
        gather_total = np.array(list(itertools.chain.from_iterable(gather_total)))

        print(f'forward time: {gather_forward.mean():.3f} s')
        print(f'backward time: {gather_backward.mean():.3f} s')
        print(f'optimizer time: {gather_optimizer.mean():.3f} s')
        print(f'communicate time: {gather_communicate.mean():.3f} s')
        print(f'total time: {gather_total.mean():.3f} s')


    if dist.is_initialized():
        dist.barrier(device_ids=[rank])
        dist.destroy_process_group()



def main():

    basic_config = {
        'vocab_size': 10000,
        'rope_theta': 10000,
        'context_length': 128
    }

    model_type = 'medium'
    world_size = 2
    backend = 'nccl'
    batch_size = 4
    warm_up_iter = 5
    test_iter = 5
    use_mix_precision = False
    use_jit = False
    use_flat = False
    use_overlap_communicate = False


    mp.spawn(
        run_beachmark,
        args=(
            backend,
            world_size,
            warm_up_iter,
            test_iter,
            batch_size,
            model_type,
            basic_config,
            use_jit,
            use_mix_precision,
            use_flat,
            use_overlap_communicate
        ),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()