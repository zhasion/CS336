import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

def _setup_process_group(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12390"

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        local_rank = None
        if device_count > 0:
            local_rank = rank % device_count
            torch.cuda.set_device(local_rank)
        else:
            raise ValueError('Unable to find CUDA devices.')
        device = f'cuda:{local_rank}'
    else:
        device = 'cpu'

    dist.init_process_group(backend=backend, world_size=world_size, rank=rank)
    return device


def _cleanup_process_group():
    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m(x)


def run_allclose(rank: int, backend: str, world_size: int, batch_size: int):
    device = _setup_process_group(rank, world_size, backend)

    print(f'device: {device}, backend: {backend}, batch_size: {batch_size}, world_size: {world_size}')
    
    # Broadcast the parameter to others from rank 0.
    model = ToyModel().to(device=device)
    with torch.no_grad():
        for p in model.parameters():
            dist.broadcast(p, src=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()


    baseline_model = None
    baseline_optimizer = None
    if rank == 0:
        baseline_model = ToyModel().to(device=device)
        baseline_model.load_state_dict(model.state_dict())
        baseline_optimizer = torch.optim.SGD(baseline_model.parameters(), lr=0.1)

    # Random generate x, y.
    torch.manual_seed(0)
    input_x = torch.randn((batch_size, 16), dtype=torch.float32, device=device)
    output_y = torch.randint(0, 4, (batch_size, ), dtype=torch.long, device=device)

    local_batch_size = batch_size // world_size
    start_idx = rank * local_batch_size
    end_idx= (rank + 1) * local_batch_size 
    local_input_x = input_x[start_idx:end_idx, ...]
    local_output_y = output_y[start_idx:end_idx, ...]

    optimizer.zero_grad()
    logits = model(local_input_x)
    loss = loss_fn(logits, local_output_y)
    loss.backward()

    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad)
            p.grad.div_(world_size)
    optimizer.step()


    if rank == 0:
        baseline_optimizer.zero_grad()
        logits = baseline_model(input_x)
        loss = loss_fn(logits, output_y)
        loss.backward()
        baseline_optimizer.step()

        for para, baseline_para in zip(model.parameters(), baseline_model.parameters()):
            assert torch.allclose(para.grad, baseline_para.grad)
        print('Successful!')

    _cleanup_process_group()

def main():
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 2
    batch_size = 4
    assert batch_size % world_size == 0

    backend = 'gloo'
    mp.spawn(run_allclose, args=(backend, world_size, batch_size), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()