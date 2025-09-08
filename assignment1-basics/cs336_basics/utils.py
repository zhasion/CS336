import torch
import numpy as np
import numpy.typing as npt
import math
import os
from torch import Tensor
from jaxtyping import Int, Float
from typing import Iterable, BinaryIO, IO


def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(0, x)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    shifted_x = x - x.max(dim=dim, keepdim=True).values
    exp_shifted_x = torch.exp(shifted_x)
    return exp_shifted_x / exp_shifted_x.sum(dim=dim, keepdim=True)


def log_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    shifted_x = x - x.max(dim=dim, keepdim=True).values
    exp_shifted_x = torch.exp(shifted_x)
    sum_exp_shifted_x = torch.sum(exp_shifted_x, dim=dim, keepdim=True)
    return shifted_x - torch.log(sum_exp_shifted_x)

def cross_entropy_loss(
    inputs: Float[torch.Tensor, " batch_size vocab_size"], 
    targets: Int[torch.Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    # log-sum-exp trick for numerical stability
    log_probs = log_softmax(inputs, dim=-1)

    # Extract loss corresponding to the target class
    negative_log_softmax_logits = -torch.gather(input=log_probs, dim=-1, index=targets.unsqueeze(-1))

    return negative_log_softmax_logits.mean()


def get_perplexity(x: float) -> float:
    return math.exp(min(x, 25))


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        # Warm-up.
        return it / warmup_iters * max_learning_rate
    elif it <= cosine_cycle_iters:
        # Cosine annealing.
        cosine_coef = math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(cosine_coef)) * (max_learning_rate - min_learning_rate)
    else:
        # Post-annealing.
        return min_learning_rate


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    parameters_with_grad = [p for p in parameters if p.grad is not None]
    
    if not parameters_with_grad:
        return
    
    l2_norm = torch.cat([p.grad.view(-1) for p in parameters_with_grad]).norm()
    # l2_norm = torch.sqrt(sum([p.grad.pow(2).sum() for p in parameters_with_grad]))

    if max_l2_norm < l2_norm:
        clip_coef = max_l2_norm / (l2_norm + eps)
        for p in parameters_with_grad:
            p.grad.mul_(clip_coef)
    return l2_norm


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    
    # Sample the start indices for the dataset.
    max_sample_indices = len(dataset) - context_length - 1
    batch_smaple_start_indices = np.random.randint(low=0, high=max_sample_indices + 1, size=batch_size, dtype=np.int64)

    # Abtain the indices table [batch_size, context_length].
    context_indices = np.arange(0, context_length)
    input_batch_indices = batch_smaple_start_indices[:, None] + context_indices[None, :]

    # Smaple data from dataset and create the Tensor.
    input_batch_data = torch.tensor(dataset[input_batch_indices], device=device, dtype=torch.long)
    target_batch_data = torch.tensor(dataset[input_batch_indices + 1], device=device, dtype=torch.long)

    return input_batch_data, target_batch_data


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes]
):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:

    # Load checkpoint.
    device = next(model.parameters()).device
    checkpoint = torch.load(src, map_location=device)

    # Resume model and optimizer.
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    iteration = checkpoint['iteration']

    return iteration