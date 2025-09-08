import sys
import pathlib
import torch
import timeit
import numpy as np
import pandas as pd
from einops import einsum
import torch.cuda.nvtx as nvtx
import math
import torch.nn as nn
from contextlib import nullcontext

sys.path.append(str(pathlib.Path(__file__).parent.parent))

import cs336_basics
from cs336_basics.cs336_basics.model import BasicsTransformerLM
from cs336_basics.cs336_basics.optimizer import AdamW
from cs336_basics.cs336_basics.nn_utils import softmax

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    with nvtx.range("final matmul"):
        result = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    return result

cs336_basics.cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention



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

def synchronize(device):
    if device == 'cuda':
        torch.cuda.synchronize()

def benchmarking_script():
    # ------------------- Basic setting ------------------
    basic_config = {
        'vocab_size': 10000,
        'rope_theta': 10000,
        'context_length': 128
    }
    
    device = 'cuda'
    batch_size = 4
    warm_up_iter = 5
    test_iter = 10

    use_backward = False
    mix_precision = True
    model_type = 'medium'

    result = {}

    # ------------------- Init model and optimizer ------------------
    model = get_model(model_type, basic_config).to(device=device)
    optimizer = AdamW(model.parameters())

    num_parameters = sum(np.prod(p.shape) for p in model.parameters())
    print(f"parameters count {num_parameters}")
    print(f"parameters memory {num_parameters * 4 / 2**30} GiB")

    # ------------------- Prepare the input data ------------------
    x = torch.randint(low=0, high=basic_config['vocab_size'], size=(batch_size, basic_config['context_length']))
    x = x.to(device=device)

    # ------------------- Use context manager ------------------
    context = torch.autocast(device_type=device, dtype=torch.bfloat16) if mix_precision is True else nullcontext()

    with context:
        # Warm up.
        print(f'start warm up {warm_up_iter} times')
        torch.cuda.nvtx.range_push("warmup")
        for _ in range(warm_up_iter):
            y = model(x)
            if use_backward:
                optimizer.zero_grad()
                loss = y.mean()
                loss.backward()
                optimizer.step()
        torch.cuda.nvtx.range_pop()

        # Record memory.
        torch.cuda.memory._record_memory_history(max_entries=1000000)

        forward_time_list = []
        backward_time_list = []
        optimizer_time_list = []
        for _ in range(test_iter):

            synchronize(device=device)

            # ------- forward time -------
            torch.cuda.nvtx.range_push(f"forward-{iter}")
            tic = timeit.default_timer()
            y = model(x)
            synchronize(device=device)
            forward_time_list.append(timeit.default_timer() - tic)
            torch.cuda.nvtx.range_pop()

            if use_backward is True:
                loss = y.mean()
                optimizer.zero_grad()

                # ------- backward time -------
                torch.cuda.nvtx.range_push(f"backward-{iter}")
                tic = timeit.default_timer()
                loss.backward()
                synchronize(device=device)
                backward_time_list.append(timeit.default_timer()- tic)
                torch.cuda.nvtx.range_pop()

                # ------- optimizer time -------
                torch.cuda.nvtx.range_push(f"optimizer-{iter}")
                tic = timeit.default_timer()
                optimizer.step()
                optimizer.zero_grad()
                synchronize(device=device)
                optimizer_time_list.append(timeit.default_timer() - tic)
                torch.cuda.nvtx.range_pop()

            synchronize(device=device)

        torch.cuda.memory._dump_snapshot(f"{model_type}-{basic_config['context_length']}-full-memory-snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

    result[model_type] = {
        'context': basic_config['context_length'],
        'warm': warm_up_iter,
        'iter': test_iter, 
        'forward': f'{float(np.array(forward_time_list).mean())*1000:.2f}±{float(np.array(forward_time_list).std())*1000:.2f}',
    }
    

    if use_backward is True:
        result[model_type].update({
            'backward': f'{float(np.array(backward_time_list).mean())*1000:.2f}±{float(np.array(backward_time_list).std())*1000:.2f}',
            'optimizer': f'{float(np.array(optimizer_time_list).mean())*1000:.2f}±{float(np.array(optimizer_time_list).std())*1000:.2f}',
        })
        
    data = pd.DataFrame.from_dict(result)
    print(data.T.round(5).to_markdown())


def mixed_precision_accumulation():
    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    print(s)

    s = torch.tensor(0, dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print(s)

    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print(s)

    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float32)
    print(s)


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f'model parameters: {next(self.parameters()).dtype}')
        print(f'input dtype: {x.dtype}')
        x = self.fc1(x)
        print(f'after fc1(x) dtype: {x.dtype}')
        x = self.relu(x)
        print(f'after relu dtype: {x.dtype}')
        x = self.ln(x)
        print(f'after ln(x) dtype: {x.dtype}')
        x = self.fc2(x)
        print(f'after fc2(x) dtype: {x.dtype}')

        return x


def run_autocast_toy_model(device: str):
    model = ToyModel(2, 3).to(device=device)
    inputs = torch.rand((2, 2), device=device, dtype=torch.float32)

    for dtype in [torch.float16, torch.float32, torch.bfloat16]:
        print('-'*10, f'autocast {dtype}', '-'*10)

        with torch.autocast(device_type=device, dtype=dtype):
            outputs = model(inputs)
            print(f"output dtype {outputs.dtype}")
            loss = outputs.mean()
            print(f"loss dtype {loss.dtype}")
        
        loss.backward()
        print("grad dtypes:", next(model.parameters()).grad.dtype)

if __name__ == '__main__':
    # argparse = argparse.ArgumentParser('cs_336 assigment2')
    # argparse.add_argument('batch_size', type=int, default=4)
    # argparse.add_argument('context_length', type=int, default=256)
    # argparse.add_argument('device', type=str, default='cuda')
    # args = argparse.parse_args()

    # benchmarking_script()
    # mixed_precision_accumulation()
    run_autocast_toy_model('cuda')