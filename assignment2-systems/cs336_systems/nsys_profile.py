import sys
import pathlib
import torch
import timeit
import numpy as np
import pandas as pd
import math
from einops import einsum
import torch.cuda.nvtx as nvtx

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
    # 'xl':     {'d_model':1600, 'd_ff': 6400, 'num_layers': 48, 'num_heads': 25},
    # '2.7B':   {'d_model':2560, 'd_ff': 10240, 'num_layers': 32, 'num_heads': 32}

}


def get_model(model_type: str, basic_config: dict) -> BasicsTransformerLM:
    config = model_zoo[model_type]
    config.update(basic_config)
    model = BasicsTransformerLM(**config)
    return model


def main():
    basic_config = {
        'vocab_size': 10000,
        'rope_theta': 10000,
        'context_length': 256
    }
    
    device = 'cuda'
    batch_size = 4
    warm_up_iter = 1
    test_iter = 10
    use_backward = True

    model_type = 'small'
    result = {}

    model = get_model(model_type, basic_config).to(device=device)
    optimizer = AdamW(model.parameters())

    with torch.autocast(device_type=device):
        print(f'start warm up {warm_up_iter} times')
        for _ in range(warm_up_iter):
            x = torch.randint(low=0, high=basic_config['vocab_size'], size=(batch_size, 256))
            x = x.to(device=device)
            y = model(x)
            if use_backward is True:
                loss = y.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        torch.cuda.synchronize()

        forward_time_list = []
        backward_time_list = []
        optimizer_time_list = []
        for _ in range(test_iter):
            x = torch.randint(low=0, high=basic_config['vocab_size'], size=(batch_size, 256))
            x = x.to(device=device)

            # ------- forward time -------
            tic = timeit.default_timer()
            y = model(x)
            torch.cuda.synchronize()
            forward_time_list.append(timeit.default_timer() - tic)

            if use_backward is True:
                optimizer.zero_grad()
                loss = y.mean()

                # ------- backward time -------
                tic = timeit.default_timer()
                loss.backward()
                torch.cuda.synchronize()
                backward_time_list.append(timeit.default_timer()- tic)

                optimizer.zero_grad()
                # ------- optimizer time -------
                tic = timeit.default_timer()
                optimizer.step()
                torch.cuda.synchronize()
                optimizer_time_list.append(timeit.default_timer() - tic)

            torch.cuda.synchronize()

    result[model_type] = {
        # 'warm_up': warm_up_iter,
        'iter': test_iter, 
        'for_avg': float(np.array(forward_time_list).mean()),
        'for_std': float(np.array(forward_time_list).std()),
    }
    

    if use_backward is True:
        result[model_type].update({
            'back_avg': float(np.array(backward_time_list).mean()),
            'back_std': float(np.array(backward_time_list).std()),
            'opt_avg': float(np.array(optimizer_time_list).mean()),
            'opt_std': float(np.array(optimizer_time_list).std()),
        })
    data = pd.DataFrame.from_dict(result)
    print(data.T.round(5).to_markdown())

if __name__ == '__main__':
    # argparse = argparse.ArgumentParser('cs_336 assigment2')
    # argparse.add_argument('batch_size', type=int, default=4)
    # argparse.add_argument('context_length', type=int, default=256)
    # argparse.add_argument('device', type=str, default='cuda')
    # args = argparse.parse_args()

    main()