def count_parameters(
    vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int
) -> int:
    embedding = vocab_size * d_model
    mha = 4 * d_model ** 2
    rms = 2 * d_model
    ffn = 3 * (d_model * d_ff)
    final_rms = d_model
    lm_head = d_model * vocab_size
    total_paras = embedding + (mha + rms + ffn) * num_layers + final_rms + lm_head
    return total_paras

def count_activateion(
    vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, bs: int
) -> int:
    # Tansformer Block
    residule = 2 * bs * context_length * d_model

    # MHA
    qkv_proj = 3 * bs * context_length * d_model
    qk_mat = bs * num_heads * context_length * context_length
    softmax_qk_mat = bs * num_heads * context_length * context_length
    qkv = bs * context_length * d_model
    mha_output = bs * context_length * d_model

    # FFN
    w1 = bs * context_length * d_ff
    w3 = bs * context_length * d_ff
    silu = bs * context_length * d_ff
    w1_w3_elementwise = bs * context_length * d_ff
    ffn_output = bs * context_length * d_ff

    # RMS
    rms = bs * context_length * d_model
    
    # Final RMS
    final_rms = bs * context_length * d_model

    # LM head
    lm_head = bs * context_length * vocab_size
    loss = bs * context_length

    mha_activation = qkv_proj + qk_mat + softmax_qk_mat + qkv + mha_output
    ffn_activation = w1 + w3 + silu + w1_w3_elementwise + ffn_output
    transformer_activation = num_layers * (residule + rms + ffn_activation + mha_activation)

    return transformer_activation + final_rms + lm_head + loss


def calculate_memory(
    vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, bs: int
) -> int:
    parameters = count_parameters(vocab_size, context_length, num_layers, d_model, num_heads, d_ff)
    activation = count_activateion(vocab_size, context_length, num_layers, d_model, num_heads, d_ff, bs)
    buffer = 2 * context_length * d_model # rope

    print(f'Parameters: {parameters * 4 / 1e9:.2f} GB')
    print(f'Gradients: {parameters * 4 / 1e9:.2f} GB')
    print(f'Optimizer State: {parameters * 2 * 4 / 1e9:.2f} GB')
    print(f'Activation: {activation * 4 / 1e9:.2f} GB')
    print(f'Buffer: {buffer * 4 / 1e9:.2f} GB')
    print(f'Total Memory: {4 * (parameters * 4 + activation + buffer) / 1e9:.2f} GB')


def calculate_flop(
    vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int
) -> int:
    embedding = 0
    
    mha = 2 * (
        context_length * d_model * d_model * 4                  # qkvo projection [l, d] @ [d, d]
        + context_length * d_model * context_length             # q @ k.T.  [l, d] @ [d, l]
        + context_length * context_length * d_model             # attn @ V  [l, l] @ [l, d]
    )

    ffn = 2 * (3 * context_length * d_model * d_ff)
    rms = 2 * (2 * context_length * d_model)                    # [l, d] * [1, d]
    final_rms = 2 * (context_length * d_model)
    lm_head  = 2 * (d_model * vocab_size * context_length)      # all for training

    total = embedding + (mha + ffn + rms) * num_layers + final_rms + lm_head

    print('embedding:', f'{embedding / 1e9 :.2f} GFLOPS', f'({embedding / total * 100:.2f}%)')
    print('mha:', f'{mha * num_layers / 1e9 :.2f} GFLOPS', f'({mha * num_layers / total * 100:.2f}%)')
    print('ffn:', f'{ffn * num_layers / 1e9 :.2f} GFLOPS', f'({ffn * num_layers / total * 100:.2f}%)')
    print('rms:', f'{rms * num_layers / 1e9 :.2f} GFLOPS', f'({rms * num_layers / total * 100:.2f}%)')
    print('final_rms:', f'{final_rms / 1e9 :.2f} GFLOPS', f'({final_rms / total * 100:.2f}%)')
    print('lm_head:', f'{lm_head / 1e9 :.2f} GFLOPS', f'({lm_head / total * 100:.2f}%)')
    
    print('TOTAL:', f'{total} FLOPS', f'({total / total * 100:.2f}%)')
    

if __name__ == '__main__':
    print("-" * 30, "GPT-2 XL bs=1", "-" * 30)
    calculate_memory(vocab_size=50257, context_length=1024, num_layers=48, d_model=1600, num_heads=25, d_ff=6400, bs=1)

    print("-" * 30, "GPT-2 XL bs=2", "-" * 30)
    calculate_memory(vocab_size=50257, context_length=1024, num_layers=48, d_model=1600, num_heads=25, d_ff=6400, bs=2)
