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
    print(f'embedding: {embedding}, mha: {mha*num_layers}, rms: {rms*num_layers}, ffn: {ffn*num_layers}, final_rms: {final_rms}, lm_head: {lm_head}')
    print(f'total: {total_paras}')
    return total_paras


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
    count_parameters(vocab_size=50257, context_length=1024, num_layers=48, d_model=1600, num_heads=25, d_ff=6400)

    print("-" * 30, "GPT-2 XL", "-" * 30)
    calculate_flop(vocab_size=50257, context_length=1024, num_layers=48, d_model=1600, num_heads=25, d_ff=6400)
    
    print("-" * 30, "GPT-2 Large", "-" * 30)
    calculate_flop(vocab_size=50257, context_length=1024, num_layers=36, d_model=1280, num_heads=20, d_ff=6400)
    
    print("-" * 30, "GPT-2 Medium", "-" * 30)
    calculate_flop(vocab_size=50257, context_length=1024, num_layers=24, d_model=1024, num_heads=16, d_ff=6400)

    print("-" * 30, "GPT-2 Small", "-" * 30)
    calculate_flop(vocab_size=50257, context_length=1024, num_layers=12, d_model=768, num_heads=12, d_ff=6400)

    print("-" * 30, "GPT-2 XL with 16484 context_length", "-" * 30)
    calculate_flop(vocab_size=50257, context_length=16_384, num_layers=48, d_model=1600, num_heads=25, d_ff=6400)

    print("-" * 30, "small model", "-" * 30)
    count_parameters(vocab_size=10000, context_length=256, num_layers=4, d_model=512, num_heads=16, d_ff=1344)
    