import os
import os.path as osp
import sys
from pathlib import Path
sys.path.append(Path(__file__).parent)

from memory_profiler import memory_usage
from cs336_basics.bpe import *

if __name__ == '__main__':

    # Basic setting.
    input_path = "data/owt_train.txt"
    maxmium_vocab_size = 32_000
    special_tokens = ['<|endoftext|>']
    use_pickle_save = True

    # Train BPE.
    mem_usage_before = memory_usage(-1, interval=0.1, timeout=5)[0]
    vocab, merges = train_bpe(
        input_path=input_path, 
        vocab_size=maxmium_vocab_size, 
        special_tokens=special_tokens,
        use_multiprocessing=True,
    )
    mem_usage_after = memory_usage(-1, interval=0.1, timeout=5)[0]
    
    print(f"RAM before use: {mem_usage_before} MB")
    print(f"RAM after use: {mem_usage_after} MB")
    print(f"Peak memory usage: {max(memory_usage(proc=-1, interval=0.1, timeout=1))} MB")
    print(f"Memory increase: {mem_usage_after - mem_usage_before} MB")

    # Save the final result to the disk.
    save_dir = osp.join(osp.dirname(osp.dirname(input_path)), 'output')
    os.makedirs(save_dir, exist_ok=True)

    vocab_save_path = osp.join(save_dir, 'owt_vocab')
    merges_save_path = osp.join(save_dir, 'owt_merges')

    save_vocab_and_merges(
        vocab=vocab,
        merges=merges,
        vocab_save_path=vocab_save_path,
        merges_save_path=merges_save_path,
    )

    print('longest token in the vocabulary is:', sorted(list(vocab.values()), key=len, reverse=True)[0])




