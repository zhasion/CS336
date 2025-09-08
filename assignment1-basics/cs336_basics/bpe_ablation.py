import os
import regex as re
from tqdm import tqdm
from collections import defaultdict
from .bpe import (
    Timer, _initialize_vocabulary, _get_bytes_pair_freq, 
    PAT, word_to_bytes_tuple, pretokenize_parallelizing, pretokenize_without_parallelizing
)

def train_bpe_without_memory_opt(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    use_multiprocessing: bool = True
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # Initialize a timer
    timer = Timer()

    # Vocabulary initialization.
    vocab, next_id = _initialize_vocabulary(special_tokens=special_tokens)
    
    # Pretokenization.
    timer.start()
    pretoken_func = pretokenize_parallelizing if use_multiprocessing else pretokenize_without_parallelizing
    pre_token_freq = pretoken_func(input_path=input_path, special_tokens=special_tokens)
    pre_token_list = list(pre_token_freq.keys())
    timer.tick('pretokenize')

    # Count the frequency of byte pairs and in which pre-token a byte pair appears.
    bytes_pair_freq, bytes_pair_map_to_pre_token_idx = _get_bytes_pair_freq(pre_token_freq, pre_token_list)

    # Compute BPE merges.
    merges = []
    for _ in tqdm(range(vocab_size - len(vocab))):
        # Select the highest frequncy bytes pair and record to merge list.
        max_freq = max(bytes_pair_freq.values())
        select_bytes_pair = max([k for k, v in bytes_pair_freq.items() if v == max_freq])
        merges.append(select_bytes_pair)
        merge_token = select_bytes_pair[0] + select_bytes_pair[1]
        vocab[next_id] = merge_token
        next_id += 1

        # Update the pretoken infomation base on merge token.
        for pre_token_idx in bytes_pair_map_to_pre_token_idx[select_bytes_pair]:
            pre_token = pre_token_list[pre_token_idx]
            freq = pre_token_freq[pre_token]
            
            # Update the pretoken in which the bytes pair appears.
            i = 0
            new_pre_token = []
            pos = 0
            position_list = []
            while i < len(pre_token):
                if i < len(pre_token) - 1 and (pre_token[i], pre_token[i+1]) == select_bytes_pair:
                    new_pre_token.append(merge_token)
                    position_list.append(pos)
                    i += 2
                else:
                    new_pre_token.append(pre_token[i])
                    i += 1
                pos += 1

            new_pre_token = tuple(new_pre_token)
            pre_token_list[pre_token_idx] = new_pre_token
            del pre_token_freq[pre_token]
            pre_token_freq[new_pre_token] = freq

            # Update the bytes pair frequency based on the merge token.
            for pos in position_list:
                # Consider left token and current token also be merged.
                if pos > 0:
                    left_token = new_pre_token[pos - 1]
                    if merge_token == left_token:
                        bytes_pair_freq[(select_bytes_pair[1], select_bytes_pair[0])] -= freq
                    else:
                        bytes_pair_freq[(left_token, select_bytes_pair[0])] -= freq
                    bytes_pair_freq[(left_token, merge_token)] += freq
                    bytes_pair_map_to_pre_token_idx[(left_token, merge_token)].add(pre_token_idx)
                
                # Consider right token and current token also be merged.
                if pos < len(new_pre_token) - 1:
                    right_token = new_pre_token[pos + 1]
                    if merge_token == right_token:
                        bytes_pair_freq[(select_bytes_pair[1], select_bytes_pair[0])] -= freq
                    else:
                        bytes_pair_freq[(select_bytes_pair[1], right_token)] -= freq
                    bytes_pair_freq[(merge_token, right_token)] += freq
                    bytes_pair_map_to_pre_token_idx[(merge_token, right_token)].add(pre_token_idx)
        
        del bytes_pair_freq[select_bytes_pair]
        del bytes_pair_map_to_pre_token_idx[select_bytes_pair]
    timer.tick('merge BEP')
    timer.print_stats()
    return vocab, merges


def train_bep_before_merge_optimization(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    When I started writing, I was rather straightforward and didn't consider 
    the issue of merge optimization. For the sake of efficiency comparison, 
    I kept this code.
    """
    # Initialize a timer
    timer = Timer()

    # Vocabulary initialization.
    vocab, next_id = _initialize_vocabulary(special_tokens=special_tokens)
    
    timer.start()
    # Read corpus data.
    with open(input_path, 'r', encoding='utf-8') as f:
        corpus_data = f.read()
    
    # Removing special tokens before pre-tokenization.
    special_token_pat = "|".join(map(re.escape, special_tokens))
    sentences = re.split(special_token_pat, corpus_data)

    # Pre_tokenization and count the frequency.
    pre_token_freq = {}
    for sentence in sentences:
        for word in re.findall(PAT, sentence):
            pre_token = word_to_bytes_tuple(word)
            pre_token_freq[pre_token] = pre_token_freq.get(pre_token, 0) + 1
    timer.tick('pretokenize')
    
    # Compute BPE merges.
    merges = []
    for _ in tqdm(range(vocab_size - len(vocab))):
        # Count the frequency of byte pairs.
        bytes_pair_freq = defaultdict(int)
        for pre_token_idx, pre_token in enumerate(pre_token_freq.keys()):
            freq = pre_token_freq[pre_token]
            for i in range(len(pre_token) - 1):
                bytes_pair = (pre_token[i], pre_token[i+1])
                bytes_pair_freq[bytes_pair] += freq

        # Select the highest frequncy bytes pair and record to merge list.
        max_freq = max(bytes_pair_freq.values())
        select_bytes_pair = max([k for k, v in bytes_pair_freq.items() if v == max_freq])
        merges.append(select_bytes_pair)
        merge_token = select_bytes_pair[0] + select_bytes_pair[1]
        vocab[next_id] = merge_token
        next_id += 1

        # Update the pretoken infomation base on merge token.
        new_pre_token_freq = {}
        for pre_token, freq in pre_token_freq.items():
            merge_indices = [i for i in range(len(pre_token) - 1) 
                             if (pre_token[i], pre_token[i+1]) == select_bytes_pair]
            if merge_indices:
                i = 0
                new_pre_token = []
                while i < len(pre_token):
                    if i in merge_indices:
                        new_pre_token.append(merge_token)
                        i += 2
                    else:
                        new_pre_token.append(pre_token[i])
                        i += 1
                new_pre_token_freq[tuple(new_pre_token)] = freq
            else:
                new_pre_token_freq[pre_token] = freq
        pre_token_freq = new_pre_token_freq
    timer.tick('merge BEP')
    timer.print_stats()

    return vocab, merges
