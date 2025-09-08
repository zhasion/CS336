import os
import regex as re
import multiprocessing
import time
import pickle
import json

from typing import BinaryIO
from collections import defaultdict, Counter
from tqdm import tqdm

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Timer():
    def __init__(self):
        self.state = defaultdict(float)
        self.start_time = time.time()
        self.current_time = None

    def start(self) -> None:
        self.current_time = time.time()

    def tick(self, func_name: str) -> None:
        assert self.current_time is not None, 'timer need to start'
        interval_time = time.time() - self.current_time
        self.state[func_name] = interval_time
        self.current_time = time.time()

    def print_stats(self) -> None:
        self.state['total_time'] = time.time() - self.start_time
        for func_name, interval_time in self.state.items():
            print(f'{func_name}: {interval_time}')


def word_to_bytes_tuple(word: str) -> tuple[bytes]:
    return tuple([bytes([x]) for x in word.encode('utf-8')])


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _initialize_vocabulary(special_tokens: list[str]) -> tuple[dict[int, bytes], int]:
    vocab = {i:bytes([i]) for i in range(256)}
    next_id = 256

    special_tokens_bytes = [token.encode('utf-8') for token in special_tokens]
    for token in special_tokens_bytes:
        vocab[next_id] = token
        next_id += 1
    return vocab, next_id


def _get_bytes_pair_freq(
    pre_token_freq: dict[tuple[bytes], int], 
    pre_token_list: list[tuple[bytes]]
) -> tuple[
    dict[tuple[bytes], int], 
    dict[tuple[bytes], set]
]:
    bytes_pair_freq = defaultdict(int)
    bytes_pair_map_to_pre_token_idx = defaultdict(set)
    for pre_token_idx, pre_token in enumerate(pre_token_list):
        freq = pre_token_freq[pre_token]
        for i in range(len(pre_token) - 1):
            bytes_pair = (pre_token[i], pre_token[i+1])
            bytes_pair_freq[bytes_pair] += freq
            bytes_pair_map_to_pre_token_idx[bytes_pair].add(pre_token_idx)
    return bytes_pair_freq, bytes_pair_map_to_pre_token_idx


def get_chunk_pre_token_freq_worker(
    args: tuple[str | os.PathLike, int, int, list[str]],
) -> dict[tuple[bytes], int]:
    input_path, start, end, special_tokens = args

    # Read corpus data acoording to the start and end position.
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    
    # Removing special tokens before pre-tokenization.
    special_token_pat = "|".join(map(re.escape, special_tokens))
    sentences = re.split(special_token_pat, chunk)

    # Pre_tokenization and count the frequency.
    pre_token_freq = dict()
    for sentence in sentences:
        for word in re.findall(PAT, sentence):
            pre_token = word_to_bytes_tuple(word)
            pre_token_freq[pre_token] = pre_token_freq.get(pre_token, 0) + 1
    return pre_token_freq


def pretokenize_without_parallelizing(
    input_path: str | os.PathLike, 
    special_tokens: list[str],
    num_chunk: int = 10,
) -> dict[tuple[bytes], int]:
    
    # Find chunk boundary.
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunk, "<|endoftext|>".encode("utf-8"))
        
        pre_token_freq = {}
    
        for start, end in zip(boundaries, boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
        
            # Removing special tokens before pre-tokenization.
            special_token_pat = "|".join(map(re.escape, special_tokens))
            sentences = re.split(special_token_pat, chunk)

            # Pre_tokenization and count the frequency.
            for sentence in sentences:
                for word in re.findall(PAT, sentence):
                    pre_token = word_to_bytes_tuple(word)
                    pre_token_freq[pre_token] = pre_token_freq.get(pre_token, 0) + 1
    
    return pre_token_freq


def pretokenize_parallelizing(
    input_path: str | os.PathLike, 
    special_tokens: list[str],
    num_chunk: int = 10,
    num_processes: int = 10,
) -> dict[tuple[bytes], int]:
    
    # Find chunk boundary.
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunk, "<|endoftext|>".encode("utf-8"))
    
    # Parallelizing pre-tokenization.
    submits = [(input_path, start, end, special_tokens) for start, end in zip(boundaries, boundaries[1:])]
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(get_chunk_pre_token_freq_worker, submits)

    # Collect the result.input_path, start, end, special_tokens
    pre_token_freq = Counter()
    for each in results:
        pre_token_freq.update(each)

    return dict(pre_token_freq)


def save_vocab_and_merges(
    vocab: dict[int, bytes], 
    merges: list[tuple[bytes, bytes]], 
    vocab_save_path: str,
    merges_save_path: str,
) -> None:
    with open(vocab_save_path + '.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    with open(merges_save_path + '.pkl', 'wb') as f:
        pickle.dump(merges, f)
    for k, v in vocab.items():
            vocab[k] = str(v)

    with open(vocab_save_path + '.json', 'w', encoding='utf-8') as f:
        json.dump(vocab, f)

    with open(merges_save_path + '.txt', 'w', encoding='utf-8') as f:
        for each in merges:
            f.write(f'{each[0]} {each[1]}\n') 


def train_bpe(
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
        # print(len(pre_token_list), len(pre_token_freq), len(bytes_pair_freq), len(bytes_pair_map_to_pre_token_idx))
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
                        reduct_bytes_pair = (select_bytes_pair[1], select_bytes_pair[0])
                    else:
                        reduct_bytes_pair = (left_token, select_bytes_pair[0])

                    plus_bytes_pair = (left_token, merge_token)

                    bytes_pair_freq[reduct_bytes_pair] -= freq
                    bytes_pair_freq[plus_bytes_pair] += freq

                    bytes_pair_map_to_pre_token_idx[plus_bytes_pair].add(pre_token_idx)

                    if bytes_pair_freq[reduct_bytes_pair] == 0:
                        del bytes_pair_freq[reduct_bytes_pair]
                
                # Consider right token and current token also be merged.
                if pos < len(new_pre_token) - 1:
                    right_token = new_pre_token[pos + 1]
                    if merge_token == right_token:
                        reduct_bytes_pair = (select_bytes_pair[1], select_bytes_pair[0])
                    else:
                        reduct_bytes_pair = (select_bytes_pair[1], right_token)

                    plus_bytes_pair = (merge_token, right_token)

                    bytes_pair_freq[reduct_bytes_pair] -= freq
                    bytes_pair_freq[plus_bytes_pair] += freq
                    
                    bytes_pair_map_to_pre_token_idx[plus_bytes_pair].add(pre_token_idx)

                    if bytes_pair_freq[reduct_bytes_pair] == 0:
                        del bytes_pair_freq[reduct_bytes_pair]
        
        del bytes_pair_freq[select_bytes_pair]
        del bytes_pair_map_to_pre_token_idx[select_bytes_pair]
    timer.tick('merge BEP')
    timer.print_stats()
    return vocab, merges


if __name__ == '__main__':
    input_path = "test.txt"
    special_tokens = ['<|endoftext|>']
    vocab_size = 256+8
    train_bpe(input_path, vocab_size, special_tokens)