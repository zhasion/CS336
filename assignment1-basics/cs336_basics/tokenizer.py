import os
import json
import regex as re
import pickle

from collections.abc import Iterable


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def word_to_bytes_tuple(word: str) -> tuple[bytes]:
    return tuple([bytes([x]) for x in word.encode('utf-8')])


def get_bytes_invisible_maps() -> dict[int, str]:
    id_to_char = dict()
    for id in list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1)):
        id_to_char[id] = chr(id)
    
    offset = 0
    for id in range(2**8):
        if id not in id_to_char.keys():
            id_to_char[id] = chr(2**8 + offset)
            offset += 1
    return id_to_char


class Tokenizer():
    def __init__(
        self,
        vocab:dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str]|None =None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self.merges_rank = {item:i for i, item in enumerate(merges)}

        if special_tokens is not None:
            for special_token in special_tokens:
                byte_special_token = special_token.encode('utf-8')
                if byte_special_token not in self.vocab.values():
                    self.vocab[len(self.vocab)] = byte_special_token
        else:
            self.special_tokens = []
        self.vocab_bytes_to_id = {bytes: id for id, bytes in self.vocab.items()}


    def load_vocab_merges_from_pkl(
        vocab_filepath: str | os.PathLike, 
        merges_filepath: str | os.PathLike, 
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        
        return vocab, merges


    def from_file(
        vocab_filepath: str | os.PathLike, 
        merges_filepath: str | os.PathLike, 
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        
        # GPT-2 remap the bytes to id.
        char_to_map_bytes_id = {v:k for k, v in get_bytes_invisible_maps().items()}

        # Load vocab.
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        vocab = {id : bytes([char_to_map_bytes_id[k] for k in item]) for item, id in vocab.items()}

        # Load merges.
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for each_line in f.readlines():
                clean_line = each_line.rstrip().split(' ')
                merges.append((
                    bytes([char_to_map_bytes_id[x] for x in clean_line[0]]),
                    bytes([char_to_map_bytes_id[x] for x in clean_line[1]]),
                ))

        return vocab, merges

    def encode_iterable(self, iterable: Iterable[str])-> Iterable[int]:
        for chunk in iterable:
            yield from self.encode(chunk)
    
    def encode(self, text:str)->list[int]:
        tokenizer_id_list = []

        # We need find the special token first.
        if self.special_tokens:
            sorted_special_token = sorted(self.special_tokens, key=len, reverse=True)
            special_token_pattern = "|".join(map(re.escape, sorted_special_token))
            sentences = re.split(f"({special_token_pattern})", text) # It's important to add () for the pattern.
        else:
            sentences = [text]
        
        # Encode for each sentence.
        for sentence in sentences:
            if sentence in self.special_tokens:
                tokenizer_id_list.append(self.vocab_bytes_to_id[sentence.encode('utf-8')])
            else:
                tokenizer_id_list.extend(self.encode_non_special(sentence))
        return tokenizer_id_list

    def encode_non_special(self, text:str) -> list[int]:

        # Pre-tokenization for the input sentence.
        pre_token_list = []
        for word in re.findall(PAT, text):
            pre_token_list.append(word_to_bytes_tuple(word))
        
        new_pre_token_list = []
        for pre_token in pre_token_list:
            while True:
                # List the bytes pairs for the pretoken and find whether is exist in merges.
                bytes_pairs = [(pre_token[i], pre_token[i+1]) for i in range(len(pre_token) - 1)]
                if not bytes_pairs:
                    break

                # The earliest byte pairs are added to the merge list, and the earliest byte pairs are used for merging.
                select_bytes_pair = min(bytes_pairs, key=lambda x: self.merges_rank.get(x, float('inf')))
                if select_bytes_pair not in self.merges_rank:
                    break
                
                # Apply the merges for pretoken.
                i = 0
                new_pre_token = []
                while i < len(pre_token):
                    if i < len(pre_token) - 1 and (pre_token[i], pre_token[i+1]) == select_bytes_pair:
                        new_pre_token.append(pre_token[i] + pre_token[i+1])
                        i+=2
                    else:
                        new_pre_token.append(pre_token[i])
                        i+=1
                if len(pre_token) == len(new_pre_token):
                    break
                else:
                    pre_token = new_pre_token

            new_pre_token_list.extend(pre_token)

        # Mapping the bytes to id.
        return [self.vocab_bytes_to_id[x] for x in new_pre_token_list]

    def decode(self, ids:list[int])->str:
        return b''.join([self.vocab[id] for id in ids]).decode('utf-8', errors='replace')


if __name__ == '__main__':
    # Use Lineprofile
    from line_profiler import LineProfiler

    vocab_input_path = "tests/fixtures/gpt2_vocab.json"
    merges_input_path = "tests/fixtures/gpt2_merges.txt"
    special_tokens = ['<|endoftext|>']

    vocab, merges = Tokenizer.from_file(vocab_input_path, merges_input_path)
    tokenizer = Tokenizer(vocab, merges, special_tokens)

    with open("tests/fixtures/tinystories_sample_5M.txt") as f:
        contents = f.read()
        lp = LineProfiler()
        lp.add_function(tokenizer.encode_non_special)
        test_func = lp(tokenizer.encode)
        test_func(contents)
        lp.print_stats(unit=1)

