import sys
import time
import os.path as osp
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from cs336_basics.tokenizer import *

root_path = str(Path(__file__).parent.parent)

def smaple_documents(
    file_path: str | os.PathLike,
    num_sample: int, 
    delimiter: str
) -> str:
    documents = []
    with open(file_path, 'r', encoding='utf8') as f:
        current_doc = ""
        while len(documents) < num_sample:
            line = f.readline()
            if delimiter in line:
                current_doc = current_doc + line[:line.index(delimiter) + len(delimiter)]
                documents.append(current_doc)
                current_doc = line[line.index(delimiter) + len(delimiter):]
            else:
                current_doc += line
    return '\n'.join(documents)


def save_tokenizer_data(corpus_path: str | os.PathLike, tokenizer: Tokenizer, save_path: str | os.PathLike) -> None:
    with open(corpus_path, 'r', encoding='utf8') as f:
        corpus = f.read()
    tokenizer_ids = tokenizer.encode(corpus)
    tokenizer_ids = np.array(tokenizer_ids, dtype=np.uint16)
    
    # np.save(save_path, tokenizer_ids)
    np.savez_compressed(save_path, tokenizer_ids)


def main():
    #------------------------------ init tokenizer ------------------------------ 
    tinystories_vocab, tinystories_merges = Tokenizer.load_vocab_merges_from_pkl(
        vocab_filepath = osp.join(root_path, 'output', 'tinystories_vocab.pkl'),
        merges_filepath = osp.join(root_path, 'output', 'tinystories_merges.pkl')
    )
    tinystories_tokenizer = Tokenizer(vocab=tinystories_vocab, merges=tinystories_merges, special_tokens=['<|endoftext|>'])

    owt_vocab, owt_merges = Tokenizer.load_vocab_merges_from_pkl(
        vocab_filepath = osp.join(root_path, 'output', 'owt_vocab.pkl'),
        merges_filepath = osp.join(root_path, 'output', 'owt_merges.pkl') 
    )
    owt_tokenizer = Tokenizer(vocab=owt_vocab, merges=owt_merges, special_tokens=['<|endoftext|>'])

    #------------------------------ sample documents ------------------------------ 
    tinystories_sample = smaple_documents(
        file_path=osp.join(root_path, 'data', 'TinyStoriesV2-GPT4-valid.txt'),
        num_sample=10,
        delimiter='<|endoftext|>'
    )

    owt_sample = smaple_documents(
        file_path=osp.join(root_path, 'data', 'owt_valid.txt'),
        num_sample=10,
        delimiter='<|endoftext|>'
    )

    #------------------------------ compression ratio ------------------------------ 
    tinystories_bytes = tinystories_sample.encode('utf-8')
    start_time = time.time()
    tinystories_token_ids = tinystories_tokenizer.encode(tinystories_sample)
    end_time = time.time()
    print('tinystories tokenizer compression ratio for tinystories sample:', len(tinystories_bytes) / len(tinystories_token_ids))
    print('tinystories tokenizer throughput:', len(tinystories_bytes) / (end_time - start_time))

    owt_bytes = owt_sample.encode('utf-8')
    start_time = time.time()
    owt_token_ids = owt_tokenizer.encode(owt_sample)
    end_time = time.time()
    print('owt tokenizer compression ratio for owt sample:', len(owt_bytes) / len(owt_token_ids))
    print('owt tokenizer throughput:', len(owt_bytes) / (end_time - start_time))


    tinystories_token_ids_encoded_by_owt = owt_tokenizer.encode(tinystories_sample)
    print('tinystories encoded by owt compression ratio bytes/token:', len(tinystories_bytes) / len(tinystories_token_ids_encoded_by_owt))

    owt_token_ids_encoded_by_tinystories = tinystories_tokenizer.encode(owt_sample)
    print('owt encoded by tinystorise compression ratio bytes/token:', len(owt_bytes) / len(owt_token_ids_encoded_by_tinystories))

    #------------------------------ saving dataset ------------------------------ 
    # exit()
    start_time = time.time()
    save_tokenizer_data(
        corpus_path=osp.join(root_path, 'data', 'owt_valid.txt'),
        tokenizer=tinystories_tokenizer,
        save_path=osp.join(root_path, 'data', 'owt_valid'),
    )
    print('Tokenizer owt_valid.txt cost time:', time.time() - start_time)
    
    start_time = time.time()
    save_tokenizer_data(
        corpus_path=osp.join(root_path, 'data', 'owt_train.txt'),
        tokenizer=tinystories_tokenizer,
        save_path=osp.join(root_path, 'data', 'owt_train'),
    )
    print('Tokenizer owt_train.txt cost time:', time.time() - start_time)

    start_time = time.time()
    save_tokenizer_data(
        corpus_path=osp.join(root_path, 'data', 'TinyStoriesV2-GPT4-valid.txt'),
        tokenizer=tinystories_tokenizer,
        save_path=osp.join(root_path, 'data', 'TinyStoriesV2-GPT4-valid'),
    )
    print('Tokenizer TinyStoriesV2-GPT4-valid.txt cost time:', time.time() - start_time)
    start_time = time.time()
    save_tokenizer_data(
        corpus_path=osp.join(root_path, 'data', 'TinyStoriesV2-GPT4-train.txt'),
        tokenizer=tinystories_tokenizer,
        save_path=osp.join(root_path, 'data', 'TinyStoriesV2-GPT4-train'),
    )
    print('Tokenizer TinyStoriesV2-GPT4-train.txt cost time:', time.time() - start_time)

if __name__ == '__main__':

    main()

