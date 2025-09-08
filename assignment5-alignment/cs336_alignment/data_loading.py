import transformers
import torch
import os
import json
import random
import gzip
import sys
import pathlib
from tqdm import tqdm
sys.path.append(str(pathlib.Path(__file__).parent.parent))


def get_alpaca_template():
    with open('cs336_alignment/prompts/alpaca_sft.prompt', 'r', encoding='utf8') as f:
        alpaca_template = f.read()
    return alpaca_template.strip()
    

class PackedSFTDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, dataset_path: str, seq_length: int, shuffle: bool):
        super().__init__()

        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.seq_length = seq_length
        self.shuffle = shuffle
        self.eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
        self.alpaca_template = get_alpaca_template()

        assert os.path.exists(dataset_path), f'{dataset_path} not exists!'

        self.input_chunks, self.label_chunks = self._load_and_process_data()


    def _load_and_process_data(self):
        assert str(self.dataset_path).endswith(('.gz', 'jsonl'))

        if str(self.dataset_path).endswith('.gz'):
            with gzip.open(self.dataset_path, 'rt') as f:
                instruct_datas = f.readlines
        else:
            with open (self.dataset_path, 'r', encoding='utf-8') as f:
                instruct_datas = f.readlines()

        instruct_datas = [json.loads(data.strip()) for data in instruct_datas]

        if self.shuffle:
            random.shuffle(instruct_datas)

        all_token_ids = []
        for data in tqdm(instruct_datas, desc="Processing instruct data"):
            formatted_text = self.alpaca_template.format(
                instruction=data['prompt'],
                response=data['response']
            )
            all_token_ids.extend(self.tokenizer.encode(formatted_text) + [self.eos_id])
        
        if len(all_token_ids) <= self.seq_length:
            return [], []

        input_sequence = all_token_ids[:-1]
        label_sequence = all_token_ids[1:]

        input_chunks = []
        label_chunks = []

        for idx in range(0, len(all_token_ids) // self.seq_length):
            start_idx = idx * self.seq_length
            end_idx = start_idx + self.seq_length
            input_chunks.append(input_sequence[start_idx:end_idx])
            label_chunks.append(label_sequence[start_idx:end_idx])

        return input_chunks, label_chunks
    

    def __len__(self):
        return len(self.input_chunks)


    def __getitem__(self, index):
        if index >= len(self.input_chunks):
            raise IndexError(f"Index {index} out of bounds for dataset with length {len(self)}")

        input_ids = self.input_chunks[index]
        labels = self.label_chunks[index]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def get_iterate_batches(dataset: torch.utils.data.Dataset, batch_size: int, shuffle: bool):
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader