import torch
import os
import yaml
import logging

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.module import TransformerLM
from cs336_basics.utils import *


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def load_config(config_path: str | os.PathLike) -> dict:
    with open(config_path, 'r', encoding='utf8') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {config_path}")
    return config

def build_tokenizer(config: dict):
    vocab, merges = Tokenizer.load_vocab_merges_from_pkl(
        vocab_filepath = config['tokenizer']['vocab_path'],
        merges_filepath = config['tokenizer']['merges_path'],
    )
    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=['<|endoftext|>'])

    return tokenizer

def build_model(config: dict):
    model = TransformerLM(
        vocab_size=config['model']['vocab_size'],
        context_length=config['model']['context_length'],
        d_model=config['model']['d_model'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        d_ff=config['model']['d_ff'],
        rope_theta=config['model']['rope_theta']
    )
    return model


def main(config: str):
    config = load_config(config)

    model = build_model(config=config)
    model.to(config.trainer.device)
    model.eval()

    tokenizer = build_tokenizer(config)
    model.load_state_dict(torch.load('output/checkpoint_iter_1000.pt')['model'])
    
    print(model.generate_text('Once upon a time', tokenizer, 100, 1, 1))

if __name__ == '__main__':
    main('config.yaml')