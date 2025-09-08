import os
import argparse
import pandas as pd
import json
from typing import List, Any
from vllm import LLM, SamplingParams


def get_dataset(dir_path: os.PathLike):
    fpath = os.path.join(dir_path, 'alpaca_eval.jsonl')
    with open(fpath, 'r', encoding='utf8') as f:
        data = [json.loads(x.strip()) for x in f]
    instructions = [x['instruction'] for x in data]
    outputs = [x['output'] for x in data]
    generators = [x['generator'] for x in data]
    datasets = [x['dataset'] for x in data]
    return instructions, outputs, generators, datasets  


def eval_llm(vllm_model: LLM, prompts: List[str], sampling_params: SamplingParams):
    outputs = vllm_model.generate(prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    return responses

def main(args: dict):
    instructions, outputs, generators, datasets = get_dataset(args.eval_dir)

    # model init
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        min_tokens=4,
        max_tokens=1024,
        include_stop_str_in_output=True,
    )
    llm = LLM(model=args.model_path)

    # llm generate and evaluate
    responses = eval_llm(llm, instructions, sampling_params)

    df = pd.DataFrame(data={
        'instruction': instructions,
        'response': responses
    })
    df['generator'] = os.path.basename(args.model_path)
    df['dataset'] = datasets
    df.to_json('data/alpaca_eval.json', orient='records', indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='models/Qwen2.5-Math-1.5B')
    parser.add_argument('--eval_dir', default='data/alpaca_eval')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)