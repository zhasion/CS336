import os
import argparse
import pandas as pd
from typing import List, Any
from vllm import LLM, SamplingParams
from glob import glob


def get_dataset(dir_path: os.PathLike):
    fpath = os.path.join(dir_path, 'simple_safety_tests.csv')
    df = pd.read_csv(fpath)
    return df

def eval_llm(vllm_model: LLM, prompts: List[str], sampling_params: SamplingParams):
    outputs = vllm_model.generate(prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    return responses

def main(args: dict):
    df = get_dataset(args.eval_dir)
    prompts_final = df['prompts_final'].tolist()

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
    responses = eval_llm(llm, prompts_final, sampling_params)

    df = pd.DataFrame(data={
        'prompts_final': prompts_final,
        'response': responses
    })
    df.to_json('data/sst_eval.jsonl', orient='records', lines=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='models/Qwen2.5-Math-1.5B')
    parser.add_argument('--eval_dir', default='data/simple_safety_tests')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)