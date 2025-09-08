import vllm
import argparse
import json
import os
import sys
import pathlib
import torch

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.eval_metric import evaluate_vllm

def main(
    model_path: os.PathLike, 
    valid_path: os.PathLike, 
    prompt_template_path: os.PathLike
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Read validation dataset. 
    with open(valid_path, 'r', encoding='utf8') as f:
        valid_data = [json.loads(x) for x in f]

    # Read the prepared prompt template.
    with open(prompt_template_path, 'r', encoding='utf8') as f:
        prompt_template = f.read()

    # Extract the problems and answers.
    prompts = [prompt_template.format(question = x['problem']) for x in valid_data]
    answers = [x['answer'] for x in valid_data]

    # Evaluate the LLM model based on the reward function.
    evaluate_vllm(
        llm=vllm.LLM(model_path, device=device, dtype=torch.bfloat16, enable_prefix_caching=True),
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        answers=answers
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-path', 
        default='models/Qwen2.5-Math-1.5B')
    parser.add_argument(
        '--valid-path', 
        default='data/MATH/validation.jsonl')
    parser.add_argument(
        '--prompt-template-path', 
        default='cs336_alignment/prompts/r1_zero.prompt')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    
    main(
        model_path = args.model_path,
        valid_path = args.valid_path,
        prompt_template_path = args.prompt_template_path
    )