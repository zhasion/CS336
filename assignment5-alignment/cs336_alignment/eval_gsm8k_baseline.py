import os
import argparse
import regex as re
import json
from typing import List, Any
from vllm import LLM, SamplingParams

gsm8k_prompt_template = (
    '{question} \n'
    'Answer:\n'
    'In the prompt above, question refers to the GSM8K question (e.g., Natalia sold clips to 48 of her '
    'friends in April, and then she sold half as many clips in May. How many clips did Natalia sell '
    'altogether in April and May?).'
)


def parse_gsm8k_response(
    model_output: str,
) -> str | None:
    output = re.findall('\d+', model_output)
    if output:
        return output[-1]
    return None



def get_dataset(dir_path: os.PathLike):
    fpath = os.path.join(dir_path, 'test.jsonl')
    with open(fpath, 'r', encoding='utf8') as f:
        data = [json.loads(x.strip()) for x in f]
    questions = [gsm8k_prompt_template.format(question = x['question']) for x in data]
    answers = [x['answer'] for x in data]
    return questions, answers  


def eval_llm(vllm_model: LLM, prompts: List[str], answers: List[str], sampling_params: SamplingParams):
    outputs = vllm_model.generate(prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    rewards = [int(answer == parse_gsm8k_response(response)) for answer, response in zip(answers, responses)]
    return responses, rewards

def main(args: dict):
    questions, answers = get_dataset(args.eval_dir)

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
    responses, rewards = eval_llm(llm, questions, answers, sampling_params)

    print('-' * 50)
    print("GSM8K Baseline Results")
    print(f'Accuracy: {sum(rewards) / len(rewards) * 100:.3f}% ({sum(rewards)}/{len(rewards)})')
    print('-' * 50)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='models/Qwen2.5-Math-1.5B')
    parser.add_argument('--eval_dir', default='data/gsm8k')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)