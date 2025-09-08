import os
import sys
import argparse
import pathlib
import pandas as pd
import regex as re
from typing import List, Any
from vllm import LLM, SamplingParams
from glob import glob
sys.path.append(str(pathlib.Path(__file__).parent.parent))

mmlu_prompt_template = (
    "Answer the following multiple choice question about {subject}. Respond with a single "
    'sentence of the form "The correct answer is _", filling the blank with the letter '
    "corresponding to the correct answer (i.e., A, B, C or D).\n"
    "\n"
    "Question: {question}\n"
    "A. {option_1}\n"
    "B. {option_2}\n"
    "C. {option_3}\n"
    "D. {option_4}\n"
    "Answer:"
)

def parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    output = re.findall('the correct answer is ([abcd])', model_output.lower())
    if output:
        return output[0].upper()
    return None


def get_dataset(dir_path: os.PathLike):
    # read all csv files
    csv_files = glob(os.path.join(dir_path, "*.csv"))
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file, header=None, names=["question", "option_1", "option_2", "option_3", "option_4", "answer"])
        df["subject"] = (os.path.basename(file).split('.')[0].replace('_val', ''))
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True,)

    # use the prompt
    df["prompt"] = df.apply(
        lambda x: mmlu_prompt_template.format(
            question=x["question"],
            option_1=x["option_1"],
            option_2=x["option_2"],
            option_3=x["option_3"],
            option_4=x["option_4"],
            subject=x["subject"],
        ),
        axis=1,
    )
    return df


def eval_llm(vllm_model: LLM, prompts: List[str], answers: List[str], sampling_params: SamplingParams):
    outputs = vllm_model.generate(prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    rewards = [int(answer == parse_mmlu_response({}, response)) for answer, response in zip(answers, responses)]
    return responses, rewards


def main(args: dict):
    # get dataset
    df = get_dataset(args.eval_dir)
    prompts, answers = df["prompt"].tolist(), df["answer"].tolist()

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
    responses, rewards = eval_llm(llm, prompts, answers, sampling_params)

    df["response"] = responses
    df["reward"] = rewards
    print('-' * 50)
    print("MMLU Baseline Results")
    print(f'Accuracy: {sum(rewards) / len(rewards) * 100:.3f}% ({sum(rewards)}/{len(rewards)})')
    print('-' * 50)
    df.to_csv("data/mmlu_eval.csv", index=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='models/Qwen2.5-Math-1.5B')
    parser.add_argument('--eval_dir', default='data/mmlu/val')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)