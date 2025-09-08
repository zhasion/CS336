import os
import vllm
import time
from typing import Callable
from pydantic import BaseModel


class MetricInfo(BaseModel):
    n_format_correct: float = 0.0
    n_answer_correct: float = 0.0
    n_correct: float = 0.0
    format_accuracy: float = 0.0
    answer_accuracy: float = 0.0
    accuracy: float = 0.0


class EvalInfo(BaseModel):
    prompt: str
    answer: str
    complete: str
    reward: dict[str, float]


def get_timestamp():
    now = time.localtime(time.time())
    return time.strftime('%Y_%m_%d_%H_%M_%S', now)


def evaluate_vllm(
    llm: vllm.LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    answers: list[str],
    eval_sampling_params: vllm.SamplingParams | None = None
):
    # Init the sampling parameters for model inference.
    if eval_sampling_params is None:
        eval_sampling_params = vllm.SamplingParams(
            temperature=1.0,
            top_p=1.0,
            min_tokens=0,
            max_tokens=1024,
            stop=['</answer>'],
            include_stop_str_in_output=True
        )

    # Inference.
    outputs = llm.generate(
        prompts,
        eval_sampling_params,
    )

    # Compare the output and the gt answer, record the reward.
    metricinfo = MetricInfo()
    eval_result_list = list()
    for i, output in enumerate(outputs):
        prompt = output.prompt
        complete = output.outputs[0].text 
        answer = answers[i]

        rewards = reward_fn(complete, answer)

        format_reward = rewards['format_reward']
        answer_reward = rewards['answer_reward']
        correct = rewards['reward']

        metricinfo.n_format_correct += format_reward
        metricinfo.n_answer_correct += answer_reward
        metricinfo.n_correct += correct

        eval_result_list.append(
            EvalInfo(
                prompt=prompt,
                answer=answer,
                complete=complete,
                reward=rewards
            )
        )
    
    # Calculate the accuracy.
    metricinfo.format_accuracy = metricinfo.n_format_correct / len(answers)
    metricinfo.answer_accuracy = metricinfo.n_answer_correct / len(answers)
    metricinfo.accuracy = metricinfo.n_correct / len(answers)

    # Save the record data.
    os.makedirs('eval_result', exist_ok=True)
    output_path = os.path.join('eval_result', 'eval_' + get_timestamp() + '.jsonl')

    with open(output_path, 'w', encoding='utf8') as f:
        f.write(metricinfo.model_dump_json() + '\n')
        f.write('\n'.join([x.model_dump_json() for x in eval_result_list]))