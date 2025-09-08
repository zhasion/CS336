import os
import json
import sys
import argparse
import pathlib
import torch
import random
import vllm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

from cs336_alignment.helper import (
    init_vllm, gradient_clip, 
    tokenize_prompt_and_output, get_response_log_probs,
    sft_microbatch_train_step, load_policy_into_vllm_instance,
    get_logger, save_checkpoints)


def get_dataset(fpath: os.PathLike) -> list:
    with open(fpath, 'r', encoding='utf8') as f:
        dataset = [json.loads(x.strip()) for x in f]
    return dataset


def get_random_batch_data(train_dataset: list, batch_size: int):
    # sample_index = np.random.choice(np.arange(len(train_dataset)), size=(batch_size))
    # sampled_data = [train_dataset[idx] for idx in sample_index.tolist()]
    sampled_data = random.sample(population=train_dataset, k=batch_size)
    prompt = [x['prompt'] for x in sampled_data]
    response = [x['response'] for x in sampled_data]
    ground_truth = [x['ground_truth'] for x in sampled_data]
    return prompt, response, ground_truth


def get_seqence_batch_data(train_dataset: list, batch_size: int, start_idx: int):
    assert batch_size + start_idx < len(train_dataset)
    sampled_data = train_dataset[start_idx: start_idx + batch_size]
    prompt = [x['prompt'] for x in sampled_data]
    response = [x['response'] for x in sampled_data]
    ground_truth = [x['ground_truth'] for x in sampled_data]
    return prompt, response, ground_truth


def sft_eval(llm: vllm.LLM, eval_dataset: list[dict], tempalte_path: os.PathLike, seed: int = 42):
    
    with open(tempalte_path, 'r', encoding='utf8') as f:
        template = f.read()
    
    questions = [template.format(question=x['problem']) for x in eval_dataset]
    answers = [x['answer'] for x in eval_dataset]

    sampling_params = vllm.SamplingParams(
        n=1,
        temperature=1.0, 
        top_p=1.0, 
        seed=seed,
        stop=['</answer>'],
        include_stop_str_in_output=True,
        min_tokens=4,
        max_tokens=1024, 
    )

    outputs = llm.generate(questions, sampling_params)
    completes = []
    eval_results = []
    for output, answer in zip(outputs, answers):
        complete = output.outputs[0].text
        rewards = r1_zero_reward_fn(complete, answer)
        completes.append(complete)
        eval_results.append(rewards)
    format_accuracy = sum([reward['format_reward'] for reward in eval_results]) / len(eval_dataset)
    answer_accuracy = sum([reward['answer_reward'] for reward in eval_results]) / len(eval_dataset)

    return format_accuracy, answer_accuracy


def set_random_seed(seed: int = 42):
    torch.manual_seed(seed)
    random.seed(seed)


def sft_train(args):
    # logger setting
    set_random_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger(os.path.join(args.output_dir, 'main.log'))

    # torch setting
    if torch.cuda.is_available() and 'cuda' in args.device_train:
        torch.set_float32_matmul_precision("high")

    # model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        torch_dtype = torch.bfloat16,
        # attn_implementation = 'eager', # if don't use flash attention
        attn_implementation = 'flash_attention_2'
    ).to(device=args.device_train)
    torch.compile(model)
    llm = init_vllm(model_id=args.model_path, device=args.device_eval, seed=args.seed)
    
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # train dataset
    train_dataset = get_dataset(fpath = args.train_dataset_path)
    if args.max_unique_sample != -1:
        train_dataset = train_dataset[:args.max_unique_sample]
    if args.filter_correct:
        train_dataset = [sample for sample in train_dataset if r1_zero_reward_fn(sample["response"], sample["ground_truth"])["reward"] == 1]

    # val dataset
    validation_datset = get_dataset(fpath = args.eval_dataset_path)
    if args.num_eval < len(validation_datset):
        validation_datset = validation_datset[:args.num_eval]

    format_acc, answer_acc = sft_eval(llm, eval_dataset=validation_datset, tempalte_path=args.template_path, seed=args.seed)
    logger.info(f'Init VALIDATION | format_acc = {format_acc}, answer_acc = {answer_acc}')

    batch_size = args.batch_size
    micro_batch_size = args.batch_size // args.gradient_accumulation_steps

    nums_step = args.base_epoch * len(train_dataset) // args.batch_size 
    eval_step = nums_step // args.base_epoch

    lr_scheduler = get_scheduler(
        'cosine',
        optimizer=optimizer,
        # num_warmup_steps=int(0.03 * nums_step),
        num_warmup_steps=0,
        num_training_steps=nums_step
    )

    for step in range(1, nums_step + 1):
        for microstep in range(args.gradient_accumulation_steps):
            start_idx = step * batch_size + microstep * micro_batch_size
            prompt, response, gt = get_random_batch_data(train_dataset, micro_batch_size)
            # prompt, response, gt = get_seqence_batch_data(train_dataset, micro_batch_size, start_idx)
            tokenizations = tokenize_prompt_and_output(prompt, response, tokenizer, device=args.device_train)
            log_probs_dict = get_response_log_probs(model, tokenizations["input_ids"], tokenizations["labels"], return_token_entropy=True)
            log_probs, token_entropy = log_probs_dict["log_probs"], log_probs_dict["token_entropy"]
            loss, metadata = sft_microbatch_train_step(log_probs, tokenizations["response_mask"], args.gradient_accumulation_steps)
            
            logger.info(f'TRAIN | step: {step}, lr: {lr_scheduler.get_last_lr()[0]}, microstep: {microstep}, loss: {loss:.4f}')
        
        del tokenizations
        torch.cuda.empty_cache()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        # evaluatioe
        if step % eval_step == 0:
            load_policy_into_vllm_instance(model, llm)
            format_acc, answer_acc = sft_eval(llm, eval_dataset=validation_datset, tempalte_path=args.template_path, seed=args.seed)
            logger.info(f'VALIDATION | step: {step}, format_acc = {format_acc}, answer_acc = {answer_acc}')

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

def parse_args():
    parser = argparse.ArgumentParser()
    # Dataset and
    parser.add_argument('--model_path', default='models/Qwen2.5-Math-1.5B')
    parser.add_argument('--train_dataset_path', default='data/MATH/sft.jsonl')
    parser.add_argument('--eval_dataset_path', default='data/MATH/validation.jsonl')
    parser.add_argument('--template_path', default='cs336_alignment/prompts/r1_zero.prompt')
    parser.add_argument('--filter_correct', default=False)
    parser.add_argument('--max_unique_sample', default=-1)

    # Device setting.
    parser.add_argument('--seed', default=42)
    parser.add_argument('--device_train', default='cuda:1')
    parser.add_argument('--device_eval', default='cuda:0')

    # Optimizer.
    parser.add_argument('--lr', default=5e-5)

    # Log and checkpoints.
    parser.add_argument('--output_dir', default='exp_output/sft')

    # Train loop.
    parser.add_argument('--base_epoch', default=20)
    parser.add_argument('--batch_size', default=64)
    # parser.add_argument('--num_train_steps', default=-1)
    parser.add_argument('--gradient_accumulation_steps', default=64) # BUG: when gradient-accumulation != num-train-steps
    parser.add_argument('--eval_step', default=1)
    parser.add_argument('--num_eval', default=5000)
    

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    configs = [
        {'max_unique_sample': 128, 'filter_correct': False},
        # {'max_unique_sample': 256, 'filter_correct': False},
        # {'max_unique_sample': 512, 'filter_correct': False},
        # {'max_unique_sample': 1024, 'filter_correct': False},
        # {'max_unique_sample': -1, 'filter_correct': False},
        # {'max_unique_sample': -1, 'filter_correct': True},
    ]
    
    for config in configs:
        args.filter_correct = config['filter_correct']
        args.max_unique_sample = config['max_unique_sample']
        num_sample = config['max_unique_sample'] if config['max_unique_sample'] != -1 else 'all'
        args.output_dir = f"exp_output/sft_{num_sample}_filter_{config['filter_correct']}"
        sft_train(args)