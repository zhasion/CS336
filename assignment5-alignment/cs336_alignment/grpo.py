import torch
import argparse
import random
import json
import os
import pathlib
import sys
from jaxtyping import Float
from typing import Callable, Literal
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

from cs336_alignment.helper import (
    init_vllm, tokenize_prompt_and_output, 
    get_response_log_probs, get_logger,
    gradient_clip, load_policy_into_vllm_instance)
from cs336_alignment.sft import sft_eval

def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
    device: str = 'cpu'
) -> tuple[torch.Tensor, dict[str, float]]:
    
    rewards = [reward_fn(rollout, gt)['answer_reward'] for rollout, gt in zip(rollout_responses, repeated_ground_truths)]
    rewards = torch.tensor(rewards, device=device).reshape(-1, group_size)

    advantage = rewards - rewards.mean(dim=-1, keepdim=True)
    if normalize_by_std is True:
        advantage = advantage / (rewards.std(dim=-1, keepdim=True) + advantage_eps)
    
    metadata = {}
    return advantage.flatten(), rewards.flatten(), metadata
    

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor, 
    policy_log_probs: torch.Tensor
) -> torch.Tensor:
    
    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor, 
    policy_log_probs: torch.Tensor, 
    old_log_probs: torch.Tensor, 
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    important_weight = (policy_log_probs - old_log_probs).exp()
    clipped_important_weight = torch.clip(important_weight, 1 - cliprange, 1 + cliprange)

    grpo_clip_loss = torch.minimum(important_weight * advantages, clipped_important_weight * advantages)
    
    metadata = {}

    return -grpo_clip_loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: Float[torch.Tensor, 'bs seq'], 
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
    raw_rewards: Float[torch.Tensor, 'bs 1'] | None = None, 
    advantages: Float[torch.Tensor, 'bs 1'] | None = None, 
    old_log_probs: Float[torch.Tensor, 'bs seq'] | None = None, 
    cliprange: float | None = None, 
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == 'no_baseline':
        metadata = {}
        return compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs
        ), metadata
    elif loss_type == 'reinforce_with_baseline':
        metadata = {}
        return compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs
        ), metadata
    elif loss_type == 'grpo_clip':
        return compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange
        )
    else:
        raise ValueError('loss_type is invalid!')
    

def masked_mean(
    tensor: torch.Tensor, 
    mask: torch.Tensor, 
    dim: int | None = None, 
) -> torch.Tensor:
    masked = torch.where(mask==1, tensor, 0)
    return masked.sum(dim=dim) / mask.sum(dim=dim)


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor, 
    response_mask: torch.Tensor, 
    gradient_accumulation_steps: int, 
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
    raw_rewards: torch.Tensor | None = None, 
    advantages: torch.Tensor | None = None, 
    old_log_probs: torch.Tensor | None = None, 
    cliprange: float | None = None, 
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    policy_gradient_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange
    )

    masked_loss = masked_mean(
        tensor=policy_gradient_loss,
        mask=response_mask,
        dim=-1
    )

    loss = masked_loss.mean() / gradient_accumulation_steps

    loss.backward()
    return loss, metadata


def get_dataset(fpath: os.PathLike) -> list:
    with open(fpath, 'r', encoding='utf8') as f:
        dataset = [json.loads(x.strip()) for x in f]
    return dataset


def get_template(fpath: os.PathLike) -> str:
    with open(fpath, 'r', encoding='utf8') as f:
        template = f.read()
    return template

def set_random_seed(seed: int = 42):
    torch.manual_seed(seed)
    random.seed(seed)


def train_grpo(args):
    set_random_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger(log_path=os.path.join(args.output_dir, 'main.log'))

    # Assert check.
    assert args.train_batch_size % args.gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps" 
    ) 
    micro_train_batch_size = args.train_batch_size // args.gradient_accumulation_steps 

    assert args.rollout_batch_size % args.group_size == 0, (
        "rollout_batch_size must be divisible by group_size" 
    ) 
    n_prompts_per_rollout_batch = args.rollout_batch_size // args.group_size 

    assert args.train_batch_size >= args.group_size, (
        "train_batch_size must be greater than or equal to group_size" 
    ) 
    # n_microbatches_per_rollout_batch = args.rollout_batch_size // micro_train_batch_size


    # Constuct model, tokenizer and optimizer.
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        torch_dtype = torch.bfloat16,
        attn_implementation = 'flash_attention_2',
        # attn_implementation = 'eager'
    ).to(device=args.device_train)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.learning_rate,
        betas=args.betas,
        weight_decay=args.weight_decay
    )
    vllm_model = init_vllm(
        model_id=args.model_path,
        device=args.device_eval,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization
    )

    sampling_params = SamplingParams(
        n=args.group_size, 
        temperature=args.sampling_temperature, 
        top_p=1.0, 
        seed=args.seed,
        stop=["</answer>"], 
        include_stop_str_in_output=True, 
        min_tokens=args.sampling_min_tokens, 
        max_tokens=args.sampling_max_tokens, 
    )


    # Compile model.
    if torch.cuda.is_available() and 'cuda' in args.device_train:
        torch.set_float32_matmul_precision('high')
        torch.compile(model)


    prompt_template = get_template(args.template_path)
    train_dataset = get_dataset(args.train_dataset_path)


    for grpo_step in range(args.n_grpo_steps):
        rollout_sammple_data = random.sample(train_dataset, k=n_prompts_per_rollout_batch)

        rollout_prompts = [prompt_template.format(question=x['problem']) for x in rollout_sammple_data]
        rollout_answers = [x["answer"] for x in rollout_sammple_data]

        outputs = vllm_model.generate(rollout_prompts, sampling_params)
                           
        repeated_answers = []
        generations = []
        prompts = []
        for output, answer in zip(outputs, rollout_answers):
            for i in range(args.group_size):
                generated_text = output.outputs[i].text
                prompts.append(output.prompt)
                generations.append(generated_text)
                repeated_answers.append(answer)

        advantages, raw_rewards, metadata = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn, 
            rollout_responses=generations, 
            repeated_ground_truths=repeated_answers, 
            group_size=args.group_size, 
            advantage_eps=args.advantage_eps, 
            normalize_by_std=args.use_std_normalization, 
            device=args.device_train
        )

        tokenizations = tokenize_prompt_and_output(prompts, generations, tokenizer, args.device_train)
        
        n_train_steps_per_epoch = args.rollout_batch_size // args.train_batch_size
        loop_shape = (n_train_steps_per_epoch, args.gradient_accumulation_steps, micro_train_batch_size)
        
        input_ids = tokenizations["input_ids"].reshape(*loop_shape, -1)
        labels = tokenizations["labels"].reshape(*loop_shape, -1)
        response_mask = tokenizations["response_mask"].reshape(*loop_shape, -1)


        with torch.no_grad():
            old_log_probs = []
            for train_step in range(n_train_steps_per_epoch):
                for micro_step in range(args.gradient_accumulation_steps):
                    micro_input_ids = input_ids[train_step][micro_step]
                    micro_labels = labels[train_step][micro_step]

                    # Compute old log_probs.
                    log_probs_dict = get_response_log_probs(model, micro_input_ids, micro_labels, return_token_entropy=True)
                    log_probs, token_entropy = log_probs_dict["log_probs"], log_probs_dict["token_entropy"]
                    old_log_probs.append(log_probs)
            old_log_probs = torch.cat(old_log_probs)
            old_log_probs = old_log_probs.reshape(*loop_shape, -1)

        advantages = advantages.reshape(*loop_shape, -1)
        raw_rewards = raw_rewards.reshape(*loop_shape, -1)

        for reply_epoch in range(args.epochs_per_rollout_batch):
            for train_step in range(n_train_steps_per_epoch):
                for micro_step in range(args.gradient_accumulation_steps):
                    micro_advantages = advantages[train_step][micro_step]
                    micro_raw_rewards = raw_rewards[train_step][micro_step]
                    micro_old_log_probs = old_log_probs[train_step][micro_step]
                    micro_input_ids = input_ids[train_step][micro_step]
                    micro_labels = labels[train_step][micro_step]
                    micro_response_mask = response_mask[train_step][micro_step]

                    # Compute new log_probs.
                    log_probs_dict = get_response_log_probs(model, micro_input_ids, micro_labels, return_token_entropy=True)
                    policy_log_probs, token_entropy = log_probs_dict["log_probs"], log_probs_dict["token_entropy"]
                    
                    loss, metadata = grpo_microbatch_train_step(
                        policy_log_probs=policy_log_probs,
                        response_mask=micro_response_mask,
                        gradient_accumulation_steps=args.gradient_accumulation_steps,
                        loss_type=args.loss_type,
                        raw_rewards=micro_raw_rewards,
                        advantages=micro_advantages,
                        old_log_probs=micro_old_log_probs,
                        cliprange=args.cliprange
                    )

                    logger.info(f'TRAIN | grpo_step: {grpo_step}, reply_epoch: {reply_epoch}, train_step: {train_step}, micro_step: {micro_step}, loss = {loss:.4f}')
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        load_policy_into_vllm_instance(model, vllm_model)  
        if grpo_step != 0 and grpo_step % args.grpo_eval_freq == 0:
            format_acc, answer_acc = sft_eval(
                llm=vllm_model, 
                eval_data_path=args.val_dataset_path, 
                num_eval=args.num_eval, 
                tempalte_path=args.template_path,
                seed=args.seed
            )
            logger.info(f"VAL | grpo_step: {grpo_step}, reply_epoch: {reply_epoch}, train_step: {train_step}, format_acc = {format_acc:.4f}, answer_acc = {answer_acc:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    # Load file path.
    parser.add_argument('--model_path', type=str,  default='models/Qwen2.5-Math-1.5B')
    parser.add_argument('--template_path', default='cs336_alignment/prompts/r1_zero.prompt')
    parser.add_argument('--train_dataset_path', type=str,  default='data/MATH/train.jsonl')
    parser.add_argument('--val_dataset_path', type=str,  default='data/MATH/validation.jsonl')

    # Device setting.
    parser.add_argument('--seed', default=42)
    parser.add_argument('--device_train', default='cuda:1')
    parser.add_argument('--device_eval', default='cuda:0')

    # Smapler and inference parameter.
    parser.add_argument('--sampling_temperature', type=float,  default=1.0)
    parser.add_argument('--sampling_min_tokens', type=int,  default=4)
    parser.add_argument('--sampling_max_tokens', type=int,  default=1024)
    parser.add_argument('--gpu_memory_utilization', type=float,  default=0.85)
    parser.add_argument("--cliprange", type=float, default=0.2)

    # Train loop.
    parser.add_argument('--n_grpo_steps', type=int,  default=200)
    parser.add_argument('--epochs_per_rollout_batch', type=int,  default=1)
    parser.add_argument('--rollout_batch_size', type=int,  default=32)
    parser.add_argument('--train_batch_size', type=int,  default=32)
    parser.add_argument('--gradient_accumulation_steps', type=int,  default=32)
    parser.add_argument('--group_size', type=int,  default=8)
    parser.add_argument('--grpo_eval_freq', default=20)
    parser.add_argument('--num_eval', default=5000)

    # Optimizer.
    parser.add_argument('--learning_rate', type=float,  default=1e-5)
    parser.add_argument('--weight_decay', type=float,  default=0.0)
    parser.add_argument('--betas', type=tuple,  default=(0.9, 0.95))

    # Log and checkpoints.
    parser.add_argument('--output_dir', default='exp_output/grpo')

    # Policy gradient parameter.
    parser.add_argument('--advantage_eps', type=float,  default=1e-6)
    parser.add_argument('--use_std_normalization', type=bool,  default=True)
    parser.add_argument(
        '--loss_type', 
        type=str, 
        default='reinforce_with_baseline', 
        choices=['no_baseline', 'reinforce_with_baseline', 'grpo_clip']
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    cfg = parse_args()
    train_grpo(cfg)