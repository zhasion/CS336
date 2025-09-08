# CS336 Assignment 5 (alignment): Alignment and Reasoning RL

[TOC]



## 3 Measuring Zero-Shot MATH Performance

### Problem (math_baseline): 4 points

(a) Write a script to evaluate Qwen 2.5 Math 1.5B zero-shot performance on MATH. This script should 

(1) load the MATH validation examples from /data/a5-alignment/MATH/validation.jsonl, (2) format them as string prompts to the language model using the r1_zero prompt, and (3) generate outputs for each example. This script should also (4) calculate evaluation metrics and (5) serialize the examples, model generations, and corresponding evaluation scores to disk for analysis in subsequent problems.

It might be helpful for your implementation to include a method evaluate_vllm with arguments similar to the following, as you will be able to reuse it later:

**Deliverable:** A script to evaluate baseline zero-shot MATH performance.

- [x] See function `evaluate_vllm` in `cs336_alignment/eval_metric.py`

(b) Run your evaluation script on Qwen 2.5 Math 1.5B. How many model generations fall into each of the following categories: (1) correct with both format and answer reward 1, (2) format reward 1 and answer reward 0, (3) format reward 0 and answer reward 0? Observing at least 10 cases where format reward is 0, do you think the issue is with the base model’s output, or the parser? Why? What about in (at least 10) cases where format reward is 1 but answer reward is 0?

**Deliverable:** Generally speaking, some examples where the reward format is 0 indicate that the model does not follow the output format specified in the instruction.

```python
# run shell
python cs336_alignment/eval_math_baseline.py
```

```json
{"n_format_correct":855.0,"n_answer_correct":145.0,"n_correct":145.0,"format_accuracy":0.171,"answer_accuracy":0.029,"accuracy":0.029}
```

| Format Reward | Answer Reward | n (of 5,000) |
| ------------- | ------------- | ------------ |
| 1             | 1             | 145          |
| 1             | 0             | 710          |
| 0             | 0             | 4,145        |

(c) How well does the Qwen 2.5 Math 1.5B zero-shot baseline perform on MATH?

**Deliverable:** The zero-shot baseline performs poorly on MATH.



## 4 Supervised Finetuning for MATH

### Problem (tokenize_prompt_and_output): Prompt and output tokenization (2 points)

**Deliverable:** Implement a method tokenize_prompt_and_output that tokenizes the question and output separately, concatenates them together, and constructs a response_mask. The following interface is recommended:

- [x] See function `tokenize_prompt_and_output` in `cs336_alignment/helper.py`

```shell
# test shell
pytest -k test_tokenize_prompt_and_output
```



### Problem (compute_entropy): Per-token entropy (1 point)

**Deliverable:** Implement a method compute_entropy that computes the per-token entropy of next-token predictions.

- [x] See function `compute_entropy` in `cs336_alignment/helper.py`

```shell
# test shell
pytest -k test_compute_entropy
```



### Problem (get_response_log_probs): Response log-probs (and entropy) (2 points)

**Deliverable:** Implement a method get_response_log_probs that gets per-token conditional log-probabilities (given the previous tokens) from a causal language model, and optionally the entropy of the model’s next-token distribution.

- [x] See function `get_response_log_probs` in `cs336_alignment/helper.py`

```shell
# test shell
pytest -k test_get_response_log_probs
```



### Problem (masked_normalize): Masked normalize (1 point)

**Deliverable:** Implement a method masked_normalize that sums over tensor elements and normalizes by a constant while respecting a boolean mask.

- [x] See function `masked_normalize` in `cs336_alignment/helper.py`

```shell
# test shell
pytest -k test_masked_normalize
```



### Problem (sft_microbatch_train_step): Microbatch train step (3 points)

**Deliverable:** Implement a single micro-batch update for SFT, including cross-entropy loss, summing with a mask, and gradient scaling.

- [x] See function `sft_microbatch_train_step` in `cs336_alignment/helper.py`

```shell
# test shell
pytest -k test_sft_microbatch_train_step
```



### Problem (log_generations): Logging generations (1 point)

pass



### Problem (sft_experiment): Run SFT on the MATH dataset (2 points) (2 H100 hrs)

1. Run SFT on the reasoning SFT examples (provided in /data/a5-alignment/MATH/sft.jsonl) using the Qwen 2.5 Math 1.5B base model, varying the number of unique examples for SFT in the range {128, 256, 512, 1024}, along with using the full dataset. Tune the learning rate and batch size to achieve at least 15% validation accuracy when using the full dataset.

	**Deliverable:** Validation accuracy curves associated with different dataset sizes.

2. Filter the reasoning SFT examples to only include examples that produce the correct answer. Run SFT on the (full) filtered dataset and report the size of the filtered dataset and the validation accuracy you achieve.

	**Deliverable:** Report the size of the dataset and the validation accuracy curve you achieve. Compare your findings to the previous SFT experiment.

- [x] See code in `cs336_alignment/sft.py`



## 5 Expert Iteration for MATH

### Problem (expert_iteration_experiment): Run expert iteration on the MATH dataset points) (6 H100 hrs)

Run expert iteration on the MATH dataset (provided at /data/a5-alignment/MATH/train.jsonl) using the Qwen 2.5 Math 1.5B Base model, varying the number of rollouts G per question and the number of epochs used in the SFT step, and using n_ei_steps = 5. Vary the batch size for each expert iteration step (i.e., the size of D b ) in {512, 1024, 2048}. (You do not need to try all possible combinations of these hyperparameters. Just enough to draw conclusions about each is fine.) Log the entropy of the model’s reponses over training. Make sure to have vLLM terminate generations at the second answer tag </answer>, as done in the SFT section.

**Deliverable:** Validation accuracy curves associated with different rollout configurations. Try at least 2 different rollout counts and epoch counts.

**Deliverable:** A model that achieves validation accuracy of at least 15% on MATH.

**Deliverable:** A brief 2 sentence discussion comparing to your SFT performance, as well as performance across EI steps.

**Deliverable:** A plot of the entropy of the model’s responses over training.

pass



## 7 Group Relative Policy Optimization

### Problem (compute_group_normalized_rewards): Group normalization (2 points)

**Deliverable:** Implement a method compute_group_normalized_rewards that calculates raw rewards for each rollout response, normalizes them within their groups, and returns both the normalized and raw rewards along with any metadata you think is useful.

- [x] See function `compute_group_normalized_rewards` in `cs336_alignment/grpo.py`

```shell
# test shell
pytest -k test_compute_group_normalized_rewards
```



### Problem (compute_naive_policy_gradient_loss): Naive policy gradient (1 point)

**Deliverable:** Implement a method compute_naive_policy_gradient_loss that computes the per-token policy-gradient loss using raw rewards or pre-computed advantages.

- [x] See function `compute_naive_policy_gradient_loss` in `cs336_alignment/grpo.py`

```shell
# test shell
pytest -k test_compute_naive_policy_gradient_loss
```



### Problem (compute_grpo_clip_loss): GRPO-Clip loss (2 points)

**Deliverable:** Implement a method compute_grpo_clip_loss that computes the per-token GRPO-Clip loss.

- [x] See function `compute_grpo_clip_loss` in `cs336_alignment/grpo.py`

```shell
# test shell
pytest -k test_compute_grpo_clip_loss
```



### Problem (compute_policy_gradient_loss): Policy-gradient wrapper (1 point)

**Deliverable:** Implement compute_policy_gradient_loss, a convenience wrapper that dispatches to the correct loss routine (no_baseline, reinforce_with_baseline, or grpo_clip) and returns both the per-token loss and any auxiliary statistics.

- [x] See function `compute_policy_gradient_loss` in `cs336_alignment/grpo.py`

```shell
# test shell
pytest -k test_compute_policy_gradient_loss
```



### Problem (masked_mean): Masked mean (1 point)

**Deliverable:** Implement a method masked_mean that averages tensor elements while respecting a boolean mask.

- [x] See function `masked_mean` in `cs336_alignment/grpo.py`

```shell
# test shell
pytest -k test_masked_mean
```



### Problem (grpo_microbatch_train_step): Microbatch train step (3 points)

**Deliverable:** Implement a single micro-batch update for GRPO, including policy-gradient loss, averaging with a mask, and gradient scaling.

- [x] See function `grpo_microbatch_train_step` in `cs336_alignment/grpo.py`

```shell
# test shell
pytest -k test_grpo_microbatch_train_step
```



### Problem (grpo_train_loop): GRPO train loop (5 points)

**Deliverable:** Implement a complete train loop for GRPO. Begin training a policy on MATH and confirm that you see validation rewards improving, along with sensible rollouts over time. Provide a plot with the validation rewards with respect to steps, and a few example rollouts over time.

- [x] See function `train_grpo` in `cs336_alignment/grpo.py`



### Problem (grpo_learning_rate): Tune the learning rate (2 points) (6 H100 hrs)

Starting with the suggested hyperparameters above, perform a sweep over the learning rates and report the final validation answer rewards (or note divergence if the optimizer diverges).

**Deliverable:** Validation reward curves associated with multiple learning rates.

**Deliverable:** A model that achieves validation accuracy of at least 25% on MATH. 

**Deliverable:** A brief 2 sentence discussion on any other trends you notice on other logged metrics.

pass



### Problem (grpo_baselines): Effect of baselining (2 points) (2 H100 hrs)

Train a policy with reinforce_with_baseline and with no_baseline.

**Deliverable:** Validation reward curves associated with each loss type.

**Deliverable:** A brief 2 sentence discussion on any other trends you notice on other logged metrics.

pass





### Problem (think_about_length_normalization): Think about length normalization (1point)

**Deliverable:** Compare the two approaches (without running experiments yet). What are the pros and cons of each approach? Are there any specific settings or examples where one approach seems better?

Normalization by sequence length makes the sequence loss and gradient contribution independent of the sequence length. If not used, it may motivate the model to output longer correct answers.





### Problem (grpo_length_normalization): Effect of length normalization (2 points) (2 H100hrs)

**Deliverable:** Compare normalization with masked_mean and masked_normalize with an end-to end GRPO training run. Report the validation answer reward curves. Comment on the findings, including any other metrics that have a noticeable trend.

Hint: consider metrics related to stability, such as the gradient norm.



### Problem (grpo_group_standard_deviation): Effect of standard deviation normalization points) (2 H100 hrs)

**Deliverable:** Compare the performance of use_std_normalization == True and use_std_ ⌋ normalization == False. Report the validation answer reward curves. Comment on the findings, including any other metrics that have a noticeable trend.

Hint: consider metrics related to stability, such as the gradient norm.



### Problem (grpo_off_policy): Implement off-policy GRPO

**Deliverable:** Implement off-policy GRPO training.

Depending on your implementation of the full GRPO train loop above, you may already have the infrastructure to do this. If not, you need to implement the following:

• You should be able to take multiple epochs of gradient steps per rollout batch, where the number of epochs and optimizer updates per rollout batch are controlled by rollout_batch_size, epochs_ ⌋ per_rollout_batch, and train_batch_size.

• Edit your main training loop to get response logprobs from the policy after each rollout batch generation phase and before the inner loop of gradient steps—these will be the old_log_probs. We suggest using torch.inference_mode().

• You should use the "GRPO-Clip" loss type.



### Problem (grpo_off_policy_sweep): Off-policy GRPO hyperparameter sweep (4 points) (12 H100 hrs)

Deliverable: Fixing rollout_batch_size = 256, choose a range over epochs_per_rollout_ ⌋ batch and train_batch_size to sweep over. First do a broad sweep for a limited number of GRPO steps (<50) to get a sense of the performance landscape, and then a more focused sweep for a larger number of GRPO steps (200). Provide a brief experiment log explaining the ranges you chose.

Compare to your on-policy run with epochs_per_rollout_batch = 1 and train_batch_size = 256, reporting plots with respect to number of validation steps as well as with respect to wall-clock time.

Report the validation answer reward curves. Comment on the findings, including any other metrics that have a noticeable trend such as entropy and response length. Compare the entropy of the model’s responses over training to what you observed in the EI experiment.

Hint: you will need to change gradient_accumulation_steps to keep memory usage constant.

pass



### Problem (grpo_off_policy_clip_ablation): Off-policy GRPO-Clip ablation (2 points) (2 H100 hrs) 

**Deliverable:** Implement the unclipped per-token loss as a new loss type "GRPO-No-Clip". Take your best performing off-policy hyperparameters from the previous problem and run the unclipped version of the loss. Report the validation answer reward curves. Comment on the findings compared to your GRPO-Clip run, including any other metrics that have a noticeable trend such as entropy, response length, and gradient norm



### Problem (grpo_prompt_ablation): Prompt ablation (2 points) (2 H100 hrs)

**Deliverable:** Report the validation answer reward curves for both the R1-Zero prompt and the question-only prompt. How do metrics compare, including any other metrics that have a noticeable trend such as entropy, response length, and gradient norm? Try to explain your findings.



## 9 Leaderboard: GRPO on MATH

pass
