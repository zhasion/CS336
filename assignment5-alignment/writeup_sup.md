# CS336 Assignment 5 Supplement (alignment): Instruction Tuning and RLHF

[TOC]



## 2 Motivation: Training Generalist LLMs

### Problem (mmlu_baseline): 4 points

(a) Write a function to parse generated language model outputs into the letter corresponding to the predicted answer. If model response cannot be parsed, return None.

- [x] See function `parse_mmlu_response` in `cs336_data/eval_mmlu_baseline.py`

```python
# test shell
pytest -k test_parse_mmlu_response
```

(b) Write a script to evaluate Llama 3.1 8B zero-shot performance on MMLU. This script should

(1) load the MMLU examples, (2) format them as string prompts to the language model, and

(3) generate outputs for each example. This script should also (4) calculate evaluation metrics and (5) serialize the examples, model generations, and corresponding evaluation scores to disk for further analysis.

**Deliverable:** A script to evaluate baseline zero-shot MMLU performance.

```python
# run shell
python cs336_alignment/mmlu_baseline.py
```

<font color="purple">**Due to resource limitations, the following questions are skipped.**</font>

(c) Run your evaluation script on Llama 3.1 8B. How many model generations does your evaluation function fail to parse? If non-zero, what do these examples look like?

**Deliverable:** Number of model generations that failed parsing. If non-zero, a few examples of generations that your function wasn’t able to parse.

(d) How long does it take the model to generate responses to each of the MMLU examples? Estimate the throughput in examples/second.

**Deliverable:** Estimate of MMLU examples/second throughput.

(e) How well does the Llama 3.1 8B zero-shot baseline perform on MMLU?

**Deliverable:** 1-2 sentences with evaluation metrics.

(f) Sample 10 random incorrectly-predicted examples from the evaluation dataset. Looking through the examples, what sort of errors does the language model make?

**Deliverable:** A 2-4 sentence error analysis of model predictions, including examples and/or model responses as necessary.



### Problem (gsm8k_baseline): 4 points

(a) Write a function to parse generated language model outputs into a single numeric prediction. If model response cannot be parsed, return None. 

**Deliverable:** A function to parse generated predictions on GSM8K into a single numeric answer.

- [x] See function `parse_mmlu_response` in `cs336_alignment/eval_gsm8k_baseline.py`

```python
# test shell
pytest -k test_parse_gsm8k_response
```

(b) Write a script to evaluate Llama 3.1 8B zero-shot performance on GSM8K. This script should

(1) load the GSM8K examples, (2) format them as string prompts to the language model, and (3) generate outputs for each example. This script should also (4) calculate evaluation metrics and (5) serialize the examples, model generations, and corresponding evaluation scores to disk for further analysis.

**Deliverable:** A script to evaluate baseline zero-shot GSM8K performance.

```shell
# run shell
python cs336_alignment/gsm8k_baseline.py
```

<font color="purple">**Due to resource limitations, the following questions are skipped.**</font>

(c) Run your evaluation script on Llama 3.1 8B. How many model generations does your evaluation function fail to parse? If non-zero, what do these examples look like?

**Deliverable:** Number of model generations that failed parsing. If non-zero, a few examples of generations that your function wasn’t able to parse.

(d) How long does it take the model to generate responses to each of the GSM8K examples? Estimate the throughput in examples/second.

**Deliverable:** Estimate of GSM8K examples/second throughput.

(e) How well does the Llama 3.1 8B zero-shot baseline perform on GSM8K?

**Deliverable:** 1-2 sentences with evaluation metrics.

(f) Sample 10 random incorrectly-predicted examples from the evaluation dataset. Looking through the examples, what sort of errors does the language model make?

**Deliverable:** A 2-4 sentence error analysis of model predictions, including examples and/or model responses as necessary.



### Problem (alpaca_eval_baseline): 4 points

(a) Write a script to collect Llama 3.1 8B zero-shot predictions on AlpacaEval. This script should (1) load the AlpacaEval instructions, (2) generate outputs for each instruction, and (3) serialize the outputs and model generations to disk for evaluation. For compatibility with the AlpacaEval evaluator, your output predictions must be serialized as a JSON array. Each entry of this JSON array should contain a JSON object with the following keys:

• instruction: the instruction.

• output: the output of the model, given the instruction.

• generator: a string identifier corresponding to the name of the model that generated the output (e.g., llama-3.1-8b-base). This should be the same across all entries in the JSON array.

• dataset: a string identifier that indicates which dataset the instruction comes from. This is provided in the original AlpacaEval dataset.

**Deliverable:** A script to generate zero-shot outputs on AlpacaEval.

```shell
# run shell
python cs336_alignment/eval_alpaca_baseline.py
```

<font color="purple">**Due to resource limitations, the following questions are skipped.**</font>

(b) How long does it take the model to generate responses to each of the AlpacaEval examples?

Estimate the throughput in examples/second.

**Deliverable:** Estimate of AlpacaEval examples/second throughput.

(c) To measure our model’s performance on AlpacaEval, we’ll use Llama 3.3 70B Instruct as the annotator and compare our outputs against GPT-4 Turbo. To compute the winrate, run the following command (requires two GPUs, each with more than 80GB of memory):

This command will load our model outputs and run Llama 3.3 70B Instruct locally to get its preference judgments and compute the corresponding winrate. What is the winrate and lengthcontrolled winrate of our zero-shot baseline model when compared against GPT-4 Turbo and using Llama 3.3 70B Instruct as the annotator?

**Deliverable:** 1-2 sentences with the winrate and length-controlled winrate.

(d) Sample 10 random examples where the baseline model’s response is dispreferred versus GPT-4 Turbo (you should be able to see the AlpacaEval annotations at scripts/alpaca_eval_vllm_llama3_3_70b_fn/annotations_seed0_configs.json). Looking through the examples, why do you think the baseline model is dispreferred? Are there any cases where you disagree with the automatic evaluator?

**Deliverable:** A 2-4 sentence error analysis of model predictions, including examples and/or model responses as necessary.



### Problem (sst_baseline): 4 points

(a) Write a script to collect Llama 3.1 8B zero-shot predictions on SimpleSafetyTests. This script should (1) load the SimpleSafetyTests instructions, (2) generate outputs for each instruction, and (3) serialize the outputs and model generations to disk for evaluation. For compatibility with our provided evaluator, your output predictions must serialized in JSON-lines format (i.e., new-line separated JSON objects). Each JSON object should contain (at least) the following keys:

• prompts_final: the instruction from SimpleSafetyTests.

• output: the output of the model, given the instruction.

**Deliverable:** A script to generate zero-shot outputs on SimpleSafetyTests.

```shell
# run shell
python cs336_alignment/eval_sst_baseline.py
```

<font color="purple">**Due to resource limitations, the following questions are skipped.**</font>

(b) How long does it take the model to generate responses to each of the SimpleSafetyTests examples?

Estimate the throughput in examples/second.

**Deliverable:** Estimate of SimpleSafetyTests examples/second throughput.

(c) To measure our model’s performance on SimpleSafetyTests, we’ll use Llama 3.3 70B Instruct to annotate responses as being safe or unsafe. To compute the proportion of safe outputs (as judged by Llama 3.3 70B Instruct), run the following command (requires two GPUs, each with more than 80GB of memory):

This command will load our model outputs and run Llama 3.3 70B Instruct locally to get annotations and compute the corresponding proportion of “safe” outputs. What proportion of model outputs are judged as safe?

**Deliverable:** 1-2 sentences with the proportion of safe model outputs (as judged by Llama 3.3 70B Instruct).

(d) Sample 10 random examples where the baseline model’s response is judged to be unsafe (you should be able to see the annotations at the output path that you specified when running the evaluator). Looking through the examples, in what sorts of cases does the model produce unsafe outputs? Are there any cases where you disagree with the automatic evaluator?

**Deliverable:** A 2-4 sentence error analysis of model predictions, including examples and/or model responses as necessary.



### Problem (look_at_sft): 4 points

Look through ten random examples in the provided instruction tuning training dataset. What sort of traditional NLP tasks are represented in this sample (e.g., question answering, sentiment analysis, etc.)? Comment on the quality of the sampled examples (both the prompt and the corresponding instruction).

**Deliverable:** 2-4 sentences with a description of what sorts of tasks are implicitly included in the instruction tuning dataset, as well commentary about the data quality. Use concrete examples whenever possible.

```shell
wget https://nlp.stanford.edu/data/nfliu/cs336-spring-2024/assignment5/safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz
wget https://nlp.stanford.edu/data/nfliu/cs336-spring-2024/assignment5/safety_augmented_ultrachat_200k_single_turn/test.jsonl.gz
```



### Problem (data_loading): Implement data loading (3 points)

(a) **Deliverable:** Implement a PyTorch Dataset subclass that generates examples for instruction tuning. The Dataset should have the following interface:

- [x] See class `PackedSFTDataset` in `cs336_alignment/data_loading.py`

```python
# test shell
pytest -k test_packed_sft_dataset
```

(b) **Deliverable:** Implement a function that returns batches from your previously-implemented Dataset. Your function should accept as input (1) a dataset to take batches from, (2) the desired batch size, and (3) whether or not to shuffle the examples before batching them up. Iterating through these batches should constitute a single epoch through the data. You may find torch.utils.data.DataLoader to be useful.

- [x] See class `get_iterate_batches` in `cs336_alignment/data_loading.py`

```python
# test shell
pytest -k test_iterate_batches
```



### Problem (sft_script): Training script: instruction tuning (4 points)

**Deliverable:** Write a script that runs a training loop fine-tune the Llama 3.1 8B base model on the provided instruction tuning data. In particular, we recommend that your training script allow for (at least) the following:

• Ability to configure and control the various model and optimizer hyperparameters.

• Ability to train on larger batch sizes than can fit in memory via gradient accumulation.

• Periodically logging training and validation performance (e.g., to console and/or an external service like Weights and Biases).a 

If you’ve completed the previous assignments (e.g., A1, mandatory part of A5), feel free to adapt the training script that you previously wrote to support fine-tuning pre-trained language models on instruction tuning data with gradient accumulation. Alternatively, you may find the provided training script from assignment 4 to be a useful starting point (though we encourage you to write the script from scratch if you haven’t already done it).

- [x] See code in `cs336_alignment/sft_script.py`



### Problem (sft): Instruction Tuning (6 points) (24 H100 hrs)

<font color="purple">**Due to resource limitations, the following questions are skipped.**</font>

Fine-tune Llama 3 8B base on the provided instruction tuning data. We recommend training single epoch using a context length of 512 tokens with a total batch size of 32 sequences per gradient step.a  Make sure to save your model and tokenizer after training, since we’ll evaluate their performance and also use them later in the assignment for further post-training on preference pairs. We used a learning rate of 2e-5 with cosine decay and a linear warmup (3% of total training steps), but it may be useful to experiment with different learning rates to get a better intuition for what values work well.

**Deliverable:** A description of your training setup, along with the final validation loss that was recorded and an associated learning curve. In addition, make sure to serialize the model and tokenizer after training for use in the next parts of the assignment.



## 4 Evaluating our instruction-tuned model

### Problem (mmlu_sft): 4 points

<font color="purple">**Due to resource limitations, the following questions are skipped.**</font>

(a) Write a script to evaluate your instruction-tuned model on MMLU, making sure to format the inputs in the same instruction tuning prompt format used for training. Run your evaluation script and measure the amount of time it takes for the model to generate responses to each of the MMLU examples. Estimate the throughput in examples/second. How does this compare to our zero-shot baseline?

**Deliverable:** 1-2 sentences with an estimate of MMLU examples/second throughput and a comparison to the zero-shot baseline.

(b) How well does the instruction-tuned model perform on MMLU? How does this compare to our zero-shot baseline?

**Deliverable:** 1-2 sentences with evaluation metrics and a comparison to the zero-shot baseline.

(c) Sample 10 random incorrectly-predicted examples from the evaluation dataset. Looking through the examples, what sort of errors does the language model make? Qualitatively, how do the outputs of the fine-tuned model differ from the outputs of the zero-shot baseline?

**Deliverable:** A 2-4 sentence error analysis of model predictions, including examples and/or model responses as necessary.



### Problem (gsm8k_sft): 4 points

<font color="purple">**Due to resource limitations, the following questions are skipped.**</font>

(a) Write a script to evaluate your instruction-tuned model on GSM8K, making sure to format the inputs in the same instruction tuning prompt format used for training. Run your evaluation script and measure the amount of time it takes the model to generate responses to each of the GSM8K examples. Estimate the throughput in examples/second. How doe sthis compare to our zero-shot baseline?

**Deliverable:** 1-2 sentences with an estimate of GSM8K examples/second throughput and a comparison to the zero-shot baseline.

(b) How well does the instruction-tuned model perform on GSM8K? How does this compare to our zero-shot baseline?

**Deliverable:** 1-2 sentences with evaluation metrics and a comparison to the zero-shot baseline.

(c) Sample 10 random incorrectly-predicted examples from the evaluation dataset. Looking through the examples, what sort of errors does the language model make? Qualitatively, how do the outputs of the fine-tuned model differ from the outputs of the zero-shot baseline?

**Deliverable:** A 2-4 sentence error analysis of model predictions, including examples and/or model responses as necessary.



### Problem (alpaca_eval_sft): 4 points

<font color="purple">**Due to resource limitations, the following questions are skipped.**</font>

(a) Write a script to collect the predictions of your fine-tuned model on AlpacaEval. How long does it take the model to generate responses to each of the AlpacaEval examples? Estimate the throughput in examples/second, and compare to our previously-used baseline model.

**Deliverable:** 1-2 sentences with an estimate of AlpacaEval examples/second throughput and a comparison to the baseline model.

(b) To measure our model’s performance on AlpacaEval, we’ll use Llama 3.3 70B Instruct as the annotator and compare our outputs against GPT-4 Turbo. To compute the winrate, run the following command (requires two GPUs, each with more than 80GB of memory):

```shell
uv run alpaca_eval --model_outputs <path_to_model_predictions.json> \
--annotators_config 'scripts/alpaca_eval_vllm_llama3_3_70b_fn' \
--base-dir '.'
```

This command will load our model outputs and run Llama 3.3 70B locally to get its preference judgments and compute the corresponding winrate. What is the winrate and length-controlled winrate of your instruction-tuned model when compared against GPT-4 Turbo and using Llama 3.3 70B Instruct as the annotator? How does this winrate compare to our zero-shot baseline?

**Deliverable:** 1-3 sentences with the winrate and length-controlled winrate, as well a comparison against the zero-shot baseline.

(c) Sample 10 random examples where your fine-tuned model’s response is dispreferred versus GPT-4 Turbo. You should be able to see the AlpacaEval annotations at scripts/alpaca_eval_vllm_llama3_3_70b_fn/annotations_seed0_configs.json, and the entries where "preference" is equal to 1.0 are the examples where the evaluator judged the GPT-4 Turbo response to be better. Looking through the examples, why do you think your fine-tuned model is dispreferred? Are there any cases where you disagree with the automatic evaluator?

**Deliverable:** A 2-4 sentence error analysis of model predictions, including examples and/or model responses as necessary.



### Problem (sst_sft): 4 points

<font color="purple">**Due to resource limitations, the following questions are skipped.**</font>

(a) Write a script to collect the predictions of your fine-tuned model on SimpleSafetyTests. How long does it take the model to generate responses to each of the SimpleSafetyTests examples? Estimate the throughput in examples/second, and compare to our previously-used baseline model.

**Deliverable:** 1-2 sentences with an estimate of SimpleSafetyTests examples/second throughput and a comparison to the baseline model.

(b) To measure our model’s performance on SimpleSafetyTests, we’ll use Llama 3.3 70B Instruct to annotate responses as being safe or unsafe. To compute the proportion of safe outputs (as judged by Llama 3.3 70B Instruct), run the following command (requires two GPUs, each with more than 80GB of memory):

```shell
uv run python scripts/evaluate_safety.py \
--input-path <path_to_model_predictions.jsonl> \
--model-name-or-path /data/a5-alignment/models/Llama-3.3-70B-Instruct \
--num-gpus 2 \
--output-path <path_to_write_output.jsonl>
```

This command will load our model outputs and run Llama 3.3 70B locally to get annotations and compute the corresponding proportion of “safe” outputs. What proportion of model outputs are judged as safe? How does this compare to the zero-shot baseline?

**Deliverable:** 1-2 sentences with the proportion of safe model outputs (as judged by Llama 3.3 70B Instruct).

(c) Sample 10 random examples where your fine-tuned model’s response is judged to be unsafe (you should be able to see the annotations at the output path that you specified when running the evaluator). Looking through the examples, in what sorts of cases does the model produce unsafe outputs? Are there any cases where you disagree with the automatic evaluator?

**Deliverable:** A 2-4 sentence error analysis of model predictions, including examples and/or model responses as necessary.



### Problem (red_teaming): 4 points

(a) Beyond the examples listed above, what are three other possible ways that language models might be misused?

**Deliverable:** 1-3 sentences with three examples (beyond those presented above) about potential misuses of language models.

(b) Try prompting your fine-tuned language model to assist you in completing three different potentially malicious applications. For each malicous application, provide a description of your methodology and the results, as well as any qualitative takeaways you drew from the experience. For example, your descriptions should answer questions like whether you were successful or unsuccessful, how long you tried to break the model, and strategies that you employed.

**Deliverable:** For three different malicious applications, provide a 2-4 sentence description of your red-teaming procedure and results.



## 5 “Reinforcement Learning” from “Human Feedback”

### Problem (look_at_hh): 2 points

pass



### Problem (dpo_loss): 2 points

Write a function that computes the per-instance DPO loss. Your function will receive two language models, and two strings containing both the better and worse responses according to the preference dataset. Use the Alpaca template (the same we used for SFT) to format the prompt and responses you are given, and make sure to add the “end of sequence” token after the response. To simplify your implementation, you can use the following observation: when computing a difference of conditional logprobabilities under the same model (e.g., $\log \pi_\theta (y_w | x) − \log \pi_\theta (y_l | x)$), this is equivalent to computing the difference of the unconditional log-probabilities (e.g., $\log \pi_\theta (x \oplus y_w )−\log \pi_\theta(x \oplus y_l )$, where $\oplus$ denotes the concatenation of sequences of tokens), since the log-probability of the prompt cancels out.

- [x] See class `get_iterate_batches` in `cs336_alignment/data_loading.py`

```python
# test shell
pytest -k test_per_instance_dpo_loss
```



### Problem (dpo_training): 4 points

<font color="purple">**Due to resource limitations, the following questions are skipped.**</font>

1. Implement your DPO training loop, and train your instruction-tuned Llama 3.1 8B model for 1 epoch over HH. Save your model with the highest validation accuracy.

	**Deliverable:** A script to train your instruction-tuned Llama model with DPO on HH, and a screenshot of your validation accuracy curve during training.

2. Now, evaluate your model after DPO on AlpacaEval, as you did in problem alpaca_eval_sft. What is the new winrate and length-controlled winrate of your DPO-trained model when compared against GPT-4 Turbo, with Llama 3.3 70B Instruct as the annotator? How does that compare to the SFT model you started with?

	**Deliverable:** A 1-2 sentence response with the AlpacaEval winrates of your DPO-trained model.

3. Evaluate your DPO-trained model on SimpleSafetyTests. How does it compare to the SFT model?

	**Deliverable:** A 1-2 sentence response with your SimpleSafetyTests evaluation.

4. Both AlpacaEval and SimpleSafetyTests test behaviours that are directly demonstrated in HH, such as instruction following and refusing potentally harmful prompts. Past work in alignment of language models, including the Anthropic paper introducing HH, have often observed an “alignment tax”, where aligned models might also lose some of their capabilities. Evaluate your DPO model on GSM8k and MMLU. What do you observe?

	**Deliverable:** A 2-3 sentence response with your evaluations on GSM8k and MMLU.
