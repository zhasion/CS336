import torch
import sys
import pathlib
from transformers import PreTrainedTokenizerBase
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from cs336_alignment.helper import get_response_log_probs

def get_alpaca_template():
    with open('cs336_alignment/prompts/alpaca_sft.prompt', 'r', encoding='utf8') as f:
        alpaca_template = f.read()
    return alpaca_template.strip()


def get_inputs_and_labels(tokenizer: PreTrainedTokenizerBase, prompt_template: str, prompt: str, response: str):
    prompt = prompt_template.format(instruction=prompt, response=response)
    input_ids = tokenizer.encode(prompt)
    labels = input_ids[1:] + [tokenizer.eos_token_id]
    input_ids = torch.tensor([input_ids])
    labels = torch.tensor([labels])
    return input_ids, labels


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:

    prompt_template = get_alpaca_template()
    
    input_ids_chosen, labels_chosen = get_inputs_and_labels(tokenizer, prompt_template, prompt, response_chosen)
    input_ids_rejected, labels_rejected = get_inputs_and_labels(tokenizer, prompt_template, prompt, response_rejected)

    log_probs_model_chosen = get_response_log_probs(lm, input_ids_chosen, labels_chosen)["log_probs"]
    log_probs_model_ref_chosen = get_response_log_probs(lm_ref, input_ids_chosen, labels_chosen)["log_probs"]
    chosen_ratio = (log_probs_model_chosen - log_probs_model_ref_chosen).sum(dim=-1)

    log_probs_model_rejected = get_response_log_probs(lm, input_ids_rejected, labels_rejected)["log_probs"]
    log_probs_model_ref_rejected = get_response_log_probs(lm_ref, input_ids_rejected, labels_rejected)["log_probs"]
    rejected_ratio = (log_probs_model_rejected - log_probs_model_ref_rejected).sum(dim=-1)

    return -torch.log(torch.sigmoid(beta * (chosen_ratio - rejected_ratio))).mean()