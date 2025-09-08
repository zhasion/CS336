import sys
import torch
import argparse
import pathlib
import pathlib
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from cs336_alignment.data_loading import get_iterate_batches, PackedSFTDataset

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

def run(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    train_dataloader: torch.utils.data.DataLoader, 
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    args: dict
):
    for epoch in range(1, args.epochs + 1):
        for idx, batch in enumerate(train_dataloader):
            model.train()
            input_ids = batch["input_ids"].to(args.device)
            labels = batch["labels"].to(args.device)
            logits = model(input_ids).logits
            loss = (
                torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
                / args.gradient_accumulation_steps
            )
            loss.backward()

            if (idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step() 
                optimizer.zero_grad()
                scheduler.step()

            if (idx + 1) % args.eval_nums == 0:
                model.eval()
                with torch.no_grad():
                    batch_loss = 0.0
                    for _, batch in val_dataloader:
                        input_ids = batch["input_ids"].to(args.device)
                        labels = batch["labels"].to(args.device)
                        logits = model(input_ids).logits
                        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1)).detach().cpu().item()
                        batch_loss += loss.detach().cpu().item()

    # model.save_pretrained(save_directory=args.output_dir) 
    # tokenizer.save_pretrained(save_directory=args.output_dir)

def main(args: dict):

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device = args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    train_dataset = PackedSFTDataset(tokenizer=tokenizer, dataset_path=args.train_fpath, seq_length=args.max_tokens, shuffle=True)
    val_dataset = PackedSFTDataset(tokenizer=tokenizer, dataset_path=args.val_fpath, seq_length=args.max_tokens, shuffle=True)
    train_dataloader = get_iterate_batches(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = get_iterate_batches(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.03 * args.epochs * len(train_dataset) // args.batch_size,
        num_training_steps=args.epochs * len(train_dataset) // args.batch_size,
    )

    run(model, tokenizer, train_dataloader, val_dataloader, optimizer, scheduler, args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='models/Qwen2.5-Math-1.5B')
    # parser.add_argument('--train_fpath', default='data/sft/train.jsonl.gz')
    # parser.add_argument('--val_fpath', default='data/sft/test.jsonl.gz')
    parser.add_argument('--train_fpath', default='data/sft/sample.jsonl')
    parser.add_argument('--val_fpath', default='data/sft/sample.jsonl')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max_tokens', default=512)
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--lr', default=2e-5)
    parser.add_argument('--epochs', default=10)
    parser.add_argument('--gradient_accumulation_steps', default=16)
    parser.add_argument('--output_dir', default='output/sft')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)