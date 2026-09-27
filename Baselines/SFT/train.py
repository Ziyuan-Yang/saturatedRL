import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

def parse_args():
    parser = argparse.ArgumentParser(description="SFT Training Script for Qwen/LLMs")

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dataset_name", type=str, default="HINT-lab/sft_Qwen_Qwen3-0.6B",
                        help="HuggingFace dataset ID or local .json/.jsonl file path")
    parser.add_argument("--response_field", type=str, default=None,
                        help="Field to use as assistant response (auto-detected if not set)")
    parser.add_argument("--output_dir", type=str, default="./results_0.6B")
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--use_gradient_checkpointing", action="store_true")
    parser.add_argument("--response_template", type=str, default="<|im_start|>assistant\n")

    parser.set_defaults(use_gradient_checkpointing=True)
    return parser.parse_args()

def main():
    args = parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print(f"--- Configuration ---")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Max Seq Len: {args.max_seq_length}")
    print(f"---------------------")

    # ---------- Load Model and Tokenizer ----------
    print("Loading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, trust_remote_code=True, enable_thinking=False
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------- Load Dataset ----------
    print("Loading and formatting dataset...")
    if args.dataset_name.endswith(".json") or args.dataset_name.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=args.dataset_name, split="train")
        print(f"Loaded local file: {args.dataset_name}, {len(dataset)} examples")
    else:
        dataset = load_dataset(args.dataset_name, split="train")

    # Auto-detect response field: prefer 'response' (full reasoning) over 'answer'
    if args.response_field:
        assistant_field = args.response_field
    elif "response" in dataset.column_names:
        assistant_field = "response"
    else:
        assistant_field = "answer"
    print(f"Using '{assistant_field}' as assistant response field")

    # ---------- Format Dataset ----------
    dataset = dataset.filter(
        lambda ex: ex["question"] is not None and len(ex["question"].strip()) > 0
            and ex[assistant_field] is not None and len(ex[assistant_field].strip()) > 0
    )
    print(f"Filtered dataset size: {len(dataset)}")

    def format_example(example):
        input_text = example["question"] + "\n" + r"Please reason step by step, and put your final answer within \boxed{}."
        return {
            "messages": [
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": example[assistant_field]},
            ]
        }

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    def apply_chat_template(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False,enable_thinking=False
            )
        }

    dataset = dataset.map(apply_chat_template, remove_columns=["messages"])
    print(f"Sample Text: {dataset[0]['text'][:300]}")

    # ---------- Data Collator (mask user tokens from loss) ----------
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=args.response_template,
        tokenizer=tokenizer,
    )

    # ---------- Trainer ----------
    training_args = SFTConfig(
        output_dir=args.output_dir,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=10,
        gradient_checkpointing=args.use_gradient_checkpointing,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")

if __name__ == "__main__":
    main()
