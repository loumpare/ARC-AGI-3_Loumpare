"""train_dpo_qlora.py — Step D of the DPO fine-tuning plan.

QLoRA 4-bit DPO training on Ministral-3-14B-Instruct-2512 using the dataset
assembled in Step C (assemble_dpo_dataset.py).

Setup (all already in .venv):
  trl==1.12.0, peft==0.20.0, transformers==5.15.1, bitsandbytes==0.50.2, accelerate==1.14.0

Memory estimate on RTX 3090 (24GB):
  - Model (int4): ~7.5GB
  - Activations + optimizer states: ~10-12GB
  - LoRA adapter: ~0.3GB
  - Should fit with max_seq_length=2048; use 1024 if OOM

Usage:
  source .venv/bin/activate
  python3 notebooks/train_dpo_qlora.py [options]

Key args:
  --model-id       HF model ID (default: mistralai/Ministral-3-14B-Instruct-2512)
  --dataset        Path to trl train.jsonl (default: latest in results/)
  --output-dir     Where to save adapter (default: checkpoints/dpo_ministral_YYYYMMDD)
  --max-seq-len    Max tokens per example (default: 2048)
  --batch-size     Per-device batch size (default: 1)
  --grad-accum     Gradient accumulation steps (default: 8)
  --epochs         Training epochs (default: 3)
  --lora-rank      LoRA rank (default: 16)
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DPOConfig, DPOTrainer


def load_dataset_from_jsonl(path: Path) -> Dataset:
    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    # trl DPOTrainer conversational format:
    # {"prompt": list[dict], "chosen": list[dict], "rejected": list[dict]}
    return Dataset.from_list(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="mistralai/Ministral-3-14B-Instruct-2512",
        help="HuggingFace model ID. Must match the base model used for Ollama inference.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to trl train.jsonl. Defaults to latest results/dpo_train_dataset_*/train.jsonl",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta (KL penalty)")
    parser.add_argument("--dry-run", action="store_true", help="Load model + dataset only, no training")
    args = parser.parse_args()

    # Resolve dataset path
    if args.dataset is None:
        candidates = sorted(Path("results").glob("dpo_train_dataset_*/train.jsonl"))
        if not candidates:
            raise FileNotFoundError("No dpo_train_dataset_*/train.jsonl found. Run assemble_dpo_dataset.py first.")
        args.dataset = str(candidates[-1])
    dataset_path = Path(args.dataset)
    print(f"Dataset: {dataset_path}")

    # Resolve output dir
    if args.output_dir is None:
        args.output_dir = f"checkpoints/dpo_ministral_{date.today().strftime('%Y%m%d')}"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output: {args.output_dir}")

    # Load dataset
    dataset = load_dataset_from_jsonl(dataset_path)
    print(f"Dataset size: {len(dataset)} pairs")

    if args.dry_run:
        print("[dry-run] Dataset loaded OK. Skipping model load and training.")
        return

    # BitsAndBytes 4-bit config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")

    # LoRA config - Mistral uses standard attention layers
    # Pass to DPOTrainer directly (handles get_peft_model internally)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    # DPO config
    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        beta=args.beta,
        max_length=args.max_seq_len,
        dataset_num_proc=4,
        # Conversational format (prompt/chosen/rejected as message lists) is
        # auto-detected by trl via is_conversational() — no explicit flag needed.
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("Starting DPO training...")
    trainer.train()

    print(f"Saving adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
