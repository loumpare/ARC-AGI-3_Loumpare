"""DPO QLoRA training — standalone script (run outside Jupyter to avoid kernel crashes).

Usage:
    cd /home/lavolpe/Bureau/Kaggle/ARC-AGI-3
    source .venv/bin/activate
    python notebooks/run_dpo_training.py
"""
import os, json, time, torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig
from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID   = 'mistralai/Ministral-3-14B-Instruct-2512'
DATASET    = Path('results/dpo_train_dataset_2026-09-21/train.jsonl')
OUTPUT_DIR = Path('checkpoints/dpo_ministral_20260921')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LORA_RANK  = 16
MAX_LENGTH = 2048   # 212/218 examples survive (vs 16/218 at 1024)
BATCH_SIZE = 1
GRAD_ACCUM = 8
EPOCHS     = 3
LR         = 5e-5
BETA       = 0.1

# ── GPU check ─────────────────────────────────────────────────────────────────
free_gb = torch.cuda.mem_get_info()[0] / 1024**3
total_gb = torch.cuda.mem_get_info()[1] / 1024**3
print(f'VRAM: {free_gb:.1f} GB free / {total_gb:.1f} GB total')
assert free_gb > 14, f'Not enough VRAM ({free_gb:.1f} GB)'

# ── Dataset ───────────────────────────────────────────────────────────────────
records = [json.loads(l) for l in DATASET.read_text().splitlines() if l.strip()]
dataset = Dataset.from_list(records)
print(f'{len(dataset)} DPO pairs loaded')

# ── Tokenizer ─────────────────────────────────────────────────────────────────
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, fix_mistral_regex=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── Model (strip FP8 config → BnB 4-bit) ─────────────────────────────────────
print('Loading model (4-bit)...')
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
config = AutoConfig.from_pretrained(MODEL_ID)
config.quantization_config = None  # remove stored FP8 config

model = Mistral3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    config=config,
    quantization_config=bnb_config,
    device_map='auto',
    dtype=torch.bfloat16,
)
model = prepare_model_for_kbit_training(model)

free_after = torch.cuda.mem_get_info()[0] / 1024**3
print(f'Model loaded. VRAM free: {free_after:.1f} GB')

# ── LoRA ─────────────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_RANK,
    lora_alpha=LORA_RANK * 2,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'],
    lora_dropout=0.05,
    bias='none',
)

# ── DPO training ─────────────────────────────────────────────────────────────
dpo_config = DPOConfig(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    gradient_checkpointing=True,
    learning_rate=LR,
    lr_scheduler_type='cosine',
    warmup_steps=8,
    bf16=True,
    logging_steps=5,
    save_steps=100,
    save_total_limit=2,
    beta=BETA,
    max_length=MAX_LENGTH,
    precompute_ref_log_probs=True,
    dataset_num_proc=4,
    report_to='none',
)

trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_p   = sum(p.numel() for p in model.parameters())
print(f'Trainable: {trainable/1e6:.1f}M / {total_p/1e9:.1f}B ({100*trainable/total_p:.2f}%)')

t0 = time.time()
print('Starting DPO training...')
trainer.train()
elapsed = time.time() - t0
print(f'\nDone in {elapsed/3600:.1f}h ({elapsed/60:.0f} min)')

# ── Save ─────────────────────────────────────────────────────────────────────
trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
print(f'Adapter saved to: {OUTPUT_DIR}')
for f in sorted(OUTPUT_DIR.glob('adapter*')):
    print(f'  {f.name}: {f.stat().st_size/1024**2:.1f} MB')
