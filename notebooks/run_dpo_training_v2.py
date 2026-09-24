"""
DPO QLoRA training v2 — loads from clean BF16 export (no FP8 conflicts).

Pre-requisite: run notebooks/export_bf16.py once to create
  checkpoints/ministral_bf16_export/  (plain BF16, no FP8 config)

Then this script:
  1. Load BF16 export with BnB NF4 (standard QLoRA, no FP8 issues)
  2. Apply LoRA to language_model attention + MLP only
  3. DPO training with precompute_ref_log_probs=True
"""
import os, gc, json, time, shutil, torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer

BF16_EXPORT = Path('checkpoints/ministral_bf16_export')
DATASET     = Path('results/dpo_train_dataset_2026-09-21/train.jsonl')
OUTPUT_DIR  = Path('checkpoints/dpo_ministral_20260921')

LORA_RANK  = 16
MAX_LENGTH = 2048
BATCH_SIZE = 1
GRAD_ACCUM = 8
EPOCHS     = 3
LR         = 5e-5
BETA       = 0.1

# ── Checks ────────────────────────────────────────────────────────────────────
assert BF16_EXPORT.exists(), 'Run notebooks/export_bf16.py first'
free_gb  = torch.cuda.mem_get_info()[0] / 1024**3
total_gb = torch.cuda.mem_get_info()[1] / 1024**3
print(f'VRAM: {free_gb:.1f} GB free / {total_gb:.1f} GB total')
assert free_gb > 18, f'Not enough VRAM ({free_gb:.1f} GB)'

# ── Dataset ───────────────────────────────────────────────────────────────────
records = [json.loads(l) for l in DATASET.read_text().splitlines() if l.strip()]
dataset = Dataset.from_list(records)
print(f'{len(dataset)} DPO pairs loaded')

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    str(BF16_EXPORT), fix_mistral_regex=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── Model: BnB NF4 from clean BF16 export ─────────────────────────────────────
# v1 failure: stripped FP8 config → scale factors ignored → NaN.
# v2 fix: export_bf16.py dequantized FP8→BF16 correctly using weight_scale_inv.
# Now BnB 4-bit applies normally (no FP8 conflict, weights are already correct).
print('Loading model (BnB NF4)...')
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = Mistral3ForConditionalGeneration.from_pretrained(
    str(BF16_EXPORT),
    quantization_config=bnb_config,
    device_map='auto',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
free_after = torch.cuda.mem_get_info()[0] / 1024**3
print(f'Model loaded. VRAM free: {free_after:.1f} GB')

# ── LoRA ──────────────────────────────────────────────────────────────────────
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# language_model path inside Mistral3ForConditionalGeneration:
#   model.model.language_model.model.layers.{i}.self_attn.{q,k,v,o}_proj
# Use regex so LoRA doesn't target vision tower linear layers.
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_RANK * 2,
    target_modules=(
        r'model\.language_model\.'
        r'(model\.)?layers\.\d+\.'
        r'(self_attn\.(q_proj|k_proj|v_proj|o_proj)'
        r'|mlp\.(gate_proj|up_proj|down_proj))'
    ),
    lora_dropout=0.05,
    bias='none',
    task_type=None,
)

# ── DPO training ──────────────────────────────────────────────────────────────
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True)

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
print(f'Trainable: {trainable/1e6:.1f}M / {total_p/1e9:.1f}B '
      f'({100*trainable/total_p:.2f}%)')

t0 = time.time()
print('Starting DPO training...')
trainer.train()
elapsed = time.time() - t0
print(f'\nDone in {elapsed/3600:.1f}h ({elapsed/60:.0f} min)')

trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
print(f'Adapter saved to: {OUTPUT_DIR}')
for f in sorted(OUTPUT_DIR.glob('adapter*')):
    print(f'  {f.name}: {f.stat().st_size/1024**2:.1f} MB')
