"""
Step E: Evaluate DPO fine-tuned adapter vs baseline (no adapter).

Loads 20 prompts from the DPO training dataset, generates responses from:
  1. Baseline: BF16 export + NF4, adapter DISABLED
  2. Fine-tuned: same model, adapter ENABLED

Metric: verification_rate = fraction of responses that access history frames
        (history[-N] or previous_frame) before calling action().

Also: direct_action_rate = action() call exists with no history comparison.

Writes per-example results to results/eval_dpo_adapter_2026-09-22.jsonl
and prints a summary.
"""
import json, re, random, torch, os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from pathlib import Path
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration
from peft import PeftModel

BF16_EXPORT  = Path('checkpoints/ministral_bf16_export')
ADAPTER_DIR  = Path('checkpoints/dpo_ministral_20260921')
DATASET      = Path('results/local_dpo_dataset_2026-09-21.jsonl')
OUT_FILE     = Path('results/eval_dpo_adapter_2026-09-22.jsonl')
N_EVAL       = 20
MAX_NEW_TOKENS = 512
SEED         = 42

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── Dataset ────────────────────────────────────────────────────────────────────
random.seed(SEED)
records = [json.loads(l) for l in DATASET.read_text().splitlines() if l.strip()]
# Sample evenly across analysis_step range to get variety
records.sort(key=lambda r: r['analysis_step'])
step = max(1, len(records) // N_EVAL)
eval_records = records[::step][:N_EVAL]
print(f'Eval set: {len(eval_records)} records, '
      f'analysis_steps {[r["analysis_step"] for r in eval_records]}')

# ── Model ─────────────────────────────────────────────────────────────────────
print('Loading model (BnB NF4 + DPO adapter)...')
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
base_model = Mistral3ForConditionalGeneration.from_pretrained(
    str(BF16_EXPORT),
    quantization_config=bnb_config,
    device_map='auto',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
model.eval()

free_gb = torch.cuda.mem_get_info()[0] / 1024**3
print(f'Model+adapter loaded. VRAM free: {free_gb:.1f} GB')

tokenizer = AutoTokenizer.from_pretrained(str(BF16_EXPORT), fix_mistral_regex=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── Generation ────────────────────────────────────────────────────────────────
def generate(messages: list, use_adapter: bool) -> str:
    if use_adapter:
        model.enable_adapter_layers()
    else:
        model.disable_adapter_layers()

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors='pt').to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    new_ids = out[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)

# ── Analysis ──────────────────────────────────────────────────────────────────
VERIFY_PATTERNS = [
    r'history\[-\d+\]',          # accessing history frames: history[-2]
    r'previous_frame',            # explicit previous_frame var
    r'prev_frame',
    r'prev_seg',
    r'compare.*frame',
    r'frame.*change',
    r'if.*boundary.*!=',          # comparing node boundaries
    r'if.*pixels.*!=',
    r'detect.*change',
    r'verify.*action',
    r'before.*action\(',
]

def has_verification(text: str) -> bool:
    """True if response checks history/frames before deciding on action."""
    for pat in VERIFY_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False

def has_direct_action(text: str) -> bool:
    """True if response calls action() in code without any verification pattern."""
    code_blocks = re.findall(r'```python\n(.*?)```', text, re.DOTALL)
    all_code = '\n'.join(code_blocks) if code_blocks else text
    return 'action(' in all_code and not has_verification(all_code)

def has_action_call(text: str) -> bool:
    code_blocks = re.findall(r'```python\n(.*?)```', text, re.DOTALL)
    all_code = '\n'.join(code_blocks) if code_blocks else text
    return 'action(' in all_code

# ── Eval loop ─────────────────────────────────────────────────────────────────
results = []
for i, rec in enumerate(eval_records):
    messages = rec['messages']
    game_id  = rec.get('game_id', '?')
    step_n   = rec.get('analysis_step', '?')

    print(f'\n[{i+1}/{N_EVAL}] game={game_id}  analysis_step={step_n}')

    base_resp = generate(messages, use_adapter=False)
    ft_resp   = generate(messages, use_adapter=True)

    base_verify = has_verification(base_resp)
    ft_verify   = has_verification(ft_resp)
    base_direct = has_direct_action(base_resp)
    ft_direct   = has_direct_action(ft_resp)
    base_action = has_action_call(base_resp)
    ft_action   = has_action_call(ft_resp)

    print(f'  Baseline:   verified={base_verify}  direct_action={base_direct}  has_action={base_action}')
    print(f'  Fine-tuned: verified={ft_verify}  direct_action={ft_direct}  has_action={ft_action}')
    if base_verify != ft_verify:
        print('  *** DIFFERENCE in verification ***')

    row = dict(
        game_id=game_id, analysis_step=step_n,
        base_verified=base_verify, ft_verified=ft_verify,
        base_direct_action=base_direct, ft_direct_action=ft_direct,
        base_has_action=base_action, ft_has_action=ft_action,
        base_response=base_resp[:600],
        ft_response=ft_resp[:600],
    )
    results.append(row)
    with open(OUT_FILE, 'a') as f:
        f.write(json.dumps(row) + '\n')

# ── Summary ───────────────────────────────────────────────────────────────────
n = len(results)
base_vr = sum(r['base_verified']      for r in results) / n
ft_vr   = sum(r['ft_verified']        for r in results) / n
base_dr = sum(r['base_direct_action'] for r in results) / n
ft_dr   = sum(r['ft_direct_action']   for r in results) / n
base_ar = sum(r['base_has_action']    for r in results) / n
ft_ar   = sum(r['ft_has_action']      for r in results) / n

print(f'\n{"="*55}')
print(f'EVALUATION SUMMARY  (n={n})')
print(f'{"="*55}')
print(f'                       Baseline    Fine-tuned')
print(f'  Verification rate:   {base_vr:.1%}       {ft_vr:.1%}')
print(f'  Direct action rate:  {base_dr:.1%}       {ft_dr:.1%}')
print(f'  Has any action():    {base_ar:.1%}       {ft_ar:.1%}')
print(f'{"="*55}')
per_game = {}
for r in results:
    gid = r['game_id']
    per_game.setdefault(gid, {'base_v': 0, 'ft_v': 0, 'n': 0})
    per_game[gid]['base_v'] += r['base_verified']
    per_game[gid]['ft_v']   += r['ft_verified']
    per_game[gid]['n']      += 1
print('Per-game breakdown:')
for gid, d in sorted(per_game.items()):
    print(f'  {gid}: base={d["base_v"]}/{d["n"]}  ft={d["ft_v"]}/{d["n"]}')
print(f'\nResults written to: {OUT_FILE}')
