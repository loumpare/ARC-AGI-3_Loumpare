"""
One-time export: FP8 checkpoint → plain BF16 safetensors.

Reads tensors from the HF cache one at a time (memory-mapped), dequantizes
FP8 weights using their weight_scale_inv factors, writes shards of ~4 GB.
Peak RAM: ~4 GB (one shard at a time). Duration: ~5 min.

Output: checkpoints/ministral_bf16_export/
  model-shard-*.safetensors  — weight shards
  model.safetensors.index.json
  config.json                — quantization_config removed → BnB 4-bit works
  tokenizer.*               — copied from original

Usage:
    python notebooks/export_bf16.py
Then update run_dpo_training_v2.py to load from checkpoints/ministral_bf16_export/
"""
import gc, json, shutil, torch
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer

MODEL_ID  = 'mistralai/Ministral-3-14B-Instruct-2512'
SHARD_GB  = 4        # target shard size in GB
SNAPSHOT  = Path.home() / '.cache/huggingface/hub' \
    / 'models--mistralai--Ministral-3-14B-Instruct-2512' \
    / 'snapshots/29439f81c2be264d8d393273f99e7db9c0961120'
OUTPUT    = Path('checkpoints/ministral_bf16_export')

OUTPUT.mkdir(parents=True, exist_ok=True)

# ── Load safetensors index ─────────────────────────────────────────────────────
idx = json.loads((SNAPSHOT / 'model.safetensors.index.json').read_text())
weight_map = idx['weight_map']  # key → shard filename

# Open all shards (memory-mapped, essentially free)
opened = {}
for shard_name in set(weight_map.values()):
    opened[shard_name] = safe_open(
        str(SNAPSHOT / shard_name), framework='pt', device='cpu'
    )

def get_tensor(key: str) -> torch.Tensor | None:
    shard = weight_map.get(key)
    return opened[shard].get_tensor(key) if shard else None


# ── FP8 dequantization ────────────────────────────────────────────────────────
def dequantize_fp8(w_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 weight using scale_inv factor(s) → BF16.

    Handles both:
    - Scalar scale (per-tensor): w_bf16 = w_fp8 * scale  (Ministral-3 format)
    - 2D scale (per 128×128 block): block-wise multiply
    """
    if scale.ndim == 0:
        # Per-tensor: simple multiply
        return (w_fp8.to(torch.float32) * scale.to(torch.float32)).to(torch.bfloat16)

    # Squeeze extra batch dims
    while scale.ndim > 2:
        scale = scale.squeeze(0)

    if scale.ndim == 2:
        w_f32 = w_fp8.to(torch.float32)
        rows, cols = w_f32.shape[-2:]
        scale_rows, scale_cols = scale.shape
        block_m = rows // scale_rows
        block_n = cols // scale_cols
        s_f32 = scale.to(torch.float32)
        q = w_f32.reshape(scale_rows, block_m, scale_cols, block_n)
        s = s_f32.unsqueeze(1).unsqueeze(3)  # (scale_rows, 1, scale_cols, 1)
        return (q * s).to(torch.bfloat16).reshape(rows, cols)

    # 1D or unexpected shape: broadcast multiply
    return (w_fp8.to(torch.float32) * scale.to(torch.float32)).to(torch.bfloat16)


# ── Iterate all keys, dequantize FP8, accumulate shards ──────────────────────
FP8_DTYPES = {torch.float8_e4m3fn, getattr(torch, 'float8_e5m2', None)}
SKIP_KEYS  = {'weight_scale_inv', 'activation_scale'}  # consumed by dequant

export_index = {'metadata': {'format': 'pt'}, 'weight_map': {}}
current_shard: dict[str, torch.Tensor] = {}
current_bytes = 0
shard_idx = 0

all_keys = sorted(weight_map.keys())
print(f'{len(all_keys)} tensors in checkpoint')

skipped = 0
for key in all_keys:
    # Skip scale factors (they're consumed during dequantization)
    if any(s in key for s in SKIP_KEYS):
        skipped += 1
        continue

    tensor = get_tensor(key)
    if tensor is None:
        print(f'  WARNING: missing tensor {key}')
        continue

    if tensor.dtype in FP8_DTYPES:
        # Dequantize using block-wise scale factors
        scale_key = key.replace('.weight', '.weight_scale_inv')
        scale = get_tensor(scale_key)
        if scale is None:
            print(f'  WARNING: no scale for {key}, casting directly')
            tensor = tensor.to(torch.bfloat16)
        else:
            tensor = dequantize_fp8(tensor, scale)
    elif tensor.dtype not in (torch.bfloat16, torch.float16):
        tensor = tensor.to(torch.bfloat16)

    tensor = tensor.contiguous()
    shard_name = f'model-shard-{shard_idx:05d}.safetensors'
    current_shard[key] = tensor
    export_index['weight_map'][key] = shard_name
    current_bytes += tensor.numel() * tensor.element_size()

    if current_bytes >= SHARD_GB * 1024**3:
        out_path = OUTPUT / shard_name
        print(f'  Writing shard {shard_idx} ({current_bytes/1024**3:.1f} GB)...')
        save_file(current_shard, str(out_path))
        current_shard = {}
        current_bytes = 0
        shard_idx += 1
        gc.collect()

if current_shard:
    shard_name = f'model-shard-{shard_idx:05d}.safetensors'
    out_path = OUTPUT / shard_name
    print(f'  Writing final shard {shard_idx} ({current_bytes/1024**3:.1f} GB)...')
    save_file(current_shard, str(out_path))

print(f'Skipped {skipped} scale/activation tensors')
(OUTPUT / 'model.safetensors.index.json').write_text(json.dumps(export_index, indent=2))
print(f'Index written: {len(export_index["weight_map"])} entries')

# ── Copy and patch config (remove FP8 quantization_config) ────────────────────
cfg = AutoConfig.from_pretrained(MODEL_ID)
cfg.quantization_config = None  # remove FP8 → BnB can now be applied
cfg.save_pretrained(str(OUTPUT))
print(f'Config saved (quantization_config removed)')

# Copy tokenizer files
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.save_pretrained(str(OUTPUT))
print(f'Tokenizer saved')

print(f'\nExport complete: {OUTPUT}')
for f in sorted(OUTPUT.glob('*.safetensors')):
    print(f'  {f.name}: {f.stat().st_size/1024**3:.1f} GB')
