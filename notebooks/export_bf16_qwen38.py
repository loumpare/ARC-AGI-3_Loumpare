"""
One-time export: Qwen3.8-27B FP8 checkpoint -> plain BF16 safetensors.

Root cause this fixes: `checkpoints/qwen38_27b_fp8` stores weights natively as
torch.float8_e4m3fn with a companion `*.weight_scale_inv` tensor per 128x128
block (true_value = fp8_value.float() * block_scale). The STaR SFT notebooks
(local and Colab) were loading this checkpoint by deleting `quantization_config`
and casting straight to bfloat16 -- which silently SKIPS the block-wise
rescaling. Verified: this produces a model that loads without NaN but generates
pure gibberish even on trivial prompts ("Hello, how are you?"). Every local and
Colab STaR SFT training run so far fine-tuned a LoRA adapter on top of this
mis-scaled base model.

Reads tensors from the local checkpoint one at a time (memory-mapped),
dequantizes FP8 weights using their weight_scale_inv factors, writes shards of
~4 GB. Output: checkpoints/qwen38_27b_bf16/

Usage:
    python notebooks/export_bf16_qwen38.py
Then point both star_sft_qwen38.ipynb (Cell 4) and star_sft_qwen38_colab.ipynb
(Cell 5) at checkpoints/qwen38_27b_bf16/ instead of qwen38_27b_fp8/, and REMOVE
the `del cfg.quantization_config` step (not needed -- this export has none).
"""
import gc, json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer

SHARD_GB = 4
SOURCE = Path('/home/lavolpe/Bureau/Kaggle/ARC-AGI-3/checkpoints/qwen38_27b_fp8')
OUTPUT = Path('/home/lavolpe/Bureau/Kaggle/ARC-AGI-3/checkpoints/qwen38_27b_bf16')
OUTPUT.mkdir(parents=True, exist_ok=True)


def find_model_root(base: Path) -> Path:
    if (base / 'config.json').exists():
        return base
    for p in sorted(base.rglob('config.json')):
        return p.parent
    raise FileNotFoundError(f'config.json not found under {base}')


SOURCE = find_model_root(SOURCE)
print(f'Source: {SOURCE}')

idx = json.loads((SOURCE / 'model.safetensors.index.json').read_text())
weight_map = idx['weight_map']

opened = {}
for shard_name in set(weight_map.values()):
    opened[shard_name] = safe_open(str(SOURCE / shard_name), framework='pt', device='cpu')


def get_tensor(key: str):
    shard = weight_map.get(key)
    return opened[shard].get_tensor(key) if shard else None


def dequantize_fp8(w_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """true_value = fp8_value * scale, block-wise if scale is 2D."""
    if scale.ndim == 0:
        return (w_fp8.to(torch.float32) * scale.to(torch.float32)).to(torch.bfloat16)

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
        s = s_f32.unsqueeze(1).unsqueeze(3)
        return (q * s).to(torch.bfloat16).reshape(rows, cols)

    return (w_fp8.to(torch.float32) * scale.to(torch.float32)).to(torch.bfloat16)


FP8_DTYPES = {torch.float8_e4m3fn, getattr(torch, 'float8_e5m2', None)}
SKIP_KEYS = {'weight_scale_inv', 'activation_scale'}

export_index = {'metadata': {'format': 'pt'}, 'weight_map': {}}
current_shard: dict[str, torch.Tensor] = {}
current_bytes = 0
shard_idx = 0

all_keys = sorted(weight_map.keys())
print(f'{len(all_keys)} tensors in checkpoint')

skipped = 0
dequantized = 0
for i, key in enumerate(all_keys):
    if any(s in key for s in SKIP_KEYS):
        skipped += 1
        continue

    tensor = get_tensor(key)
    if tensor is None:
        print(f'  WARNING: missing tensor {key}')
        continue

    if tensor.dtype in FP8_DTYPES:
        scale_key = key.replace('.weight', '.weight_scale_inv')
        scale = get_tensor(scale_key)
        if scale is None:
            print(f'  WARNING: no scale for {key}, casting directly (WRONG, but only affects unquantized tensors)')
            tensor = tensor.to(torch.bfloat16)
        else:
            tensor = dequantize_fp8(tensor, scale)
            dequantized += 1
    elif tensor.dtype not in (torch.bfloat16, torch.float16):
        tensor = tensor.to(torch.bfloat16)

    tensor = tensor.contiguous()
    shard_name = f'model-shard-{shard_idx:05d}.safetensors'
    current_shard[key] = tensor
    export_index['weight_map'][key] = shard_name
    current_bytes += tensor.numel() * tensor.element_size()

    if current_bytes >= SHARD_GB * 1024**3:
        out_path = OUTPUT / shard_name
        print(f'  [{i+1}/{len(all_keys)}] Writing shard {shard_idx} ({current_bytes/1024**3:.1f} GB)...')
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

print(f'Skipped {skipped} scale tensors, dequantized {dequantized} FP8 tensors')
(OUTPUT / 'model.safetensors.index.json').write_text(json.dumps(export_index, indent=2))
print(f'Index written: {len(export_index["weight_map"])} entries')

cfg = AutoConfig.from_pretrained(str(SOURCE), trust_remote_code=True)
if hasattr(cfg, 'quantization_config'):
    cfg.quantization_config = None
cfg.save_pretrained(str(OUTPUT))
print('Config saved (quantization_config removed)')

tokenizer = AutoTokenizer.from_pretrained(str(SOURCE))
tokenizer.save_pretrained(str(OUTPUT))
print('Tokenizer saved')

print(f'\nExport complete: {OUTPUT}')
for f in sorted(OUTPUT.glob('*.safetensors')):
    print(f'  {f.name}: {f.stat().st_size/1024**3:.1f} GB')
