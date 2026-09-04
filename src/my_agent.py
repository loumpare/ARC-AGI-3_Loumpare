"""ARC-AGI-3 submission agent, per the official `arcprize/ARC-AGI-3-Kaggle-Starter`
contract (2026-08-31): subclass `agents.agent.Agent`, class named `MyAgent`,
implement `is_done`/`choose_action`. This is the file that gets dropped into
`ARC-AGI-3-Agents/agents/templates/my_agent.py` and registered in
`agents/__init__.py`'s `AVAILABLE_AGENTS` during the Kaggle competition rerun.

Model classes are copied verbatim from src/stage2_grpo_vanilla.py (kept in
sync by hand, matching that file's own convention) -- loads the trained
Eyes+Brain checkpoint once at import time, shared across every per-game
Agent instance the framework creates (avoids reloading 110x).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

device = torch.device("cpu")


# ---- model (verbatim from src/stage2_grpo_vanilla.py) ----------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=4096, theta=10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x):
        x1 = x[..., :self.dim // 2]
        x2 = x[..., self.dim // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope(self, x, cos, sin):
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)
        return (x * cos) + (self._rotate_half(x) * sin)


class patch_embedding(nn.Module):
    def __init__(self, input_dim, patch_size, output_dim, image_size=64):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.proj = nn.Conv2d(input_dim, output_dim, kernel_size=patch_size, stride=patch_size)
        rows = torch.arange(self.grid_size).repeat_interleave(self.grid_size)
        cols = torch.arange(self.grid_size).repeat(self.grid_size)
        self.register_buffer("rows", rows, persistent=False)
        self.register_buffer("cols", cols, persistent=False)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class RoPEAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, grid_size):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim % 4 == 0
        self.axial_dim = self.head_dim // 2
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.row_rope = RotaryEmbedding(dim=self.axial_dim, max_len=grid_size)
        self.col_rope = RotaryEmbedding(dim=self.axial_dim, max_len=grid_size)

    def _apply_2d_rope(self, x, rows, cols):
        x_row, x_col = x[..., :self.axial_dim], x[..., self.axial_dim:]
        row_cos, row_sin = self.row_rope.cos_cached[rows], self.row_rope.sin_cached[rows]
        col_cos, col_sin = self.col_rope.cos_cached[cols], self.col_rope.sin_cached[cols]
        x_row = self.row_rope.apply_rope(x_row, row_cos, row_sin)
        x_col = self.col_rope.apply_rope(x_col, col_cos, col_sin)
        return torch.cat([x_row, x_col], dim=-1)

    def forward(self, x, rows, cols):
        batch_size, seq_len, embed_dim = x.shape
        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self._apply_2d_rope(q, rows, cols)
        k = self._apply_2d_rope(k, rows, cols)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out_proj(attn_out)


class RoPETransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, grid_size, ff_dim=None):
        super().__init__()
        ff_dim = ff_dim or embed_dim * 4
        self.attention = RoPEAttention(embed_dim, num_heads, grid_size)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, embed_dim))

    def forward(self, x, rows, cols):
        x = x + self.attention(self.norm1(x), rows, cols)
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers, patch_size):
        super().__init__()
        self.embedding = patch_embedding(input_dim, patch_size, output_dim)
        self.layers = nn.ModuleList([
            RoPETransformerEncoderLayer(output_dim, num_heads, self.embedding.grid_size)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        rows, cols = self.embedding.rows, self.embedding.cols
        for layer in self.layers:
            x = layer(x, rows, cols)
        return self.fc_out(x)


class Eyes(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        return self.encoder(x).mean(dim=1)


class Brain(nn.Module):
    def __init__(self, latent_dim, num_actions, num_queries=4, num_heads=4, num_layers=2, cot_steps=1):
        super().__init__()
        self.num_actions = num_actions
        self.cot_steps = cot_steps
        self.latent_proj = nn.Linear(latent_dim, latent_dim)
        self.imagined_proj = nn.Linear(latent_dim, latent_dim)
        self.queries = nn.Parameter(torch.randn(num_queries, latent_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.action_head = nn.Linear(latent_dim, num_actions)

    def forward(self, z, imagined=None):
        batch_size = z.shape[0]
        query_tok = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        imagined_tok = self.imagined_proj(imagined) if imagined is not None else None
        thought = z
        for _ in range(self.cot_steps):
            z_tok = self.latent_proj(thought).unsqueeze(1)
            token_list = [z_tok, query_tok]
            if imagined_tok is not None:
                token_list.append(imagined_tok)
            tokens = torch.cat(token_list, dim=1)
            out = self.transformer(tokens)
            thought = out.mean(dim=1)
        return self.action_head(thought)


class Policy(nn.Module):
    def __init__(self, input_dim, latent_dim, patch_size, num_heads, eyes_layers,
                 brain_layers, num_actions, num_queries=4, cot_steps=1):
        super().__init__()
        self.eyes = Eyes(Transformer(input_dim=input_dim, output_dim=latent_dim, num_heads=num_heads,
                                      num_layers=eyes_layers, patch_size=patch_size))
        self.brain = Brain(latent_dim, num_actions, num_queries=num_queries,
                            num_heads=num_heads, num_layers=brain_layers, cot_steps=cot_steps)

    def load_checkpoint(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.eyes.load_state_dict(ckpt["eyes"])
        self.brain.load_state_dict(ckpt["brain"])
        return ckpt.get("config", {})

    def forward(self, x):
        with torch.no_grad():
            z = self.eyes(x)
        return self.brain(z)


# ---- action space + checkpoint (loaded once, module level) -----------------
ACTION_SPACE = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                 GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION7]
ACTION_ID_TO_INDEX = {a.value: i for i, a in enumerate(ACTION_SPACE)}

_CHECKPOINT_CANDIDATES = [
    Path("/kaggle/working/submission_brain_v1.pt"),
    Path("/kaggle/input/datasets/loumitrmas/arc-agi-3-submission-brain-v1/submission_brain_v1.pt"),
    Path(__file__).resolve().parent.parent / "checkpoints" / "submission_brain_v1.pt",
]
_CHECKPOINT_PATH = next((p for p in _CHECKPOINT_CANDIDATES if p.exists()), None)

_policy = Policy(input_dim=1, latent_dim=192, patch_size=4, num_heads=6, eyes_layers=6,
                  brain_layers=1, num_actions=len(ACTION_SPACE), num_queries=4).to(device)
if _CHECKPOINT_PATH is not None:
    _policy.load_checkpoint(_CHECKPOINT_PATH)
_policy.eval()


def _frame_to_tensor(frame: FrameData) -> torch.Tensor:
    import numpy as np
    grid = np.array(frame.frame[0], dtype=float)
    return torch.from_numpy(grid).float().unsqueeze(0).unsqueeze(0)


def _pick_action(frame: FrameData) -> GameAction | None:
    legal = [a for a in frame.available_actions if a in ACTION_ID_TO_INDEX]
    if not legal:
        return None
    x = _frame_to_tensor(frame)
    with torch.no_grad():
        logits = _policy(x).squeeze(0)
    mask = torch.full((len(ACTION_SPACE),), float("-inf"))
    for action_id in legal:
        mask[ACTION_ID_TO_INDEX[action_id]] = 0.0
    action_idx = int(torch.argmax(logits + mask))
    return ACTION_SPACE[action_idx]


class MyAgent(Agent):
    """Trained Eyes+Brain policy (GRPO with PPO-clip trust region + Go-Explore
    archive, --archive-prob 0.9 -- see DESIGN_LOG.md 2026-08-27/29). No win
    ever found on ls20 in ~536k+ local episodes; this submission validates
    the pipeline with a real trained agent, not a claim of a strong score.
    Greedy (argmax) at inference -- no exploration needed once deployed."""

    MAX_ACTIONS = 500  # ~2-3 full attempts at our trained max_steps=200 budget

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        action = _pick_action(latest_frame)
        if action is None:
            # ACTION6/click-only game -- unsupported by this policy (TODO.md item 2).
            return GameAction.RESET
        action.reasoning = f"trained policy (greedy): {action.value}"
        return action
