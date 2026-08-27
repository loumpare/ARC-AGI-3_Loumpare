"""Standalone training script mirroring notebooks/vision_encoder.ipynb's
eyes/brain/intuition + GRPO/JEPA pipeline, parameterized for CLI use so
several hypotheses can be run in parallel background processes (see
DESIGN_LOG 2026-08-16: ablating epochs budget vs curiosity_coef).

Mirrors the notebook exactly -- keep the two in sync by hand if either changes.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--game", default="ls20")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--group-size", type=int, default=24)
    parser.add_argument("--t-horizon", type=int, default=4)
    parser.add_argument("--max-chunks", type=int, default=12)
    parser.add_argument("--jepa-weight", type=float, default=1.0)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--curiosity-coef", type=float, default=0.0)
    parser.add_argument("--replay-coef", type=float, default=0.0)
    parser.add_argument("--replay-batch", type=int, default=4)
    parser.add_argument("--replay-buffer-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent.parent / "results"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    COMP_DIR = Path(__file__).resolve().parent.parent / "data"
    os.environ["OPERATION_MODE"] = "offline"
    os.environ["ENVIRONMENTS_DIR"] = str(COMP_DIR / "environment_files")

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from arc_agi import Arcade
    from arcengine import GameAction, GameState

    torch.manual_seed(args.seed)

    # ---- model (verbatim from vision_encoder.ipynb) ------------------------
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
            assert self.head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D axial RoPE"
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
        def __init__(self, latent_dim, num_actions, t_horizon, num_queries=4, num_heads=4, num_layers=2):
            super().__init__()
            self.t_horizon = t_horizon
            self.num_actions = num_actions
            self.latent_proj = nn.Linear(latent_dim, latent_dim)
            self.queries = nn.Parameter(torch.randn(num_queries, latent_dim) * 0.02)
            layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_heads, batch_first=True)
            self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.action_head = nn.Linear(latent_dim, num_actions * t_horizon)

        def forward(self, z):
            batch_size = z.shape[0]
            z_tok = self.latent_proj(z).unsqueeze(1)
            query_tok = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
            tokens = torch.cat([z_tok, query_tok], dim=1)
            out = self.transformer(tokens)
            pooled = out.mean(dim=1)
            return self.action_head(pooled).view(batch_size, self.t_horizon, self.num_actions)

    class Intuition(nn.Module):
        def __init__(self, latent_dim, num_actions, t_horizon, num_heads=4, num_layers=2):
            super().__init__()
            self.t_horizon = t_horizon
            self.action_embed = nn.Embedding(num_actions, latent_dim)
            self.step_embed = nn.Embedding(t_horizon, latent_dim)
            layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_heads, batch_first=True)
            self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.predict_head = nn.Linear(latent_dim, latent_dim)

        def forward(self, z, action_indices):
            batch_size = z.shape[0]
            z_tok = z.unsqueeze(1)
            act_tok = self.action_embed(action_indices)
            step_ids = torch.arange(self.t_horizon, device=z.device).unsqueeze(0).expand(batch_size, -1)
            act_tok = act_tok + self.step_embed(step_ids)
            tokens = torch.cat([z_tok, act_tok], dim=1)
            out = self.transformer(tokens)
            return self.predict_head(out[:, 1:, :])

    class ARC_AGI_3(nn.Module):
        def __init__(self, input_dim, latent_dim, patch_size, num_heads, eyes_layers, brain_layers,
                     intuition_layers, num_actions, t_horizon, num_queries=4, ema_decay=0.99):
            super().__init__()
            self.eyes = Eyes(Transformer(input_dim=input_dim, output_dim=latent_dim, num_heads=num_heads,
                                          num_layers=eyes_layers, patch_size=patch_size))
            self.target_eyes = Eyes(Transformer(input_dim=input_dim, output_dim=latent_dim, num_heads=num_heads,
                                                 num_layers=eyes_layers, patch_size=patch_size))
            self.target_eyes.load_state_dict(self.eyes.state_dict())
            for p in self.target_eyes.parameters():
                p.requires_grad_(False)
            self.ema_decay = ema_decay
            self.brain = Brain(latent_dim, num_actions, t_horizon, num_queries=num_queries,
                                num_heads=num_heads, num_layers=brain_layers)
            self.intuition = Intuition(latent_dim, num_actions, t_horizon,
                                        num_heads=num_heads, num_layers=intuition_layers)

        @torch.no_grad()
        def update_target(self):
            for p_online, p_target in zip(self.eyes.parameters(), self.target_eyes.parameters()):
                p_target.mul_(self.ema_decay).add_(p_online, alpha=1 - self.ema_decay)

        def forward(self, x):
            z = self.eyes(x)
            logits = self.brain(z)
            return z, logits

        def imagine(self, z, action_indices):
            return self.intuition(z, action_indices)

        @torch.no_grad()
        def target_latent(self, x):
            return self.target_eyes(x)

    # ---- env / reward / rollout (verbatim from vision_encoder.ipynb) -------
    ACTION_SPACE = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                     GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION7]
    ACTION_ID_TO_INDEX = {a.value: i for i, a in enumerate(ACTION_SPACE)}

    def frame_to_tensor(frame):
        grid = frame.frame[0]
        return torch.from_numpy(grid).float().unsqueeze(0).unsqueeze(0)

    def compute_reward(prev_frame, frame):
        reward = float(frame.levels_completed - prev_frame.levels_completed)
        if frame.state == GameState.WIN:
            reward += 1.0
        elif frame.state == GameState.GAME_OVER:
            reward -= 1.0
        reward -= 0.01
        return reward

    def rollout_chunk(env, model, t_horizon, max_chunks, curiosity_coef):
        frame = env.reset()
        chunks = []
        for _ in range(max_chunks):
            if frame.state != GameState.NOT_FINISHED:
                break
            x = frame_to_tensor(frame)
            z, logits = model(x)
            logits = logits.squeeze(0)

            mask = torch.zeros(t_horizon, len(ACTION_SPACE))
            first_mask = torch.full((len(ACTION_SPACE),), float("-inf"))
            for action_id in frame.available_actions:
                if action_id in ACTION_ID_TO_INDEX:
                    first_mask[ACTION_ID_TO_INDEX[action_id]] = 0.0
            mask[0] = first_mask

            dist = torch.distributions.Categorical(logits=logits + mask)
            entropy = dist.entropy()
            action_indices = dist.sample()
            executed_indices = action_indices.tolist()

            extrinsic_reward = 0.0
            prev_frame = frame
            real_frames = []
            for i in range(t_horizon):
                if frame.state != GameState.NOT_FINISHED:
                    break
                step_idx = executed_indices[i]
                action_id = ACTION_SPACE[step_idx].value
                if action_id not in frame.available_actions:
                    legal = [a for a in frame.available_actions if a in ACTION_ID_TO_INDEX]
                    if not legal:
                        break
                    step_idx = ACTION_ID_TO_INDEX[random.choice(legal)]
                    executed_indices[i] = step_idx
                action = ACTION_SPACE[step_idx]
                frame = env.step(action)
                real_frames.append(frame_to_tensor(frame))
                extrinsic_reward += compute_reward(prev_frame, frame)
                prev_frame = frame

            curiosity_bonus = 0.0
            if real_frames and curiosity_coef > 0:
                with torch.no_grad():
                    preds = model.imagine(z, torch.tensor(executed_indices).unsqueeze(0))
                    targets = torch.cat([model.target_latent(f) for f in real_frames], dim=0)
                    n_real = targets.shape[0]
                    surprise = F.mse_loss(preds[0, :n_real], targets, reduction="none").mean(dim=-1)
                curiosity_bonus = curiosity_coef * surprise.sum().item()
            total_reward = extrinsic_reward + curiosity_bonus

            # Clamped at the source: an illegal-action fallback forces an
            # action the current policy may assign near-zero probability to,
            # sending log_prob -> -inf and grpo_loss's magnitude with it.
            # Grad clipping alone only bounds the update's *norm*, not which
            # term dominates its *direction* (this swamped self-imitation's
            # replay signal in the first A/B test).
            executed_log_probs = dist.log_prob(torch.tensor(executed_indices)).clamp(min=-5.0)
            chunks.append(dict(z=z, action_indices=torch.tensor(executed_indices),
                                log_probs=executed_log_probs, reward=total_reward,
                                extrinsic_reward=extrinsic_reward, real_frames=real_frames,
                                entropy=entropy, x=x))
            if frame.state != GameState.NOT_FINISHED:
                break
        return chunks

    def collect_group(env, model, group_size, t_horizon, max_chunks, curiosity_coef):
        return [rollout_chunk(env, model, t_horizon, max_chunks, curiosity_coef) for _ in range(group_size)]

    def compute_losses(rollouts, model):
        episode_rewards = torch.tensor([sum(c["reward"] for c in chunks) for chunks in rollouts])
        advantages = (episode_rewards - episode_rewards.mean()) / (episode_rewards.std() + 1e-8)

        grpo_loss = torch.tensor(0.0)
        jepa_loss = torch.tensor(0.0)
        entropy_sum = torch.tensor(0.0)
        n_jepa_terms = 0
        n_entropy_terms = 0

        for chunks, advantage in zip(rollouts, advantages):
            for c in chunks:
                grpo_loss = grpo_loss - advantage * c["log_probs"].sum()
                entropy_sum = entropy_sum + c["entropy"].sum()
                n_entropy_terms += c["entropy"].numel()
                if c["real_frames"]:
                    preds = model.imagine(c["z"], c["action_indices"].unsqueeze(0))
                    with torch.no_grad():
                        targets = torch.cat([model.target_latent(f) for f in c["real_frames"]], dim=0)
                    n_real = targets.shape[0]
                    jepa_loss = jepa_loss + F.mse_loss(preds[0, :n_real], targets)
                    n_jepa_terms += 1

        grpo_loss = grpo_loss / len(rollouts)
        jepa_loss = jepa_loss / max(n_jepa_terms, 1)
        entropy_mean = entropy_sum / max(n_entropy_terms, 1)
        return grpo_loss, jepa_loss, entropy_mean

    def update_replay_buffer(buffer, rollouts, capacity):
        for chunks in rollouts:
            for c in chunks:
                buffer.append({"x": c["x"], "action_indices": c["action_indices"],
                                "extrinsic_reward": c["extrinsic_reward"]})
        buffer.sort(key=lambda e: e["extrinsic_reward"], reverse=True)
        del buffer[capacity:]
        return buffer

    def self_imitation_loss(model, buffer, replay_batch):
        if not buffer:
            return torch.tensor(0.0)
        sample = random.sample(buffer, min(replay_batch, len(buffer)))
        loss = torch.tensor(0.0)
        for entry in sample:
            _, logits = model(entry["x"])
            dist = torch.distributions.Categorical(logits=logits.squeeze(0))
            loss = loss - dist.log_prob(entry["action_indices"]).sum()
        return loss / len(sample)

    # ---- run -----------------------------------------------------------
    arc = Arcade()
    env = arc.make(args.game)

    model = ARC_AGI_3(input_dim=1, latent_dim=16, patch_size=4, num_heads=2, eyes_layers=1,
                       brain_layers=1, intuition_layers=1, num_actions=len(ACTION_SPACE),
                       t_horizon=args.t_horizon, num_queries=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    reward_history, extrinsic_reward_history = [], []
    grpo_loss_history, jepa_loss_history, entropy_history, replay_loss_history = [], [], [], []
    best_extrinsic, best_total = float("-inf"), float("-inf")
    replay_buffer = []

    t0 = time.time()
    for step in range(args.epochs):
        rollouts = collect_group(env, model, args.group_size, args.t_horizon, args.max_chunks, args.curiosity_coef)
        grpo_loss, jepa_loss, entropy = compute_losses(rollouts, model)
        replay_loss = self_imitation_loss(model, replay_buffer, args.replay_batch)
        loss = (grpo_loss + args.jepa_weight * jepa_loss - args.entropy_coef * entropy
                + args.replay_coef * replay_loss)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        model.update_target()

        for chunks in rollouts:
            ext = sum(c["extrinsic_reward"] for c in chunks)
            tot = sum(c["reward"] for c in chunks)
            if ext > best_extrinsic:
                best_extrinsic, best_total = ext, tot
        replay_buffer = update_replay_buffer(replay_buffer, rollouts, args.replay_buffer_size)

        mean_reward = sum(sum(c["reward"] for c in chunks) for chunks in rollouts) / len(rollouts)
        extrinsic_mean_reward = sum(sum(c["extrinsic_reward"] for c in chunks) for chunks in rollouts) / len(rollouts)
        reward_history.append(mean_reward)
        extrinsic_reward_history.append(extrinsic_mean_reward)
        grpo_loss_history.append(grpo_loss.item())
        jepa_loss_history.append(jepa_loss.item())
        entropy_history.append(entropy.item())
        replay_loss_history.append(replay_loss.item())

        if step % 10 == 0:
            print(f"[{args.run_name}] step {step:4d} | grpo_loss {grpo_loss.item():8.4f} | "
                  f"jepa_loss {jepa_loss.item():7.4f} | entropy {entropy.item():6.3f} | "
                  f"replay_loss {replay_loss.item():7.3f} | "
                  f"reward(total) {mean_reward:7.3f} | reward(task) {extrinsic_mean_reward:7.3f} | "
                  f"best(task) {best_extrinsic:7.3f}", flush=True)

    elapsed = time.time() - t0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "config": vars(args),
        "elapsed_seconds": elapsed,
        "best_extrinsic_reward": best_extrinsic,
        "best_total_reward": best_total,
        "reward_history": reward_history,
        "extrinsic_reward_history": extrinsic_reward_history,
        "grpo_loss_history": grpo_loss_history,
        "jepa_loss_history": jepa_loss_history,
        "entropy_history": entropy_history,
        "replay_loss_history": replay_loss_history,
    }
    out_path = out_dir / f"{args.run_name}.json"
    with open(out_path, "w") as f:
        json.dump(result, f)
    print(f"[{args.run_name}] DONE in {elapsed:.1f}s -- best_extrinsic={best_extrinsic:.3f} -- saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
