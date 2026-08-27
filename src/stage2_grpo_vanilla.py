"""Standalone CLI mirror of notebooks/jepa_pretrain.ipynb's Stage 2 section
(DESIGN_LOG 2026-08-21 "restarted Stage 2 from scratch" entry): minimal
vanilla GRPO, `eyes` frozen on the Stage 1 JEPA checkpoint, no intuition/
curiosity/PPO-clip/archive/replay. Ported verbatim for a background run
against the real (offline-mode) environment -- keep in sync by hand if the
notebook's Stage 2 cells change.

Adds only observability on top of the notebook's logic (no algorithm
change): per-episode win/game-over/timeout outcome tracking, so the run can
answer the two questions the 2026-08-21 reset was meant to test --
win frequency vs. the ~1/500-episode reference, and whether the success
rate on episodes *after* the first win rises (retention) or falls back to
zero.
"""
import argparse
import json
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
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--reward-std-eps", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=str(Path(__file__).resolve().parent.parent
                                                      / "checkpoints" / "world_model_pretrained.pt"))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent.parent / "results"))
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    import os
    COMP_DIR = Path(__file__).resolve().parent.parent / "data"
    os.environ["OPERATION_MODE"] = "offline"
    os.environ["ENVIRONMENTS_DIR"] = str(COMP_DIR / "environment_files")

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from arc_agi import Arcade
    from arcengine import GameAction, GameState

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.run_name}] device={device} game={args.game} epochs={args.epochs} "
          f"group_size={args.group_size} max_steps={args.max_steps}", flush=True)

    # ---- model (verbatim from jepa_pretrain.ipynb) --------------------------
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
        def __init__(self, latent_dim, num_actions, num_queries=4, num_heads=4, num_layers=2):
            super().__init__()
            self.num_actions = num_actions
            self.latent_proj = nn.Linear(latent_dim, latent_dim)
            self.queries = nn.Parameter(torch.randn(num_queries, latent_dim) * 0.02)
            layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_heads, batch_first=True)
            self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.action_head = nn.Linear(latent_dim, num_actions)

        def forward(self, z):
            batch_size = z.shape[0]
            z_tok = self.latent_proj(z).unsqueeze(1)
            query_tok = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
            tokens = torch.cat([z_tok, query_tok], dim=1)
            out = self.transformer(tokens)
            pooled = out.mean(dim=1)
            return self.action_head(pooled)

    class Policy(nn.Module):
        def __init__(self, input_dim, latent_dim, patch_size, num_heads, eyes_layers,
                     brain_layers, num_actions, num_queries=4):
            super().__init__()
            self.eyes = Eyes(Transformer(input_dim=input_dim, output_dim=latent_dim, num_heads=num_heads,
                                          num_layers=eyes_layers, patch_size=patch_size))
            for p in self.eyes.parameters():
                p.requires_grad_(False)
            self.brain = Brain(latent_dim, num_actions, num_queries=num_queries,
                                num_heads=num_heads, num_layers=brain_layers)

        def load_pretrained_eyes(self, checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            self.eyes.load_state_dict(ckpt["eyes"])

        def forward(self, x):
            self.eyes.eval()
            with torch.no_grad():
                z = self.eyes(x)
            return self.brain(z)

    ACTION_SPACE = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                     GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION7]
    ACTION_ID_TO_INDEX = {a.value: i for i, a in enumerate(ACTION_SPACE)}

    policy = Policy(input_dim=1, latent_dim=192, patch_size=4, num_heads=6, eyes_layers=6,
                     brain_layers=1, num_actions=len(ACTION_SPACE), num_queries=4).to(device)
    policy.load_pretrained_eyes(Path(args.checkpoint))

    def count_parameters(module):
        return sum(p.numel() for p in module.parameters())

    print(f"[{args.run_name}] eyes (frozen, pretrained) {count_parameters(policy.eyes):,} | "
          f"brain (fresh, trainable) {count_parameters(policy.brain):,}", flush=True)

    def frame_to_tensor(frame):
        grid = frame.frame[0]
        return torch.from_numpy(grid).float().unsqueeze(0).unsqueeze(0).to(device)

    def compute_reward(prev_frame, frame):
        reward = float(frame.levels_completed - prev_frame.levels_completed)
        if frame.state == GameState.WIN:
            reward += 1.0
        elif frame.state == GameState.GAME_OVER:
            reward -= 1.0
        reward -= 0.01
        return reward

    def collect_group(envs, model, group_size, max_steps, nan_frame_counter):
        frames, trajectories = [], []
        for i in range(group_size):
            frame = envs[i].reset()
            frames.append(frame)
            trajectories.append([{"grid": frame.frame[0].copy(), "action": None, "reward": 0.0}])

        steps_per_rollout = [[] for _ in range(group_size)]
        active = [True] * group_size

        for _ in range(max_steps):
            idxs, legal_lists = [], {}
            for i in range(group_size):
                if not active[i]:
                    continue
                if frames[i].state != GameState.NOT_FINISHED:
                    active[i] = False
                    continue
                legal = [a for a in frames[i].available_actions if a in ACTION_ID_TO_INDEX]
                if not legal:
                    active[i] = False
                    continue
                legal_lists[i] = legal
                idxs.append(i)

            if not idxs:
                break

            xs = torch.cat([frame_to_tensor(frames[i]) for i in idxs], dim=0)
            if not torch.isfinite(xs).all():
                nan_frame_counter[0] += 1
            with torch.no_grad():
                logits = model(xs)

            masks = torch.full((len(idxs), len(ACTION_SPACE)), float("-inf"), device=device)
            for row, i in enumerate(idxs):
                for action_id in legal_lists[i]:
                    masks[row, ACTION_ID_TO_INDEX[action_id]] = 0.0

            dist = torch.distributions.Categorical(logits=logits + masks)
            action_idxs = dist.sample()

            for row, i in enumerate(idxs):
                action_idx = action_idxs[row]
                prev_frame = frames[i]
                frame = envs[i].step(ACTION_SPACE[action_idx.item()])
                frames[i] = frame
                reward = compute_reward(prev_frame, frame)
                trajectories[i].append({"grid": frame.frame[0].copy(),
                                         "action": ACTION_SPACE[action_idx.item()], "reward": reward})
                steps_per_rollout[i].append(dict(x=xs[row:row + 1], mask=masks[row],
                                                  action_idx=action_idx.detach(), reward=reward))

        rollouts = []
        for i in range(group_size):
            total_reward = sum(s["reward"] for s in steps_per_rollout[i])
            outcome = ("win" if frames[i].state == GameState.WIN
                       else "game_over" if frames[i].state == GameState.GAME_OVER
                       else "timeout")
            rollouts.append(dict(steps=steps_per_rollout[i], trajectory=trajectories[i],
                                  reward=total_reward, outcome=outcome))
        return rollouts

    def live_rollouts_of(rollouts):
        return [r for r in rollouts if r["steps"]]

    def episode_rewards_of(rollouts):
        return torch.tensor([r["reward"] for r in live_rollouts_of(rollouts)])

    def flatten_steps_with_advantage(rollouts):
        live = live_rollouts_of(rollouts)
        if not live:
            return []
        episode_rewards = torch.tensor([r["reward"] for r in live])
        advantages = (episode_rewards - episode_rewards.mean()) / (episode_rewards.std() + 1e-8)
        flat = []
        for r, advantage in zip(live, advantages):
            for s in r["steps"]:
                flat.append((s, advantage))
        return flat

    def compute_losses(minibatch, model):
        xs = torch.cat([s["x"] for s, _ in minibatch], dim=0)
        masks = torch.stack([s["mask"] for s, _ in minibatch])
        action_idxs = torch.stack([s["action_idx"] for s, _ in minibatch])
        advantages = torch.stack([adv for _, adv in minibatch]).to(xs.device)

        logits = model(xs)
        dist = torch.distributions.Categorical(logits=logits + masks)
        log_probs = dist.log_prob(action_idxs)

        policy_loss = -(advantages * log_probs).mean()
        entropy = dist.entropy().mean()
        return policy_loss, entropy

    # ---- run --------------------------------------------------------------
    arc = Arcade()
    envs = [arc.make(args.game) for _ in range(args.group_size)]

    optimizer = torch.optim.Adam(policy.brain.parameters(), lr=0.001)

    reward_history, loss_history, entropy_history = [], [], []
    outcome_log = []  # one entry per live episode, in order: "win" | "game_over" | "timeout"
    best_run = {"reward": float("-inf"), "outcome": None}
    skipped_updates = 0
    nan_skips = 0
    nan_frame_counter = [0]  # non-finite input frame seen during collection (mutable box)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.run_name}.json"

    def first_win_index():
        for idx, o in enumerate(outcome_log):
            if o == "win":
                return idx
        return None

    def retention_report():
        idx = first_win_index()
        if idx is None:
            return None
        before = outcome_log[:idx]
        after = outcome_log[idx + 1:]
        wins_before = before.count("win")
        wins_after = after.count("win")
        rate_before = wins_before / len(before) if before else None
        rate_after = wins_after / len(after) if after else None
        return {
            "first_win_episode_index": idx,
            "episodes_before": len(before),
            "episodes_after": len(after),
            "win_rate_before": rate_before,
            "win_rate_after": rate_after,
            "total_wins": outcome_log.count("win"),
        }

    t0 = time.time()
    for step in range(args.epochs):
        rollouts = collect_group(envs, policy, args.group_size, args.max_steps, nan_frame_counter)

        for r in live_rollouts_of(rollouts):
            outcome_log.append(r["outcome"])
            if r["reward"] > best_run["reward"]:
                best_run["reward"] = r["reward"]
                best_run["outcome"] = r["outcome"]

        mean_reward = sum(r["reward"] for r in rollouts) / len(rollouts)
        live = live_rollouts_of(rollouts)
        task_std = episode_rewards_of(rollouts).std().item() if live else 0.0

        if task_std <= args.reward_std_eps:
            skipped_updates += 1
            loss = entropy = torch.tensor(0.0)
        else:
            flat_steps = flatten_steps_with_advantage(rollouts)
            random.shuffle(flat_steps)
            for mb_start in range(0, len(flat_steps), args.minibatch_size):
                minibatch = flat_steps[mb_start:mb_start + args.minibatch_size]
                loss, entropy = compute_losses(minibatch, policy)
                total_loss = loss - args.entropy_coef * entropy

                optimizer.zero_grad()
                if torch.isfinite(total_loss):
                    total_loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(policy.brain.parameters(),
                                                                 max_norm=args.grad_clip_norm)
                    if torch.isfinite(grad_norm):
                        optimizer.step()
                    else:
                        optimizer.zero_grad()
                        nan_skips += 1
                else:
                    nan_skips += 1

        reward_history.append(mean_reward)
        loss_history.append(loss.item())
        entropy_history.append(entropy.item())

        if (step + 1) % args.log_every == 0 or step == 0 or step == args.epochs - 1:
            n_win = outcome_log.count("win")
            n_ep = len(outcome_log)
            win_rate = n_win / n_ep if n_ep else 0.0
            skip_rate = skipped_updates / (step + 1)
            print(f"[{args.run_name}] step {step+1:5d}/{args.epochs} | loss {loss.item():+7.3f} | "
                  f"entropy {entropy.item():5.3f} | reward {mean_reward:+6.3f} | best {best_run['reward']:+6.3f} | "
                  f"episodes {n_ep:6d} | wins {n_win:3d} ({win_rate:.5f}) | "
                  f"skip_rate {skip_rate:.3f} | nan_skips {nan_skips} | nan_frames {nan_frame_counter[0]}",
                  flush=True)
            ret = retention_report()
            if ret is not None:
                print(f"[{args.run_name}]   retention: first_win@ep{ret['first_win_episode_index']} | "
                      f"rate_before={ret['win_rate_before']} | rate_after={ret['win_rate_after']} | "
                      f"total_wins={ret['total_wins']}", flush=True)

        if (step + 1) % 50 == 0 or step == args.epochs - 1:
            elapsed = time.time() - t0
            result = {
                "config": vars(args),
                "elapsed_seconds": elapsed,
                "step": step + 1,
                "best_reward": best_run["reward"],
                "best_outcome": best_run["outcome"],
                "reward_history": reward_history,
                "loss_history": loss_history,
                "entropy_history": entropy_history,
                "outcome_log": outcome_log,
                "skipped_updates": skipped_updates,
                "nan_skips": nan_skips,
                "nan_frames_seen": nan_frame_counter[0],
                "retention": retention_report(),
            }
            with open(out_path, "w") as f:
                json.dump(result, f)

    elapsed = time.time() - t0
    print(f"[{args.run_name}] DONE in {elapsed:.1f}s -- episodes={len(outcome_log)} "
          f"wins={outcome_log.count('win')} best={best_run['reward']:.3f} -- saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
