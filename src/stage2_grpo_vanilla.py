"""Standalone CLI mirror of notebooks/jepa_pretrain.ipynb's Stage 2 section
(DESIGN_LOG 2026-08-21 "restarted Stage 2 from scratch" entry): `eyes`
frozen on the Stage 1 JEPA checkpoint, no intuition/curiosity/archive/
replay. Ported verbatim for a background run against the real
(offline-mode) environment -- keep in sync by hand if the notebook's
Stage 2 cells change.

Adds two things on top of the notebook's original vanilla-GRPO logic:
1. Per-episode win/game-over/timeout outcome tracking, so a run can answer
   win frequency vs. the ~1/500-episode reference, and whether the success
   rate on episodes *after* the first win rises (retention) or falls back
   to zero.
2. A PPO-style clipped-ratio surrogate (2026-08-27 diagnostic,
   `grad_debug_c_ppo_clip_test.py`): vanilla GRPO here processes several
   minibatches sequentially per outer step, each with its own
   optimizer.step() -- by minibatch 2+, the advantage/actions were
   sampled under the *old* policy but graded against the *already-updated*
   one, with nothing bounding how far a single update can move the policy.
   Confirmed empirically to cause collapse (100% zero-variance-skip within
   10 steps, every entropy_coef from 0.001 to 1.0) vs. sustained,
   monotonic improvement over 80 steps with the ratio-clip added -- same
   seed/data, only the loss changed. `old_log_prob` is captured at
   collection time (frozen policy, matches PPO's importance-sampling
   assumption) and used to clip the surrogate objective.
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
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--cot-steps", type=int, default=1,
                         help="recurrent latent 'chain of thought' passes in Brain before deciding (1 = off/original)")
    parser.add_argument("--use-imagination", action="store_true",
                         help="feed Brain one-step-ahead Intuition predictions per candidate action")
    parser.add_argument("--curiosity-coef", type=float, default=0.0,
                         help="intrinsic bonus = coef * MSE(Intuition's 1-step prediction, actual next latent)")
    parser.add_argument("--use-archive", action="store_true",
                         help="Go-Explore-style: some rollouts resume from a replayed prefix of a past good trajectory instead of a fresh reset")
    parser.add_argument("--archive-size", type=int, default=8)
    parser.add_argument("--archive-prob", type=float, default=0.5,
                         help="probability a given rollout starts from an archived prefix rather than a fresh reset")
    parser.add_argument("--archive-min-live-steps", type=int, default=40,
                         help="archived replay prefix is cut short enough to leave at least this many live policy steps")
    parser.add_argument("--replay-coef", type=float, default=0.0,
                         help="self-imitation: behavior-cloning loss weight toward the best steps ever seen")
    parser.add_argument("--replay-buffer-size", type=int, default=64)
    parser.add_argument("--replay-batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--reward-std-eps", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=str(Path(__file__).resolve().parent.parent
                                                      / "checkpoints" / "world_model_pretrained.pt"))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent.parent / "results"))
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save-checkpoint", default=None,
                         help="path to save {eyes, brain, config} state_dicts to (every 50 steps + at the end); "
                              "no run before 2026-08-29 ever persisted a trained brain -- only JSON metric logs")
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

    class Intuition(nn.Module):
        """World model (Stage 1 JEPA, verbatim from jepa_pretrain.ipynb): given
        z_t and an action, predicts z_hat_{t+1}. t_horizon=1 to match the
        pretrained checkpoint. Frozen when used in Stage 2 -- only consulted
        for one-step imagination, never fine-tuned here (avoids the
        eyes<->policy coupling instability from the pre-08-21 joint runs)."""
        def __init__(self, latent_dim, num_actions, t_horizon=1, num_heads=4, num_layers=2):
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

    class Brain(nn.Module):
        """2026-08-29: added `cot_steps` (recurrent "thinking" passes over the
        pooled latent before deciding -- continuous-latent chain of thought,
        no tokenized text/language prior) and an optional `imagined` input
        (one-step-ahead latent per candidate action, from `Intuition`) fed in
        as extra tokens. Both default to their original no-op values
        (cot_steps=1, imagined=None) so existing behavior is unchanged unless
        explicitly enabled."""
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
                     brain_layers, num_actions, num_queries=4, cot_steps=1,
                     use_imagination=False, needs_intuition=False, intuition_layers=4):
            super().__init__()
            self.num_actions = num_actions
            self.use_imagination = use_imagination
            self.has_intuition = use_imagination or needs_intuition
            self.eyes = Eyes(Transformer(input_dim=input_dim, output_dim=latent_dim, num_heads=num_heads,
                                          num_layers=eyes_layers, patch_size=patch_size))
            for p in self.eyes.parameters():
                p.requires_grad_(False)
            self.brain = Brain(latent_dim, num_actions, num_queries=num_queries,
                                num_heads=num_heads, num_layers=brain_layers, cot_steps=cot_steps)
            if self.has_intuition:
                self.intuition = Intuition(latent_dim, num_actions, t_horizon=1,
                                            num_heads=num_heads, num_layers=intuition_layers)
                for p in self.intuition.parameters():
                    p.requires_grad_(False)
            if use_imagination:
                # Learned stand-in for an action's imagined outcome when that
                # action is currently illegal -- Intuition was never trained
                # on out-of-distribution (illegal-state) transitions for it.
                self.no_action_placeholder = nn.Parameter(torch.randn(latent_dim) * 0.02)

        def load_pretrained_eyes(self, checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            self.eyes.load_state_dict(ckpt["eyes"])
            if self.has_intuition:
                self.intuition.load_state_dict(ckpt["intuition"])

        def forward(self, x, masks=None):
            self.eyes.eval()
            with torch.no_grad():
                z = self.eyes(x)

            imagined = None
            if self.use_imagination:
                self.intuition.eval()
                with torch.no_grad():
                    batch_size = z.shape[0]
                    z_rep = z.repeat_interleave(self.num_actions, dim=0)
                    action_ids = torch.arange(self.num_actions, device=z.device).unsqueeze(0)
                    action_ids = action_ids.expand(batch_size, -1).reshape(-1, 1)
                    imagined_flat = self.intuition(z_rep, action_ids).squeeze(1)  # (batch*num_actions, latent_dim)
                    imagined = imagined_flat.view(batch_size, self.num_actions, -1)
                    if masks is not None:
                        legal = (masks == 0.0).unsqueeze(-1)  # (batch, num_actions, 1)
                        imagined = torch.where(legal, imagined, self.no_action_placeholder)

            return self.brain(z, imagined=imagined)

        @torch.no_grad()
        def encode(self, x):
            self.eyes.eval()
            return self.eyes(x)

        @torch.no_grad()
        def predict_next_latents(self, z, action_idxs):
            """Batched: action_idxs is (batch,), one action per row (2026-08-29
            fix -- the original per-row Python-loop version of this call was
            the actual bottleneck in curiosity-enabled runs, not GPU
            contention: ~24 rollouts x up to 200 steps x 3 tiny unbatched
            model calls per outer step made those runs 100x+ slower than
            archive/replay-only ones at the same GPU load)."""
            self.intuition.eval()
            return self.intuition(z, action_idxs.unsqueeze(1)).squeeze(1)

    ACTION_SPACE = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                     GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION7]
    ACTION_ID_TO_INDEX = {a.value: i for i, a in enumerate(ACTION_SPACE)}

    policy = Policy(input_dim=1, latent_dim=192, patch_size=4, num_heads=6, eyes_layers=6,
                     brain_layers=1, num_actions=len(ACTION_SPACE), num_queries=4,
                     cot_steps=args.cot_steps, use_imagination=args.use_imagination,
                     needs_intuition=(args.curiosity_coef > 0), intuition_layers=4).to(device)
    policy.load_pretrained_eyes(Path(args.checkpoint))
    print(f"[{args.run_name}] cot_steps={args.cot_steps} use_imagination={args.use_imagination} "
          f"curiosity_coef={args.curiosity_coef} use_archive={args.use_archive} "
          f"replay_coef={args.replay_coef}", flush=True)

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

    def collect_group(envs, model, group_size, max_steps, nan_frame_counter, action_counts,
                       archive=None, archive_prob=0.0, archive_min_live_steps=40, curiosity_coef=0.0):
        """`archive`, if given, is a shared list of {"actions": [...], "reward": float}
        dicts (Go-Explore-style, 2026-08-29): with probability `archive_prob` a
        rollout resumes from a replayed prefix of a past good trajectory instead
        of a fresh reset -- deterministic since `ls20` has no RNG of its own
        (verified 2026-08-27), so replaying a stored action sequence reliably
        reaches the same state. The replayed prefix is not used for training
        (it wasn't sampled from the current policy); only the live continuation
        counts toward `reward`/advantage. `full_actions`/`archive_value` on each
        returned rollout are the whole trajectory (replay + live), for the
        caller to update the archive with."""
        frames, trajectories, full_actions, replayed_reward = [], [], [], []
        for i in range(group_size):
            frame = envs[i].reset()
            actions_i, replay_r = [], 0.0
            if archive and random.random() < archive_prob:
                entry = random.choice(archive)
                seq = entry["actions"]
                if seq:
                    cut = random.randint(0, min(len(seq) - 1, max(0, max_steps - archive_min_live_steps)))
                    for a in seq[:cut]:
                        prev = frame
                        frame = envs[i].step(a)
                        replay_r += compute_reward(prev, frame)
                        actions_i.append(a)
                        if frame.state != GameState.NOT_FINISHED:
                            break
            frames.append(frame)
            full_actions.append(actions_i)
            replayed_reward.append(replay_r)
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

            masks = torch.full((len(idxs), len(ACTION_SPACE)), float("-inf"), device=device)
            for row, i in enumerate(idxs):
                for action_id in legal_lists[i]:
                    masks[row, ACTION_ID_TO_INDEX[action_id]] = 0.0

            with torch.no_grad():
                logits = model(xs, masks)

            dist = torch.distributions.Categorical(logits=logits + masks)
            action_idxs = dist.sample()
            old_log_probs = dist.log_prob(action_idxs).detach()

            next_frames = [None] * len(idxs)
            for row, i in enumerate(idxs):
                action_idx = action_idxs[row]
                action_counts[action_idx.item()] += 1
                prev_frame = frames[i]
                action = ACTION_SPACE[action_idx.item()]
                frame = envs[i].step(action)
                frames[i] = frame
                next_frames[row] = frame
                full_actions[i].append(action)
                reward = compute_reward(prev_frame, frame)
                trajectories[i].append({"grid": frame.frame[0].copy(), "action": action, "reward": reward})
                steps_per_rollout[i].append(dict(x=xs[row:row + 1], mask=masks[row],
                                                  action_idx=action_idx.detach(),
                                                  old_log_prob=old_log_probs[row], reward=reward))

            if curiosity_coef > 0:
                # Batched (2026-08-29 fix): one call for the whole active
                # group per timestep, not one per row -- env.step() itself
                # stays sequential (the real engine is stateful), only the
                # model calls are batched.
                xs_next = torch.cat([frame_to_tensor(f) for f in next_frames], dim=0)
                z_t = model.encode(xs)
                z_next = model.encode(xs_next)
                predicted = model.predict_next_latents(z_t, action_idxs)
                bonuses = curiosity_coef * torch.nn.functional.mse_loss(
                    predicted, z_next, reduction="none").mean(dim=1)
                for row, i in enumerate(idxs):
                    bonus = bonuses[row].item()
                    steps_per_rollout[i][-1]["reward"] += bonus
                    trajectories[i][-1]["reward"] += bonus

        rollouts = []
        for i in range(group_size):
            total_reward = sum(s["reward"] for s in steps_per_rollout[i])
            outcome = ("win" if frames[i].state == GameState.WIN
                       else "game_over" if frames[i].state == GameState.GAME_OVER
                       else "timeout")
            rollouts.append(dict(steps=steps_per_rollout[i], trajectory=trajectories[i],
                                  reward=total_reward, outcome=outcome,
                                  full_actions=full_actions[i],
                                  archive_value=replayed_reward[i] + total_reward))
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

    def compute_losses(minibatch, model, clip_eps):
        xs = torch.cat([s["x"] for s, _ in minibatch], dim=0)
        masks = torch.stack([s["mask"] for s, _ in minibatch])
        action_idxs = torch.stack([s["action_idx"] for s, _ in minibatch])
        old_log_probs = torch.stack([s["old_log_prob"] for s, _ in minibatch])
        advantages = torch.stack([adv for _, adv in minibatch]).to(xs.device)

        logits = model(xs, masks)
        dist = torch.distributions.Categorical(logits=logits + masks)
        new_log_probs = dist.log_prob(action_idxs)

        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        surrogate = torch.min(ratio * advantages, clipped_ratio * advantages)
        policy_loss = -surrogate.mean()
        entropy = dist.entropy().mean()
        return policy_loss, entropy

    # ---- run --------------------------------------------------------------
    arc = Arcade()
    envs = [arc.make(args.game) for _ in range(args.group_size)]

    optimizer = torch.optim.Adam(policy.brain.parameters(), lr=args.lr)

    reward_history, loss_history, entropy_history = [], [], []
    task_std_history, grad_norm_history = [], []
    outcome_log = []  # one entry per live episode, in order: "win" | "game_over" | "timeout"
    best_run = {"reward": float("-inf"), "outcome": None}
    skipped_updates = 0
    nan_skips = 0
    nan_frame_counter = [0]  # non-finite input frame seen during collection (mutable box)
    action_counts = [0] * len(ACTION_SPACE)  # cumulative actions sampled, for distribution sanity checks
    archive = [] if args.use_archive else None  # Go-Explore-style: {"actions": [...], "reward": float}
    replay_buffer = []  # self-imitation: best individual steps ever seen, {"x", "action_idx", "episode_reward"}
    replay_loss_history = []

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
        rollouts = collect_group(envs, policy, args.group_size, args.max_steps,
                                  nan_frame_counter, action_counts,
                                  archive=archive, archive_prob=args.archive_prob,
                                  archive_min_live_steps=args.archive_min_live_steps,
                                  curiosity_coef=args.curiosity_coef)

        for r in live_rollouts_of(rollouts):
            outcome_log.append(r["outcome"])
            if r["reward"] > best_run["reward"]:
                best_run["reward"] = r["reward"]
                best_run["outcome"] = r["outcome"]

        if archive is not None:
            for r in live_rollouts_of(rollouts):
                archive.append({"actions": r["full_actions"], "reward": r["archive_value"]})
            archive.sort(key=lambda e: e["reward"], reverse=True)
            del archive[args.archive_size:]

        if args.replay_coef > 0:
            for r in live_rollouts_of(rollouts):
                for s in r["steps"]:
                    replay_buffer.append({"x": s["x"], "action_idx": s["action_idx"], "episode_reward": r["reward"]})
            replay_buffer.sort(key=lambda e: e["episode_reward"], reverse=True)
            del replay_buffer[args.replay_buffer_size:]

        mean_reward = sum(r["reward"] for r in rollouts) / len(rollouts)
        live = live_rollouts_of(rollouts)
        task_std = episode_rewards_of(rollouts).std().item() if live else 0.0
        grad_norm_this_step = 0.0

        if task_std <= args.reward_std_eps:
            skipped_updates += 1
            loss = entropy = torch.tensor(0.0)
        else:
            flat_steps = flatten_steps_with_advantage(rollouts)
            random.shuffle(flat_steps)
            for mb_start in range(0, len(flat_steps), args.minibatch_size):
                minibatch = flat_steps[mb_start:mb_start + args.minibatch_size]
                loss, entropy = compute_losses(minibatch, policy, args.clip_eps)
                total_loss = loss - args.entropy_coef * entropy

                optimizer.zero_grad()
                if torch.isfinite(total_loss):
                    total_loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(policy.brain.parameters(),
                                                                 max_norm=args.grad_clip_norm)
                    if torch.isfinite(grad_norm):
                        optimizer.step()
                        grad_norm_this_step = grad_norm.item()
                    else:
                        optimizer.zero_grad()
                        nan_skips += 1
                else:
                    nan_skips += 1

        replay_loss_val = 0.0
        if args.replay_coef > 0 and replay_buffer:
            sample = random.sample(replay_buffer, min(args.replay_batch, len(replay_buffer)))
            xs_r = torch.cat([e["x"] for e in sample], dim=0)
            action_idxs_r = torch.stack([e["action_idx"] for e in sample])
            logits_r = policy(xs_r)
            dist_r = torch.distributions.Categorical(logits=logits_r)
            replay_loss = -dist_r.log_prob(action_idxs_r).mean()
            optimizer.zero_grad()
            (args.replay_coef * replay_loss).backward()
            replay_grad_norm = torch.nn.utils.clip_grad_norm_(policy.brain.parameters(), max_norm=args.grad_clip_norm)
            if torch.isfinite(replay_grad_norm):
                optimizer.step()
                replay_loss_val = replay_loss.item()
            else:
                optimizer.zero_grad()

        reward_history.append(mean_reward)
        loss_history.append(loss.item())
        entropy_history.append(entropy.item())
        task_std_history.append(task_std)
        grad_norm_history.append(grad_norm_this_step)
        replay_loss_history.append(replay_loss_val)

        if (step + 1) % args.log_every == 0 or step == 0 or step == args.epochs - 1:
            n_win = outcome_log.count("win")
            n_ep = len(outcome_log)
            win_rate = n_win / n_ep if n_ep else 0.0
            skip_rate = skipped_updates / (step + 1)
            n_actions = sum(action_counts)
            action_dist = ([round(c / n_actions, 3) for c in action_counts] if n_actions else action_counts)
            print(f"[{args.run_name}] step {step+1:5d}/{args.epochs} | loss {loss.item():+7.3f} | "
                  f"entropy {entropy.item():5.3f} | task_std {task_std:6.4f} | grad_norm {grad_norm_this_step:6.3f} | "
                  f"reward {mean_reward:+6.3f} | best {best_run['reward']:+6.3f} | "
                  f"episodes {n_ep:6d} | wins {n_win:3d} ({win_rate:.5f}) | "
                  f"skip_rate {skip_rate:.3f} | nan_skips {nan_skips} | nan_frames {nan_frame_counter[0]} | "
                  f"action_dist {action_dist} | replay_loss {replay_loss_val:6.3f} | "
                  f"archive_n {len(archive) if archive is not None else 0} | replay_buf_n {len(replay_buffer)}",
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
                "task_std_history": task_std_history,
                "grad_norm_history": grad_norm_history,
                "replay_loss_history": replay_loss_history,
                "archive_final_size": len(archive) if archive is not None else 0,
                "action_counts": action_counts,
                "outcome_log": outcome_log,
                "skipped_updates": skipped_updates,
                "nan_skips": nan_skips,
                "nan_frames_seen": nan_frame_counter[0],
                "retention": retention_report(),
            }
            with open(out_path, "w") as f:
                json.dump(result, f)

            if args.save_checkpoint:
                torch.save({"eyes": policy.eyes.state_dict(), "brain": policy.brain.state_dict(),
                            "config": vars(args), "step": step + 1}, args.save_checkpoint)

    elapsed = time.time() - t0
    print(f"[{args.run_name}] DONE in {elapsed:.1f}s -- episodes={len(outcome_log)} "
          f"wins={outcome_log.count('win')} best={best_run['reward']:.3f} -- saved to {out_path}", flush=True)
    if args.save_checkpoint:
        print(f"[{args.run_name}] checkpoint saved to {args.save_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
