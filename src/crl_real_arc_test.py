"""Second validation stage for goal_conditioned_crl.py (see
crl_sanity_check.py for the first, synthetic-gridworld stage, which PASSed
-- mean goal-reaching distance dropped from ~6 to ~0.5 after enough
training). This stage swaps the synthetic 2D state for a REAL ARC-AGI-3
game's raw grid, encoded by a small shared CNN, and otherwise reuses the
exact same hindsight-relabeling + InfoNCE + soft-policy-gradient training
loop -- if the mechanism transfers, embedding-space distance from a
policy-driven rollout to a genuinely-reached held-out goal state should
drop with training, same as the synthetic case.

Deliberately NOT reusing the project's existing RoPE-transformer "Eyes"
encoder from stage2_grpo_vanilla.py -- this is a first feasibility check of
a NEW objective (contrastive, not PPO-clip), kept minimal/fast on purpose;
swapping in the heavier existing encoder is a natural next step only if
this passes.

No reward function, no game-specific logic anywhere (see
feedback_no_game_hacking) -- purely self-supervised from raw grids and
hindsight-relabeled goals, exactly like the synthetic check.

Usage: python3 src/crl_real_arc_test.py --game ls20 --epochs 400
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "data", "ARC-AGI-3-Agents"))
os.environ.setdefault("OPERATION_MODE", "offline")
os.environ.setdefault("ENVIRONMENTS_DIR", os.path.join(REPO, "data", "environment_files"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import arc_agi  # noqa: E402
from arcengine import GameAction  # noqa: E402

from goal_conditioned_crl import CRLCritic, CRLPolicy, info_nce_critic_loss, policy_loss_soft_action  # noqa: E402

NUM_COLORS = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTION_ENUM = {1: GameAction.ACTION1, 2: GameAction.ACTION2, 3: GameAction.ACTION3,
               4: GameAction.ACTION4, 5: GameAction.ACTION5, 7: GameAction.ACTION7}


class GridEncoder(nn.Module):
    """Small shared CNN: one-hot(16 colors) grid -> latent vector. Used for
    BOTH state and goal (they're literally the same kind of object -- a
    grid). Deliberately lightweight (this is a feasibility check, not the
    final architecture) -- 64x64 -> 16x16 -> 4x4 -> flatten -> linear."""

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(NUM_COLORS, 32, kernel_size=4, stride=4), nn.SiLU(),   # 64 -> 16
            nn.Conv2d(32, 64, kernel_size=4, stride=4), nn.SiLU(),          # 16 -> 4
        )
        self.out = nn.Linear(64 * 4 * 4, latent_dim)
        # unlike MLPResNet's residual blocks (which already have internal
        # LayerNorm), this plain conv stack had NOTHING bounding its output
        # scale -- confirmed empirically (2026-09-08) that without this,
        # embedding norms grow unboundedly over training (mean output norm
        # went from ~0.2 at init to embedding distances ~8 after 1500
        # epochs, versus ~3-5 early in training), destabilizing the whole
        # InfoNCE distance comparison. A final LayerNorm keeps the latent
        # scale bounded regardless of how long training runs.
        self.final_norm = nn.LayerNorm(latent_dim)

    def forward(self, grid_int: torch.Tensor) -> torch.Tensor:
        # grid_int: [B, 64, 64] integer color ids
        onehot = F.one_hot(grid_int.long(), NUM_COLORS).permute(0, 3, 1, 2).float()
        x = self.conv(onehot)
        return self.final_norm(self.out(x.flatten(1)))


def collect_rollouts(env_factory, game_id: str, n_rollouts: int, max_steps: int,
                      action_to_idx: dict[int, int]):
    """Uniform-random rollouts (pure exploration, no goal/reward involved
    yet) -- returns a list of (grids [T+1, 64, 64] int, action_idxs [T] int)
    tuples. `action_to_idx` maps a raw GameAction.value to a dense 0..N-1
    index matching CRLCritic's action_dim."""
    trajectories = []
    for _ in range(n_rollouts):
        env = env_factory.make(game_id)
        env.reset()
        obs = env.observation_space
        grids = [np.array(obs.frame[0], dtype=np.int64)]
        actions = []
        for _ in range(max_steps):
            legal = [a for a in obs.available_actions if a in ACTION_ENUM]
            if not legal:
                break
            raw_a = int(np.random.choice(legal))
            obs = env.step(ACTION_ENUM[raw_a])
            actions.append(action_to_idx[raw_a])
            grids.append(np.array(obs.frame[0], dtype=np.int64))
        if actions:
            trajectories.append((np.stack(grids), np.array(actions, dtype=np.int64)))
    return trajectories


def sample_hindsight_batch(trajectories: list, batch: int):
    idxs = np.random.randint(0, len(trajectories), size=batch)
    states, actions, goals = [], [], []
    for i in idxs:
        grids, acts = trajectories[i]
        t = np.random.randint(0, len(acts))          # a valid action index (0..T-1)
        g_t = np.random.randint(t + 1, len(grids))    # a future frame index (t+1..T)
        states.append(grids[t])
        actions.append(acts[t])
        goals.append(grids[g_t])
    return np.stack(states), np.array(actions), np.stack(goals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="ls20")
    ap.add_argument("--n-rollouts", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    # actions actually seen (fixed per-game action set assumed constant across the
    # short rollout window -- true for ls20; a production version would need to
    # handle a per-frame available_actions mask explicitly)
    action_ids = sorted(ACTION_ENUM.keys())
    num_actions = len(action_ids)
    action_to_idx = {a: i for i, a in enumerate(action_ids)}

    print(f"Collecting {args.n_rollouts} random rollouts on {args.game} "
          f"({args.max_steps} steps each)...", flush=True)
    arcade = arc_agi.Arcade()
    trajectories = collect_rollouts(arcade, args.game, args.n_rollouts, args.max_steps, action_to_idx)
    print(f"done, {sum(len(g) for g, _ in trajectories)} total frames across "
          f"{len(trajectories)} usable rollouts.", flush=True)

    encoder = GridEncoder(latent_dim=64).to(DEVICE)
    critic = CRLCritic(state_dim=64, action_dim=num_actions, goal_dim=64,
                        hidden_dim=128, num_blocks=2, embed_dim=32).to(DEVICE)
    policy = CRLPolicy(state_dim=64, goal_dim=64, num_actions=num_actions,
                        hidden_dim=128, num_blocks=2).to(DEVICE)
    params = list(encoder.parameters()) + list(critic.parameters())
    critic_opt = torch.optim.Adam(params, lr=args.lr)
    policy_opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    def eval_policy_vs_random(n: int = 200):
        """Does the TRAINED policy's chosen action move the (s,a) embedding
        closer to a genuinely-reached held-out goal than a RANDOM action
        would, in the critic's own embedding space? One-step-lookahead proxy
        (not a real multi-step rollout, to keep eval fast) -- the real
        metric of interest, analogous to crl_sanity_check.py's L1-distance
        eval but in learned-embedding space since raw grids have no natural
        distance metric the way (row, col) coordinates do."""
        with torch.no_grad():
            s_np, _, g_np = sample_hindsight_batch(trajectories, n)
            s = torch.tensor(s_np, device=DEVICE)
            g = torch.tensor(g_np, device=DEVICE)
            s_lat, g_lat = encoder(s), encoder(g)
            action_idx = policy.logits(s_lat, g_lat).argmax(-1)
            action_onehot = F.one_hot(action_idx, num_actions).float()
            sa_embed = critic.embed_sa(s_lat, action_onehot)
            g_embed = critic.embed_g(g_lat)
            trained_dist = (sa_embed - g_embed).pow(2).sum(-1).sqrt().mean().item()
            rand_onehot = F.one_hot(torch.randint(0, num_actions, (n,), device=DEVICE), num_actions).float()
            sa_embed_rand = critic.embed_sa(s_lat, rand_onehot)
            random_dist = (sa_embed_rand - g_embed).pow(2).sum(-1).sqrt().mean().item()
        return trained_dist, random_dist

    print("Training goal-conditioned CRL on real ARC grids...", flush=True)
    for epoch in range(1, args.epochs + 1):
        s_np, a_np, g_np = sample_hindsight_batch(trajectories, args.batch_size)
        state_grid = torch.tensor(s_np, device=DEVICE)
        goal_grid = torch.tensor(g_np, device=DEVICE)
        action_idx = torch.tensor(a_np, device=DEVICE)
        action_onehot = F.one_hot(action_idx, num_actions).float()

        critic_opt.zero_grad()
        state_lat = encoder(state_grid)
        goal_lat = encoder(goal_grid)
        sa_embed = critic.embed_sa(state_lat, action_onehot)
        g_embed = critic.embed_g(goal_lat)
        logits = critic.logits(sa_embed, g_embed)
        labels = torch.arange(logits.shape[0], device=DEVICE)
        c_loss = F.cross_entropy(logits, labels)
        c_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        critic_opt.step()

        for p in params:
            p.requires_grad_(False)
        policy_opt.zero_grad()
        with torch.no_grad():
            state_lat_d = encoder(state_grid)
            goal_lat_d = encoder(goal_grid)
        p_loss = policy_loss_soft_action(policy, critic, state_lat_d, goal_lat_d)
        p_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        policy_opt.step()
        for p in params:
            p.requires_grad_(True)

        if epoch % 20 == 0 or epoch == 1:
            trained_dist, random_dist = eval_policy_vs_random()
            print(f"epoch {epoch:4d}  critic_loss={c_loss.item():.3f}  policy_loss={p_loss.item():.3f}  "
                  f"trained_action_dist={trained_dist:.3f}  random_action_dist={random_dist:.3f}", flush=True)

    trained_dist, random_dist = eval_policy_vs_random(n=1000)
    print(f"\nFINAL: trained_action_dist={trained_dist:.3f}  random_action_dist={random_dist:.3f}")
    if trained_dist < random_dist * 0.8:
        print("PASS: the trained policy's action moves consistently closer to the "
              "goal (in embedding space) than a random action -- signal transfers "
              "to real ARC grids.")
    else:
        print("FAIL (or not clearly better) -- mechanism didn't clearly transfer "
              "to real ARC grids with this encoder/budget.")


if __name__ == "__main__":
    main()
