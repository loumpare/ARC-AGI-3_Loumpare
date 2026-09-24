"""Fast, decisive sanity check for goal_conditioned_crl.py's mechanism,
BEFORE touching the real (slow, complex) ARC-AGI-3 environment -- same
practice as this project's own smoke_test_vanilla_grpo.py precedent
(DESIGN_LOG 2026-08-21).

Minimal synthetic gridworld: an NxN grid, agent at (row, col), 4 actions
(up/down/left/right, clipped at the border), no walls, no reward function
at all. Trains purely via hindsight goal-relabeling (a state the agent
actually visited later in the SAME random rollout becomes the "goal" an
earlier step is credited with reaching) -- if goal-conditioned CRL is
implemented correctly, average final distance-to-a-COMMANDED-goal should
drop well below a random policy's over training.

Usage: python3 src/crl_sanity_check.py
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from goal_conditioned_crl import CRLCritic, CRLPolicy, info_nce_critic_loss, policy_loss_soft_action

GRID_N = 8
NUM_ACTIONS = 4  # up, down, left, right
EPISODE_LEN = 20
BATCH_TRAJECTORIES = 256
EPOCHS = 300
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DELTAS = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])  # up, down, left, right


def state_feat(pos: np.ndarray) -> torch.Tensor:
    """Normalized (row, col) in [-1, 1] -- kept deliberately simple/low-dim
    since this check is only validating the CRL mechanism, not perception."""
    return torch.tensor((pos / (GRID_N - 1)) * 2 - 1, dtype=torch.float32, device=DEVICE)


def sample_hindsight_batch(batch: int):
    """For each of `batch` fresh random rollouts, pick a random step t and a
    random FUTURE step t' > t within the SAME trajectory -- (state@t,
    action@t, state@t' as goal) is a valid self-supervised training triple,
    zero reward function needed (see module docstring)."""
    pos = np.random.randint(0, GRID_N, size=(batch, 2))
    traj = np.zeros((batch, EPISODE_LEN + 1, 2), dtype=np.int64)
    traj[:, 0] = pos
    actions = np.zeros((batch, EPISODE_LEN), dtype=np.int64)
    for t in range(EPISODE_LEN):
        a = np.random.randint(0, NUM_ACTIONS, size=batch)
        actions[:, t] = a
        pos = np.clip(pos + DELTAS[a], 0, GRID_N - 1)
        traj[:, t + 1] = pos

    t_idx = np.random.randint(0, EPISODE_LEN, size=batch)
    future_offset = (np.random.rand(batch) * (EPISODE_LEN - t_idx)).astype(np.int64) + 1
    g_idx = np.minimum(t_idx + future_offset, EPISODE_LEN)

    s = traj[np.arange(batch), t_idx]
    a = actions[np.arange(batch), t_idx]
    g = traj[np.arange(batch), g_idx]
    return state_feat(s), torch.tensor(a, device=DEVICE), state_feat(g)


@torch.no_grad()
def eval_goal_reaching(policy: CRLPolicy, n_episodes: int = 200) -> float:
    """Runs the CURRENT greedy policy from a random start toward a random
    COMMANDED goal for EPISODE_LEN steps, returns mean final L1 distance
    (grid cells) to that goal -- the actual metric of interest, not a proxy
    loss."""
    start = np.random.randint(0, GRID_N, size=(n_episodes, 2))
    goal = np.random.randint(0, GRID_N, size=(n_episodes, 2))
    pos = start.copy()
    goal_feat = state_feat(goal)
    for _ in range(EPISODE_LEN):
        logits = policy.logits(state_feat(pos), goal_feat)
        action = logits.argmax(dim=-1).cpu().numpy()
        pos = np.clip(pos + DELTAS[action], 0, GRID_N - 1)
    return float(np.abs(pos - goal).sum(axis=1).mean())


def main() -> None:
    critic = CRLCritic(state_dim=2, action_dim=NUM_ACTIONS, goal_dim=2,
                        hidden_dim=128, num_blocks=2, embed_dim=32).to(DEVICE)
    policy = CRLPolicy(state_dim=2, goal_dim=2, num_actions=NUM_ACTIONS,
                        hidden_dim=128, num_blocks=2).to(DEVICE)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=LR)
    policy_opt = torch.optim.Adam(policy.parameters(), lr=LR)

    random_baseline = eval_goal_reaching(policy)  # policy is still random-init here
    print(f"random-init policy baseline: mean final L1 distance = {random_baseline:.2f} "
          f"(grid is {GRID_N}x{GRID_N}, max possible ~{2 * (GRID_N - 1)})")

    for epoch in range(1, EPOCHS + 1):
        state, action, goal = sample_hindsight_batch(BATCH_TRAJECTORIES)
        action_onehot = F.one_hot(action, NUM_ACTIONS).float()

        critic_opt.zero_grad()
        c_loss = info_nce_critic_loss(critic, state, action_onehot, goal)
        c_loss.backward()
        critic_opt.step()

        for p in critic.parameters():
            p.requires_grad_(False)
        policy_opt.zero_grad()
        p_loss = policy_loss_soft_action(policy, critic, state, goal)
        p_loss.backward()
        policy_opt.step()
        for p in critic.parameters():
            p.requires_grad_(True)

        if epoch % 20 == 0 or epoch == 1:
            dist = eval_goal_reaching(policy)
            print(f"epoch {epoch:4d}  critic_loss={c_loss.item():.3f}  "
                  f"policy_loss={p_loss.item():.3f}  eval_mean_L1_dist={dist:.2f}")

    final = eval_goal_reaching(policy, n_episodes=1000)
    print(f"\nFINAL: mean final L1 distance to commanded goal = {final:.3f} "
          f"(random baseline was {random_baseline:.2f})")
    if final < random_baseline * 0.5:
        print("PASS: policy reaches commanded goals much better than random -- "
              "the CRL mechanism is working correctly.")
    else:
        print("FAIL (or not clearly better) -- do not trust this implementation "
              "on the real ARC environment yet.")


if __name__ == "__main__":
    main()
