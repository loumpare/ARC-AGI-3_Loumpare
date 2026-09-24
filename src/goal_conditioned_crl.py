"""Goal-conditioned Contrastive RL (CRL, Eysenbach et al. 2022) with the
depth-scaling recipe from "1000 Layer Networks for Self-Supervised RL"
(Wang et al., NeurIPS 2025, arXiv:2503.14858): residual MLP blocks (Dense ->
LayerNorm -> Swish, x4 per block, skip connection around the whole block) +
depth as the primary scaling axis, instead of width.

Why this track at all (see DESIGN_LOG.md 2026-09-07/08 entries): the
project's OTHER track (GRPO/PPO-clip on a 1-layer "Brain" transformer,
src/stage2_grpo_vanilla.py) has been stuck on a hard exploration ceiling
since 2026-08-29, yet still holds the all-time-best REAL Kaggle score
(0.09) across both tracks. The 2503.14858 paper's own ablation (Appendix
A.2, Figure 13) shows depth ALONE does not help TD/actor-critic methods
like SAC/TD3+HER (flat or negative past 4 layers, same architecture) --
only a genuinely contrastive/classification-style objective benefits from
depth. So this is deliberately a NEW objective (goal-conditioned InfoNCE),
not just a deeper version of the existing PPO-clip GRPO.

Adaptation for discrete actions (the paper uses continuous robotics
control via Brax/MJX): the paper's policy loss reparameterizes a continuous
action and differentiates the critic through it directly. For a discrete
action space, this file uses the analogous "soft" trick -- feed the
policy's full categorical probability vector (not a hard/sampled one-hot)
into the critic's phi(s,a) encoder, so the critic score is directly
differentiable w.r.t. the policy's logits, no REINFORCE/advantage
estimator needed.

Goals are entirely self-supervised via hindsight relabeling (Andrychowicz
et al. 2017 HER, the same trick the CRL paper's own reward function
`r_g(s,a) = (1-gamma) p(s'=g|s,a)` is built on): a state the agent actually
reached later in an episode is treated as the goal it was "trying" to
reach at an earlier step, with zero hand-crafted reward function -- see
feedback_no_game_hacking, this never references a specific game_id or
mechanic.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """One residual block: `block_size` repeated (Dense, LayerNorm, Swish)
    units, with a skip connection around the whole block -- see Figure 2 of
    2503.14858. Total network depth = block_size * num_blocks."""

    def __init__(self, dim: int, block_size: int = 4):
        super().__init__()
        layers = []
        for _ in range(block_size):
            layers += [nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MLPResNet(nn.Module):
    """Generic depth-scalable encoder: an input projection, `num_blocks`
    ResidualBlocks (each `block_size` layers deep), and an output
    projection. This is the single building block reused for the critic's
    phi/psi encoders and for the policy network -- depth is controlled by
    `num_blocks`, independent of `hidden_dim` (width)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_blocks: int, block_size: int = 4):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, block_size) for _ in range(num_blocks)])
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.out_proj(x)

    @property
    def depth(self) -> int:
        return len(self.blocks) * self.blocks[0].net.__len__() // 3 if self.blocks else 0


class CRLCritic(nn.Module):
    """f(s, a, g) = -||phi(s, a) - psi(g)||_2 (Eysenbach et al. 2022) --
    trained via InfoNCE to classify which goal a (state, action) pair's
    trajectory actually reached, among a batch of candidates."""

    def __init__(self, state_dim: int, action_dim: int, goal_dim: int,
                 hidden_dim: int = 256, num_blocks: int = 2, embed_dim: int = 64,
                 block_size: int = 4):
        super().__init__()
        self.phi = MLPResNet(state_dim + action_dim, hidden_dim, embed_dim, num_blocks, block_size)
        self.psi = MLPResNet(goal_dim, hidden_dim, embed_dim, num_blocks, block_size)

    def embed_sa(self, state: torch.Tensor, action_probs: torch.Tensor) -> torch.Tensor:
        return self.phi(torch.cat([state, action_probs], dim=-1))

    def embed_g(self, goal: torch.Tensor) -> torch.Tensor:
        return self.psi(goal)

    @staticmethod
    def logits(sa_embed: torch.Tensor, g_embed: torch.Tensor) -> torch.Tensor:
        """[B, B] matrix of -L2 distance between every (s,a) embed and every
        goal embed in the batch -- diagonal is the true (positive) pair,
        off-diagonal are in-batch negatives (InfoNCE)."""
        return -torch.cdist(sa_embed, g_embed, p=2)


class CRLPolicy(nn.Module):
    """Categorical policy pi(a | s, g) over a discrete action space."""

    def __init__(self, state_dim: int, goal_dim: int, num_actions: int,
                 hidden_dim: int = 256, num_blocks: int = 2, block_size: int = 4):
        super().__init__()
        self.net = MLPResNet(state_dim + goal_dim, hidden_dim, num_actions, num_blocks, block_size)

    def logits(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, goal], dim=-1))

    def sample(self, state: torch.Tensor, goal: torch.Tensor):
        logits = self.logits(state, goal)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)


def info_nce_critic_loss(critic: CRLCritic, state: torch.Tensor, action_onehot: torch.Tensor,
                          goal: torch.Tensor) -> torch.Tensor:
    """Standard batch InfoNCE: row i's (s,a) should match row i's goal (the
    goal hindsight-relabeled FROM that same trajectory), not any other row's
    goal in the batch."""
    sa_embed = critic.embed_sa(state, action_onehot)
    g_embed = critic.embed_g(goal)
    logits = critic.logits(sa_embed, g_embed)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


def policy_loss_soft_action(policy: CRLPolicy, critic: CRLCritic, state: torch.Tensor,
                             goal: torch.Tensor) -> torch.Tensor:
    """Discrete-action adaptation of the paper's `max E[f(phi(s,a),psi(g))]`
    policy update: feed the policy's full softmax probability vector (not a
    hard sample) into the critic's phi(s,a) -- directly differentiable, no
    REINFORCE/advantage estimator needed. The critic's own parameters are
    frozen for this loss (only the policy should move to satisfy the
    critic, not the other way around)."""
    action_probs = F.softmax(policy.logits(state, goal), dim=-1)
    with torch.no_grad():
        g_embed = critic.embed_g(goal)
    sa_embed = critic.embed_sa(state, action_probs)
    return (sa_embed - g_embed).pow(2).sum(-1).mean()
