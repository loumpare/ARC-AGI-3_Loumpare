"""GRPO fine-tuning for ToolsAgent's "brain" (goal-selection LLM). Real
implementation as of 2026-09-03 (user gave explicit go-ahead to launch a real
run, scoped to ls20 only for now).

Group = `group_size` independent ls20 episodes played with the CURRENT
policy. Reward per episode = sum of `compute_reward` (copied verbatim from
notebooks/vision_encoder.ipynb cell 36ddc704 -- levels_completed delta +
WIN/GAME_OVER +/- step penalty, nothing else, see feedback_no_game_hacking)
over every frame transition. Every goal-selection decision made during an
episode is credited with that episode's TOTAL reward (same trajectory-level
credit assignment stage2_grpo_vanilla.py already uses -- see its
`flatten_steps_with_advantage`/`compute_losses`, group-normalized advantage +
PPO-clip surrogate, copied here verbatim from that proven pattern).

Perception/pathfinding is untouched: every rollout reuses ToolsAgent's real
self-tracking/wall-learning/BFS code from src/llm_tools_agent.py exactly as
in production. Only the goal-selection decision is answered by the model
being trained instead of a static Qwen2.5-7B prompt call.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "data" / "ARC-AGI-3-Agents"))

os.environ.setdefault("OPERATION_MODE", "offline")
os.environ.setdefault("ENVIRONMENTS_DIR", str(REPO / "data" / "environment_files"))

BASE_MODEL_PATH = str(REPO / "checkpoints" / "base_models" / "Qwen2.5-0.5B-Instruct")
CHECKPOINT_DIR = REPO / "checkpoints" / "brain_grpo_ls20"

SYSTEM_PROMPT = (
    "You are the decision module for a game-playing agent. A perception layer has "
    "already computed, with certainty, your own position and a list of distinct "
    "landmarks on the grid, plus any discovered cause-and-effect. Prefer unvisited "
    "landmarks unless effects suggest otherwise. Respond in exactly two parts: "
    "1) a short reasoning (1 sentence), 2) on the LAST line, exactly one landmark id "
    "(e.g. blob_2) and nothing else."
)


def compute_reward(prev_frame, frame) -> float:
    """Verbatim from notebooks/vision_encoder.ipynb cell 36ddc704 -- see
    feedback_no_game_hacking. Do not add any other term to this function."""
    from arcengine import GameState
    reward = float(frame.levels_completed - prev_frame.levels_completed)
    if frame.state == GameState.WIN:
        reward += 1.0
    elif frame.state == GameState.GAME_OVER:
        reward -= 1.0
    reward -= 0.01
    return reward


class LLMBrainAgent:
    """Not an Agent subclass -- a thin driver around llm_tools_agent's pure
    functions (blob/self tracking, BFS) that asks the passed-in model+tokenizer
    for every goal-selection decision (no MAX_BRAIN_CALLS cap: the whole point
    is this model is cheap enough to call every time), and records
    (prompt, completion_text, completion_token_ids) for each decision."""

    MAX_ACTIONS = 150

    def __init__(self, game_id: str, env, model, tokenizer, device, sample: bool = True):
        import llm_tools_agent as lta
        self._lta = lta
        self.game_id = game_id
        self.env = env
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.sample = sample

        self.self_colors: set[int] = set()
        self.attached_colors: set[int] = set()
        self.self_bbox = None
        self.prev_touched_keys: set = set()
        self.action_deltas: dict = {}
        self.pending_deltas: dict = {}
        self.tried_actions: set[str] = set()
        self.blocked_values: set[int] = set()
        self.effects_log: list = []
        self.visited_blob_keys: set = set()
        self.current_path: list = []
        self.current_goal_key = None
        self.goal_fail_count = 0
        self.prev_grid = None
        self.prev_action_name = None
        self.prev_self_pos = None
        self.prev_levels_completed = None
        self.action_counter = 0

        self.decisions: list[dict] = []  # collected (prompt, completion_ids, ...) per decision
        self.frames: list = []

    def _ask_model(self, prompt: str) -> tuple[str, "torch.Tensor", "torch.Tensor"]:
        import torch
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        enc = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        ).to(self.device)
        prompt_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=40,
                do_sample=self.sample, temperature=0.8 if self.sample else None,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        completion_ids = out[0][prompt_len:]
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        return completion_text, enc["input_ids"][0], completion_ids

    def run_episode(self) -> float:
        lta = self._lta
        raw = self.env.reset()
        latest = self._to_frame(raw)
        self.frames = [latest]
        total_reward = 0.0

        while latest.state.name not in ("WIN",) and self.action_counter <= self.MAX_ACTIONS:
            action = self._choose_action(latest)
            raw = self.env.step(action)
            new_frame = self._to_frame(raw)
            total_reward += compute_reward(latest, new_frame)
            self.frames.append(new_frame)
            latest = new_frame
            self.action_counter += 1

        return total_reward

    def _to_frame(self, raw):
        from agents.agent import Agent
        # reuse the exact conversion logic the real framework uses
        return Agent._convert_raw_frame_data(self, raw)  # unbound call, no Agent instance needed

    def _choose_action(self, latest_frame):
        from arcengine import GameAction, GameState
        lta = self._lta

        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.prev_grid = None
            self.prev_action_name = None
            self.self_bbox = None
            self.current_path = []
            return GameAction.RESET

        legal = [a for a in latest_frame.available_actions if a in lta.ACTION_NAMES]
        if not legal:
            return GameAction.RESET
        legal_names = [lta.ACTION_NAMES[a] for a in legal]

        grid = lta._grid_of(latest_frame)
        level_changed = (self.prev_levels_completed is not None and
                          latest_frame.levels_completed != self.prev_levels_completed)
        if level_changed:
            self.prev_grid = None
        self.prev_levels_completed = latest_frame.levels_completed

        # reuse ToolsAgent's real update logic by way of a throwaway bound-method call
        lta.ToolsAgent._update_from_last_transition(self, grid)

        bulk = lta._bulk_colors(grid)
        blobs = lta._find_blobs(grid, bulk | self.self_colors | self.attached_colors)

        if not self.self_colors:
            untried = [a for a in legal_names if a not in self.tried_actions]
            chosen_name = untried[0] if untried else legal_names[self.action_counter % len(legal_names)]
            self.tried_actions.add(chosen_name)
            self.prev_grid = grid
            self.prev_action_name = chosen_name
            return lta.ACTION_NAME_TO_ENUM[chosen_name]

        self_pos = lta._bbox_centroid(self.self_bbox) if self.self_bbox else self.prev_self_pos
        self.prev_self_pos = self_pos
        self_pos_int = (int(round(self_pos[0])), int(round(self_pos[1]))) if self_pos else (0, 0)

        if not self.current_path:
            if blobs:
                id_map = {}
                blob_lines = []
                for i, b in enumerate(blobs):
                    bid = f"blob_{i}"
                    id_map[bid] = b
                    visited = "visited" if (b["color"], b["bbox"][0], b["bbox"][2]) in self.visited_blob_keys else "unvisited"
                    blob_lines.append(f"{bid}: color={b['color']} pos={b['centroid']} size={b['size']} ({visited})")
                effects_lines = [
                    f"- touching color {e['touched_color']} at {e['touched_pos']} changed region "
                    f"rows{e['effect_region'][0]}-{e['effect_region'][1]} cols{e['effect_region'][2]}-{e['effect_region'][3]}"
                    for e in self.effects_log if "effect_region" in e
                ] or ["(none discovered yet)"]
                prompt = (
                    f"Your position: {self_pos}\n\nLandmarks visible now:\n" + "\n".join(blob_lines) +
                    "\n\nEffects discovered so far:\n" + "\n".join(effects_lines) +
                    "\n\nWhich landmark should you move toward next?"
                )
                completion_text, prompt_ids, completion_ids = self._ask_model(prompt)
                self.decisions.append(dict(prompt_ids=prompt_ids, completion_ids=completion_ids))

                chosen_id = None
                for line in reversed(completion_text.strip().splitlines()):
                    line = line.strip()
                    if line in id_map:
                        chosen_id = line
                        break
                target = id_map.get(chosen_id)
                if target is None:
                    unvisited = [b for b in blobs
                                 if (b["color"], b["bbox"][0], b["bbox"][2]) not in self.visited_blob_keys]
                    target = lta._closest_blob(unvisited, self_pos) or lta._closest_blob(blobs, self_pos)
                self.visited_blob_keys.add((target["color"], target["bbox"][0], target["bbox"][2]))
                path = lta._bfs_path(grid, self_pos_int, target["bbox"], self.blocked_values, self.action_deltas)
                self.current_path = path or []
            if not self.current_path:
                untried = [a for a in legal_names if a not in self.tried_actions]
                chosen_name = untried[0] if untried else legal_names[self.action_counter % len(legal_names)]
                self.tried_actions.add(chosen_name)
                self.prev_grid = grid
                self.prev_action_name = chosen_name
                return lta.ACTION_NAME_TO_ENUM[chosen_name]

        action = self.current_path.pop(0)
        self.prev_grid = grid
        self.prev_action_name = action.name
        return action


def sequence_logprob(model, prompt_ids, completion_ids, device, return_entropy=False):
    """Sum of log P(token | context) over just the completion tokens. If
    `return_entropy`, also returns the mean per-token entropy of the policy's
    distribution over those tokens -- used as a regularization bonus (see
    `train`) after the first real run visibly collapsed into a narrow,
    repetitive behavior (staying near the start, never reaching the door)
    within 8 steps, with nothing to resist that."""
    import torch
    full = torch.cat([prompt_ids, completion_ids]).unsqueeze(0).to(device)
    with torch.set_grad_enabled(model.training):
        logits = model(full).logits[:, :-1, :]
    target = full[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    completion_start = prompt_ids.shape[0] - 1  # -1: logits[i] predicts token[i+1]
    seq_logprob = token_log_probs[0, completion_start:].sum()
    if not return_entropy:
        return seq_logprob
    probs = log_probs.exp()
    token_entropy = -(probs * log_probs).sum(-1)
    mean_entropy = token_entropy[0, completion_start:].mean()
    return seq_logprob, mean_entropy


def reference_logprob(model, prompt_ids, completion_ids, device):
    """Log-prob under the FROZEN base model (LoRA adapters disabled via peft's
    disable_adapter(), no second model copy needed) -- the reference policy
    for a KL penalty. Standard RLHF/PPO safeguard, omitted from the first real
    run: keeps the fine-tuned policy from drifting into a degenerate pattern
    while advantage estimates are still noisy (small group_size, few steps)."""
    import torch
    with model.disable_adapter():
        with torch.no_grad():
            return sequence_logprob(model, prompt_ids, completion_ids, device)


def train(group_size: int = 4, n_steps: int = 12, clip_eps: float = 0.2, lr: float = 1e-5,
          max_actions: int = 80, entropy_coef: float = 0.01, kl_coef: float = 0.05):
    import torch
    from train_brain_grpo import BASE_MODEL_PATH  # self-import ok, avoids re-deriving path

    from arc_agi import Arcade
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, dtype=torch.bfloat16, device_map=device)
    lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    arc = Arcade()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = CHECKPOINT_DIR / "train_log.jsonl"

    for step in range(n_steps):
        t0 = time.time()
        model.eval()
        episode_rewards = []
        episode_decisions = []  # list of list-of-decision-dicts, one per episode
        levels_this_step = []

        for _ in range(group_size):
            env = arc.make("ls20")
            driver = LLMBrainAgent("ls20", env, model, tokenizer, device, sample=True)
            driver.MAX_ACTIONS = max_actions
            reward = driver.run_episode()
            episode_rewards.append(reward)
            episode_decisions.append(driver.decisions)
            levels_this_step.append(driver.frames[-1].levels_completed)

        rewards_t = torch.tensor(episode_rewards, dtype=torch.float32)
        advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)

        # old_log_prob: computed with the SAME (not-yet-updated-this-step) weights,
        # right after collection -- valid "old" policy by construction since no
        # optimizer.step() has happened yet this iteration.
        model.eval()
        flat = []
        for decisions, adv in zip(episode_decisions, advantages):
            for d in decisions:
                with torch.no_grad():
                    old_lp = sequence_logprob(model, d["prompt_ids"], d["completion_ids"], device)
                flat.append((d["prompt_ids"], d["completion_ids"], old_lp, adv))

        if not flat:
            print(f"[step {step}] no decisions collected this step (self never bootstrapped?), skipping update")
            continue

        # gradient accumulation, one decision at a time -- NOT `torch.stack([... for ... in flat])`
        # followed by one loss.backward(): that keeps every decision's forward-pass
        # computation graph alive simultaneously until the single backward call, and
        # with ~50-250 decisions/step (no MAX_BRAIN_CALLS cap here, deliberately, see
        # module docstring) that OOM'd a 3090 on the first real run today. Processing
        # one decision's forward+backward at a time (loss scaled by 1/n so the summed
        # gradients still average correctly) bounds memory to ~1 decision's graph
        # regardless of how many decisions a step collects.
        model.train()
        optimizer.zero_grad()
        n = len(flat)
        total_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        for p_ids, c_ids, old_lp, adv in flat:
            new_lp, entropy = sequence_logprob(model, p_ids, c_ids, device, return_entropy=True)
            ref_lp = reference_logprob(model, p_ids, c_ids, device)
            adv_t = adv.to(device)
            ratio = torch.exp(new_lp - old_lp)
            clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
            surrogate = torch.min(ratio * adv_t, clipped_ratio * adv_t)
            # k3 KL estimator (Schulman, "Approximating KL Divergence") -- unbiased,
            # always >=0, lower variance than the naive (new_lp - ref_lp) difference.
            log_ratio_ref = ref_lp - new_lp
            kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1
            step_loss = (-surrogate - entropy_coef * entropy + kl_coef * kl) / n
            step_loss.backward()
            total_loss += float(step_loss.detach())
            total_entropy += float(entropy.detach()) / n
            total_kl += float(kl.detach()) / n
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        optimizer.step()
        loss = total_loss

        elapsed = round(time.time() - t0, 1)
        win_rate = sum(1 for lv in levels_this_step if lv >= 1) / group_size
        log_entry = dict(step=step, mean_reward=float(rewards_t.mean()), win_rate=win_rate,
                          levels=levels_this_step, loss=float(loss), entropy=total_entropy, kl=total_kl,
                          n_decisions=len(flat), elapsed=elapsed)
        print(f"[step {step}] mean_reward={log_entry['mean_reward']:.3f} win_rate={win_rate:.2f} "
              f"levels={levels_this_step} loss={log_entry['loss']:.4f} entropy={total_entropy:.3f} "
              f"kl={total_kl:.4f} n_decisions={len(flat)} elapsed={elapsed}s", flush=True)
        with open(log_path, "a") as f:
            import json
            f.write(json.dumps(log_entry) + "\n")

        if step % 5 == 0 or step == n_steps - 1:
            model.save_pretrained(str(CHECKPOINT_DIR / f"step_{step}"))

    model.save_pretrained(str(CHECKPOINT_DIR / "final"))
    tokenizer.save_pretrained(str(CHECKPOINT_DIR / "final"))
    print("training complete, final checkpoint saved to", CHECKPOINT_DIR / "final")


if __name__ == "__main__":
    train()
