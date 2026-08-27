# TODO

Pre-planned task list. Check items off (`- [x]`) as they're done — click the
checkbox directly in VS Code's Markdown Preview (`Ctrl+Shift+V`, pin the tab
alongside the editor). See `ARCHITECTURE.md` for design rationale and
`DESIGN_LOG.md` for the running discussion of open architecture questions.

## 0. Submission plumbing (open items from GUIDELINE.md)
- [x] Generate `kernel-metadata.json` for `notebooks/` (`kaggle kernels init -p notebooks/`) — filled in: `code_file=starter.ipynb`, `kernel_type=notebook`, `enable_internet=false` (matches offline-only choice + no-internet-at-eval rule), `competition_sources=["arc-prize-2026-arc-agi-3"]` (attaches the competition dataset on Kaggle). `kaggle kernels push -p notebooks/` will now work when you're ready to actually publish — not run yet, that's a visible action on your Kaggle account, do it explicitly when there's something worth submitting.
- [x] Review `data/ARC-AGI-3-Agents/agents/README.md` and `agents/templates/README.md` — both are stub files pointing to `three.arcprize.org/docs`, no real content. Reviewed the actual template code instead: **6 of 8 templates (`llm_agents.py`, `multimodal.py`, `reasoning_agent.py`, `smolagents.py`, `langgraph_thinking/`, `langgraph_functional_agent.py`) import `openai`/hit a live LLM API — unusable as a submission basis given no-internet-at-eval.** Only `random_agent.py` / `langgraph_random_agent.py` are pure offline. `random_agent.py` is a useful reference for the `Agent` base-class contract (`choose_action`, `is_done`, RESET-on-`GAME_OVER`, `is_simple()`/`is_complex()`, `set_data({"x","y"})` for ACTION6, `.reasoning` field) to implement the baseline policy against (item 1 below).

## 1. Baseline — validate the loop end-to-end
- [x] Small transformer policy trained directly on raw grids (no JEPA yet), via PPO or plain GRPO — `notebooks/baseline.ipynb`: RoPE ViT-style encoder + GRPO (group-relative advantage, no clip/KL yet), trained on `ls20` with ls20-specific reward shaping (diagnostic only, see cell `7aacda0b`). 500-step run: mean group reward `-0.31 → ~0.11`, converges without collapsing (note: a separate 500-step run collapsed into a degenerate fixed-action policy around step 225 — entropy bonus not yet added, see caveat below).
- [x] Validate action loop + reward signal end-to-end against real env steps, **offline mode** — `notebooks/vision_encoder.ipynb` now sets `OPERATION_MODE=offline` / `ENVIRONMENTS_DIR=../data/environment_files` before creating `Arcade()`; confirmed no network calls (log shows local load, not "Got anonymous API key"), and ~14x faster than online (20 epochs in ~20s vs. minutes).
- [x] Confirm `available_actions` masking works correctly per-step — `select_action` masks unavailable action logits with `-inf` before sampling, exercised every step across all training runs with no invalid-action errors.

## 2. Vision encoder ("eyes")
- [x] Grid tokenizer — built differently than originally spec'd: 4x4 Conv2d patches (not 8x8 + explicit color-id embedding), 2D axial RoPE inside attention instead of a learned additive positional embedding. See `vision_encoder.ipynb` `patch_embedding`/`RoPEAttention`.
- [x] From-scratch encoder, trained from scratch, no pretrained backbone — much smaller than the original "~10-20M params" estimate (`eyes` is 3,824 params, `latent_dim=16`); kept intentionally tiny for fast iteration, revisit sizing once the pipeline is trusted end-to-end.

## 3. Stage 1 — JEPA world model pretraining ("intuition")
- [x] Log `(frame_t, action_t, frame_{t+1})` transitions with a random policy — `notebooks/jepa_pretrain.ipynb`. Only 19/25 games usable (6 games never expose the 6 supported SimpleActions, likely ACTION6/click-only — see DESIGN_LOG 2026-08-16/17).
- [x] Train encoder + EMA target encoder + action-conditioned predictor (stop-grad MSE) — `WorldModel` class, same `Eyes`/`Intuition` mechanism as the joint pipeline, trained alone this time (no policy gradient touching `eyes`).
- [ ] Validate k-step latent prediction error on held-out trajectories before trusting it for planning — in progress: held-out error is low for most games but persistently high for `wa30` (~13x train error) across every run so far; a `UserWarning` about a tensor shape mismatch also appeared once in the latest 15000-step run, not yet confirmed as real vs. a stale artifact from a long-lived Jupyter kernel. Next: restart kernel, rerun clean, break down loss per-game before trusting a checkpoint (see DESIGN_LOG 2026-08-16/17 for full detail).

## 4. Stage 2 — GRPO policy training
- [ ] Policy/value head: action-type logits (masked by `available_actions`) + pointer head over 64 tokens for ACTION6 (x,y)
- [ ] Group rollouts from a fixed starting state, group-relative advantage — 2026-08-20 added encoder isolation, Go-Explore archive, zero-variance resampling, PPO clip/KL; a real run (2026-08-21) showed the clip/KL + joint JEPA fine-tuning combination had a real bug (shared `clip_grad_norm_` over eyes+intuition+brain let a saturated JEPA gradient crush the policy gradient to near-zero, see DESIGN_LOG 2026-08-21). **Restarted Stage 2 from scratch same day**: minimal vanilla GRPO, `eyes` frozen (no JEPA loss, no curiosity, no PPO clip/KL, no archive, no replay) — validated via mock smoke test, not yet run against the real env
- [x] Real-env rollouts only first (no latent imagination yet)

## 5. Stretch — latent imagination
- [ ] Once world-model accuracy is validated, generate most GRPO rollout groups via latent imagination (predictor), spend real env steps only on top-k candidates

## 6. Open architecture question — frozen-LLM "brain" (see DESIGN_LOG.md, 2026-08-14)
- [ ] Ablation: frozen pretrained LLM (+ vision adapter) vs. from-scratch reasoning transformer of matched adapter-param count, on the same logged trajectories — decide before committing to the frozen-LLM idea
- [ ] If pursuing it: pin down a concrete candidate (param count + OSI-approved license, or accept the incompatible-license carve-out for that component)
- [ ] If pursuing it: use the JEPA world model to verify/score the LLM's proposed action(s) before spending a real env step, rather than executing long open-loop action chunks blindly
