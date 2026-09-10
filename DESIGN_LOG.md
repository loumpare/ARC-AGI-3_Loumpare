# Design log

Running, dated log of architecture ideas and how they evolve. Not the
settled design (that's `ARCHITECTURE.md`) — this is the discussion trail.
Append new dated entries; don't rewrite old ones.

## 2026-08-14 — Eyes / frozen brain / intuition proposal

**Proposal (user):**
- **Eyes**: a ViT vision encoder turns the raw frame into a latent.
- **Brain**: that latent (+ learned query tokens) is fed into a **frozen**
  small LLM, which predicts `k` action vectors for future timesteps `t..t+k`.
- **Intuition**: right after the LLM's output, a second ViT (I-JEPA style)
  learns to predict the next frame(s) latent — a world model sitting
  downstream of the LLM's action choices.

**Debate / open risks raised:**
1. **Frozen-LLM domain mismatch** — same reasoning that already killed a
   pretrained natural-image encoder (`ARCHITECTURE.md`'s DINOv3 rejection)
   plausibly applies to a language-pretrained frozen LLM repurposed for
   grid/action sequences. Frozen-transformer-as-generic-sequence-processor
   literature (Tsimpoukelli et al. "Frozen"; Lu et al. "Pretrained
   Transformers as Universal Computation Engines") shows real but modest
   transfer, task-dependent. Treat as a hypothesis to ablate against a
   from-scratch reasoning transformer of matched adapter-parameter count —
   not a foundation to commit to upfront.
2. **Competition constraints** — no internet at eval (LLM weights must be
   bundled offline); Section 2.5 "open source weights" requirement for prize
   winners means LLM license matters (e.g. Llama's license isn't
   OSI-approved — usable under the incompatible-license carve-out, but that
   component then can't be open-sourced). Also: per-step frozen-LLM latency
   vs. scorecard time windows (`MAX_OPEN_FOR_MINUTES`/`STALE_MINUTES`,
   already an open question in `ARCHITECTURE.md`) needs a concrete model-size
   number before it's a real plan.
3. **Open-loop k-action prediction is risky** — `available_actions` is a
   per-frame legality mask and a single `ACTION6` click can irreversibly
   change level state; committing to k actions before seeing intermediate
   frames risks queuing illegal/wasted actions. Suggested fix: keep k small,
   or use the JEPA world model to check the predicted next latent's
   legality/value *before* spending a real env step on each queued action.

**Strongest part, worth keeping regardless of the frozen-LLM question:**
using the downstream JEPA world model to cheaply score/verify the policy's
proposed actions before touching the real env is exactly the "latent
imagination" mechanism already sketched for GRPO stage 2 in
`ARCHITECTURE.md` — just fed by an LLM's proposals instead of a
from-scratch policy's. Reuse this regardless of which policy module wins.

**Suggested sequencing:** validate the closed-loop baseline (from-scratch
policy + real env steps) and the JEPA next-frame predictor on real data
first (already the build order in `ARCHITECTURE.md`). Only then swap the
policy head for a frozen-LLM + adapter and compare directly against that
baseline.

**Status:** undecided — no experiment run yet.

## 2026-08-16/17 — GRPO ablations on ls20 exhausted, pivot to JEPA-first pretraining

**What was tried, in order, on `notebooks/vision_encoder.ipynb` (joint GRPO+JEPA on `ls20` only):**
1. Baseline joint run, no entropy term → policy collapsed hard ~step 200-230, never recovered (no gradient pressure left to escape once near-deterministic).
2. Added entropy bonus → policy stayed spread out, but `grpo_loss` pinned at exactly `0.0000` for hundreds of updates: every episode in a GRPO group scored identically (ran out the step budget, no win/loss), so group-relative advantage was always 0. "Zero-variance trap."
3. Raised `group_size` 8→24, lowered `entropy_coef` 0.01→0.001 → `grpo_loss` stopped being zero but instead exploded (-64 → -1688 over 490 steps) while reward stayed flat. Cause: only step-0 action legality is masked in the open-loop `t_horizon`-action chunk; a narrowing policy assigns near-zero probability to a forced-legal fallback action on later steps → `log_prob → -inf`.
4. Added `clip_grad_norm_(max_norm=1.0)` (bounds update *norm*, not which loss term dominates *direction*) and a generic curiosity/intrinsic reward (`curiosity_coef × MSE(Intuition's prediction, actual next latent)` — no per-game hacking, uses only the model's own surprise).
5. Controlled 4-way ablation (`src/train.py`, same seed, 500 vs 50 epochs × curiosity on/off): **budget and curiosity made no measurable difference** — every config found the exact same single win in ~12,000 episodes and never repeated it (`1/500` epochs ever beat the always-fail baseline). Real bottleneck: GRPO never *retains* a rare success, not that it can't find one.
6. Implemented self-imitation replay (Oh et al. 2018 — buffer of best-ever chunks, replayed each update to imitate what already worked) + clamped `log_prob` at the source (fixed the risk-#3 explosion properly instead of just clipping the symptom). Tested `replay_coef` at 0.1 and 2.0, both against a no-replay control, same seed: **no measurable benefit either time** — likely because the buffer only ever contains one real success out of 500 epochs, so there's little coherent signal to imitate, and only a chunk's *starting* frame (the fixed reset state) is reliably revisited by a changing policy.

**Decision: stop tuning GRPO knobs on a single hard-exploration game.** Pivoted to `notebooks/jepa_pretrain.ipynb`: train **Eyes + Intuition only** (no Brain, no policy gradient, no reward) on `(frame_t, action_t, frame_{t+1})` transitions collected by a **random** policy across as many of the 25 offline games as possible. JEPA only needs state coverage, not successful episodes, so the exploration difficulty that sank GRPO doesn't apply to this stage. This also directly removes the eyes↔policy coupling risk that was never actually eliminated (only worked around) in the joint runs.

**Findings from `jepa_pretrain.ipynb` so far:**
- **6 of 25 games (`ft09`, `lp85`, `r11l`, `s5i5`, `tn36`, `vc33`) never expose any of the 6 supported SimpleActions** (ACTION1-5, ACTION7) — likely ACTION6/click-only games, unsupported by the current action space. 19/25 usable. Worth revisiting ACTION6 support later, since 24% of games are currently out of reach entirely, not just for this notebook.
- Held-out validation must be picked *after* filtering to usable games — picking from the raw sorted game list deterministically landed on `vc33` (unusable) every run, wasting half the validation set.
- First run (17k transitions, 2000 steps): train MSE 0.44, held-out `tu93` 0.37, held-out `wa30` 6.07 (13x worse).
- Bumped data 8x (136k transitions) without bumping `epochs` proportionally → train MSE got *worse* (0.59), as expected: each transition was seen <1x on average vs ~7.5x before. Not a sign of anything wrong, just an accounting mismatch (`epochs × batch_size ÷ num_transitions` = average times each example is seen — keep this in mind before comparing runs).
- Latest run: 136k transitions, 15000 steps (~42 min). Loss curve: drops fast to near-0 in the first ~500 steps, but noisy large spikes (up to ~6, matching the initial loss scale) recur throughout the *entire* run, not damping over time. `wa30`'s persistently high held-out error across all runs so far suggests this game's dynamics may be the recurring source of those spikes (worth a per-game loss breakdown rather than more epochs). A `UserWarning` about mismatched tensor shapes in `F.mse_loss` also appeared once during the 15000-step run; the shapes in the code are fixed by construction (`preds` and `targets` are always `(batch_size, 1, latent_dim)` given `batch_size=64` and `t_horizon=1` are constant) so this is most likely a stale artifact from the long-lived Jupyter kernel (reused across many parameter changes without a restart) rather than an active bug — **not yet confirmed**, restart the kernel and rerun cleanly to rule it out before trusting a checkpoint.

**Next steps, in order:**
1. Restart the Jupyter kernel, rerun `jepa_pretrain.ipynb` clean, confirm the `UserWarning` is gone (or track down a real cause if it recurs).
2. Break down `jepa_loss` per-game (not just aggregate train/held-out) to check whether the persistent spikes trace back to specific games (`wa30` suspected) — decide whether that's acceptable or needs addressing (e.g. per-game loss weighting, more capacity) before trusting the encoder.
3. Once satisfied: load `checkpoints/world_model_pretrained.pt` into `vision_encoder.ipynb`'s `ARC_AGI_3` — remember to copy the loaded `eyes` weights into `target_eyes` too (not just `eyes`), since `ARC_AGI_3.__init__` only syncs them at random-init time.
4. Resume Stage 2 GRPO (TODO item 4) on top of the pretrained representation, with a real trust region (clipped-ratio + KL) instead of the ad-hoc gradient clipping used throughout this session.

**Guardrails that held throughout (keep enforcing):** no per-game reward shaping (generic `levels_completed`/`WIN`/`GAME_OVER` reward only, curiosity uses only the model's own prediction error); no pretrained vision/language backbones; log non-obvious decisions here as they happen, not after the fact — this entry exists because token budget ran low mid-session and the next session (human or AI) needs to pick this up cheaply.

## 2026-08-18 — Stage 2 (`jepa_pretrain.ipynb`) run: architecture fix confirmed, retention problem is structural to GRPO, not representation quality

Loaded the Stage 1 checkpoint into `ARC_AGI_3`, trained `Brain` with GRPO+JEPA
on `ls20` (`t_horizon=1`, closed-loop -- one `Brain` call per real env step,
per the design note in `jepa_pretrain.ipynb`'s Stage 2 markdown cell).

**Confirmed: `t_horizon=1` genuinely fixes risk #3.** `grpo_loss` stayed
bounded in roughly [-3, +3.5] across 500 updates -- no repeat of the
`vision_encoder.ipynb` explosion (-64 → -1688). Masking every action's
legality before sampling (possible now that every step is closed-loop)
removes the illegal-action-fallback path entirely, not just its symptom.

**Not fixed: task reward stayed flat at the always-fail baseline (~-0.48)
for the entire 500-update run, with a single isolated success (reward 0.52,
same value as every previous ls20 run) around update 300 -- never retained.**
`jepa_loss` and policy entropy both show a transient blip at the same
update, then return to their prior trend, exactly the same "found once,
never kept" signature seen in the H1-H4 budget/curiosity ablation and both
self-imitation attempts on the un-pretrained encoder (see the entry above).

**This is the important part: pretraining Eyes+Intuition on 17 games first
did not change this outcome at all.** Same failure mode, same reward value,
whether the encoder started random or already understood a fair amount
about game dynamics. That's strong evidence the retention problem is
specific to GRPO's mechanism on this game (no memory of a rare success
across updates), not a symptom of poor representations -- pretraining was
worth doing for other reasons (stability, generalization, TODO item 3) but
was never going to fix this by itself.

**Why self-imitation deserves a second, fairer try:** the previous two
self-imitation attempts (`replay_coef` 0.1 and 2.0, `vision_encoder.ipynb`)
failed to show any effect, but `grpo_loss` there was still exploding
(-64 to -1688+) even with the log_prob clamp fix, so the replay term
(magnitude ~2-20) never had a chance to compete for gradient direction.
Now that `grpo_loss` is naturally bounded to a comparable scale (~±3) by the
`t_horizon=1` fix, a replay term of similar magnitude can actually compete
this time -- worth retrying before concluding self-imitation doesn't work
here at all.

**Next:** re-add the self-imitation buffer (same mechanism as before: top-K
best steps ever seen by extrinsic reward, replayed each update against the
*current* policy) to `jepa_pretrain.ipynb`'s Stage 2 loop, on top of the
pretrained + `t_horizon=1` setup.

## 2026-08-20 — Literature search on the open problems above

Ran a targeted lit search (7 questions, 3 parallel agents) against the
open problems accumulated in this log. Full references below; only citing
what's directly load-bearing for a decision.

**1. Shared-encoder interference (rising `jepa_loss` during GRPO fine-tuning,
2026-08-18 entry).** Confirmed as a known failure mode, not a bug: Stooke et
al. 2021 "Decoupling Representation Learning from RL"
([arXiv:2009.08319](https://arxiv.org/abs/2009.08319)) show RL gradients
degrade shared visual features unless isolated — their fix is a separate
optimizer for the SSL loss with detached features fed to the policy. Liu et
al. 2023 ([arXiv:2307.04887](https://arxiv.org/abs/2307.04887)) measure this
interference directly and tie it to instability. **Action: stop-grad `z`
into the policy head, or a much lower LR / periodic freeze on `Eyes`, during
the GRPO phase** — currently `Eyes` gets gradient from both losses with no
isolation.

**2. Curiosity/intrinsic reward, if revisited.** Raw JEPA prediction error
as reward is a known "noisy-TV" trap (Pathak et al. 2017). Two fixes with
evidence: RND's fixed random target (Burda et al. 2018,
[arXiv:1810.12894](https://arxiv.org/abs/1810.12894)) sidesteps it
structurally since the target can't degrade; or reward *learning progress*
(Δerror) instead of raw error level (2025,
[arXiv:2509.25438](https://arxiv.org/abs/2509.25438)). Not urgent — curiosity
is currently unused (H1-H4 ablation, 2026-08-16/17 entry, found no measurable
effect either way) — but if it comes back, don't reuse the raw-JEPA-error
form.

**3. Retention of rare sparse-reward successes (core unsolved problem, both
self-imitation attempts so far found no effect).** Go-Explore (Ecoffet et
al. 2019/2021, [arXiv:1901.10995](https://arxiv.org/abs/1901.10995)) names
the exact symptom — "detachment" (policy drifts away from a promising state
between updates) and "derailment" (exploration noise prevents reliably
returning to it) — and its fix is architectural: an explicit archive of
(state, trajectory) with deterministic **return-to-state** before exploring
further, not a loss-function term. This is a different category of fix than
either self-imitation attempt tried. Self-Imitation Learning (Oh et al.
2018, [arXiv:1806.05635](https://arxiv.org/abs/1806.05635)) also clips
advantage to positive-only transitions, which the current replay
implementation may not be doing.

**4. GRPO stability.** The zero-variance-group dead-gradient issue
(2026-08-16/17 entry, step 2) is a documented, inherent property of the
GRPO estimator (Shao et al. 2024,
[arXiv:2402.03300](https://arxiv.org/abs/2402.03300)), quantified at 28-45%
of batches on hard tasks by a 2026 follow-up
([arXiv:2605.21125](https://arxiv.org/abs/2605.21125)). DAPO (2025,
[arXiv:2503.14476](https://arxiv.org/abs/2503.14476)) filters zero-variance
groups before the update rather than let them contribute a dead gradient,
plus an asymmetric "Clip-Higher" bound borrowed from PPO. TODO item 4
("clipped-ratio + KL-penalty update") already plans the PPO-style clip;
group-filtering is the piece not yet planned.

**5. JEPA pretrain-then-freeze.** Real precedent for freezing after
pretraining (I-JEPA linear probing; MVP for robots, Radosavovic et al. 2022,
[arXiv:2210.03109](https://arxiv.org/abs/2210.03109), freezes the encoder
and trains only a small control head). But the strongest RL-specific
numbers (SPR, Schwarzer et al. 2021,
[arXiv:2007.05929](https://arxiv.org/abs/2007.05929); DreamerV3) come from
*continued* joint adaptation, not a frozen-at-init encoder. Current setup
(2026-08-18 entry) uses fully-frozen `eyes` for Stage 2 — worth an ablation
against low-LR continued fine-tuning once retention (item 3) is no longer
the confound.

**6. Action masking.** Huang & Ontañón
([arXiv:2006.14171](https://arxiv.org/abs/2006.14171)) confirms pre-softmax
logit masking (already the implementation, TODO item 1) is the
theoretically-justified approach and matters more as the invalid-action
space grows — relevant if/when ACTION6 (click, 64-token space) is added.

**7. 2D RoPE.** Heo & Park 2024
([arXiv:2403.13298](https://arxiv.org/abs/2403.13298)) directly confirms
axial 2D RoPE beats learned absolute PE for resolution-extrapolation on
ViTs, and that "RoPE-Mixed" (diagonal-aware) does better than pure axial
when diagonal spatial structure matters — plausible for ARC grids. No
change needed now; worth trying Mixed over pure-axial if the encoder is
revisited.

**Net read:** items 1, 2, and 5 all point the same direction — the two
observed pathologies (rising `jepa_loss`, and a hypothetical rising
curiosity reward) are both textbook symptoms of not isolating the encoder's
gradient from the policy gradient. Item 3 (retention) is the one true
unknown with no literature-confirmed fix yet tried here — Go-Explore's
archive-and-return is a structurally different approach worth prototyping
next, separately from tuning self-imitation further.

## 2026-08-20 — Implemented: encoder isolation, Go-Explore archive, GRPO group filtering

Implemented all three of the above session's action items directly in
`jepa_pretrain.ipynb`'s Stage 2 cells (not `src/train.py` — that script
mirrors the older `vision_encoder.ipynb` open-loop `t_horizon=4` ablations,
a different and now-superseded apparatus; left untouched to avoid
conflating the two). Not yet run against the real ARC engine this session —
verified structurally with a standalone mock-env/mock-model smoke test
instead (deterministic toy env, 200 updates, asserts on gradient isolation
and archive growth). **Next session should run it for real on `ls20` before
trusting any of this.**

1. **Encoder gradient isolation** — `ARC_AGI_3.forward` now does
   `logits = self.brain(z.detach())`. `z` itself (non-detached) is still
   returned and still used by `jepa_loss` in `compute_losses`, so `eyes`
   keeps learning from JEPA, just not from the policy gradient. Smoke test
   confirms: a policy-only backward pass leaves `eyes`' `.grad` as `None`.
2. **Go-Explore archive-and-return** — `collect_group` takes an `archive`
   (list of `{"actions", "reward"}`, capped top-K by `update_archive`) and an
   `archive_frac`; that fraction of each group replays an archived
   action-prefix from `reset` before live policy sampling resumes, instead
   of always starting fresh. No save-state API is exposed by the env, so
   "return" is approximated as deterministic action replay — relies on the
   still-unconfirmed no-hidden-randomness assumption noted in
   `ARCHITECTURE.md`'s open questions; if wrong, replay lands *near* the
   archived state rather than exactly on it, which should still help more
   than never returning. Rollouts changed from `(steps, trajectory)` tuples
   to dicts (`steps`, `trajectory`, `actions`, `reward`) so archive/best-run
   scoring can use full-trajectory reward (prefix + live) while GRPO's
   advantage still uses live-only reward (no gradient on replayed actions,
   since no `log_prob` was recorded for them).
3. **GRPO zero-variance group filtering** — new `collect_group_dynamic`
   resamples the whole group (up to `max_resamples=3`) while live-reward std
   stays ≤ `reward_std_eps=1e-4`; the main loop skips the optimizer step
   entirely (but still updates the archive/replay-buffer/best-run
   bookkeeping) if it's still zero-variance after resampling, logging a
   running `skipped_updates` / `resample_total` count instead of silently
   spending updates on a dead gradient.

**Known limitation carried over, not addressed this pass:** still no
clipped-ratio/KL trust region (TODO item 4) — `compute_losses` is unchanged
plain REINFORCE-style group-relative advantage. Add that after this run
confirms (or rules out) whether the archive fixes retention.

**First real run on `ls20`, 500 steps, results:**

```
step    1 | grpo  +0.641 | jepa 2.574 | reward(task) -0.480 | best(task) -0.480 | archive 4
step  100 | grpo  +0.019 | jepa 0.973 | reward(task) -0.480 | best(task) -0.480 | archive 4
step  200 | grpo +15.791 | jepa 1.229 | reward(task) -0.830 | best(task) +0.520 | archive 4
step  300 | grpo +14.277 | jepa 0.995 | reward(task) -0.830 | best(task) +0.520 | archive 4
step  400 | grpo +12.030 | jepa 0.786 | reward(task) -0.830 | best(task) +0.520 | archive 4
step  500 | grpo +13.898 | jepa 0.690 | reward(task) -0.830 | best(task) +0.520 | archive 4
```
`skipped=0`, `resamples=0` for the entire run.

**Fix #1 (encoder isolation) confirmed working:** `jepa_loss` trends down
(2.574 → 0.690) instead of the 2026-08-18 run's rise — the policy gradient
is no longer corrupting `eyes`. This is the one clean win from this run.

**Fix #2 (Go-Explore archive) has a design bug, found by this run, not the
mock smoke test:** a win was found and archived somewhere before step 200
(matches the historical single win value, 0.52). But archived entries store
the *full* action sequence including the terminal (WIN-triggering) action —
replaying it during "return" lands the episode already in a terminal state,
so the live sampling loop's very first check (`if frame.state !=
NOT_FINISHED: break`) exits immediately with **zero live steps**. With
`archive_frac=0.5`, roughly half the group (12/24) becomes one of these
"phantom" zero-step rollouts once the archive holds a win: no `log_prob`,
no gradient contribution, but their `reward=0` (from
`episode_rewards_of`, which sums an empty list) still enters the group's
mean/std used for every other rollout's advantage. This is the likely cause
of `grpo_loss` jumping from ~0 to +12..+16 right at step 200 (the advantage
statistics for the surviving 12 real rollouts got skewed by 12 phantom
zeros) and `reward(task)` getting *worse* (-0.48 → -0.83) afterward, not
better. Net effect: **retention is still not solved** -- `best(task)` never
moves past the single +0.520 win found once, same "found once, never
repeated" signature as every previous attempt (2026-08-16/17, 2026-08-18),
now compounded by half the sampling budget being wasted on phantom replays.

**Fix applied (same day):** `update_archive` now stores `actions[:-1]`
(drops the terminal action), so "return" lands one step *before*
completion — non-terminal, so live sampling resumes there and the decisive
final action is chosen (and gradient-credited) by the live policy itself.
Also added `live_rollouts_of` as a belt-and-suspenders guard: `compute_losses`
and `episode_rewards_of` now exclude any zero-live-step rollout from the
group's mean/std entirely (not just from the gradient-accumulation loop,
which already excluded them implicitly), so a phantom rollout can no longer
skew the surviving rollouts' advantage even if one slips through some other
way in the future.

**Second real run on `ls20`, 500 steps, results:**

```
step    1 | grpo -0.085 | jepa  3.505 | reward(task) -0.480 | best(task) -0.480 | archive 4
step  100 | grpo +0.117 | jepa  0.694 | reward(task) -0.480 | best(task) -0.480 | archive 4
step  200 | grpo -0.328 | jepa  0.627 | reward(task) -0.480 | best(task) -0.480 | archive 4
step  300 | grpo -1.026 | jepa  0.574 | reward(task) -0.480 | best(task) -0.480 | archive 4
step  400 | grpo -0.857 | jepa 12.573 | reward(task) -0.240 | best(task) +0.040 | archive 4
step  500 | grpo +0.654 | jepa  1.213 | reward(task) -0.240 | best(task) +0.040 | archive 4
```
`skipped=0`, `resamples=0` for the entire run again.

**Trim fix confirmed:** `grpo_loss` stayed in the expected ~[-1, +1] range
the whole run -- no repeat of the previous run's +12..+16 blowup. No phantom
rollouts this time.

**Archive is chaining generations, not just replaying one fixed win:** the
new best trajectory is **96 frames**, exactly ~2x a single `max_steps=48`
episode -- consistent with the run building a new success on top of an
already-archived ~48-step partial trajectory (return to it, then extend
further live), rather than rediscovering the same short win from scratch.
This is Go-Explore's intended behavior (return-then-extend compounding
progress across updates), not something the previous single-episode setup
could do at all.

**First real signal that the group's typical behavior, not just the
single best episode, improved:** `reward(task)` (group mean) moved from
-0.480 to -0.240 at step 400 and *stayed* there through step 500 -- the
previous run's group mean got permanently worse after a win was archived
(-0.48 → -0.83); this time it's better and held for 100+ steps. Weak
evidence, one run, but the opposite direction from before.

**`jepa_loss` spiked to 12.573 exactly at step 400** (vs. ~0.6-0.7
immediately before/after) -- coincides with `reward(total)` spiking to
+1.305 (curiosity bonus dominating), i.e. the newly-explored region past
frame ~48 has real, temporary prediction error (understandable: Stage 1
pretraining never saw this deep into the game) rather than the "noisy TV"
pathology from the lit search (which would show *persistent or growing*
error, not a one-step spike that recovers immediately, as this one did by
step 500). Worth watching over longer runs, not concerning yet.

**Still unresolved:** `resamples=0`/`skipped=0` both runs -- the
zero-variance-trap fix (#3) remains untested in practice. No clip/KL trust
region yet. Only one archive-chained success found in 500 steps; too early
to call retention solved, but the mechanism is visibly doing something
different (and better-looking) than the naive full-replay version.

## 2026-08-20 (later same day) — Added PPO clipped-ratio + KL-penalty trust region (TODO item 4's last piece)

Implemented in `jepa_pretrain.ipynb` Stage 2 (cells `f63cc0ae`, `166582e9`,
`2498d8a9`). Not yet run against the real env -- validated with mock-env/
mock-model smoke tests (epoch-0 ratio == 1.0 exactly, epoch-0 KL == 0.0
exactly, KL grows across epochs within an outer step, stop-grad still
intact, no crashes over 100 mock updates).

**Mechanism change:** previously one gradient step per collected group,
using `log_prob` recorded at collection time directly in the REINFORCE loss
(`grpo_loss = -mean(advantage * log_prob)`). Now `compute_losses` takes
`ppo_epochs` (default 4) gradient steps per collected group, **re-evaluating
every stored step under the current policy each epoch** (`z`, `logits`
recomputed fresh from stored `x` -- never reused from collection time, since
reusing the old computational graph across multiple `backward()` calls
would be stale/incorrect after the first `optimizer.step()`).

**Math, per step:**
- Old (collection-time) policy produced `log π_old(a|s)` for the action
  actually taken, stored as `log_prob_old`, and the full distribution
  `log π_old(·|s)` over all `len(ACTION_SPACE)` actions, stored as
  `old_log_probs_full` (`dist.logits` from the masked Categorical --
  PyTorch normalizes this internally, i.e. it's already `log_softmax(logits
  + mask)`).
- Each epoch, current policy gives `log π_new(a|s)` by re-running
  `model(x)` and rebuilding the Categorical with the *same stored* `mask`
  (must be the same object, not recomputed from `x`, since legality isn't
  derivable from pixels alone).
- Importance ratio: `r = exp(log π_new(a|s) - log π_old(a|s))` -- equals
  `π_new(a|s) / π_old(a|s)`, `r = 1` at epoch 0 by construction (same
  weights, same input, deterministic forward pass).
- Clipped surrogate (Schulman et al. 2017): `L_clip = -min(r · A, clip(r,
  1-ε, 1+ε) · A)`, `ε = clip_eps = 0.2`. `grpo_loss` is the mean of this
  over all live steps (normalized by episode count, same convention the
  plain REINFORCE version used, not step count).
- KL penalty, computed **exactly** rather than sampled (tiny discrete
  action space, so this is cheap): `KL(π_old ‖ π_new) = Σ_a π_old(a) ·
  (log π_old(a) - log π_new(a))`, using the full stored `old_log_probs_full`
  against the fresh `dist.logits` each epoch. Added to the total loss as
  `kl_coef · KL`, `kl_coef = 0.02`.
- Total loss per epoch: `L = L_clip + kl_coef·KL + jepa_weight·L_jepa −
  entropy_coef·H(π) + replay_coef·L_replay`, same combination as before
  with the two new terms folded in.

**Cost:** `ppo_epochs`x more forward/backward passes per outer step (~4x at
the default) for the same number of real-env rollouts collected -- real-env
sampling was always the expensive part, this just extracts more gradient
signal per sample instead of discarding the group after one step. If a run
needs to match the old wall-clock budget, lower `epochs` roughly
proportionally.

**Next:** run it for real on `ls20`, check that `KL`/`grpo_loss` stay
bounded and `KL` doesn't run away across epochs (would indicate `clip_eps`/
`kl_coef` need retuning), and see whether the extra gradient mileage per
group changes anything about the retention picture from the last two runs.

## 2026-08-20 (same day) — Bug found on first real run: NaN KL from the very first update

3000-epoch `ls20` run showed `kl = nan` already at step 1. Root cause:
`compute_losses`'s exact-KL formula summed over *all* `len(ACTION_SPACE)`
actions, including illegal ones. An illegal action is masked to `-inf` in
both `old_log_probs_full` and the current epoch's `dist.logits` (same
stored mask replayed each epoch), and `-inf - (-inf) = nan` in float
arithmetic -- not the mathematically-correct 0, even though `pi_old(a) = 0`
there means that action's true contribution to KL is 0 by definition
(`0 * log(0/x) = 0` is a standard KL convention, but `0 * nan` is `nan` in
floating point, not 0 -- the `.exp()` zeroing never gets a chance to apply
since the subtraction itself is already indeterminate).

The mock smoke test (`smoke_test_v2.py`) didn't catch this because its toy
env made every action legal at every step (`available_actions = [0,1,2]`
always) -- the `-inf` branch was never exercised. `ls20`, like essentially
every real game here, has partial legality almost every step, so the bug
fired on the very first real update. **Lesson: a mock env used to validate
RL-loop logic needs to exercise partial action legality, not just be
structurally similar** -- added to the smoke test (`smoke_test_v3.py`,
action 2 always illegal) and confirmed clean afterward.

**Fix:** restrict the KL sum to the legal-action subset, computed by
indexing rather than masking-then-multiplying (avoids ever forming the
`-inf - -inf` expression at all, not just zeroing its result after the
fact):
```python
legal = torch.isfinite(old_log_probs)
kl_sum = kl_sum + (old_log_probs[legal].exp() * (old_log_probs[legal] - new_log_probs[legal])).sum()
```
This is also the mathematically correct definition -- `KL(pi_old || pi_new)`
is a sum over `pi_old`'s support, which by construction excludes actions
`pi_old` assigns zero probability to.

**The 3000-epoch run collected before this fix is invalid from update 1
onward** (`nan` loss -> `nan` gradients -> `nan` weights propagate through
every subsequent forward pass) -- discard it, re-run after this fix.

**Follow-up:** 3000 epochs at `group_size=24`/`ppo_epochs=4` estimated
~35h wall-clock (real-env rollout collection, not the added PPO epochs, is
almost certainly the dominant cost -- ~1150 real engine `step()` calls per
outer step at that group size/max_steps). Re-ran instead at `epochs=300`,
`group_size=16`, `ppo_epochs=2` for a faster sanity check:

```
step    1/300 | grpo +0.368 | jepa 2.560 | kl 0.0295 | reward(task) -0.480 | best(task) -0.480
step  100/300 | grpo -0.145 | jepa 0.166 | kl 0.0022 | reward(task) -0.240 | best(task) +0.040
step  200/300 | grpo -0.059 | jepa 0.235 | kl 0.0016 | reward(task) -0.240 | best(task) +0.040
step  300/300 | grpo -0.097 | jepa 0.066 | kl 0.0015 | reward(task) -0.240 | best(task) +0.040
```

**Good: fix confirmed clean.** `kl` stays finite and shrinks toward 0
(0.0295 → 0.0015), `grpo_loss` stays small and bounded, no `nan` anywhere --
the trust region is behaving as designed, not diverging.

**Not good: flat plateau for 200 of 300 steps.** `best(task)` and
`reward(task)` both froze at their step-100 values (+0.040 / -0.240) through
step 300 -- identical to three decimal places, no new archive-chained
progress, same "found once, then stuck" shape as every previous attempt,
just clean instead of chaotic this time. `kl` shrinking toward 0 across
the run says each successive update is barely moving the policy at all --
plausibly too little gradient signal per group to keep making progress,
not (this time) an instability artifact.

**Read:** the stability fixes (encoder isolation, archive-return, zero-var
filtering, clip/KL) are doing their job -- nothing is corrupting itself or
exploding anymore. But they don't by themselves make `ls20` (chosen
specifically as a stress-test hard-exploration game, see every prior entry
in this log) solvable in a few hundred updates at a shrunk `group_size=16`.
Two candidate confounds worth separating before concluding anything about
the mechanisms themselves: (1) `group_size` was cut 24→16 for wall-clock
reasons, which directly weakens the group-relative advantage signal and the
odds of a group containing a genuinely novel outcome each step; (2) the
archive picks its top-4 by reward alone (documented simplification, no
novelty weighting) -- if those 4 entries are just near-duplicates of the
one known 96-frame lineage, `archive_frac=0.5` spends half the group
re-exploring the same path instead of genuinely new ones.

**Suggested next step, not yet done:** before tuning `ls20` further,
validate the same pipeline on a different (ideally easier) offline game --
`ls20` was deliberately picked as a hard-exploration stress test back in
the 2026-08-16/17 ablations, so a flat plateau here doesn't cleanly
distinguish "the mechanisms aren't helping" from "this specific game is
still just very hard." If a different game shows the archive genuinely
chaining multiple distinct improvements within a normal budget, that's a
much stronger signal the retention mechanisms work than anything further
squeezed out of `ls20` alone.

## 2026-08-20 (same day) — Fixed the curiosity-masked zero-variance check

Confirmed the suspicion raised earlier: `episode_rewards_of` summed
`s["reward"]` (extrinsic + curiosity_bonus), and that's what
`collect_group_dynamic`'s zero-variance-trap check and the main loop's
skip decision both used. Since `curiosity_coef=0.005` is on, and curiosity
varies per rollout even when every rollout hits the identical task outcome
(e.g. all 16-24 rollouts time out at `max_steps` with zero level
completions -- the "-0.480 always-fail baseline" seen at the start of
every run), the check's input was never actually zero-variance even when
the *task* signal was -- explaining why `skipped=0`/`resamples=0` on every
real run so far didn't mean the trap was truly absent, just masked.

**Fix:** `episode_rewards_of` now sums `extrinsic_reward` only (task
reward, no curiosity), used exclusively for the trap check --
`compute_losses`'s actual GRPO advantage is intentionally left unchanged
(still the combined extrinsic+curiosity reward; blending curiosity into
the policy gradient is a separate, unquestioned design choice, not the bug
here). Added `reward_std_diagnostic(rollouts) -> (task_std, total_std)`
and wired both into the periodic log line, so future runs show directly
how much curiosity noise would have masked the check.

Verified with a standalone mock (16 rollouts, identical task reward
`-0.48`, curiosity bonus varying `0.05-0.125`): before the fix, std over
the combined reward was `0.0238` (>> `reward_std_eps=1e-4`, trap
invisible); `episode_rewards_of` now reports the true `task_std=0.0`
directly. Not yet re-run against the real env -- next run's periodic log
will show how often `std(task)` actually hits near 0 in practice, which
was invisible until now.

**Fix #3 (zero-variance filtering) untested by this run** -- `resamples`
stayed 0 throughout, i.e. the mechanism was never exercised (the group
always had enough natural reward variance, partly *because* of the phantom
rollouts above). Can't yet confirm or rule out whether it works on a real
zero-variance group; revisit once fix #2's phantom-rollout issue is fixed
and doesn't accidentally keep manufacturing variance.

## 2026-08-20 (same day) — Two external public Kaggle agents inspected for reference

Not integrated into the pipeline, kept as notes only.

- **"FORGE v31" (public kernel, uradkr/v31-code-pro, self-reported 0.36)**
  is not a learned or fair-play agent: it reads the offline games' hidden
  `.py` source directly (`importlib` + regex to extract the exact win
  condition), then does BFS/A* by `copy.deepcopy()`-ing the internal engine
  object and calling `perform_action()` on the copy directly -- bypassing
  the frame/action API entirely. Depends on local access to game source
  that almost certainly isn't available for the real held-out eval games,
  and is exactly the per-game hacking this project has ruled out from the
  start ([[feedback_no_game_hacking]] memory). Not a meaningful comparison
  point for this project. One reusable idea, generic and non-cheating:
  connected-component blob centroids as ACTION6 click candidates instead of
  raw (x,y) sampling.
- **"Explore2" (public agent code shared by user, self-reported 0.54)** is
  legitimate -- only uses `frame`/`available_actions`, no hidden-source
  reads, no engine-internals access. A deterministic frontier-graph search
  agent (no neural net, no gradients at all): state = frame hash with
  border/counter cells masked out, BFS to the nearest node with untried
  actions when the current one is exhausted, connected-component click
  candidates for ACTION6, plus two generic online-learned heuristics: an
  auto-detected counter/animation mask (cells that change in >=80% of
  observed transitions, not hardcoded) and per-action effectiveness
  ordering (empirical P(action changed state), tried first). One
  game-specific override (`cn04: USE_REORDER=False`) is a minor exception
  to "no per-game special-casing." A 0.54 score from pure search (no
  learning at all) suggests many offline games have a small enough
  reachable state space for frontier BFS to solve outright -- a different
  problem-solving regime than this project's learned Eyes/Brain/Intuition,
  not a comparable number. Two reusable *generic* ideas if ever revisited:
  masked-state-hash visitation counts as a count-based exploration bonus
  (alternative/complement to the current prediction-error curiosity), and
  the same connected-component click-candidate idea as above for the
  still-unbuilt ACTION6 pointer head (TODO item 4).

## 2026-08-20 (same day) — Community signal (Discord): reported competition "meta" is an LLM-based agent, not from-scratch

Secondhand, not verified firsthand -- screenshot of the competition Discord,
not read directly. Danny Wu (community member, unclear if organizer or
participant) states the current "meta" is a solution nicknamed **"duck"**
built on **Qwen3.6**, with a write-up described as "really good," listing
areas the team hasn't explored but thinks could be meaningful. No link
captured -- if this matters later, worth a deliberate search for the actual
write-up rather than acting on the secondhand summary alone.

Directly relevant to the still-open **TODO item 6 / DESIGN_LOG
2026-08-14** question (frozen-pretrained-LLM "brain" vs. from-scratch
reasoning transformer, never resolved, ablation never run) -- if the
reported meta is genuinely LLM-based and outperforming from-scratch
approaches, that's a real signal worth weighing before continuing to defer
that ablation indefinitely. Not acted on yet -- just a data point.

Also from the same thread: Danny Wu says **training (including SFT) is
allowed** as long as your own data is attached, and **compute resources are
available if used genuinely and solely for the competition** -- relevant to
whether a heavier training budget (e.g. actually running the frozen-LLM
ablation, or more Stage 1 JEPA pretraining) is realistic, but per-rules
detail (which resources, how to request them) not captured here either.

## 2026-08-20 (same day) — Empirically tested DINOv2/v3 vs. from-scratch (new notebook: `dino_probe.ipynb`)

User's request, given the `ls20` plateau: stop reasoning about whether a
pretrained natural-image encoder would help and actually test it, on a
handful of real `ls20` frames. New standalone notebook, not wired into the
pipeline -- a probe. Three encoders compared: frozen `facebook/dinov2-small`
(22M params), a random-init ViT of the *identical* architecture (isolates
"pretraining helps" from "a big enough network already helps"), and this
project's own trained `Eyes` (3,824 params, Stage 1 JEPA checkpoint).

**Quantitative (correlation between embedding distance and true pixel-level
difference, 40-frame pairwise, Pearson/Spearman):**
```
DINOv2-small (pretrained)              | pearson r=+0.621 | spearman rho=+0.596
random-init ViT (same arch)            | pearson r=+0.750 | spearman rho=+0.781
our Eyes (from-scratch, Stage 1 JEPA)  | pearson r=+0.140 | spearman rho=+0.121
```
A same-sized **random-init** ViT correlates *better* with raw pixel
difference than pretrained DINOv2 -- natural-image pretraining isn't adding
anything here, consistent with `ARCHITECTURE.md`'s original (until now,
untested) rejection. Concrete mechanism, not just "domain mismatch": DINOv2
patches are 14px; upscaled to 224, one ARC cell is ~3.5px, so a 2-cell
sprite is smaller than a single DINO patch and gets averaged away before
attention ever sees it. This project's own `Eyes` already uses patch=4 on
the *native* 64x64 grid for exactly this reason.

**Qualitative (patch-PCA, the standard DINO-paper visualization) -- first
pass was misleading, fixed same session:** the first version fit a fresh
PCA per frame independently, which makes colors incomparable across frames
(PCA components are only defined up to sign/rotation) -- **fixed** to fit
one shared PCA (+ shared min/max) across all shown frames per encoder, so
frame-to-frame color changes are actually meaningful. Result, 6 frames
spanning the collected rollout:
- DINOv2 and random-init ViT rows look **essentially frame-invariant** --
  same static pattern regardless of which frame, dominated by the big
  static background/border, no visible response to the small moving sprite.
- **Our `Eyes` row visibly changes frame to frame**, and a colored blob's
  position in the PCA-RGB image appears to track the small pink/orange
  sprite's position in the raw frame reasonably well across several of the
  6 samples -- qualitatively, `Eyes` looks like it's localizing the one
  game-relevant moving element that neither pretrained nor random ViT show
  any sign of noticing.

**This reframes the "our Eyes scores worse" quantitative result from
earlier the same day.** `ls20`'s raw pixel-difference signal is likely
dominated by things that don't matter for the game (a bottom progress bar
that fills in incrementally every step, touching many pixels) rather than
the small sprite that actually matters. `Eyes` trained via JEPA
next-state-under-action prediction has real pressure to represent
*action-relevant* change and can legitimately learn to de-emphasize
bulk-but-irrelevant pixel churn -- which would show up as *low* correlation
with raw Hamming pixel-distance while still being the more useful
representation, not a worse one. DINOv2/random-init have no such pressure,
so their higher raw-pixel correlation may just mean they're reacting to the
progress bar, not tracking the game.

**Not rigorously confirmed** -- this is a qualitative read of one PCA plot,
not a targeted metric. A real follow-up (not done yet) would isolate the
sprite specifically (e.g. correlate embedding change with sprite (x,y)
displacement alone, masking out the progress-bar region) rather than
inferring it by eye from 6 frames.

**Net conclusion:** empirical evidence supports staying with the
from-scratch approach over swapping in a pretrained DINOv2/v3 backbone --
not just on the original "domain mismatch" reasoning, but because the
tiny trained-from-scratch encoder shows a real (if only qualitatively
observed) sign of tracking the actual game-relevant signal that a 22M-param
pretrained network, patch-size-mismatched to this domain, does not.

**Correction (same session, minutes later) -- the qualitative "Eyes tracks
the sprite" read above was wrong.** User pushback, confirmed empirically:
even confirmed-static patches (e.g. the reset icon, never changes pixel
value across any of the 40 collected frames) visibly shift color in
`Eyes`'s shared-PCA row -- that's instability, not selective tracking. Added
a rigorous test: found the 212/256 patches (83%) that are byte-identical
across all 40 frames (ground truth, not inferred), then measured each
encoder's per-patch embedding std across frames, split static vs. dynamic:

```
DINOv2-small      | static std 0.0030 | dynamic std 0.0097 | dynamic/static ratio  3.19x
random-init ViT   | static std 0.0002 | dynamic std 0.0113 | dynamic/static ratio 46.68x
our Eyes          | static std 0.0197 | dynamic std 0.0581 | dynamic/static ratio  2.94x
```

`Eyes` drifts 66x more than random-init and 6.5x more than DINOv2 on
patches that provably never change -- and has the *worst* dynamic/static
signal-to-noise ratio of the three, not the best. The random-init ViT's
extremely clean separation makes sense mechanically (a frozen deterministic
network gives identical output for identical input; near-zero static std is
close to guaranteed unless attention leaks information in from patches that
did change elsewhere, and that leak is small for random-init, moderate for
DINOv2, large for `Eyes`).

**Corrected conclusion:** `Eyes` is the weakest of the three encoders on
every rigorous (non-eyeballed) metric today, most plausibly because it's
severely undertrained/under-capacity (3,824 params, 1 layer) rather than
because it's smartly de-emphasizing irrelevant pixels. This does *not*
change the DINOv2/v3-rejection conclusion (patch-granularity mismatch
argument is untouched, and random-init's clean static-stability is likely a
generic property of any frozen deterministic net, not evidence of learned
understanding). It does mean **`Eyes`'s representation quality itself is
now a real, evidenced open question** -- worth more Stage 1 JEPA
training/data or added capacity before trusting it further, rather than
assumed adequate. Retracting the earlier "Eyes looks like it's tracking the
sprite" read entirely -- it doesn't hold up.

## 2026-08-21 — Scaled up Stage 1 (bigger Eyes+Intuition, GPU, more data), re-ran the DINO probe

Direct response to the above: local dev machine has an idle RTX 3090
(discovered mid-session), so installed a CUDA torch build
(`torch==2.13.0+cu130`, `requirements.txt` updated) instead of the CPU-only
one used until now. Scaled `WorldModel` from 3,824/7,613-param
Eyes/Intuition (1 layer each, latent_dim=16) to **2,709,504/3,789,056**
(eyes_layers=6, intuition_layers=4, latent_dim=192, num_heads=6) -- still
tiny next to DINOv2-small's 22M, but ~700x this project's previous size.
Data: `STEPS_PER_GAME` 8,000 -> 30,000 (~510k train transitions). Training:
batch_size 64->256, epochs 15,000->20,000, on GPU. Stage 2's `ARC_AGI_3`
instantiation updated to match (untested this session -- not run).

**Mid-run scare, resolved:** at 25% (step 5018/20000) the loss readout
showed `jepa_loss=15.3768`, alarming next to prior tiny-model runs' ~0.5-2.5
range -- but the loop only plotted the full curve at the very end, so
there was no way to tell "high but stable/still descending" from "actually
diverging" from a single snapshot. Not resolved by inspection this session
(user restarted before checking the trend) -- but the loop itself was fixed
regardless: lowered `lr` 1e-3 -> 3e-4 with a 500-step linear warmup (no
warmup at all was a real gap for a 6-layer transformer trained from
scratch, even if it wasn't confirmed to be the cause here), and added
`pbar.write` logging every 500 steps plus a log-scale final plot, so this
ambiguity can't recur silently.

**Re-ran `dino_probe.ipynb` against the new checkpoint (also updated its
Eyes config to match). Mixed, genuinely interesting result:**

```
Pixel-distance correlation (global, pooled embedding):
DINOv2-small (pretrained)     | pearson r=+0.621 | spearman rho=+0.596
random-init ViT (same arch)   | pearson r=+0.750 | spearman rho=+0.781
our Eyes (NEW, 2.7M, trained) | pearson r=+0.768 | spearman rho=+0.847   <- now highest of all three
                                 (was r=+0.140 with the old 3.8K-param Eyes)

Static-vs-dynamic patch stability (confirmed-static-patch temporal std):
DINOv2-small       | static 0.0030 | dynamic 0.0097 | ratio  3.19x
random-init ViT    | static 0.0002 | dynamic 0.0113 | ratio 46.68x
our Eyes (NEW)     | static 0.0399 | dynamic 0.0467 | ratio  1.17x        <- worse than the old
                                                                             Eyes' 2.94x, and static
                                                                             std nearly doubled (was 0.0197)
```

**Real progress on one axis, a new problem on another.** The pooled/global
embedding is now genuinely the *best* of the three at reflecting true
pixel-level state differences -- a real, sizeable improvement, not noise.
But patch-level stability on confirmed-static content got *worse*, not
better, despite 700x the capacity and ~3.75x the data. Qualitatively (patch-
PCA, shared basis): the new Eyes row shows strong **vertical banding** --
color gradients running left-to-right -- that shifts color dramatically
frame to frame while keeping the same banded *structure*. Doesn't resemble
the scene's actual object layout the way random-init's clean segmentation
does. Working hypothesis, not confirmed: could be the 2D axial RoPE (row/col
split) dominating the representation with a positional mode that doesn't
track content, rather than a content-driven signal -- plausible given the
architecture, not verified.

**Net read:** don't declare this solved. The global discriminability number
moving in the right direction is real and worth keeping, but the static-
patch regression is new evidence of an unresolved (possibly different)
stability problem that "bigger + more data" alone didn't fix -- and might
even have made more visible by giving the model enough capacity to express
a strong-but-wrong mode (the banding) that a tinier model couldn't. Next: (1)
inspect whether the banding correlates with the RoPE row/col axes
specifically (e.g. ablate row-RoPE vs col-RoPE contribution), (2) longer
training / check whether the mid-run loss scare was real divergence now
that the loop logs properly, before drawing firm conclusions from this one
run.

## 2026-08-21 (same day) — Confirmed: the loss scare was real divergence, not just a different scale

User checked the full curve (log-scale plot, now that the loop logs
properly): `jepa_loss` drops fast to ~0.05 by step ~2,500, then climbs
steadily and noisily to ~0.9-1.0 by step 20,000. Not "high but stable" --
genuinely got worse for 87% of the run. The saved checkpoint was the
*final* (most-degraded) weights, explaining the DINOv2-probe's static-patch
stability regression after "scaling up" -- the model that got probed was
well past its optimum.

**Fixed:** `jepa_pretrain.ipynb`'s Stage 1 loop now tracks the smoothed
loss (mean of last `log_every=500` steps) every `log_every` steps, keeps a
CPU copy of `eyes`/`intuition` weights whenever that smoothed loss improves,
and stops after `PATIENCE_CHECKS=4` (2,000 steps) with no improvement.
Validation and checkpoint-saving cells now restore the *best* tracked
weights into `model` first, instead of using whatever the loop happened to
end on. Verified the bookkeeping logic on a synthetic curve shaped like the
real one (fast drop to ~0.05 by step 2,500, then linear climb): correctly
stopped at step 5,000, best recorded at step 3,000 -- matches the real
curve's visible minimum.

**RoPE-domination hypothesis confirmed on the (degraded, pre-fix) checkpoint.**
Added a diagnostic to `dino_probe.ipynb`: correlate each patch's leading PCA
direction with its raw row index vs. column index, averaged over all 40
frames.
```
our Eyes         | |corr(PC1, row)| = 0.078 +/- 0.052 | |corr(PC1, col)| = 0.820 +/- 0.035
DINOv2-small     | |corr(PC1, row)| = 0.394 +/- 0.018 | |corr(PC1, col)| = 0.047 +/- 0.013
random-init ViT  | |corr(PC1, row)| = 0.286 +/- 0.003 | |corr(PC1, col)| = 0.179 +/- 0.008
```
`Eyes`' leading direction is overwhelmingly explained by column index alone
(0.82) and barely by row (0.08) -- both DINOv2 and random-init show much
weaker, less lopsided row/col correlations (and DINOv2's, if anything, leans
row not column). Too extreme to be genuine content tracking (the sprite
visibly moves in both row and column across the collected frames) --
consistent with the column-axis RoPE component dominating attention
regardless of content, though not proven (could also just be what a
badly-overtrained/degraded model collapses toward, unrelated to RoPE
specifically -- the two hypotheses aren't yet disentangled).

**Not yet known: is the banding an over-training artifact, a RoPE
architecture issue, or both.** Re-running Stage 1 with the early-stopping
fix now to get a real best-checkpoint, then re-running both the
static-patch-stability and row/col-correlation diagnostics on it. If the
early-stopped checkpoint's col-correlation is still ~0.8, that points at
RoPE specifically (worth an ablation: zero out `col_rope`'s contribution and
see if attention/representation quality changes) rather than just "trained
too long."

## 2026-08-21 (same day) — RoPE hypothesis result + Stage 2 was ~15h/300-steps, fixed

Re-ran Stage 1 with the new early-stopping loop: stopped at step 3,500, best
checkpoint step 1,500 (`mean(last 500)=0.0684`). Re-ran both `dino_probe.ipynb`
diagnostics against it:
```
static-patch stability:  static 0.0073 | dynamic 0.0229 | ratio 3.11x  (was 1.17x -- fixed, ~= DINOv2's 3.19x)
RoPE row/col correlation: |corr(PC1,row)|=0.118 | |corr(PC1,col)|=0.184  (was 0.078/0.820 -- collapse gone)
pixel-distance corr:     pearson r=+0.481, spearman rho=+0.576  (was +0.768 on the over-trained checkpoint)
```
**RoPE-domination hypothesis resolved: it was an over-training artifact, not
an architecture flaw.** The column-collapse (0.82 correlation) is gone at
the early-stopped checkpoint (back to 0.18, comparable to DINOv2/random-init's
own row/col correlations). Good news for the architecture; no RoPE ablation
needed.

**New trade-off surfaced:** early stopping (picks the step with lowest raw
JEPA loss) fixed stability but landed very early (step 1,500 of a 20,000
budget) and gave up the pixel-correlation gain the over-trained checkpoint
had (0.768 -> 0.481, now below both DINOv2 and random-init again). Raw JEPA
loss is an imperfect proxy for downstream representation quality -- it can
dip early (target EMA still close to online encoder, task is briefly easier)
before the encoder has developed much real structure. Proposed but not yet
done: snapshot checkpoints periodically through a longer run and evaluate
each with both DINO-probe diagnostics directly, instead of trusting raw loss
alone to pick the checkpoint. Deferred -- user chose to proceed to Stage 2
with the current (stable, RoPE-collapse-free, moderate-correlation)
checkpoint instead of tuning Stage 1 further.

**Stage 2 was effectively broken for the new encoder size:** first real run
showed `181.18s/it`, ETA ~15 hours for a 300-step run. Root cause: Stage 2
was never moved to GPU (Stage 1 was, Stage 2 wasn't touched), and
`compute_losses` ran a Python loop calling `model(...)` once per stored step
-- fine for the original 7.6K-param encoder, but `ppo_epochs` re-runs that
loop from scratch every epoch, and the encoder is now ~6.5M params combined.

**Fixed:** `policy = ARC_AGI_3(...).to(device)`, `frame_to_tensor`/the
sampling `mask` now created directly on `device`, and `compute_losses`
rewritten to stack every stored step across the whole group into single
tensors and do **one** forward pass instead of one per step. Verified
against a mock env/model with partial action legality (exercises the
-inf/-inf KL guard): batched output matches the old unbatched loop exactly
(diff < 1e-4 on all four returned values), gradients/stop-grad unchanged.
Not yet re-run against the real env to confirm the actual speedup -- next
step.

## 2026-08-21 (same day) — Batched compute_losses wasn't enough: collection itself vectorized

The `compute_losses` batching fix cut the ETA from ~15h to ~7h12 for 300
steps (~86s/it) -- better, still far too slow. Benchmarked single-sample
`Eyes` forward pass directly: GPU is actually *faster* than CPU even at
batch=1 (5.4ms vs 13.2ms) for the current 6-layer/192-dim size, so the
remaining cost isn't "GPU overhead on tiny batches" -- it's the sheer
*number* of sequential single-sample calls during collection:
`collect_group` ran `group_size` (24) rollouts fully sequentially, each up
to `max_steps` (48) real-env steps, each doing up to 3 model calls (main
forward + curiosity's `imagine`/`target_latent`) one at a time -- up to
~3,456 individual forward calls per outer step before any resampling.

**Fixed: `collect_group` rewritten to be lockstep-vectorized.** Instead of
one rollout fully to completion before the next starts, all `group_size`
rollouts now step together: at each timestep, gather the current frame of
every still-*active* rollout (not yet terminated/out of legal actions) into
one batched tensor, do ONE forward pass for all of them, sample all their
actions, then step each rollout's own env individually (real-engine
stepping is inherently sequential/stateful, can't be batched) and record.
Needs one independent env instance per group slot now (`envs = [arc.make(GAME)
for _ in range(group_size)]`, created once, reused every outer step) --
a single shared `env` object can't represent `group_size` simultaneously
in-progress games. Archive-prefix replay stays a simple per-rollout
sequential loop first (deterministic, no model calls needed), only the
*live* policy-sampled phase is batched. Cuts the number of individual model
forward calls during collection by ~`group_size`x.

Verified with a mock env pool (staggered termination steps per rollout,
partial action legality, archive + curiosity both exercised): each
rollout's recorded step count matches its own termination point exactly (no
cross-rollout interference from the shared timestep loop), reward/action
bookkeeping correct, archive prefix replay correct, no crashes. Not yet
re-run against the real env to confirm the actual wall-clock speedup --
next step, expect on the order of the group_size reduction in forward-call
count (~24x fewer calls), though real speedup will be less than that
1:1 since larger batches take somewhat longer per call than tiny ones.

## 2026-08-21 (same day) — Vectorized collection OOM'd, then a real "zero gradient updates" blocker found

Vectorized `collect_group` sped things up (15.26s/it, down from 86s/it) but
OOM'd 24GB of VRAM at `compute_losses`: batching the whole group in one
forward pass means ~1,150 "samples," but each is a 256-patch-token sequence
(64x64 grid, patch_size=4), so batched-attention activation memory for
backward blew up well past parameter count. **Fixed:** minibatched --
`flatten_steps_with_advantage` computes the group-relative advantage once
over the full group (needs the full episode rewards to normalize correctly),
then `compute_losses_minibatch` processes `minibatch_size=128`-step chunks,
`ppo_epochs` full shuffled passes over all of them, one optimizer step per
minibatch. Normalizes by minibatch step count now, not episode count (a
minibatch won't align with episode boundaries) -- a deliberate, noted
convention change. Verified via mock (24 rollouts x ~40 steps, minibatch
128): flatten count matches raw step count exactly, all minibatch
backward+step calls run without error, gradients/stop-grad intact.

**After restarting the kernel to reclaim the ~23GB stuck from the OOM
crash, ran again: 48 consecutive outer steps, `skipped=48` -- 100%,
zero gradient updates applied at all.** Not a bug -- confirmed by reading
the skip branch: `grpo_loss = jepa_loss = entropy = kl = replay_loss =
torch.tensor(0.0)` is a hardcoded placeholder printed on every skip, not a
computed (let alone NaN) value; `compute_losses`/`compute_losses_minibatch`
were never even called. The zero-variance-trap fix (DESIGN_LOG 2026-08-20)
is doing exactly its job -- and revealing something the curiosity-noise
masking previously hid: literally every collected episode across
thousands of samples (24 rollouts x up to 4 attempts x 48 outer steps) hit
the exact same task outcome (`-0.480`, i.e. ran out `max_steps` with no
death, no win, ever).

**Likely root cause: `max_steps=48` structurally capped every rollout
below the length any known win requires.** Every win found this entire
session (archive-chained or not) needed 48-96 steps -- with a hard 48-step
live budget, no live rollout could ever physically reach one, only ever
time out identically. **Fixed:** raised `max_steps` to 120 (margin above
the known 96-step ceiling). Costs ~2.5x collection time, absorbable now
that collection is vectorized (16s/it observed, not 86-181s/it). Not yet
confirmed this actually breaks the 100%-skip streak -- next run's
`skipped`/`resamples` counts and whether `best(task)` ever moves off
-0.480 will tell.

## 2026-08-21 (same day) — max_steps=120 broke the skip streak, then a NaN crash from an OOD outlier

`max_steps=120` worked: by step 60, `skipped` was back to normal (no
longer 100%), and `best(task)=-1.200` -- a genuine death (GAME_OVER, not
just a timeout) got found and used for a real gradient update for the
first time this run. Confirms the earlier diagnosis: `max_steps=48` was
structurally preventing any outcome diversity.

**New crash right after: `jepa_loss=1023.839` (vs. the usual 0.1-10 range)
just before `Categorical(logits=...)` got NaN logits and raised a
ValueError.** Read as: some minibatch contained a catastrophically
mispredicted sample -- almost certainly a WIN/GAME_OVER frame, since Stage
1's random-walk pretraining data essentially never reaches those (a random
policy rarely intentionally wins or dies), so the world model has ~no
calibration there. That one sample's huge MSE, averaged into the
minibatch's `jepa_loss`, produced a gradient that (even after
`clip_grad_norm_`) corrupted the weights to NaN -- clipping bounds the
gradient *norm* assuming finite values; it doesn't help once a component is
already NaN/Inf, and a large-but-finite loss can still produce a NaN
downstream (e.g. through GELU/softmax overflow) after one bad step.

**Fixed, two layers:**
1. **Per-sample JEPA loss clamp** (`JEPA_LOSS_CLAMP=50.0`) inside
   `compute_losses_minibatch` -- clamps each sample's MSE *before*
   averaging, so one extreme outlier can't dominate the minibatch mean the
   way it did at 1023 (would still average high with only an aggregate
   clamp on the mean itself). Verified on a synthetic outlier: mean MSE
   8333 (dominated by one bad sample) -> 16.7 clamped.
2. **Finite-loss/grad safety net** in the main loop -- skip the
   optimizer step entirely (via `torch.isfinite(loss)` then
   `torch.isfinite(grad_norm)` checks) rather than let a non-finite loss or
   post-clip gradient ever reach `optimizer.step()`. New counters
   `nan_loss_skips`/`nan_grad_skips`, logged periodically. Verified: a NaN
   loss now leaves model weights completely untouched instead of
   corrupting them.

Belt-and-suspenders: (1) stops the *routine* version of this problem before
it starts, (2) catches anything (1) doesn't, without ever letting NaN reach
the weights either way. Not yet re-run against the real env -- next step.

## 2026-08-21 (same day) — Clamp fix ran, but revealed two more problems: dead gradient + noisy-TV

Ran with the clamp/safety-net fix, 229/300 steps in before user caught two
red flags:

1. **`jepa_loss` pinned at exactly `JEPA_LOSS_CLAMP` (50.000) for many
   consecutive logged steps** -- not a rare outlier being caught anymore,
   routine. `torch.clamp` has *zero* gradient past its ceiling, so every
   sample landing above 50 was getting no learning signal at all --
   precisely the samples most in need of correction, silently excluded.
   Confirmed directly: a synthetic huge-error sample gets gradient `0.0`
   through hard-clamped MSE vs. a bounded, nonzero `-1.0` through Huber
   loss on the same input.
2. **Mean episode reward spiked to ~2000** in the live plot. Root cause:
   `collect_group`'s curiosity bonus uses the *same* raw JEPA prediction
   error as a reward signal, but was never bounded by the clamp fix (which
   only touched `compute_losses_minibatch`'s GRPO-side copy). With
   `max_steps=120` pushing rollouts into states the early-stopped (step
   1,500) Stage 1 checkpoint predicts badly, that error -- and the
   curiosity reward built from it -- exploded, rewarding the policy for
   *surprising the world model* rather than task progress. This is exactly
   the "noisy TV" pathology flagged in the 2026-08-20 literature search
   (topic 2), now observed concretely instead of theoretically.

**Fixed:**
1. `compute_losses_minibatch`'s JEPA term switched from clamped MSE to
   **Huber loss** (`delta=1.0`), outer clamp kept as a final safety net for
   truly pathological cases (now a much higher bar since Huber's per-sample
   value grows linearly, not quadratically, past `delta`). Verified: Huber
   keeps nonzero (if bounded) gradient even for a synthetic error of 1000,
   where clamped MSE goes fully dead.
2. **`curiosity_coef` set to 0.0** (was 0.005) -- disables the
   contaminated reward signal entirely rather than patching it in place.
   Noted to re-enable only after also switching `collect_group`'s surprise
   computation to the same bounded (Huber) loss, not raw MSE.

Not yet re-run against the real env with both fixes together -- next step.

## 2026-08-21 (same day) — ran with both fixes, found the real culprit, and restarted Stage 2 from scratch

Ran a full 300-step pass with the Huber loss + `curiosity_coef=0.0` fixes
together. **Result: mean episode reward jumped from -1.2 to -0.677 by
step ~30, then flat-lined bit-for-bit for the remaining 270 steps.**
`jepa_loss` shot up to the clamp ceiling (50.0) by step ~60 and stayed
pinned there continuously for the rest of the run — not an occasional
outlier anymore, routine. KL and entropy both collapsed early and stopped
moving. Read as: training effectively stopped learning after step ~75.

**Root cause: `clip_grad_norm_(policy.parameters(), max_norm=1.0)` clips
across `eyes` + `intuition` + `brain` jointly, by a single global-norm
scale factor.** `jepa_loss` chronically saturated implies a huge raw
gradient norm on `eyes`/`intuition`; `grpo_loss` (~0.2-0.3) implies a tiny
one on `brain`. The scale factor needed to bring the *combined* norm down
to 1.0 is dominated almost entirely by the JEPA side — and that same
factor gets applied to `brain`'s gradient too, crushing it to near-nothing.
Stop-grad (`z.detach()` before `brain`) makes the two losses *functionally*
independent (no shared computation graph), but they were never independent
at the clipping step, which sits after both backward passes have already
accumulated into `.grad`. This fully explains the flat-line: `brain` kept
receiving a gradient, just one scaled down by whatever factor the JEPA
blowup demanded, every single step.

Likely secondary cause of the JEPA saturation itself: the early-stopped
Stage 1 checkpoint was trained only on random-walk transitions; once GRPO
started pushing rollouts into policy-driven states (archive replay,
near-win/death), the frozen-at-Stage-1-start world model had ~no
calibration there and stayed miscalibrated despite continued fine-tuning.

**Decision (user call, not a unilateral patch): stop layering fixes on
top of fixes and restart Stage 2 from scratch instead** — the user
independently flagged (looking at `dino_probe.ipynb`'s shared-PCA
comparison again) that the *trained* `Eyes` encoder's patches look less
visually legible than the random-init control's, on top of "on s'enfonce
dans un puits sans fond" (this had become a bottomless pit of one-bug-at-a-
time patches). Chose the **minimal vanilla GRPO** option over three
offered: no PPO clip/KL, no Go-Explore archive, no self-imitation replay,
no curiosity, and critically — **`eyes` frozen entirely during Stage 2, no
JEPA loss at all**. This doesn't just work around the grad-clipping bug,
it removes the class of problem structurally: with only `brain` ever
receiving a gradient, there is no second loss term to fight it for budget,
shared or otherwise.

Rewrote `jepa_pretrain.ipynb`'s Stage 2 section (`Brain`/`Policy` classes,
instantiation, `collect_group`/`compute_losses`/main loop) accordingly.
Kept from the previous version: the vectorized lockstep `collect_group`
(a real perf fix, not part of the complexity spiral), `max_steps=120`
(structural fix, not optional), the zero-variance-trap skip check, and the
finite-loss/grad safety net (cheap, prevents weight corruption). Verified
via a mock smoke test (`smoke_test_vanilla_grpo.py`) before touching the
real notebook: confirmed `eyes.weight.grad is None` after backward (not
just unchanged after a step — actually never receives a gradient at all),
`brain`'s weights do change, the zero-variance check still fires correctly,
and a multi-step loop runs without crashing. Not yet run against the real
env — next step.

## 2026-09-07 — LLM-relay track: brain-execution bugs, cross-game memory, and a human-heuristic-driven structural detector

Full-day session on the parallel LLM-relay track (`src/llm_*.py`), not the
GRPO/JEPA track above. Starting point: `UnifiedVisionAgent` (qwen3.8,
unified perception+decision) + calibration + real GPU, real Kaggle Phase B
score **0.06** (2026-09-06/07, best LLM-relay score to date). Local 25-game
benchmark going in: self-identified the avatar on 13/25 games but only
completed a level on 2-3/25 — this session's investigation was into that
gap.

### Kaggle scoring methodology (clarified, not previously documented)

Pulled from `docs.arcprize.org/methodology.md` (the Kaggle competition page
itself is a JS-rendered React app, not fetchable directly). Per-level score
= `(human_baseline_actions / ai_actions)^2`, capped at 1.15x. Per-game score
= weighted average of per-level scores, weight = 1-indexed level number
(later levels count more); an uncompleted level contributes 0 to the
numerator but its weight still counts in the denominator. **Global score =
plain average across ALL games** (55 public / 55 private) — a game that
never completes a single level scores a flat 0% and weighs exactly as much
as a game scored 100%. Implication: breadth (getting ≥1 level on more
games) is worth more than depth (shaving actions off already-won games),
at least until most games clear at least one level.

### Tooling built

- `src/benchmark_agent.py` — reusable N-game local runner (self_found /
  levels_completed / error per game), so this and future sessions stop
  rebuilding throwaway scratchpad harnesses from zero every time.
- `src/replay_reflection.py` — plays a game, then separately shows the
  model several sampled frames across the WHOLE recorded trajectory + the
  full action log, and asks it to infer the game's goal/mechanic/strategy
  in hindsight (diagnostic only, not used in real gameplay).
- `src/run_vlm_raw.py` / `src/run_vlm_raw_memory.py` — ablation agents:
  qwen3.8 with zero code scaffolding (just an image + tiny history +
  available actions), with and without a minimal single-step
  previous-frame + action-taken memory.
- `src/interactive_play.py` — lets a human play a local game turn by turn
  through the conversation (replays the action log from a fresh reset each
  call, no persistent process needed), built to capture a human's own
  problem-solving process for comparison. Tried once live in-conversation
  on `cd82`; too slow that way, user will record themselves playing later
  instead and bring notes to a future session.

### Root cause found: brain-call budget bug, and two related execution fixes

Benchmarked the 12 games where the avatar is identified but zero levels
ever complete (`cd82, cn04, g50t, m0r0, re86, sk48, sp80, tr87, tu93, wa30,
ar25, bp35`). `brain_call_count` was **exactly 4** in every one, with
`action_counter=151` — i.e. the LLM was consulted only 4 times total, then
the agent ran on pure code-heuristic autopilot for the remaining ~90+
actions regardless of how confused its own notes still were. Root cause:
`VisionToolsAgent`'s periodic/fresh-goal triggers (`llm_tools_vision_agent.py`)
gated on `ToolsAgent`'s `MAX_BRAIN_CALLS=4`, documented THERE as a
**per-level** cap, but checked against `brain_call_count`, the **whole-game**
running total — silently a 4-calls-per-entire-150-action-game budget, not
4-per-level. Fixed with a separate `MAX_BRAIN_CALLS_GAME=10` class constant.

Two more execution-layer bugs fixed the same day, all independently
verified against real gameplay/frames, none reproduced as a clean win-rate
gain in single-run local testing (see "honest results" below):
- **Direct-action-repeat ported to the main branch**: a brain-suggested
  non-spatial action was tried exactly once before the very next turn
  re-consulted the brain again, burning the whole call budget testing
  single untested actions one at a time. `ToolsAgent`'s bootstrap branch
  already had a fix for this (`DIRECT_ACTION_REPEAT=4`, retry the same
  suggested action 4x) but it had never been ported to either file's main
  goal-directed branch.
- **`state_graph.py` win-path replay wired into the main branch**: this
  generic discrete-state-graph replay mechanism (record `(state, action) ->
  next_state` transitions, mark goal states, replay a known path to one)
  previously only ran during bootstrap (self not yet identified) — a win
  discovered by BFS/brain reasoning after self-ID was never recorded for
  replay at all within the same episode.
- **`src/shared_game_memory.py`**: process-wide, thread-safe cross-game
  prior on generic ACTION1-7 role (movement vs non-spatial), shared across
  all concurrent per-game threads within one real Swarm run (confirmed via
  `agents/swarm.py`: all ~110 games run as threads in the SAME process,
  started nearly simultaneously). Surfaced to the brain prompt as an
  explicitly-labeled weak hint from other games, never substituting for a
  game's own bootstrap/calibration verification, never keyed by game_id.
  Verified thread-safe under real concurrency (4 games in parallel threads,
  0 crashes, correct vote aggregation, appropriately non-unanimous votes).
- **Generic progress/stall signal + periodic trajectory reflection**: track
  whether the current state (via `state_graph`'s own `state_signature`) has
  already appeared in a bounded recent-history window (a cycle/stall with
  no real progress), and the pixel-coverage trend of any already-detected
  HUD/bar color over time — both surfaced explicitly in the brain prompt.
  Separately, every 40 actions, sample 8 frames across the whole episode so
  far (via the base `Agent` class's own `self.frames`, already free) + the
  full action log, and rewrite `brain_notes` with a trajectory-grounded
  reflection instead of the usual single-frame guess (local-dev/Ollama
  only for now — silently skipped on the Kaggle GGUF path, which is
  single-image only today).

### Retrospective reflection experiment — and a self-correction

Ran `replay_reflection.py` on 5 games (`ls20` control + `cd82/re86/sk48/tr87`,
150 actions each). Initial read: the retrospective reflections were far more
specific/detailed than live turn-by-turn notes, seemingly confirming
"comprehension isn't the bottleneck." **A follow-up manual spot-check —
actually opening extracted GIF frames and comparing them to the reflection
text, not just re-reading the LLM's own confident prose — partially walked
that back**: 3/4 games' claims held up under visual inspection (cd82's
piece rotation, sk48's paint/brush trail, tr87's independent column
cursors + glyph cycling), but `re86`'s claim was **partly fabricated** — it
asserted a "steady downward drift, reset at step 112" that the actual
frames (0/74/147) flatly contradict (the crosses end up almost exactly
where they started; the real mid-run change was horizontal, not
vertical). Revised verdict: the model reads STATIC scene structure fairly
reliably given the full trajectory, but its causal/dynamic narrative is not
reliably grounded and can confabulate specific, confident-sounding details
— the same "verify before asserting" discipline applies to the model's own
output, not just to claims made to the user.

### Bare-qwen ablation (how much does the scaffolding buy?)

`run_vlm_raw.py` (qwen3.8, zero code tools) on `ls20/cd82/tr87`, 80 actions
each: 0/3 completed a level, including `ls20` (which the scaffolded agent
wins reliably in a similar budget). Visual inspection of the `ls20` GIF:
real spatial progress happened (the token got most of the way to the goal
by the last frame) but very inefficiently — only 8 visually-distinct
frames across 81 actions (most actions produced no visible change at all),
and at the FINAL turn the model still wrote "I have no idea what each
action does" — zero convergence on action-effect knowledge despite 80
tries, with no persistent notes and only a 4-frame rolling state-tag
history. Adding ONE piece of memory (`run_vlm_raw_memory.py`: previous
frame + action taken, shown each turn) still didn't win in the same
budget, but measurably fixed the "blind repeat" pattern: 82/82
visually-distinct frames (vs 8/82), explicit causal reasoning even at the
last turn ("ACTION2 produced no visible change, I'll try a different
action"), and the HUD progress bar reached ~90% (vs barely moving) before
plateauing around the halfway point. Independent confirmation, from a
completely different angle (ablating everything away instead of adding
more), of the same running theme: the real gap is *exploiting* a direction
once found, not understanding the game.

### User's own play heuristics → shared-palette region detector (best result of the day)

Asked the user how they personally approach an unfamiliar ARC-AGI-3
puzzle. Their answer, paraphrased: (1) sparse/rare colors are likely
interactive objects, majority/bulk colors are background/path, not the
main objects; (2) test a few movements to see what moves and how; (3) the
mechanic is usually about relating one sparse object to ANOTHER sparse
object (go to it, move/push/click/change it), not treating each object in
isolation. (1) and (2) already matched the existing code closely
(`_bulk_colors`/`_find_blobs`'s sparse/bulk split, the calibration phase) —
good independent confirmation those are sound. (3) was a genuinely missing
capability: built `_detect_shared_palette_regions` (union-find clustering
of blobs by spatial proximity, then flags region pairs sharing ≥2 colors —
a generic "legend vs workbench" signal) and `_find_unfiltered_blobs` (same
extraction as the existing `_find_blobs` but without the
`MAX_FRAGMENTS_PER_COLOR` cutoff, needed because `cd82`'s and `tr87`'s
actual legend/key colors fragment past that cutoff and were invisible to
any blob-based check before this). Wired into the already-shared
`_structural_fact_lines` path, so all 3 agent classes' prompts pick it up
with no extra plumbing. Two real bugs caught and fixed before trusting it
(clustering can transitively chain far-apart fragments into one sprawling
"region" spanning most of the grid; fixed with a span filter that only
rejects a region large in BOTH axes at once, since a real legend/header
strip is often thin in one axis but spans the full width/height in the
other). Verified against real initial frames of all 25 local games: fires
correctly and only on games with this structure (`cd82, sk48, tr87, re86,
wa30`), silent on the rest.

**Result, real 150-action 13-game benchmark
(`results/unified_sharedpalette_13game_20260907.json`)**: still 2/13
completed a level (`ls20`+`sp80`, same tally as the `state_graph` run), but
a clear qualitative jump in `brain_notes` on 3-4 of the 4 closely-tracked
stuck games: `cd82` → *"a color-matching puzzle where the center workbench
... must be made to match the top-left target"*; `tr87` → *"the top 3x2
grid defines pink->purple partner pairs ... I must cycle each purple
symbol to match its pink partner per the reference grid"* (more precise
than even the earlier whole-trajectory reflection got); `re86` → shifted
from "collect green squares" to *"pattern-alignment puzzle ... squares may
represent target patterns to match"*; `sk48` → more structured
("beam-alignment puzzle", explicitly recognizes it's stalling without
firing the beam). This is the first fix all day where improving the INPUT
(not the budget, not the retry mechanism) visibly changed what the brain
concludes, in a real full-budget run, on most of the tracked games — still
didn't convert into a new win in 150 actions, but it's the strongest
positive qualitative signal of the day, and it came directly from a
human's stated play strategy rather than another automated ablation.

### Honest bottom line

Six fixes shipped and committed (`feature/brain-persistent-notes`:
`df8aea1`, `a6ec84b`, `aa8b74e`, `7ab94b8`, `daa6391`, `780c2be`), each
independently verified correct/safe (no regression on `ls20`'s reliable
win across every run today), **none yet proven to raise the local win rate
beyond existing single-run noise** — `sp80` alone accounted for the entire
observed variance across 6 separate 12-13-game benchmark runs today (won
in runs 1, 3 [state_graph], and 6 [shared-palette]; lost in runs 2, 4, 5),
every other of the 12 stuck games scored 0 in literally every run,
regardless of which fix was active. This does not mean the fixes don't
help — each addresses a real, separately-confirmed bug or gap — it means
12 games × 1 run each is not enough data to see a win-rate signal through
that much noise. Two honest paths forward, not mutually exclusive: (a)
proper repeated-trial statistics (N≥3 runs per game per config) before
drawing more conclusions locally, or (b) judge future changes against the
real Kaggle Phase B score (55 public games, one ground truth) instead of
noisy local reruns. Nothing was submitted to Kaggle today.

## 2026-09-08 — Two new agent architectures from current field leaders (REPL harness + executable world model)

**Motivation:** rather than keep iterating the stalled CRL/goal-conditioned
track (see 2026-09-07 CRL entries — synthetic sanity check passed, real-ARC
transfer test unstable/inconclusive), asked what's actually winning on the
real ARC-AGI-3 leaderboard right now and read the two strongest publicly
documented approaches in full before writing any code (user: "essaye d'en
savoir plus sur leur publication... avant de faire quoi que ce soit", then
"travail sur les deux approche séparement et test les"):

- **Duck Harness** (Tufa Labs, current #1 Kaggle leaderboard, 11.04):
  github.com/Tufalabs/duck-harness, blog at tufalabs.ai/research/duck-harness/.
  ONE tool (`python` REPL) exposing game state as inspectable variables
  (`current_frame.ascii`/`.segmentation`, `history`, `valid_actions`,
  `action(...)`), Qwen 3.6 27B FP8. Their own stated finding: hand-crafted
  domain tools **hurt** vs. letting the model write its own inspection code.
  Their `.segmentation` view (id/color/shape-hash/pixels/boundary/adjacency)
  is functionally the same idea as our own `_find_unfiltered_blobs`/
  `_bulk_colors` (llm_tools_agent.py) — independent convergent validation of
  that earlier design. Their prompt also explicitly warns against mistaking
  a HUD/timer edge-strip for clickable objects — the exact failure mode we
  found and fixed ourselves in 2026-09-07's shared-palette work.
- **Executable World Models** (Sergey Rodionov, arXiv:2605.05138, AGI-2026):
  github.com/astroseger/arc-3-agents-baseline1. A coding agent (Codex CLI +
  GPT-5.5 "high reasoning effort") maintains an executable Python world
  model (engine/state-io/planner files), verifies it by exact-replaying
  recorded observations, refactors toward simpler abstractions each cycle
  (practical MDL-like bias), and plans by simulating through the model
  before spending a real action. 58.12% mean RHAE, 15/25 public games fully
  solved with GPT-5.5; only 41.29%/8-25 with the weaker GPT-5.4 — very
  reasoning-quality-sensitive. Their README (checked 2026-09-08, postdates
  the paper) already reports a follow-up hitting ~99% RHAE / full 25-game
  saturation with GPT-5.6-sol, explicitly flagged by the authors themselves
  as "saturation of the public set, not evidence ARC-AGI-3 is solved
  generally" — i.e. even they don't trust public-set numbers at face value.

New branch: `feature/repl-worldmodel-agents` (off `feature/brain-persistent-notes`,
so the new agents can reuse the segmentation/blob utilities already built there).

**Built, both integrated into `src/benchmark_agent.py` as `--agent repl` /
`--agent worldmodel`:**

- `src/llm_repl_agent.py` (`ReplToolsAgent`) — direct adaptation of Duck
  Harness to our stack (local Ollama, not their inference server). One
  fenced ```python``` code block per turn, exec'd in a reduced-builtins
  sandbox exposing `current_frame`/`previous_frame`/`history`/
  `valid_actions`/`action(...)`. Honest simplifications vs. the original,
  documented in the module docstring: `action(...)` only QUEUES (our
  Agent.main() contract is one real env step per choose_action() call, we
  can't let the model step the real env mid-snippet like their harness
  does), no Docker/subprocess sandbox isolation (just a restricted
  `__builtins__`), no attached image by default. `.segmentation` reuses
  `llm_tools_agent._find_unfiltered_blobs`/`_bulk_colors` directly rather
  than reimplementing.
- `src/llm_world_model_agent.py` (`WorldModelAgent`) — scaled-down
  adaptation: ONE function `predict_next_ascii(ascii_grid, action) ->
  ascii_grid` instead of three files, exact-string-match + character-level
  soft-match verification against a capped transition log, 1-ply lookahead
  planning (prefer the action predicted to change the state most, novelty
  proxy) with round-robin fallback while the model is still identity-like,
  budget-capped refinement calls (`MAX_REFINE_CALLS`) with a regression
  guard (reject a refined version if it scores worse than the current one
  on the same recorded transitions — same spirit as their replay verifier
  gating acceptance).

**Bugs caught during smoke-testing, fixed before the real test (both real,
both would have silently broken results if left in):**
1. `ReplToolsAgent` never fed the previous turn's exec error back to the
   model — it repeated the identical `TypeError: 'int' object is not
   subscriptable` (wrong assumption about the bbox format) across
   consecutive calls with zero ability to self-correct. Fixed by appending
   `self.last_error` + `self.last_code` to the next prompt when non-empty;
   confirmed empirically afterward that the model DOES recover within 1-2
   turns once it can see its own error.
2. `WorldModelAgent._run_predict` had NO runtime guard on the LLM-generated
   `predict_next_ascii` function — only compile-time syntax was checked. An
   accidental infinite loop in generated code would silently freeze the
   whole agent forever (this function runs once per legal action for
   planning, plus once per logged transition for verification, so a hang
   anywhere is a hang everywhere). Fixed with a `signal.alarm`-based 2s
   timeout per call; verified with a deliberately-infinite `while True:
   pass` stub that it now returns cleanly instead of hanging.

**Results (local qwen3.8, which `ollama ps` reveals is actually a 27.3B
model at Q4_K_M, fully resident in VRAM — same rough class as Tufa Labs'
own Qwen 3.6 27B, just running through Ollama on this machine's single GPU
instead of their production inference stack):**

| Agent | Game | Actions | Levels | Won | Notes |
|---|---|---|---|---|---|
| UnifiedVisionAgent (baseline, 2026-09-07 run) | ls20 | 150 | 1/7 | No | reference point |
| UnifiedVisionAgent (baseline, 2026-09-07 run) | cd82 | 150 | 0/6 | No | |
| UnifiedVisionAgent (baseline, 2026-09-07 run) | tr87 | 150 | 0/6 | No | |
| ReplToolsAgent | ls20 | 21 | 0/7 | No | smoke test only |
| ReplToolsAgent | tr87 | 60 | 0/6 | No | no crash, ~4.3s/action |
| ReplToolsAgent | cd82 | 60 | 0/6 | No | no crash, ~4.3s/action |
| WorldModelAgent | ls20 | 60 | 0/7 | No | 0/4 refinement calls succeeded (see below) |

Budgets are NOT matched to the baseline's 150 (60, sometimes less) —
local-model latency made a fair like-for-like comparison impractical today;
these numbers say "didn't crash, didn't yet win at a smaller budget," not
"worse than baseline at equal budget."

**REPL agent**: works end-to-end, self-corrects from sandbox exceptions
after the error-feedback fix, roughly 4.3s/action (~260s for 60 actions on
tr87/cd82). Still hits the bbox-format exec error close to half the time
on a fresh code snippet — burns real turns on failed inspection rather than
progress. No level completed at these small budgets; not evidence either
way about the mechanism's ceiling, just an honest "ran clean, too small a
budget/too few runs to see a signal" result, same caveat as every local
sweep this project has run.

**World model agent — the more decisive finding of the day**: EVERY
refinement attempt (4/4 across a 60-action ls20 run, each given a generous
240s timeout) timed out waiting for qwen3.8 to finish generating a
refined `predict_next_ascii`. The agent therefore ran the ENTIRE test in
its round-robin fallback, having never once obtained a working transition
model. All the mechanism's individual pieces (compile-check,
regression-guard accept/reject, replay verification, the new SIGALRM
timeout guard) were unit-tested and confirmed correct in isolation — the
bottleneck is pure LLM generation latency for a prompt containing two full
64x64 grids (~4KB of text each) on local hardware, not a logic bug. This
plausibly explains why the source paper needed Codex CLI + GPT-5.5/5.6
cloud-scale infrastructure rather than a claim that the approach itself is
unsound — worth retrying with a smaller/faster or code-specialized local
model for the refinement step specifically (e.g. `qwen2.5-coder:7b`,
already pulled locally) before concluding anything about the mechanism's
real value here. Not attempted today.

### Honest bottom line

Two new, functionally-complete agent architectures added, both directly
traceable to the two strongest publicly-documented ARC-AGI-3 approaches
found today, both committed on `feature/repl-worldmodel-agents`. Neither
beat the existing baseline today, but the comparison wasn't apples-to-apples
(much smaller action budgets, single runs, no repeated trials — same
"12 games × 1 run each is not enough data" caveat as 2026-09-07). The REPL
agent is immediately usable and cheap to keep iterating on. The world-model
agent's core loop is implemented and unit-verified but has not yet
completed a single real refinement cycle locally — its next step is
infrastructure (faster/smaller refinement model), not more agent logic,
before it can be judged fairly. Nothing was submitted to Kaggle today.

### Follow-up, same day: coder-model retry (world model) + baseline-matched test (REPL)

User confirmed: keep pushing on both, but prioritize the REPL/Duck-Harness
track specifically since it's the real current #1 leaderboard strategy.

**World model, retried with `qwen2.5-coder:7b` instead of `qwen3.8` (27B)
for the refinement call only:** dramatic latency fix — 6/6 refinement calls
completed in 2.7-6.5s each (vs. 4/4 timing out at 240s before), full
60-action ls20 run in 25.7s (was 1208.7s). **But this surfaced a DIFFERENT,
more fundamental problem, not solved by fixing latency**: traced the actual
learning curve call-by-call and printed the final accepted function — the
coder model's `predict_next_ascii` writes `# No change: return ascii_grid`
for literally every real action in this game's action space (ACTION1-6),
and instead spends its effort hallucinating handlers for action names that
don't exist in this game at all (`"CLICK"`, `"MOVE_UP"`, `"MOVE_DOWN"`,
`"MOVE_LEFT"`, `"MOVE_RIGHT"`) — dead code that never executes. The
regression guard correctly never rejected any of these versions because
none of them ever score worse than identity (they ARE identity for every
action that's ever actually called). The reported 93.9-98.7% "soft match"
is exactly what an identity/no-op function gets for free on a mostly-static
64x64 grid where only a small region changes per frame — not evidence of
learning. **Honest conclusion: the coder-model swap fixed the engineering
problem (latency) but did not fix the actual capability problem (the model
never engages with real transition dynamics)** — next lever here isn't
speed anymore, it's prompt quality (e.g. showing an explicit computed diff
of changed cells between before/after instead of two full walls of grid
text and expecting the model to spot the change itself) or a stronger model
for this specific abstraction task. Not attempted further today, per the
user's explicit priority on the REPL track.

**REPL agent, re-tested at the SAME 150-action budget as the baseline** (the
earlier comparison used 21-60 actions, not a fair test) — added a concrete
worked-example code block to the system prompt first, targeting the
recurring `'int' object is not subscriptable` bbox-format confusion.
Result on `ls20`/`tr87`/`cd82`, 150 actions each: **0/7, 0/6, 0/6 levels
completed — 0 wins on all three, matching baseline on cd82/tr87 but WORSE
than baseline on `ls20` specifically (baseline: 1/7 levels)**. Could not
precisely re-measure the exec-error rate for this run (a leftover `| tail
-200` on the launch command truncated most of the mid-run log before it
reached the output file — a self-inflicted logging mistake, not a result);
per-action latency did drop from ~4.3s to ~2.0s versus the pre-fix run,
suggestive but not proof the worked example reduced failed exec calls.
**Honest bottom line: our from-scratch local adaptation of Duck Harness does
not yet match, let alone beat, our own existing baseline at equal budget**,
despite Duck Harness itself being the real #1 on the actual Kaggle
leaderboard. Most likely explanations, not yet isolated: (a) Tufa Labs runs
their own production Qwen 3.6 27B FP8 inference server, not a
heavily-loaded local Ollama instance sharing a GPU with everything else on
this machine; (b) documented simplifications in our port (no live
mid-snippet real-env re-stepping, simpler sandbox, no attached image per
turn — Duck Harness's own writeup credits gains partly to multimodality,
which this port deliberately left out for a first test). Attaching an image
per turn (matching `llm_vlm_agent.py`'s existing renderer) is the most
direct untried lever if this track continues.

### Second follow-up, same day: multi-round tool loop + real root-cause bug hunt

User confirmed ("yes") both proposed fixes: matching Duck Harness's sampling
params and, more importantly, letting the model call the Python tool
MULTIPLE times per decision turn (their `LOCAL_ANALYZER_TOOL_STEPS`,
default 12 in their source) instead of once — a real architectural gap
this port had, missed despite already having read the exact prompt line
saying so ("You can call the python tool as many times as you want per
step") when first building it.

**Implemented in `src/llm_repl_agent.py`**: `MAX_TOOL_CALLS_PER_TURN`
(started at 4, cut to 2 after cost measurements below), stdout capture via
`contextlib.redirect_stdout` so each round's output/error feeds into the
next round's prompt within the SAME turn, sampling params changed to match
their Qwen defaults (temperature 0.6, top_p 0.95, top_k 20, was an
arbitrary 0.4).

**First smoke test (4 rounds/turn) was a regression, not an improvement**:
1229.8s for just 15 actions (vs. ~65-260s before), 26 exec errors logged.
New error types appeared that had never shown up before: `NameError` for
names like `cell`, `get_cell`, `char_to_color`, `ascii_str` — the model was
referencing variables/functions it had defined in an EARLIER round's code
as if the session persisted, even though the prompt said otherwise. Showing
prior-round code as context apparently reinforced the wrong mental model
("this is a continuing REPL") instead of the intended one. Also found: `ord`
was missing from the sandbox's safe-builtins whitelist, breaking any
ascii-char/color-id conversion outright.

**Fixed**: `MAX_TOOL_CALLS_PER_TURN` cut 4→2 (cost control), added `ord`/
`chr` to safe builtins, and made the "no persistence between rounds" prompt
language much more explicit + relabeled the shown transcript as "context
only, do NOT reference these names."

**Re-test after that fix: real speedup (1229.8s→156.8s) and the
cross-round-confusion errors were gone, but 16 of the remaining 17 errors
were still the exact same `TypeError: 'int' object is not subscriptable`
from before the original "worked example" fix** — meaning that earlier fix
had never actually worked. Traced the ACTUAL generated code (not
inference) via a direct instrumented run and found the real cause for the
first time: the model was writing `n["hash"][:8]` to truncate a shape hash
for display — a completely reasonable assumption that a field called
"hash" is a string-like digest — but `_shape_hash()` returned a raw Python
`int`, which isn't subscriptable. **This was never actually a bbox-format
problem** (the earlier hypothesis from the first REPL agent build), it was
a raw-int-vs-string schema mismatch the whole time; the bbox worked example
added earlier was solving a problem that wasn't the real one. **Fixed**:
`_shape_hash()` now returns a hex string (`format(..., "016x")`) instead of
a raw int — same position-independent shape signature, just sliceable like
the model already expected. Also added `hasattr`/`getattr` to safe builtins
(found via the same trace) and later `repr` (found in the final run below).

**Final 150-action, 3-game comparative run (`ls20`/`tr87`/`cd82`, matching
the baseline's budget), all fixes applied**, ~52 min total wall time:

| Game | Exec errors (out of 150 actions, ≤2 rounds each) | Levels | Won |
|---|---|---|---|
| ls20 | 1 | 0/7 | No |
| tr87 | 15 (mostly `ImportError: __import__ not found` — model kept trying to `import` despite the rule against it) | 0/6 | No |
| cd82 | 5 | 0/6 | No |

**Honest bottom line**: the exec-error rate dropped from effectively 100%
(the original bbox/hash misdiagnosis, then the multi-round regression) down
to roughly 1-10% per game — three real, confirmed bugs fixed
(cross-round variable confusion, the hash int/string schema mismatch, and
several missing sandbox builtins), each independently verified via direct
instrumented traces of actual generated code rather than guessed. **But
none of this moved the actual outcome**: still 0 levels completed on all
three games, identical to every earlier attempt today, and still worse
than the existing baseline on `ls20` specifically (which reliably gets
1/7 there). This matches a pattern already seen elsewhere in this
project's LLM-relay track (see [[llm_relay_agent_experiments]]): fixing
real, confirmed execution/mechanics bugs is necessary but has repeatedly
NOT been sufficient to close a win-rate gap — the comprehension/strategy
gap (does the model actually understand what to DO once it can reliably
inspect state) looks like the dominant remaining bottleneck for this port,
not remaining exec-error noise. The untried image-attachment lever from
the previous entry is still the most likely next thing to actually move
outcomes, since Duck Harness's own writeup credits it as a real
contributor, not just a nice-to-have.

## 2026-09-08/09 — Ran Duck Harness's REAL implementation against our local ls20

User: "je veux que tu essaies LEUR Implémentation complète" -- rather than
keep guessing at gaps from reading source, clone and actually run their
real code (`github.com/Tufalabs/duck-harness`) against our own offline
environment, using our local Ollama as the model server.

**Feasibility check, confirmed real not assumed**: their `tufa-arc-agi-framework`
(TAAF) package's `pyproject.toml` pins `arc_agi>=0.9.8`/`arcengine>=0.9.3` --
this repo's venv already has the exact same versions (`arc-agi==0.9.8`,
`arcengine==0.9.3`) installed. `TAAF.game_api.ArcadeSpec(operation_mode=OFFLINE)`
is the identical no-network/no-API-key mode this project's own agents already
use, and accepts an explicit `environments_dir` -- pointed it straight at our
own `data/environment_files`, no adapter needed. All required Python deps
(`dotenv`, `matplotlib`, `requests`) were already present.

**Set up a minimal driver** (bypassing their Makefile/`uv`/Slurm/Kaggle
deployment scaffolding entirely, which none of this local setup needs):
`taaf.game_api.GameAPI(env_name="ls20", arcade_spec=...)` +
`inference.framework.solver.HarnessSolver(model="qwen3.8:latest",
start_local_server=False)` + `taaf.benchmark.Benchmark(games=[game],
solver=solver).run()`, with `LOCAL_ANALYZER_BASE_URL=http://localhost:11434/v1`
pointed at Ollama's own OpenAI-compatible endpoint (not a real vLLM/OpenRouter
server, which is what they actually built this for).

**First real run hit a genuine interop bug, not a config mistake**: Ollama's
OpenAI-compat layer rejects `content: null` on a chat message
(`400 ... "invalid message content type: <nil>"`), which is valid on
real OpenAI/vLLM for an assistant message that's pure reasoning with no
text content (`inference/agent/tool_agent.py` line ~1900:
`assistant_message["content"] = None`). Confirmed by direct trace, not
guessed. **Patched the local clone**: `None` -> `""` at that one line.
This is a real Ollama-strictness gap, not evidence their code is wrong.

**Before the patch, the run had already reached `levels=1.0/7` in exactly
18 actions before crashing on that bug at action 19** (score 3.57 in their
own metric). This is the first time ANY agent -- ours or a port of theirs --
has completed a level of ls20 today, and did it in roughly a quarter of our
baseline's ~78 actions.

**After the patch, three more real attempts, same local model, same game,
same offline environment, no code changes between them (temperature=0.6
sampling is the only source of variation)**:

| Attempt | Actions | Levels | Notes |
|---|---|---|---|
| Pre-patch (crashed at the bug) | 18 (crashed at 19) | 1/7 | score 3.57 |
| Post-patch, capped at 30 | 30 (ran to cap) | 0/7 | state=`gave_up`, score 0.00 |
| Post-patch, capped at 150 | 18 to reach level 1, then stalled hard through ~102-103+ before being cut off after ~1.5h real wall-clock with zero further progress | 1/7 (never advanced past it) | interrupted, not a clean finish |

**Honest findings, all three genuinely new data points, not re-runs of the
same thing**:
1. **Level 1 was reached in exactly 18 actions in both attempts that got
   there at all** -- a striking, consistent number, suggesting the model
   converges on a similar (possibly near-optimal) strategy when it correctly
   understands the level, not random luck.
2. **But it is NOT reliable**: 1 of 3 real attempts got 0 levels entirely.
   This nuances (and for our specific local model, contradicts) any
   assumption that "their agent solves ls20 near-perfectly" -- that claim
   traces to a DIFFERENT system entirely (Executable World Models'
   `ewma_sv_v1.6` + GPT-5.6-sol, a closed frontier model via Codex CLI, not
   Duck Harness) and does not transfer to Duck Harness's own architecture
   running our local 27B model.
3. **No attempt advanced past level 1**, even the one given up to 150
   actions -- 84+ additional actions past the level-1 win produced zero
   further progress before the run was manually cut off for practicality
   (see next point). This matches our own baseline's pattern of "level 1
   is learnable, level 2's larger/harder layout is a much bigger jump" --
   i.e. level 2 being hard is not specific to our own agent's weaknesses.
4. **Severe, real slowdown observed on the long run**: 18 actions in 342s,
   then 84 more in ~850s (still reasonable pace), then only ~1-2 more
   actions across the next ~50+ minutes before being cut off -- confirmed
   via the real transcript file's timestamp (fresh writes throughout, so
   NOT frozen/hung, genuinely still working) and round count (153 total
   model-response rounds logged for only ~102-103 real actions, i.e. it
   was spending many tool-call rounds per turn investigating without
   committing to a real action). Root cause not fully diagnosed (could be
   context growth degrading local inference speed, could be the model
   genuinely stuck reasoning in circles on a harder obstacle) -- flagged
   as unresolved, not asserted.
5. Given point 4, **the full 3×150-action verification the user asked for
   was not completed as originally planned** -- stopped after ~1.5h on
   the first trial once it became clear a full run could take many more
   hours at this degraded rate, a practical call given how much of today
   was already spent on this whole investigation, not a technical failure.

### Honest bottom line

Confirmed the real Duck Harness code runs against our real local
environment and real local model after one small, genuine interop patch --
this is not a simulation or an inference from reading source, it is their
actual harness actually playing our actual `ls20`. It reached level 1 in
18 actions (vs. our baseline's ~78) when it worked, which is a real,
meaningful efficiency gap consistent with the architectural analysis
earlier in this log (multi-round per-turn tool calls, live multi-action
batching, no hardcoded exploration bootstrap). But it is not reliable with
our specific local model (1/3 clean failures, 0/3 progressed past level 1),
and one run degraded severely in speed for reasons not yet root-caused.
The efficiency advantage is real; the "near-perfect ls20 solve" claim
belongs to a different, closed-model system and should not be expected
here. Vendored clone (with the one-line patch) left at
`/tmp/.../scratchpad/vendor/duck-harness` -- not committed to this repo,
purely an investigation artifact, MIT-licensed third-party code.

## 2026-09-09 — Level-skip + cross-game battery on the real Duck Harness

User: "continue les tests sur plusieurs niveaux avec set level et sur
d'autres jeux aussi" -- test the real Duck Harness starting mid-game
(later ls20 levels directly) and on other games (tr87, cd82), not just
ls20 from level 0.

**Built a level-jump hook for TAAF's `GameAPI`**, same underlying mechanism
as this project's own `src/level_skip_harness.py` (`arcengine`'s
`ARCBaseGame.set_level(index)` + re-render + republish a frame so the next
`observation_space` read reflects it), adapted as a `LevelSkipGameAPI`
subclass overriding `_start_game` to call the real startup then jump.

**Validated the jump actually works, not just assumed**: TAAF's own
`actions_per_level` diagnostic field looked like it hadn't moved (still
attributing actions to slot 0) after jumping to level index 1 -- turned
out to be a red herring: that field naively attributes by `levels_completed`
value, which our jump doesn't touch (score-wise it's still "0 levels
completed", regardless of which level's sprites are rendered). Confirmed
the ACTUAL raw grid the harness received matches our own already-validated
`level_skip_harness.jump_to_level`'s level-1 grid exactly
(`np.array_equal` on the real board, not a randomized `hash()` -- caught
and fixed a mistake mid-check where Python's per-process hash
randomization on bytes made two genuinely-identical grids compare as
"different" until switched to a direct array comparison), and does NOT
match level 0's grid. The level-skip mechanism works correctly.

**Also found and used `HarnessSolver.max_runtime_s_per_game`** (a real,
documented field, not something added) as a hard wall-clock cap per game
-- necessary because two earlier attempts each stalled for 50+ minutes on
a single turn with no forward progress (confirmed via fresh transcript
timestamps that it was still genuinely working, not hung, just very slow
mid-turn reasoning). Set to 600s (10 min) per game for this battery so one
stuck game can't block the rest indefinitely.

**Battery result (local qwen3.8, 60-action cap, 600s wall-clock cap,
each a single run -- not repeated trials, same noise caveat as always):**

| Game | Start level | Actions taken | Wall time | Levels | Notes |
|---|---|---|---|---|---|
| ls20 | 1 (2nd level, cold start) | 5 | 10m43s | 0/7 | near-zero progress |
| ls20 | 2 (3rd level, cold start) | 4 | 10m00s | 0/7 | hit a real Ollama read-timeout mid-run |
| tr87 | 0 (natural start) | 23 | 10m12s | 0/6 | no level won |
| cd82 | 0 (natural start) | 23 | 10m12s | 0/6 | no level won |

**Honest findings**:
1. **ls20 levels 2 and 3, entered cold via level-skip, were dramatically
   slower per-action** (~120-150s/action) than tr87/cd82's fresh level-0
   starts (~26s/action) or ls20's own level 0 (~19s/action across earlier
   runs). Two competing explanations, NOT disentangled by this battery:
   (a) jumping in cold loses whatever mechanic-understanding the model
   would have carried forward from actually playing level 1 first
   (their own prompt explicitly encourages "Cross-level notes"), or
   (b) ls20's levels 2/3 layouts are just intrinsically much harder
   puzzles regardless of how they're reached. The EARLIER interrupted
   150-action natural-progression trial (2026-09-08 entry) also stalled
   hard once past level 1 with zero further progress across 84+ actions
   -- consistent with (b) being at least part of the story, since that
   run reached level 2 "warmed up" and still got stuck. Don't assert
   which explanation dominates without a cleaner controlled test (e.g.
   ls20 level-skip to level 1 preceded by a synthetic "already solved
   level 0" note injected into history, vs. not).
2. **First real cross-game data for Duck Harness on this project's other
   two tracked games**: 0/6 on both tr87 and cd82 within budget, matching
   this project's own agents' long-standing struggle with those two
   specific games (see [[llm_relay_agent_experiments]] -- `cd82`'s
   non-Cartesian rotational-selector mechanic and `tr87`'s
   reference-grid matching mechanic were already flagged as structurally
   hard for OUR heuristics; this is the first evidence they're also hard
   for a strong general-purpose LLM reasoning approach, not just for
   heuristics that assume simple avatar movement).
3. `max_runtime_s_per_game` is a good practical tool for any future local
   testing of this harness -- bounds wall-clock cost per game without
   needing to guess an action-count cap that might cut off a genuinely
   slow-but-working run too early.

### Honest bottom line

The level-skip mechanism works and is now reusable for future targeted
testing. But this battery raises more questions than it closes: level 2+
of ls20 remains an unsolved wall for this local-model + harness
combination whether reached naturally or directly, and Duck Harness does
not show an advantage over our own agents on tr87/cd82 specifically --
both struggle equally. Given how much of today was already spent on this
whole Duck Harness investigation (four separate live-testing sessions), a
natural stopping point for now: further work here needs either a cleaner
controlled experiment design (isolating cold-start vs. intrinsic
difficulty) or a decision to move on to a different track, not more ad hoc
single runs.

## 2026-09-09 — Root-causing the gap to their claims, the vLLM saga, and a measured multimodal test

User asked directly: "pourquoi cela ne fonctionne pas aussi bien que leur claim et
résultats?" -- a systematic accounting of every real, evidenced gap between
this local setup and Duck Harness's actual deployed config, then "corrige
tout les points le mieux possible."

**Gaps identified, each backed by something actually read/checked, not guessed:**
1. **Multimodal was never enabled** -- their `configs/inference.json` ships
   `multimodal.context=current_grid` by default; every prior test today ran
   text-only because `MULTIMODAL_CONTEXT` was simply never set in the driver
   script. A real oversight, not a hardware limit.
2. **Model precision**: their config serves `vrfai/Qwen3.6-27B-FP8`; local
   Ollama runs Q4_K_M -- meaningfully more lossy quantization.
3. **Serving infra**: their config has `enable_prefix_caching: true` on
   vLLM; Ollama has no equivalent for this workload, plausibly explaining
   the repeated severe mid-run slowdowns seen all day (context reprocessed
   from scratch every call as it grows).
4. **Statistical methodology**: their reported numbers (mean score 1.6002,
   Symbolica's 36%, etc.) are averages over `n_passes=20`; today's tests
   were 1-3 single runs -- much noisier.
5. **Time budget**: their config allows `max_runtime_minutes=45`/game;
   today's tests used 600s (10 min) caps for practicality.

User chose the most ambitious fix for every point: multimodal always on,
try a less-quantized model despite VRAM risk, attempt a real local vLLM
install despite effort/risk, and match their real n_passes/runtime config
as closely as practical.

### The vLLM attempt: three distinct, fully-diagnosed failures, correctly abandoned

Confirmed early via `nvidia-smi` that their exact model
(`vrfai/Qwen3.6-27B-FP8`) cannot run on this machine's RTX 3090 (Ampere,
compute capability 8.6) regardless of installation effort: FP8 tensor-core
acceleration requires Ada Lovelace/Hopper (8.9+/9.0), and 27B params at
1 byte/param (~27GB) exceeds the card's 24GB VRAM outright -- verified via
the model's real HuggingFace metadata (`w8a8`/`float-quantized` format,
confirmed size), not assumed. Pivoted to a 4-bit alternative.

**Safety note**: two actions were correctly blocked by the auto-mode
classifier during this work and required explicit user sign-off before
proceeding -- downloading model weights from an agent-selected unverified
HuggingFace account (first pick was `Lorbus`, an unverified account; user
redirected to the well-established `unsloth` instead), and launching vLLM
with `--trust-remote-code` (turned out unnecessary once checked --
`Qwen3_5ForConditionalGeneration` is natively registered in vLLM, no
`auto_map`/custom `.py` in the repo). Both were legitimate stops, not
false positives.

Set up an isolated venv (`.venv-vllm-DONOTCOMMIT/`, gitignored, never
touching the project's main `.venv`) specifically so a risky install
couldn't damage the working submission pipeline.

**Failure 1 -- `unsloth/Qwen3.6-27B-NVFP4` (22GB, mixed NVFP4/FP8/BF16
compressed-tensors format)**: `nvrtc: error: failed to open
libnvrtc-builtins.so.13.0` on first load attempt. Root-caused precisely
(not guessed): the library file existed inside the venv's own
`nvidia-cuda-nvrtc` pip package, just not on `LD_LIBRARY_PATH` --
fixed by exporting it explicitly. Real, second failure after that fix:
`CUDACachingAllocator... memory allocation failed with OOM`, reproduced
identically across two configurations (default settings, then
`--enforce-eager` + reduced `--max-model-len 8192` + higher
`--gpu-memory-utilization 0.96`) -- the model's mixed-precision weight
REPACKING step needs scratch memory beyond its own 22GB resident
footprint, and that doesn't fit in the ~25.3GB actually available. Not a
tuning problem; a real ceiling for this specific model on this specific
card, confirmed by two independent configurations failing at the same
point.

**Failure 2 -- `unsloth/Qwen3.6-27B-GGUF` Q4_K_M (16.8GB)**: different
failure mode entirely -- Linux OOM-killer terminated the process at
~27.5GB **system RAM** (not VRAM), confirmed via `journalctl`
(`Out of memory: Killed process ... anon-rss:27495484kB`). vLLM's GGUF
loading path apparently materializes substantially more than the file
size in host RAM before transferring to GPU, and this machine has only
31GB total system RAM.

**Failure 3 -- `unsloth/Qwen3.6-27B-GGUF` Q3_K_S (12.4GB, smaller to fit
under the RAM ceiling)**: yet another distinct failure -- vLLM 0.29.0
itself errored trying to parse the raw `.gguf` binary file as a JSON
config (`OSError: It looks like the config file at '....gguf' is not a
valid JSON file`), inside an internal `maybe_override_with_speculators`
pre-check. A real vLLM version bug/edge-case with this GGUF+separate-tokenizer
combination, not a resource limit.

**Three genuinely different, independently-confirmed blockers in a row**
(missing library -> fixed; VRAM ceiling on one format; system RAM ceiling
on another format; then a code bug on a third) is a strong, honest signal
that this specific machine (24GB Ampere GPU, 31GB system RAM) is not
well-suited to locally serving a 27B model via vLLM at any precision level
tried, independent of tuning effort. Abandoned after the user's own agreed
bounded "one last attempt" criterion was met. Cleaned up: killed all vLLM
processes, deleted both downloaded model directories (51GB combined -- disk
had filled to 100%/8.1GB free at one point), left the vLLM venv itself
installed (small, harmless) in case of a future attempt on different
hardware.

### Multimodal, actually measured this time (not just enabled and assumed to help)

Fixed `run_duck_harness_final.py` to point back at Ollama (the only
working backend) with `MULTIMODAL_CONTEXT=current_grid` set. First
verified via direct transcript inspection that the image is genuinely
reaching the model, not silently ignored: the model's own reasoning text
explicitly references visual details ("bottom-left has a blue square with
an 'L' shape", "Let me look at the image again... the orange/blue square
appears to be in the same position") -- real evidence of vision use, not
inferred from config alone.

**3-pass battery on `ls20` (150-action/20-min caps), multimodal ON:**

| Pass | Actions | Levels | Score | Notes |
|---|---|---|---|---|
| 1 | 21 (crashed) | 1/7 | 3.57 | level 1 took 20 actions; hit a real Ollama read-timeout entering level 2 |
| 2 | 31 (crashed) | 1/7 | 2.06 | level 1 took 29 actions (worse than the 22-action human baseline); another read-timeout |
| 3 | 14 (ran out of budget) | 0/7 | 0.00 | didn't even finish level 1 in 20 min |

Mean levels 0.667, mean score 1.876.

**Honest comparison to yesterday's non-multimodal 3-trial result**: 2/3
level-1 completions both took EXACTLY 18 actions (better than the 22-action
human baseline, consistent and efficient); 1/3 total failure. **With
multimodal on, the two successful completions took 20 and 29 actions --
both less efficient than the non-multimodal runs, one of them (29) actually
WORSE than the human baseline itself** -- and both multimodal runs that
progressed further hit real connection timeouts, consistent with the
already-diagnosed context-growth slowdown pattern seen all day, plausibly
made WORSE by multimodal (an image adds real payload/token weight to every
single turn, growing the context faster). **This directly contradicts this
morning's working hypothesis that missing multimodal was a likely
significant contributor to underperformance** -- enabling it did not show a
clear benefit in this small sample, and plausibly has a real cost via
faster context growth. N=3 per condition is still very noisy; this is not
proof multimodal hurts, only that it did not clearly help here, which is
itself a useful, honest correction to this morning's assumption.

### Honest bottom line

Every one of the five gaps raised this morning was investigated concretely
rather than left as speculation: multimodal is now fixed and confirmed
genuinely working (verified via the model's own visual references) but
did not show a measured benefit; the precision/serving-infrastructure gap
(vLLM) was pursued in good faith through three distinct real failures
before concluding this hardware isn't currently suited to it; the
statistical-methodology and time-budget gaps remain real and unaddressed
(today's tests are still far short of their 20-pass/45-min real config).
On the specific question "why doesn't it work as well as their claims" --
the most defensible answer after today's work is a combination of (a)
genuinely less capable local serving/precision than their production
stack, evidenced concretely rather than assumed, and (b) the small-sample
noise this project has flagged repeatedly all week, not any single fixable
bug in this port. `run_duck_harness_final.py` (Ollama + multimodal) is the
now-current best local driver for this line of testing if it continues.

## 2026-09-09 (later) — Their REAL setup, on Kaggle's actual RTX Pro 6000 grant

User: "passe sur la version en ligne on va faire leur setup complet mais
sur les gpu de kaggle" -- rather than keep working around this machine's
hardware ceiling, run Duck Harness's actual public Kaggle notebook, with
their actual FP8 weights and actual vLLM H100-class wheelhouse, on
whatever GPU Kaggle grants for this competition.

**Found their real accelerator via a community reproduction kernel's
metadata** (not guessed): `kaggle kernels pull
kevin250304/arc3-duck-v7-reproducible-baseline -m` showed
`"machine_shape": "NvidiaRtxPro6000"` -- an NVIDIA RTX PRO 6000 (Blackwell,
96GB VRAM, native FP8/FP4 tensor cores), confirming the user's own
"rtx6000" recollection and explaining why their FP8 setup works at all:
this Kaggle-granted accelerator has none of this local machine's blockers
(fits 27B at FP8 trivially in 96GB; Blackwell has real FP8 hardware
support unlike this machine's Ampere 3090). Multiple different kernel
authors use the same setup, indicating a real compute grant tied to this
competition, not one author's special access.

**All three of their referenced datasets are public**, confirmed via
`kaggle datasets list -s ...`: `jeroencottaar/taaf-kaggle-source-share`
(code bundle), `driessmit1/arc3-vllm-h100-wheelhouse-v3` (5.16GB, prebuilt
vLLM wheels), `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot` (29GB, the
actual FP8 weights). Read their real `taaf-duck-harness-kaggle-share.ipynb`
cell-by-cell -- it installs the ARC runtime from the competition
wheelhouse, mounts the bundled source dataset by marker file (not a fixed
path, learned from this project's own earlier mount-path instability
findings), and in non-submission mode plays the **bundled offline
environment files** (no gateway needed) -- exactly the reproducible,
zero-submission-cost mode needed here.

**Pushed our own kernel** (`loumitrmas/duck-harness-real-setup-offline-test`)
via `kaggle kernels push` with the same three dataset sources + the
competition data source + `"machine_shape": "NvidiaRtxPro6000"`, using
their notebook unmodified. Confirmed the CLI's own `--accelerator`/
`machine_shape` mechanism works as expected: the pushed kernel came up
with `Accelerator: GPU RTX Pro 6000` in the web UI.

**No new safety-classifier stops this time**, worth noting for contrast
with the local vLLM section above (which hit two legitimate ones) -- this
Kaggle path only used public, already-verified-via-`kaggle datasets list`
datasets and the unmodified public notebook, so there was no
agent-selected untrusted source or unauthorized flag to trigger one.

**Operational note for next time**: `kaggle kernels logs`/`kaggle kernels
output` via the CLI (v2.2.4) returned nothing for this in-progress
notebook-type kernel throughout its ~2h12m run, despite the Kaggle web UI
showing live log lines the whole time -- relied on the user's own
browser screenshots for live monitoring. Only after the kernel reached
`KernelWorkerStatus.COMPLETE` did `kaggle kernels logs` return the full
transcript. Don't assume the CLI log/output commands are useless for a
kernel type generally -- they appear to only work post-completion for
notebook kernels specifically, not scripts (unconfirmed for scripts).

**Real result, 25 public games, single pass, 2h12m38s actual wall-clock
(concurrent), mean score 1.69, median 0.25:**

| Game | Score | Levels | Actions | Game | Score | Levels | Actions |
|---|---|---|---|---|---|---|---|
| ar25 | 10.10 | 3/8 | 273 | m0r0 | 0.00 | 0/6 | 304 |
| bp35 | 0.25 | 1/9 | 520 | r11l | 4.76 | 1/6 | 81 |
| cd82 | 0.00 | 0/6 | 93 | re86 | 7.21 | 2/8 | 147 |
| cn04 | 0.00 | 0/6 | 121 | s5i5 | 0.00 | 0/8 | 51 |
| dc22 | 0.00 | 0/6 | 195 | sb26 | 2.78 | 1/8 | 64 |
| ft09 | 0.00 | 0/6 | 72 | sc25 | 0.00 | 0/6 | 237 |
| g50t | 0.00 | 0/7 | 128 | sk48 | 0.00 | 0/8 | 694 |
| ka59 | 0.30 | 1/7 | 125 | sp80 | 2.58 | 1/6 | 178 |
| lf52 | 0.31 | 1/10 | 94 | su15 | 2.22 | 1/9 | 118 |
| lp85 | 2.78 | 1/8 | 18 | tn36 | 2.99 | 1/7 | 73 |
| ls20 | 0.00 | 0/7 | 276 | tr87 | 0.00 | 0/6 | 169 |
| | | | | tu93 | 3.34 | 2/9 | 58 |
| | | | | vc33 | 2.56 | 2/7 | 121 |
| | | | | wa30 | 0.00 | 0/9 | 234 |

**13/25 games (52%) got at least one level completed** -- a completely
different picture from every local attempt today, where only `ls20` ever
worked, and unreliably (1/3 to 2/3 depending on the run). Mean score 1.69
also lands close to Duck Harness's own self-reported "mean score
1.6002" from their blog writeup (2026-09-08 entry) -- this real run is
consistent with their own published number, unlike anything achieved
locally.

**Genuinely surprising, worth stating plainly rather than smoothing
over**: `ls20` -- the one game this whole investigation has repeatedly
succeeded on locally, including with a badly under-resourced setup --
scored **0/7 here**, despite spending 276 actions on it (far more than
the ~18-20 that sufficed locally). `cd82` and `tr87` also stayed at 0,
consistent with every local finding today that these two specifically
resist this general approach regardless of resources. Real variance is
real: more compute and the correct model do not guarantee success on the
specific game a smaller setup happened to solve, and don't fix every
game either.

### Honest bottom line

This directly and conclusively answers the day's central question
("pourquoi cela ne fonctionne pas aussi bien que leur claim"): **it was
the local model/hardware gap, not the architecture.** Given the actual
model and actual appropriate hardware Duck Harness was built for, it
reaches a mean score matching their own published number and succeeds
broadly across more than half the public game set -- something no local
configuration tried today came close to. The REPL/multi-round-tool-call
architecture itself was never the bottleneck; this machine's inability to
serve a 27B FP8 model with real serving optimizations was. `cd82`/`tr87`
resisting even the real setup, and `ls20` failing here despite past local
success, are the two most important nuances not to lose in that headline
finding.

## 2026-09-09 (later still) — Critical-factor ablation: isolating what actually matters

User: "prépare un Long benchmark sur un nombre de jeux témoin (pas beaucoup)"
to isolate which specific factor (model precision, weight count, sandbox
tools) is actually responsible for the Kaggle-vs-local gap, rather than
resting on the combined "hardware+model" finding above. 4 witness games:
`ls20`, `cd82`, `tr87` (heavily tested all day) + `re86` (a positive
control that worked well on the real Kaggle run). Designed as staged
one-factor-at-a-time isolation, not full factorial, given real time/quota
cost. User: run everything, in whatever order is smartest; also: run the
Kaggle-dependent stages in parallel with the local one to save time.

### Stage 3 (tools/sandbox), COMPLETE -- local, no Kaggle quota spent

Same weak local model (Ollama qwen3.8, Q4_K_M) and same ~60-action/no-multimodal
budget throughout, only the AGENT/TOOLS differ:

| Condition | ls20 | cd82 | tr87 | re86 |
|---|---|---|---|---|
| A: our own baseline (`UnifiedVisionAgent`, hardcoded tools) | 0/7 | 0/6 | 0/6 | 0/8 |
| B: our simplified REPL port (`llm_repl_agent.py`) | 0/7 | 0/6 | 0/6 | 0/8 |
| C: the REAL unmodified Duck Harness code | **1/7** | 0/6 | 0/6 | 0/8 |

**Clean, isolated result: only condition C won anything, on the exact same
weak model everything else used.** This directly confirms the
tools/sandbox architecture itself has real, independent value -- it is
not purely a "better model" story. Condition B (our own REPL port) was
also markedly slower than both A and C (874-1225s/game vs. A's 71-249s and
C's 461-958s) and hit far more sandbox exec errors, consistent with
[[repl_worldmodel_agents_2026-09-08]]'s known gaps versus the real
implementation (single/dual tool-call-per-turn cap, in-process exec,
regex-parsed code blocks vs. their subprocess sandbox, native tool-calling,
much larger per-turn tool-call budget). `cd82`/`tr87`/`re86` staying at 0
across all three conditions again confirms these resist the general
approach regardless of tooling sophistication, consistent with every
other finding today.

### Stage 1 (model size) and Stage 4 (serving infra), Kaggle -- real engineering
detour, most of it now resolved

Read the actual (public) `setup_commands.json` from the `jeroencottaar/
taaf-kaggle-source-share` dataset to get the REAL, exact vLLM invocation
their production pipeline uses (not guessed) -- confirms
`LOCAL_ANALYZER_TOOL_STEPS=0` really does mean unlimited tool calls per
turn (not literally zero, as hypothesized on 2026-09-08), the exact flag
set (`--tool-call-parser qwen3_coder --reasoning-parser qwen3
--enable-prefix-caching --default-chat-template-kwargs
'{"preserve_thinking": true}'`), and that `arc-agi` installs from the
competition's offline wheelhouse.

Built a combined custom kernel (not their notebook -- needed full control
over which model/precision to load) reusing their proven vLLM wheelhouse
dataset. Two real, safety-classifier-correct stops during setup: an
agent-selected unverified HF account for a GGUF download (redirected to
`unsloth`, an established publisher, per the same rule as the earlier
vLLM section), and creating a brand-new Kaggle dataset from this project's
own `data/environment_files/` without prior explicit authorization for
that exact destination -- paused, explained, got explicit "la meilleure
selon toi" go-ahead before proceeding (private dataset,
`loumitrmas/arc-agi-3-offline-environment-files`, 267KB zipped, public
competition game definitions only, nothing sensitive).

**Two real, fully-diagnosed infrastructure problems found and fixed:**
1. `competition_sources` in `kernel-metadata.json` made every push fail
   with an opaque `400 Client Error` and zero detail (unlike the earlier
   `taaf-duck-harness-kaggle-share` push, which used the identical field
   successfully) -- root cause not identified, but isolated by bisection
   (removing it fixed the push) and worked around by uploading the small
   (4.2MB local, 267KB zipped) `environment_files` as our own dataset
   instead of depending on the competition attachment.
2. **`/kaggle/working` has a fixed 21GB quota, confirmed via
   `shutil.disk_usage`** -- the first combined D1+D4a+D4b run cascaded:
   vLLM's own site-packages install alone consumes ~10GB, the 8B model
   download used another ~8GB, leaving only ~10GB free by the time D4a
   tried to download a 16.8GB GGUF (`Not enough free disk space` ->
   corrupted partial download -> `File reconstruction error` -> D4b
   cascaded failure from the missing GGUF path -> final "No space left on
   device" during the teardown notebook-conversion step). Added explicit
   `shutil.disk_usage` logging and `shutil.rmtree` cleanup between stages
   -- confirmed the numbers precisely (10.2GB free after D1+cleanup, need
   16.8GB for the GGUF) rather than guessing a fix. **Given vLLM's ~10GB
   footprint plus a 16.8GB Q4_K_M GGUF structurally cannot both fit in
   21GB, Stage 4 (serving infra) was DROPPED from this run rather than
   forced through with a much-lower-quality ~9GB quant that would
   confound precision with serving infra** -- left as a deliberately
   separate, not-yet-attempted follow-up if this axis is still wanted.

**D1 (model size, 8B FP8 vs. the 27B FP8 anchor) also hit a real, distinct
bug on the first two attempts**: vLLM's OpenAI server never became ready
within 900s despite the model being much smaller than the 27B one that
loaded fine the day before. Root cause not fully isolated before the
day's session ended, but the fix applied (before final results were in)
was reverting to the EXACT flag set from the real production
`setup_commands.json` (`--tool-call-parser qwen3_coder
--reasoning-parser qwen3 --generation-config vllm
--default-chat-template-kwargs '{"preserve_thinking": true}'`, rather
than the simplified `--tool-call-parser hermes` this script had guessed)
-- plausible that the parser mismatch was silently stalling server
startup. **Check this memory/DESIGN_LOG's next entry or the actual kernel
result for whether this fix worked before trusting the model-size
ablation's outcome; not confirmed as of this entry.**

### Stage 1 follow-up: the real cause was never the flags -- wrong GPU allocated

The flag fix (v4 push) revealed the ACTUAL error immediately instead of a
bare timeout: `pydantic_core._pydantic_core.ValidationError: ... The
quantization method fp8 is not supported for the current GPU. Minimum
capability: 75. Current capability: 60.` -- and `nvidia-smi` in the same
log confirmed the allocated GPU was a **Tesla P100-PCIE-16GB (Pascal,
compute capability 6.0)**, not the requested `NvidiaRtxPro6000`. The
`machine_shape` field in `kernel-metadata.json` was silently NOT honored
for this kernel, even though it was set identically to the one working
kernel from earlier today.

Added `_assert_rtx_pro_6000()` -- a fast nvidia-smi check at the very
start of the script, aborting in seconds instead of burning ~20 minutes
on a doomed-from-the-start install+load. Hypothesized `kernel_type:
"script"` might not honor `machine_shape` the way `"notebook"` does (the
one working RTX Pro 6000 kernel was a notebook) -- converted this script
into a minimal single-cell notebook and re-pushed (v5). **The fast-fail
check worked exactly as designed (15s, not 20 minutes) but still reported
a Tesla P100** -- ruling out kernel_type as the actual variable.

**New, more likely hypothesis, not yet tested**: the one kernel that DID
get the real RTX Pro 6000 (`loumitrmas/duck-harness-real-setup-offline-test`,
2026-09-09 earlier) had `competition_sources: ["arc-prize-2026-arc-agi-3"]`
attached; this ablation kernel does not (removed earlier after it caused
an opaque `400 Client Error` on push, worked around with a private
`environment_files` dataset instead -- see above). The RTX Pro 6000 may be
a genuine per-competition compute grant that Kaggle only allocates to
kernels actually attached to that competition, not a general account-wide
perk -- a kernel without `competition_sources` may simply fall back to
the platform's standard free-tier GPU (P100) regardless of `machine_shape`.
**Not confirmed** -- the `400 Client Error` on `competition_sources` still
has no known root cause; whether it's the same underlying issue or a
separate, unrelated bug is unknown. Test this hypothesis specifically
(get `competition_sources` working, even if it means a different dataset
combination) before spending more quota on Stage 1/4.

### Honest bottom line for the ablation benchmark, end of day

**Stage 3 (tools/sandbox) is a clean, complete, valuable result**: holding
the model fixed (weak local Ollama qwen3.8), only the real unmodified Duck
Harness code won anything (`ls20` 1/7) -- neither our own baseline nor a
simplified REPL port did. This is real, independent evidence that the
tools/sandbox architecture itself matters, not just model quality.

**Stage 1 (model size) is unresolved, not negative** -- five Kaggle kernel
pushes today hit five distinct, each individually real and diagnosed
infrastructure problems (an unexplained `competition_sources` 400 error,
a `/kaggle/working` 21GB disk quota too small for vLLM+a 16.8GB GGUF
together, silently-wrong vLLM flags causing an opaque timeout instead of
the real FP8-unsupported error, and finally a GPU allocation that didn't
honor `machine_shape` at all) without ever getting the 8B model to
actually run a single game. **This is not evidence the model-size
hypothesis is wrong -- it is evidence this specific Kaggle automation
path needs more infrastructure work before it can produce a trustworthy
result.** Stopped here for today per the user's own call, given the
mounting infra friction; Stage 3's result stands on its own as today's
solid ablation deliverable. Stage 4 (serving infra) was deliberately
dropped earlier given the disk-quota finding, not attempted at all today.

## 2026-09-10 — Stage 1, four more real fixes, and a final clean-but-empty result

User: "trouve des solutions pour kaggle c'est le plus important" -- kept
pushing on Stage 1 rather than accepting yesterday's stop. Root-caused and
fixed the `competition_sources` 400 error via bisection (not guessed):

**Found the real constraint: `competition_sources` requires
`enable_internet: false`.** Tested by pushing an otherwise-identical
kernel with `enable_internet: false` -- succeeded immediately. Kaggle
rejects internet-enabled kernels attached to a competition, which tracks
with this competition's "no internet during evaluation" rule bleeding
into kernel validation generally. This explains the entire earlier
mystery: every prior push had `enable_internet: true` (needed for the
HuggingFace model download) alongside `competition_sources`.

**Fix, following the same pattern already used for `environment_files`**:
pre-download `Qwen/Qwen3-8B-FP8` locally (8.9GB) and re-upload it as a new
private Kaggle dataset (`loumitrmas/qwen3-8b-fp8-snapshot`) instead of
pulling it at kernel runtime -- removes the need for `enable_internet` at
all. With `competition_sources` + `enable_internet: false` + the
pre-staged model dataset, pushed a fresh kernel id
(`loumitrmas/ablation-model-size-v3`).

**Two more distinct, real bugs found and fixed in sequence, each via an
actual error message, not guessed:**
1. First run got past the GPU check (implying it DID get the real RTX Pro
   6000 this time -- FP8 loaded and the vLLM server came up cleanly,
   which a P100 would have rejected outright) but then failed every game
   identically with `RuntimeError('asyncio.run() cannot be called from a
   running event loop')`. Root cause: Jupyter/papermill notebook execution
   already runs its own asyncio event loop, so a plain top-level
   `asyncio.run(bench.run())` -- which works fine as a plain script --
   fails when executed as a notebook cell. Fixed with a small stdlib-only
   helper (`_run_coro_isolated`) that runs the coroutine via `asyncio.run`
   inside a fresh thread, avoiding any dependency on `nest_asyncio` (this
   kernel has no internet, so an unavailable pip package would be a dead
   end) -- verified locally first with a synthetic nested-event-loop test
   before spending more Kaggle quota on it.
2. Second run got past both the GPU check AND the asyncio fix (confirmed
   via real HTTP traffic to the analyzer in the logs) but then every
   single analyzer call failed with `400 ... "/kaggle/input/
   qwen3-8b-fp8-snapshot is not a multimodal model"`. Root cause: this is
   a genuine experimental-design gap, not an infra bug --
   `MULTIMODAL_CONTEXT=current_grid` is set (matching the real production
   config used for the 27B FP8 anchor run), but plain `Qwen/Qwen3-8B-FP8`
   is text-only (unlike the `Qwen3.6-27B` family, which is natively
   `image-text-to-text`). Every analyzer call was rejected outright,
   burning the full 15-minute-per-game timeout on 4/4 games with **zero
   actions taken on any of them** -- a clean, total washout, not a
   negative performance signal.

**Final honest status for Stage 1 (model size)**: seven distinct real
problems diagnosed and mostly fixed across this two-day pursuit
(`competition_sources`+internet conflict, disk quota, wrong vLLM flags,
GPU allocation not honoring `machine_shape` twice, the notebook asyncio
conflict, and finally this multimodal/text-only model mismatch) -- six of
seven are now genuinely resolved and reusable for a future attempt, but
the actual model-size question remains **unanswered**: 0 actions on all 4
witness games is not evidence about 8B vs. 27B capability, only evidence
that a text-only model can't be tested through a harness hardcoded to
attach images. **A real retry needs a genuinely multimodal small model**
(e.g. `Qwen3-VL-8B`, seen earlier today's HF searches) pre-staged the
same way. Stopping here for real this time -- the infrastructure path is
now well-understood and mostly reusable (GPU check, thread-isolated
asyncio runner, competition_sources+no-internet+pre-staged-dataset
pattern), so a future attempt should be much faster, but today's actual
Stage 1 data point is still zero.
