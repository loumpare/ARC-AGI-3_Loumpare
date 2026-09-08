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
