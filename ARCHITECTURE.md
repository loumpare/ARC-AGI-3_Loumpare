# ARC-AGI-3 — Candidate Architecture Sketch (Grid-JEPA + GRPO)

Design sketch, not yet implemented. Supersedes the original "DINOv3 + I-JEPA" idea
(see rationale below) — kept here so future sessions don't re-litigate it.

## Why not a natural-image pretrained encoder (DINOv3 etc.)

Checked the actual frame format: `env.step()` / `env.reset()` return
`frame.frame` as a `(1, 64, 64)` array of **discrete integers 0-15** (a small
fixed color palette), identical shape across games (`ls20`, `cd82`, `sp80`,
`tr87` all confirmed 64x64). This is a symbolic/categorical grid, not a
photograph — flat color blocks, hard edges, no texture, no lighting, no
natural-image statistics. See `docs/examples/cd82_example.png`: a rotating
rectangle sprite (ACTION3 spins it 45°) sitting on flat gray/green/yellow
fields. A photo-pretrained backbone like DINOv3 (self-supervised on web-scale
natural images) has never seen anything like this distribution — its
inductive biases (texture/object semantics, photographic invariances) don't
transfer, and it would need to be transplanted in rather than genuinely
warm-starting the model. Standard ARC-style approaches instead tokenize the
grid directly (one token per cell/patch of discrete color ids).

Dropping DINOv3 also sidesteps the custom-license carve-out question noted in
`GUIDELINE.md` (Meta's DINOv3 license) — training everything from scratch
means the whole stack can be openly licensed without needing that exemption
for prize eligibility.

## Action space (confirmed from `arcengine.GameAction`)

- `RESET` (0), `ACTION1`-`ACTION5`, `ACTION7`: `SimpleAction`, no payload.
- `ACTION6`: `ComplexAction`, carries `(x, y)` click coords, each `0-63`
  (matches the 64x64 grid — this is a "click somewhere on the board" action).
- Each `FrameData.available_actions` lists which action ids are legal in the
  current state — used to mask the policy's logits.

## Pipeline

```
                         +-------------------------+
 raw frame (1,64,64)     |   Grid Tokenizer         |
 int grid, values 0-15 ->|  8x8 non-overlap patches |
                         |  + learned color-id emb  |
                         |  + learned 2D row/col PE |
                         +------------+--------------+
                                      | 64 tokens
                                      v
                         +-------------------------+
                         |  Context Encoder (ViT-S) | <- trained from scratch,
                         |  from scratch, ~10-20M   |    no external weights
                         +------------+--------------+
                                      | z_t (tokens + pooled [CLS])
                        +-------------+-------------+
                        v                            v
          +------------------------+   +----------------------------+
          |  Action-conditioned    |   |  Policy / Value head        |
          |  JEPA Predictor        |   |  ("reasoning" transformer)  |
          |  z^_{t+1} = P(z_t,a_t) |   |  - action-type logits       |
          |  target: EMA encoder   |   |    (masked by available_    |
          |  of real frame_{t+1}   |   |    actions)                 |
          |  loss: stop-grad L2    |   |  - pointer head over 64     |
          +------------------------+   |    tokens for ACTION6 (x,y) |
                                        +----------------------------+
```

## Stage 1 — self-supervised pretraining (JEPA idea, on our own domain)

Play all 25 offline games with a random/heuristic policy, log
`(frame_t, action_t, frame_{t+1})` transitions. Train encoder + predictor:
online encoder produces `z_t`, EMA target encoder produces `z_{t+1}`
(stop-grad), predictor takes `(z_t, action_t)` and is pushed to match
`z_{t+1}`. This is action-conditioned next-latent prediction (closer to
V-JEPA's recipe than image-I-JEPA's masking), and it doubles as a cheap
**world model** — can roll forward in latent space without touching the real
engine.

## Stage 2 — policy training with GRPO instead of PPO

- Group unit: fix a starting state (game + level checkpoint reachable via
  `RESET`/replay), sample **G** action-sequences from the current policy from
  that same state.
- Score each rollout with a scalar reward from `levels_completed`/
  `win_levels` (+ small step penalty for efficiency).
- Advantage is **group-relative**: `(r_i - mean(r_group)) / std(r_group)` —
  no learned critic needed. Matters here because reward is sparse/delayed
  (only fires on level completion), which makes a PPO-style value function
  noisy to train.
- Update with the usual clipped-ratio + KL-penalty-to-reference-policy
  machinery, just with the group-relative advantage swapped in for GAE.
- Payoff of pairing with the Stage-1 world model: generate most of the G
  rollouts in **latent imagination** via the predictor (cheap, no real env
  steps), only spend real env steps on the top-1/top-k candidates. JEPA
  supplies cheap groups, GRPO turns groups into a gradient without a critic.

## Open questions / caveats (unverified, check before relying on them)

- Whether `RESET` is "free" at real eval time or counts against
  score/budget — scorecards are time-windowed
  (`MAX_OPEN_FOR_MINUTES`/`STALE_MINUTES` in `arc_agi/scorecard.py`) and
  `RESET` is a legal in-episode action, which is suggestive but not
  confirmed to mean eval-time group rollouts are viable, not just
  training-time ones.
- Whether these games have any hidden randomness — GRPO's group-relative
  advantage assumes rollouts from an identical starting state are actually
  comparable.
- Imagined rollouts via an imperfect world model compound error over
  multi-step horizons — validate k-step latent prediction error on held-out
  trajectories before trusting the predictor to generate GRPO groups; fall
  back to real-env rollouts if it's not tight enough.

## Suggested build order (not started)

1. Simple baseline first: small transformer policy trained directly on raw
   grids via PPO (or even GRPO without the JEPA world model), to validate the
   action loop and reward signal end-to-end before investing in the two-stage
   architecture.
2. Grid tokenizer + from-scratch ViT-S encoder.
3. Stage 1 JEPA pretraining on logged transitions from the 25 offline games.
4. Stage 2 GRPO policy training, real-env rollouts only.
5. Add latent-imagination rollouts once world-model accuracy is validated.
