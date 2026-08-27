# ARC Prize 2026 — ARC-AGI-3 — Project Guideline

Context file for future sessions. Read this first before touching the project.

## What this competition actually is

**Not a static dataset competition.** ARC-AGI-3 is an **interactive agent** benchmark:
you write an agent that plays a sequence of small games (`environment_files/`) by
choosing actions, and it's scored on how many games/levels it completes. There is
no train.csv/test.csv to fit a model against.

Kaggle competition: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3

## Environment

- Python **3.12.13** via pyenv, pinned in `.python-version`. Must be 3.12 — the
  offline wheels Kaggle bundles (`data/arc_agi_3_wheels/`) are built `cp312`.
- venv in `.venv/` (gitignored). Jupyter kernel registered as `arc-agi-3`.
- `requirements.txt` covers: the agent framework core (`arc-agi`, `arcengine`,
  pinned to match the offline wheels), the optional LLM-agent stack
  (`langchain`, `langgraph`, `smolagents`, `openai` — dev-only, not bundled
  offline by Kaggle), and notebook/dev tooling (`jupyter`, `kaggle`, `torch` CPU,
  `scikit-learn`).
- Kaggle API token stored at `~/.kaggle/access_token` (new-style KGAT token, not
  the old username+key `kaggle.json`). Already authenticated, works transparently.

To rebuild from scratch:
```bash
pyenv install -s 3.12.13
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name arc-agi-3 --display-name "Python (arc-agi-3)"
```

## Directory structure

```
.
├── data/                       # gitignored — downloaded via `kaggle competitions download`
│   ├── ARC-AGI-3-Agents/       # reference agent framework (has its own .git, vendored as-is)
│   │   ├── main.py             # CLI entrypoint: python main.py --agent=X --game=Y
│   │   ├── agents/
│   │   │   ├── agent.py        # base Agent class
│   │   │   ├── swarm.py        # multi-agent orchestration (uses arc_agi.Arcade)
│   │   │   └── templates/      # 8 example agents: random, langgraph_*, llm_agents,
│   │   │                       # multimodal, reasoning_agent, smolagents
│   │   └── .env                # OPERATION_MODE=offline already set by us (see Gotchas)
│   ├── environment_files/      # 25 playable games, each <id>/<version_hash>/{metadata.json,<id>.py}
│   └── arc_agi_3_wheels/       # 31 offline pip wheels (cp312) for arc_agi/arcengine + deps
├── notebooks/
│   └── starter.ipynb           # working example: plays ls20 fully offline, no API key needed
├── src/                        # empty so far — put shared code here if the agent grows
├── requirements.txt
├── README.md                   # setup + how-to instructions (English... actually French, see note)
├── RULES.md                    # full official Kaggle rules text, gitignored (3rd-party copyright)
└── GUIDELINE.md                # this file
```

Note: `README.md` is currently written in French (matches how the project was set up).
This file (`GUIDELINE.md`) is the English reference for future context/sessions.

## Two operation modes

Set via env vars read by `arc_agi.Arcade()`:

| | `OPERATION_MODE=offline` | `OPERATION_MODE=online`/`normal` |
|---|---|---|
| Network needed | No | Yes, hits `three.arcprize.org` |
| Auth needed | No | `ARC_API_KEY` (separate account at https://three.arcprize.org/) |
| Games source | Local `environment_files/` | Downloaded/played on the official server |

**We deliberately have not created an ARC_API_KEY account** (user's choice — offline-only
for now). Everything so far has been validated in offline mode.

## Gotcha: `main.py` is not fully offline-capable as shipped

`data/ARC-AGI-3-Agents/main.py` always calls the **online** `/api/games` endpoint to
resolve/validate the `--game` argument, regardless of `OPERATION_MODE`. Without a valid
`ARC_API_KEY` it fails with `401 unauthorized` → `No games available to play`, even
though the actual gameplay engine (`arc_agi.Arcade` with `OperationMode.OFFLINE`)
works perfectly locally.

**Workaround used**: bypass `main.py`/`Swarm` entirely and call `arc_agi.Arcade` +
`arcengine.GameAction` directly, as done in `notebooks/starter.ipynb`:

```python
import os
os.environ["OPERATION_MODE"] = "offline"
os.environ["ENVIRONMENTS_DIR"] = "data/environment_files"

from arc_agi import Arcade
from arcengine import GameAction

arc = Arcade()
env = arc.make("ls20")
frame = env.reset()
frame = env.step(GameAction.ACTION1)
```

## Gotcha: vendored Pillow/LangGraph bug (already patched)

`data/ARC-AGI-3-Agents/agents/templates/langgraph_thinking/vision.py` uses
`ImageDraw.Coords` as a runtime type annotation. In Pillow 12.2.0 (the version
pinned by Kaggle's offline wheels), `Coords` only exists under `TYPE_CHECKING`,
so importing the module crashes with `AttributeError`. Since `agents/__init__.py`
eagerly imports *all* templates, this blocks importing `agents` at all — including
the plain `random` agent.

**Fix applied** (local only, this file lives in gitignored `data/`, so the patch
is not persisted if you re-download): added `from __future__ import annotations`
at the top of `vision.py`. Re-apply this if you re-run
`kaggle competitions download` and re-extract the zip.

## The 25 games — rough complexity ranking

Not a difficulty measure (game logic complexity ≠ puzzle-solving difficulty), just
engine code size / sprite count / level count as a proxy for where to start:

**Simplest to explore first**: `cd82` (781 lines, 13 sprites, 6 levels), `sp80` (874, 18, 6),
`m0r0` (908, 23, 6), `sk48` (986, 16, 8), `tr87` (1102, 36, 6), `sb26` (1152, 21, 8).

**Avoid at first** — much larger: `dc22` (10,875 lines), `lp85` (21,429 lines),
`ka59` (**41,446 lines**, ~20x the average).

`metadata.json`'s `baseline_actions` field (6-10 ints) looks like a short demo
snippet, not a full solution — don't treat it as a difficulty signal.

Game action modality (`tags` in `metadata.json`): `keyboard`, `click`, or
`keyboard_click` (both) — relevant when designing the agent's action space per game.

## Competition rules — key points (full text in gitignored `RULES.md`)

- **No internet access during Kaggle evaluation** (no GPT/Claude/API-based systems)
  → any pretrained model/weights must be downloaded ahead of time and bundled into
  the notebook/dataset, not fetched at eval time.
- **External data/models are allowed by default** ("acceptable unless specifically
  prohibited by the Host"), subject to a Reasonableness Standard: reasonably
  accessible to all participants, minimal cost.
- **Open source requirement for prize winners** (Section 2.5): your own code must be
  CC-BY 4.0 / OSI-approved with no commercial-use restriction. Third-party
  dependencies need at least a permissive open-source license (Apache-2.0, GPLv3, etc).
  **Carve-out**: pretrained models/data with an *incompatible* license (e.g. custom
  research licenses like Meta's DINOv3 license) are explicitly exempted from this
  open-sourcing obligation if used to generate a winning solution — you just don't
  open-source *that* component; everything else you built still must be.
- Max 1 submission/day, up to 2 Final Submissions for judging, teams up to 8.
- Total prizes: $850,000 ($150k top-score track incl. 2026-06-30 and 2026-09-30
  milestones, $700k bonus for 100% accuracy split among top 5).

## Open items / not yet done

- No `ARC_API_KEY` — online mode untested, by user's choice.
- `notebooks/starter.ipynb` only demonstrates the offline gameplay loop with random
  actions; no real agent/scoring logic yet.
- No `kernel-metadata.json` yet for `kaggle kernels push` (needed to actually submit).
- `data/ARC-AGI-3-Agents/agents/README.md` and `agents/templates/README.md` haven't
  been reviewed in detail — worth reading before picking an agent template to build on.
- Candidate model architecture (from-scratch grid encoder + action-conditioned
  JEPA world model + GRPO-trained policy) is sketched in `ARCHITECTURE.md`,
  not yet implemented.
