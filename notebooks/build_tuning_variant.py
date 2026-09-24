#!/usr/bin/env python3
"""Build a Cell-13-only variant of the real Duck Harness submission notebook.

The submission notebook is Tufa Labs' public notebook, used unmodified except for
Cell 13 (its sanctioned "customization hook"). This script swaps ONLY that cell's
source and writes a ready-to-push kernel directory, so every tuning experiment is
a single-variable change against the same pristine baseline.

Usage:  python3 build_tuning_variant.py <variant> [--out DIR]
        variants: baseline | B_timeout | C_adaptive | D_both | E_batching

Why these levers (all measured on the v6 baseline dev run, 25 public games):
  - every one of 25 games burned exactly one 900s analyzer timeout (11.4% of its
    7920s budget) -> lever B
  - all 25 games were killed by the per-game cap with ZERO finishing naturally,
    while only ~8.8h of the 12h kernel budget was used -> lever C
  - actions per analyzer turn ranged 2.1 (ft09) to 7.1 (wa30) -> lever E
"""
from __future__ import annotations

import argparse
import ast
import json
import shutil
from pathlib import Path

# The pristine notebook + metadata pulled from the live kernel.
SOURCE_NOTEBOOK = Path("/tmp/baseline-submission/arc-agi-3-real-notebook-submission.ipynb")
SOURCE_METADATA = Path("/tmp/baseline-submission/kernel-metadata.json")
CELL13_INDEX = 13
NOTEBOOK_NAME = "arc-agi-3-real-notebook-submission.ipynb"

BASELINE_CELL13 = (
    "# Make one-off changes to `bm`, `bm.games`, or `bm.solver` here before the run starts.\n"
)

# --- Lever B ---------------------------------------------------------------
# Measured: 25/25 games logged exactly one "Read timed out" at analyzer_timeout=900s.
# A typical *successful* analyzer call took ~83s (~660 generated tokens at the ~8 tok/s
# each game gets under 28-way concurrency), so 300s is ~3.6x normal -- short enough to
# stop bleeding 900s, long enough not to kill healthy calls. A RequestException ends
# only that analysis turn; the play loop immediately starts a fresh one, so recovery
# is automatic and a shorter timeout returns to play sooner.
#
# MEASURED REGRESSION at 300s (kernel v7, D_both): 199 timeouts across 25 games
# (16.6h of cumulative wasted time, up from 6.25h at 900s) -- 300s was too
# aggressive under real contention variance, not just too short on average.
# ft09 alone burned 79% of its whole per-game budget on failed calls.
def _lever_b(timeout_s: float) -> str:
    return (
        f"bm.solver.analyzer_timeout = {timeout_s}\n"
        'print(f"[cell13] analyzer_timeout={bm.solver.analyzer_timeout}", flush=True)\n'
    )


LEVER_B = _lever_b(300.0)
LEVER_B_600 = _lever_b(600.0)

# --- Lever C ---------------------------------------------------------------
# The per-game cap must be computed at RUN time, not here: Cell 13 runs before
# Cell 15 assigns `bm.games`, so the game count is unknown at this point. Wrapping
# `bm.run` defers the calculation until the gateway's real game list exists, which
# also makes it adapt to the unknown hidden-set size (55 or 110 games) instead of
# hardcoding an assumption. Benchmark is a plain dataclass (no __slots__), so the
# instance attribute shadows the bound method -- verified locally.
LEVER_C = """import math as _math, time as _time

_solver = bm.solver
_KERNEL_BUDGET_S = 12 * 3600   # Kaggle's hard kernel limit
_RESERVE_S = 3000              # teardown + diagnostics + parquet write + slack
_CAP_FLOOR_S = 5400
# Keep dev runs affordable: a 25-game dev run is a single wave, so the run's
# wall-clock == the cap. Only the real submission gets the full ceiling.
_CAP_CEILING_S = 14400 if TRUE_SUBMISSION else 11000

_orig_run = bm.run


async def _run_with_adaptive_cap(*args, **kwargs):
    n_games = len(bm.games)
    waves = max(1, _math.ceil(n_games / max(1, _solver.concurrency)))
    elapsed = _time.time() - NOTEBOOK_START_EPOCH
    cap = (_KERNEL_BUDGET_S - elapsed - _RESERVE_S) / waves
    cap = max(_CAP_FLOOR_S, min(_CAP_CEILING_S, cap))
    _solver.max_runtime_s_per_game = float(cap)
    print(
        f"[cell13] games={n_games} conc={_solver.concurrency} waves={waves} "
        f"elapsed={elapsed:.0f}s cap={cap:.0f}s (was 7920)",
        flush=True,
    )
    return await _orig_run(*args, **kwargs)


bm.run = _run_with_adaptive_cap
"""

# --- Lever C, big-cap variant (~6h dev ceiling) -----------------------------
# D_both/F_timeout600 both used the ~11000s dev ceiling and neither clearly
# beat baseline (see sb26_variance_and_new_levers memory: sb26 stalled at
# ~240 ungrounded-guessing actions regardless of extra budget). This variant
# doubles the ceiling to directly test whether time is still the limiter, or
# whether the model just burns the extra time equally unproductively --
# distinguishing the throughput hypothesis from the reasoning-limit hypothesis.
# Deliberately does NOT combine with lever B (timeout) -- isolates the budget
# question alone, leaving analyzer_timeout at baseline (900s).
LEVER_C_BIGCAP = LEVER_C.replace(
    "_CAP_CEILING_S = 14400 if TRUE_SUBMISSION else 11000",
    "_CAP_CEILING_S = 14400 if TRUE_SUBMISSION else 21600  # ~6h dev ceiling",
)

# --- Lever E ---------------------------------------------------------------
# `_build_system_prompt` is a module-level function in tool_agent.py, called by
# ToolAgent.__init__ -- and analyzers are constructed during bm.run(), i.e. after
# this cell. Wrapping it here therefore reaches every game. Same monkeypatch shape
# that was used to inject the 7 sandbox tools previously.
# NOTE: every previous prompt addition REGRESSED the real score, so this lever is
# deliberately tested last and in isolation.
LEVER_E = '''import inference.agent.tool_agent as _ta

_orig_build_system_prompt = _ta._build_system_prompt
_BATCH_ADDENDUM = (
    "\\n\\nAction batching: when your code has derived a reliable multi-step plan "
    "(for example an ordered path returned by your own search), pass that whole "
    "sequence to `action(...)` in ONE call instead of one action per turn. Each "
    "analyzer turn is expensive, so batching steps you have already verified is the "
    "main way to make real progress inside the time budget. Stop immediately if a "
    "result reports level_completed, game_over, run_complete or done.\\n"
)


def _build_system_prompt_with_batching(*, tool_output_tokens: int) -> str:
    return _orig_build_system_prompt(tool_output_tokens=tool_output_tokens) + _BATCH_ADDENDUM


_ta._build_system_prompt = _build_system_prompt_with_batching
print("[cell13] batching addendum installed", flush=True)
'''

# --- Lever H ---------------------------------------------------------------
# MULTIMODAL_CONTEXT=current_grid is printed at t≈16s (Cell 8 solver setup).
# Cell 13 runs after that init print, but the solver may re-read the env var
# lazily at game time (during bm.run() in Cell 15). We therefore set the env
# var AND defensively clear the solver attribute if it exists.
# Hypothesis: fewer image tokens per turn → faster generation → more actions
# before the per-game timeout under 28-way concurrency contention.
LEVER_H = '''import os as _os
_os.environ["MULTIMODAL_CONTEXT"] = "none"
_os.environ.pop("MULTIMODAL_UPSCALE", None)
if hasattr(bm.solver, 'multimodal_context'):
    bm.solver.multimodal_context = None
print(
    f"[cell13] multimodal disabled: MULTIMODAL_CONTEXT={_os.environ.get('MULTIMODAL_CONTEXT')!r}",
    flush=True,
)
'''

# --- Lever I ---------------------------------------------------------------
# 28-way concurrency means each game gets ~1 GPU slice (8 tok/s) and every game
# dies on the per-game cap.  Halving to 14 gives each game ~2× more throughput
# (≈16 tok/s), so each turn completes faster and the model can take more turns
# within the same wall-clock budget.  Downside: submission waves double (55/14≈4
# vs 55/28≈2), consuming more of the 12h kernel budget on overhead -- this is
# why we test it rather than assume it helps.
LEVER_I = (
    "bm.solver.concurrency = 14\n"
    'print(f"[cell13] concurrency={bm.solver.concurrency}", flush=True)\n'
)

# --- Lever K (max-output cap) -----------------------------------------------
# Data check on results/reasoning_dataset_2026-09-21.jsonl (4065 real turns):
# the shortest-reasoning quintile has ~3x the progress rate of every other
# quintile (3.18% vs 0.74-1.23%), spread across 12 games -- long reasoning
# looks more like the model spinning in circles than thinking harder
# productively. Tests that hypothesis directly by capping generation length.
#
# LOCAL_ANALYZER_MAX_OUTPUT is read ONCE as a module constant at import time
# (tool_agent.py), but each game builds its own ToolAgent lazily during
# bm.run() (Cell 15, after Cell 13 runs) via HarnessSolver._make_analyzer ->
# ToolAgent(...) -- so patching the module constant here still lands before
# any ToolAgent is actually constructed. 512 matches the harness's own
# internal fallback reserve (_reply_reserve_tokens = max_output_tokens or 512),
# not an arbitrary number.
LEVER_K_MAXOUT = '''import inference.agent.tool_agent as _ta

_MAX_OUTPUT_TOKENS = 512
_ta._LOCAL_ANALYZER_MAX_OUTPUT = _MAX_OUTPUT_TOKENS
print(f"[cell13] LOCAL_ANALYZER_MAX_OUTPUT patched to {_MAX_OUTPUT_TOKENS}", flush=True)
'''

# --- Lever J: single-game debug variant of Lever K --------------------------
# Restricted to ONE game (sb26) -- used to smoke-test Lever K cheaply inside a
# tiny GPU-quota remnant (~1h30) before spending a full 25-game run on it.
# sb26 has a documented 240+-action ungrounded-guessing stall (see
# sb26_variance_and_new_levers memory) -- a real prior failure case to check
# against. env_name is set at Game construction (unlike game_id, which is
# empty until _start_game()), so it's filterable before bm.run() begins.
# RESULT (2026-09-24, kernel v14): sb26 alone scored 8.33 in 94 actions vs the
# 2.22-2.78 (430-631 actions) seen in every other run of this game -- see
# reasoning_length_vs_outcome memory. Promising, but n=1 -- Lever K alone
# (below) is the real test, on the full 25-game set.
LEVER_J = (
    LEVER_K_MAXOUT
    + '''
_TARGET_GAME = "sb26"
_orig_run = bm.run


async def _run_single_game(*args, **kwargs):
    before = len(bm.games)
    all_names = [getattr(g, "env_name", None) for g in bm.games]
    matched = [g for g, name in zip(bm.games, all_names) if name and name.split("-")[0] == _TARGET_GAME]
    if not matched:
        raise RuntimeError(
            f"Target game {_TARGET_GAME!r} not found among {before} games. "
            f"Available env_name values: {all_names!r}"
        )
    bm.games = matched
    bm.solver.max_runtime_s_per_game = 3000.0
    print(
        f"[cell13] restricted run to game={_TARGET_GAME} (matched {[g.env_name for g in matched]!r} "
        f"of {before} games), max_runtime_s_per_game={bm.solver.max_runtime_s_per_game}",
        flush=True,
    )
    return await _orig_run(*args, **kwargs)


bm.run = _run_single_game
'''
)

VARIANTS: dict[str, str] = {
    "baseline": "",
    "B_timeout": LEVER_B,
    "C_adaptive": LEVER_C,
    "D_both": LEVER_B + "\n" + LEVER_C,
    "E_batching": LEVER_B + "\n" + LEVER_C + "\n" + LEVER_E,
    "F_timeout600": LEVER_B_600 + "\n" + LEVER_C,
    "G_bigcap": LEVER_C_BIGCAP,
    "H_nomodal": LEVER_C + "\n" + LEVER_H,
    "I_concur14": LEVER_C + "\n" + LEVER_I,
    "J_maxout_sb26": LEVER_J,
    # The real test: Lever K alone (no other changes) on the full 25-game set,
    # directly comparable against the two known pure-baseline local means
    # (2.85 from v6, 1.36 from v12 -- see project_state_pending_submission and
    # sb26_variance_and_new_levers memories) since it's a single-variable
    # change from the exact same pristine Cell 13. Push this first when the
    # Kaggle GPU quota resets (2026-09-26).
    "K_maxout512": LEVER_K_MAXOUT,
}


def build_cell13(variant: str) -> str:
    body = VARIANTS[variant]
    if not body:
        return BASELINE_CELL13
    return f"{BASELINE_CELL13}# --- tuning variant: {variant} ---\n{body}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=sorted(VARIANTS))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out or Path(f"/tmp/variant-{args.variant}")
    out_dir.mkdir(parents=True, exist_ok=True)

    notebook = json.loads(SOURCE_NOTEBOOK.read_text())

    # Guard: the hook must be where we think it is before we overwrite anything.
    existing = "".join(notebook["cells"][CELL13_INDEX]["source"])
    if "one-off changes" not in existing:
        raise SystemExit(
            f"Cell {CELL13_INDEX} is not the customization hook (got: {existing[:80]!r}). "
            "Re-pull the notebook before building variants."
        )

    cell13 = build_cell13(args.variant)
    notebook["cells"][CELL13_INDEX]["source"] = cell13.splitlines(keepends=True)

    # Preflight: kernelspec must survive (a missing one killed an earlier push in 4s),
    # and every code cell must parse. Cell 15 legitimately uses top-level await, which
    # Jupyter supports but ast.parse does not, so allow it explicitly.
    if not notebook.get("metadata", {}).get("kernelspec"):
        raise SystemExit("kernelspec missing from notebook metadata")

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        try:
            ast.parse(source)
        except SyntaxError as exc:
            if "await" in source and "outside function" in str(exc):
                continue  # top-level await: valid in Jupyter
            raise SystemExit(f"Cell {index} failed to parse: {exc}") from exc

    (out_dir / NOTEBOOK_NAME).write_text(json.dumps(notebook, indent=1))
    shutil.copy(SOURCE_METADATA, out_dir / "kernel-metadata.json")

    metadata = json.loads((out_dir / "kernel-metadata.json").read_text())
    assert metadata["enable_gpu"] is True, "enable_gpu must be true"
    assert metadata["machine_shape"] == "NvidiaRtxPro6000", "wrong machine_shape"

    print(f"variant  : {args.variant}")
    print(f"out dir  : {out_dir}")
    print(f"gpu      : {metadata['enable_gpu']} / {metadata['machine_shape']}")
    print(f"kernel   : {metadata['id']}")
    print("--- cell 13 ---")
    print(cell13)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
