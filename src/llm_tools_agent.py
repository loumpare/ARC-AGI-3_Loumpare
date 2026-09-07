"""Comparison agent (2026-09-01), "tools" variant: replaces Gemma's fuzzy
visual guessing for self-identification, wall detection, and pathfinding
with direct, reliable code operating on the raw grid array -- diffing
matrices instead of asking a VLM to describe what moved. The brain
(qwen2.5:7b) only has to do the part text reasoning is actually good at:
picking which discovered landmark is worth pursuing as a goal, and
noticing patterns in a log of observed cause/effect. Everything spatial
(who is "self", which values block movement, how to path from A to B) is
computed in Python from `frame.frame` directly, not inferred by an LLM.

Nothing here is ls20-specific: colors/positions/tile-size are all learned
from the current episode's own grid, not hardcoded (see
feedback_no_game_hacking). The agent discovers wall values, its own
identity, and cause/effect relationships (e.g. "touching landmark X changes
region Y elsewhere") purely by acting and diffing -- the same generic
method used to empirically verify ls20's mechanics during debugging, now
built into the agent itself instead of being a one-off diagnostic script.

Implements the same `agents.agent.Agent` contract so it can be tested
through the real ARC-AGI-3-Agents framework.
"""
from __future__ import annotations

import atexit
import os
import queue
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from arcengine import FrameData, GameAction, GameState
from scipy import ndimage

from agents.agent import Agent

from state_graph import StateGraph, state_signature

# Two inference backends for the "brain" (Qwen2.5-7B), picked automatically:
#   - local Ollama server (http://localhost:11434) for local dev/testing, where
#     the model is already pulled and GPU-served
#   - a bundled GGUF file run in-process via llama-cpp-python (CPU), for the
#     Kaggle competition rerun, which has no internet access and no Ollama
#     server -- see feedback_verify_before_asserting: this path was verified
#     to load and answer correctly locally before being relied on here
_GGUF_FILENAME = "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
_GGUF_SEARCH_ROOTS = [Path("/kaggle/input"), Path("/kaggle/working"),
                      Path(__file__).resolve().parent.parent / "notebooks"]
OLLAMA_URL = "http://localhost:11434/api/chat"
BRAIN_MODEL = "qwen2.5:7b"

_llama_singleton = None
# The ARC-AGI-3-Agents framework's Swarm (agents/swarm.py) runs one thread PER
# GAME with no concurrency limit -- during the real competition rerun that's
# up to ~110 threads racing to touch this module. llama-cpp-python is not
# safe for concurrent inference calls on one Llama instance, and the naive
# "if singleton is None: create it" lazy-init is itself a race (multiple
# threads could all see None and each load a separate ~4.6GB copy at once,
# OOM-killing the process) -- this almost certainly caused the opaque
# "Kaggle Error / system error" on the first real submission (no Python
# traceback was surfaced, consistent with a native crash or an OOM kill
# rather than a clean exception). Serialize both the lazy init and every
# inference call behind one lock -- see feedback_verify_before_asserting,
# this was root-caused by reading agents/swarm.py, not guessed.
_llama_lock = threading.Lock()

# swarm.py's Swarm.main() does `for t in threads: t.join()` with NO timeout on any
# of the ~110 per-game threads -- if a single brain call ever hung (native stall in
# llama-cpp-python under the CPU thread contention of ~110 live game threads, a
# gateway hiccup, anything), the whole submission would block forever until Kaggle's
# own external time limit kills the process, surfacing as exactly the opaque
# "ERROR" we've now seen twice with no traceback. Bound every brain call by running
# it on a throwaway daemon thread and waiting on it with a timeout: if it doesn't
# finish in time, the calling game thread gets a normal (catchable) exception and
# moves on with its deterministic fallback instead of hanging.
# NOTE: a `concurrent.futures.ThreadPoolExecutor` was tried first and rejected after
# directly testing a simulated hang (see feedback_verify_before_asserting) -- its
# worker threads are non-daemon, and `concurrent.futures.thread` registers an
# `atexit` hook that joins every such worker before the interpreter can exit, so a
# genuinely stuck call would still hang the whole `python main.py` process at exit
# even though `.result(timeout=...)` itself returned on time in the calling thread.
# A plain `threading.Thread(daemon=True)` has no such hook -- it cannot block
# process exit no matter how long the underlying call stays stuck.
BRAIN_CALL_TIMEOUT_S = 120  # a 25s budget was found, empirically, to be too tight even under
                             # mild local contention (3-4 concurrent games): most calls timed
                             # out despite a real single-call baseline of ~15.5s uncontended --
                             # queueing behind the shared lock alone pushed several past 25s.
                             # A reference "3rd place" community submission (mbmmurad's, GPU/vLLM)
                             # uses 400s per LLM request; 120s is a more conservative middle
                             # ground given our smaller CPU-only model, still generous enough
                             # that legitimate calls under real ~110-way contention aren't
                             # spuriously abandoned before they'd have succeeded. Worst case
                             # per game is still hard-bounded at MAX_BRAIN_CALLS * this value.


def _usable_cpu_count() -> int:
    """`os.cpu_count()` reports the HOST's total core count, not necessarily what
    this container is actually allowed to use -- on cgroup-limited infra (which
    Kaggle's competition rerun containers plausibly are) that can wildly overstate
    the real budget. `os.sched_getaffinity(0)` respects the process's actual CPU
    affinity mask instead. Setting llama.cpp's n_threads too high oversubscribes
    real cores, both slowing every single brain call (compounding the timeout risk
    BRAIN_CALL_TIMEOUT_S already guards against) and adding CPU contention against
    the ~109 other concurrently-running game threads. sched_getaffinity is
    Linux-only (AttributeError elsewhere, e.g. local macOS dev) -- fall back to
    cpu_count() there."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 4


def _find_gguf() -> Path:
    # Kaggle's exact mount path for model_sources/dataset_sources varies
    # (casing, version numbers) -- search by filename under the plausible
    # roots instead of hardcoding one guessed path, so a Kaggle-side naming
    # detail can't silently break loading (see feedback_verify_before_asserting)
    for root in _GGUF_SEARCH_ROOTS:
        if not root.exists():
            continue
        for match in root.rglob(_GGUF_FILENAME):
            return match
    raise FileNotFoundError(f"{_GGUF_FILENAME} not found under any of: {_GGUF_SEARCH_ROOTS}")


def _gpu_offload_available() -> bool:
    """True only when the bundled llama-cpp-python wheel was actually built with
    CUDA support AND a GPU is present at runtime -- both the brain (this module)
    and the vision module (llm_tools_vision_agent._query_eyes_gguf) gate their
    GPU offload on this, since the default Kaggle CPU-only wheel has no GPU
    kernels compiled in at all (n_gpu_layers would silently be a no-op there,
    not an error) and the vision path is deliberately GPU-only (see that
    module's comment on _VISION_GGUF_FILENAME for why a CPU fallback there
    would blow the latency budget)."""
    try:
        import llama_cpp
        return bool(llama_cpp.llama_cpp.llama_supports_gpu_offload())
    except Exception:
        return False


def _get_local_llama():
    global _llama_singleton
    if _llama_singleton is not None:
        return _llama_singleton
    with _llama_lock:
        if _llama_singleton is None:  # re-check: another thread may have won the race
            from llama_cpp import Llama
            path = _find_gguf()
            # offload to GPU when the wheel/hardware actually support it (real
            # RTX PRO 6000 benchmark, notebooks/gpu_bench/: qwen3.8-27B went from
            # ~170s/call CPU to ~2-6s/call GPU -- this 7B brain model is smaller,
            # so the same offload should help at least as much); falls back to
            # n_gpu_layers=0 (pure CPU, the only path tested/used before
            # 2026-09-06) when no CUDA-enabled wheel/GPU is present, e.g. local
            # dev without Ollama, or a Kaggle run without the GPU accelerator.
            n_gpu_layers = -1 if _gpu_offload_available() else 0
            _llama_singleton = Llama(model_path=str(path), n_ctx=4096,
                                      n_gpu_layers=n_gpu_layers,
                                      n_threads=_usable_cpu_count(), verbose=False)
        return _llama_singleton


def _ollama_available() -> bool:
    try:
        import requests
        requests.get("http://localhost:11434/api/tags", timeout=0.5)
        return True
    except Exception:
        return False

ACTION_SPACE = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION7]
ACTION_NAMES = {a.value: a.name for a in ACTION_SPACE}
ACTION_NAME_TO_ENUM = {a.name: a for a in ACTION_SPACE}

# ACTION6 ("click") is deliberately NOT added to the plain ACTION_SPACE above --
# every game tested that mixes ACTION6 with movement actions would otherwise
# have it show up in the round-robin bootstrap too, wasting a slot on a
# context-free click (x=0,y=0) nothing aims. Kept as separate lookup tables,
# used only where explicitly handled (see _choose_click_action) -- same
# pattern already validated in llm_tools_vision_agent.py and tested on vc33.
_CLICK_ACTION_NAMES = dict(ACTION_NAMES)
_CLICK_ACTION_NAMES[GameAction.ACTION6.value] = "ACTION6"
_CLICK_ACTION_NAME_TO_ENUM = dict(ACTION_NAME_TO_ENUM)
_CLICK_ACTION_NAME_TO_ENUM["ACTION6"] = GameAction.ACTION6

BACKGROUND_FRACTION = 0.10   # colors covering more of the grid than this are "bulk" (background/walls)
MIN_BLOB_SIZE = 1
MAX_BRAIN_CALLS = 4           # PER-LEVEL cap on Qwen calls for ToolsAgent (see brain_calls_this_level
                               # below) -- was a flat per-GAME cap until 2026-09-04; confirmed
                               # empirically that a flat per-game budget leaves an agent permanently
                               # stuck once a new level's layout needs fresh reasoning and the budget
                               # is already spent (ls20 never got past level 1 even given 1000
                               # actions). Still imported as-is by llm_tools_vision_agent.py, where it
                               # keeps its original per-GAME meaning for that class's own budget
                               # logic -- the name is shared but each class decides its own semantics.
                               # Kept at 4 for the SAME latency-safety reasoning as before (see
                               # feedback_verify_before_asserting: measured 98 calls/game before the
                               # first fix); MAX_BRAIN_CALLS_TOTAL below is the new safety net now that
                               # a single game could rack up a fresh budget per level across many levels.
MAX_BRAIN_CALLS_TOTAL = 20     # hard ceiling across the WHOLE game regardless of how many levels are
                               # reached -- with ~110 games sharing one locked/serialized model on the
                               # real competition rerun, worst case 20 * 110 * BRAIN_CALL_TIMEOUT_S(120s)
                               # ~= 7.3h serialized if every single call timed out; under normal
                               # ~10-15s/call conditions, ~1.8h. Bounds the SAME total-submission-timeout
                               # risk the original per-game cap was created to avoid, just no longer
                               # assuming one level is the only puzzle a game will ever present.

DIRECT_ACTION_REPEAT = 4  # when the brain suggests trying an action directly (self never
                           # identified -- movement laws show nothing spatial), commit to
                           # repeating that SAME action for this many turns instead of just
                           # once. A single press can't reveal or complete a discrete/cumulative
                           # mechanic (a dial that needs several turns to reach the right index,
                           # a mass-push that needs several pushes in one direction) -- taking it
                           # once then immediately cycling to a different untried action (the old
                           # behavior) never gave any one direction enough of a chance to show an
                           # effect. Confirmed empirically (2026-09-04) that cd82/ka59 never
                           # identify a self at all, so this bootstrap path is ALL either game
                           # ever runs on for the whole 150-action budget -- see
                           # llm_relay_agent_experiments memory.

STUCK_WINDOW = 6      # if self's position stays within STUCK_MAX_DISTINCT_BBOX distinct spots
                       # across this many consecutive turns, declare it stuck and force a
                       # replan. Deliberately position-based, NOT action-based (an earlier
                       # version required the SAME action repeated -- empirically confirmed via
                       # a 10-game instrumented trace, 2026-09-04, that 8/8 real stuck episodes
                       # on ls20 were multi-action ping-pong/cycle patterns, none were a single
                       # action repeated, so that version caught ZERO of them). Also confirmed
                       # 7/8 episodes were re-attempts of an already-failed goal -- consistent
                       # with the brain's own "prefer finishing an in-progress landmark"
                       # instruction backfiring on a genuinely-unreachable one, not just a BFS
                       # desync -- this position-based watchdog is a backstop for that too,
                       # independent of what causes the specific desync/re-attempt.
STUCK_MAX_DISTINCT_BBOX = 2

EXTRA_STEP_BUDGET = 2  # see pending_arrival_action's comment: how many extra repeats of the
                        # arrival-causing action to try before giving up on "walk further in"

MODEL_TRIGGER_INTERVAL = 15  # periodic brain check-in, independent of bootstrap/goal state --
                              # ported from llm_tools_vision_agent.py's "shared" mode (empirically
                              # confirmed the best of 3 tested variants there, see
                              # brain_grpo_finetuning_plan memory). Without this, a game where
                              # self-identification never succeeds, or where blobs is empty even
                              # after self is found (confirmed on ka59/sc25/sk48 in the 2026-09-03
                              # 25-game sweep: brain_call_count stayed 0 all game), NEVER consults
                              # the brain at all.

BRAIN_SYSTEM_PROMPT = """\
You are the decision module for a game-playing agent. A perception layer has \
already computed, with certainty (not guesswork): your own position (if found \
yet), a list of distinct landmarks visible on the grid, which actions have a \
CONFIRMED spatial movement effect on your position (and by how much), and a \
log of any cause-and-effect discovered so far (e.g. "touching landmark A \
changed region near landmark B").

Landmarks are labeled unvisited, in-progress, or visited. "In-progress" means \
you've already worked toward this landmark more than once without it being \
fully resolved -- that often means it's something you push or move gradually \
rather than a one-touch pickup. Prefer FINISHING an in-progress landmark over \
starting a fresh unvisited one, unless the effects log gives a specific \
reason not to. If the effects log shows a landmark type that GREW a \
resource/time bar, treat similar-looking unvisited landmarks as likely to do \
the same -- collecting more of them before you run low is probably worth \
prioritizing.

Normally, pick a landmark to move toward -- a pathfinder handles getting you \
there using the confirmed movement laws, you do not need to plan the route \
yourself. But if the movement laws show NO action reliably moves anything (or \
your own position hasn't even been found yet), "move toward a landmark" may \
not be a meaningful plan -- the mechanic might not be about walking around at \
all (e.g. a selector/dial, or a button you press). In that case, suggest \
trying a specific ACTION directly instead of a landmark.

You keep a short, persistent scratchpad of your own best current hypothesis \
about how THIS SPECIFIC game works -- what the goal seems to be, which kind \
of landmark is worth prioritizing, anything that tripped you up -- carried \
forward across calls instead of being rederived from scratch each time. You \
MUST write a concrete, specific sentence describing what THIS game actually \
seems to be about, based on what you see below -- not a generic restatement \
of these instructions and not a placeholder. Revise it as you learn more; \
only repeat the exact same wording if it is still fully accurate and there is \
truly nothing new to add.

Your goal is to reach a WIN state / complete a level.

Respond in exactly three parts, each on its own line:
1. A short reasoning (1 sentence).
2. A line starting with "NOTES:" followed by your specific, concrete \
one-sentence hypothesis about this game -- refine or extend your previous \
notes, don't discard them unless they turned out wrong.
3. On the LAST line, write EITHER one landmark id (e.g. "blob_2") OR one \
available action name (e.g. "ACTION3") -- whichever you're recommending -- \
and nothing else.
"""

NOTES_MAX_CHARS = 400  # bounds how much the persistent-notes line can grow the prompt over a
                        # long game -- a small CPU model summarizing its own prior summary each
                        # turn could otherwise drift longer without limit


def _grid_of(frame: FrameData) -> np.ndarray:
    return np.array(frame.frame[0], dtype=int)


def _bulk_colors(grid: np.ndarray) -> set[int]:
    total = grid.size
    values, counts = np.unique(grid, return_counts=True)
    return {int(v) for v, c in zip(values, counts) if c / total > BACKGROUND_FRACTION}


MAX_FRAGMENTS_PER_COLOR = 3  # a color fragmenting into more separate blobs than this looks
                              # like a bar/gauge/dashed pattern (many repeated pips), not a
                              # single meaningful landmark -- exclude it from targets generically


def _find_blobs(grid: np.ndarray, exclude_colors: set[int]) -> list[dict]:
    """Connected-component blobs per non-bulk color, each a distinct landmark.
    Colors that fragment into many separate small blobs (a bar/gauge/dashed
    pattern of repeated pips, e.g. a depleting resource bar) are dropped
    entirely -- not one meaningful object, just decoration/HUD clutter."""
    blobs = []
    for color in np.unique(grid):
        color = int(color)
        if color in exclude_colors:
            continue
        mask = grid == color
        labeled, n = ndimage.label(mask)
        if n > MAX_FRAGMENTS_PER_COLOR:
            continue
        for i in range(1, n + 1):
            ys, xs = np.where(labeled == i)
            if len(ys) < MIN_BLOB_SIZE:
                continue
            blobs.append({
                "color": color,
                "bbox": (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())),
                "centroid": (float(ys.mean()), float(xs.mean())),
                "size": int(len(ys)),
            })
    return blobs


def _bar_fragment_counts(grid: np.ndarray, exclude_colors: set[int]) -> dict[int, int]:
    """Fragment count for each color EXCLUDED from _find_blobs as bar-like
    (>MAX_FRAGMENTS_PER_COLOR components) -- tracked separately so a change in
    a bar's length (e.g. a step/move budget getting refilled) can be reported
    as its own explicit, labeled effect instead of a vague pixel-diff blob."""
    counts = {}
    for color in np.unique(grid):
        color = int(color)
        if color in exclude_colors:
            continue
        n = ndimage.label(grid == color)[1]
        if n > MAX_FRAGMENTS_PER_COLOR:
            counts[color] = n
    return counts


EDGE_MARGIN = 2        # a blob within this many pixels of a grid border is "pinned to the edge"
BAR_ASPECT_RATIO = 5   # width >= 5x height (or vice versa) is "long and thin" -- a HUD strip shape


def _edge_aspect_bar_colors(grid: np.ndarray, exclude_colors: set[int]) -> set[int]:
    """Second, COMPLEMENTARY HUD-bar signal to _bar_fragment_counts, added
    2026-09-05 after a real instrumented replay showed ls20's own step-budget
    bar looping an agent for ~20 turns despite confirmed_bar_colors already
    being wired in (see llm_relay_agent_experiments memory) -- root cause: that
    specific bar drains as ONE solid, unfragmented blob (never more than
    MAX_FRAGMENTS_PER_COLOR pieces), so the fragment-count check never once
    excluded it, and its shifting bbox (1px narrower every drain tick) also
    defeated the visited-landmark/attempt-count dedup, since each tick looked
    like a "new" object.

    This catches the complementary case (solid, non-fragmented bars) using a
    genuinely different, purely geometric signal, verified against the real
    offending bar before trusting it: measured bbox was 1px from the bottom
    edge, 34px wide x 2px tall (17:1 aspect ratio) -- exactly what a HUD strip
    looks like in virtually any game's rendering convention, and NOT a shape
    real gameplay landmarks tend to take (pinned flush to a border AND
    extremely thin in one dimension). Position+shape based, not fragment-count
    based, so it doesn't duplicate _bar_fragment_counts -- it fills the gap
    that heuristic structurally cannot see."""
    h, w = grid.shape
    found = set()
    for color in np.unique(grid):
        color = int(color)
        if color in exclude_colors:
            continue
        mask = grid == color
        labeled, n = ndimage.label(mask)
        if n != 1:
            continue  # fragmented bars are _bar_fragment_counts's job; this
                       # function only needs to catch the SOLID single-blob case
        ys, xs = np.where(labeled == 1)
        r0, r1, c0, c1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
        touches_edge = (r0 <= EDGE_MARGIN or r1 >= h - 1 - EDGE_MARGIN or
                        c0 <= EDGE_MARGIN or c1 >= w - 1 - EDGE_MARGIN)
        height, width = r1 - r0 + 1, c1 - c0 + 1
        extreme_aspect = width >= BAR_ASPECT_RATIO * height or height >= BAR_ASPECT_RATIO * width
        if touches_edge and extreme_aspect:
            found.add(color)
    return found


def _bbox_overlaps_or_adjacent(b1: tuple, b2: tuple, margin: int = 1) -> bool:
    r0a, r1a, c0a, c1a = b1
    r0b, r1b, c0b, c1b = b2
    return not (r1a + margin < r0b or r0a - margin > r1b or
                c1a + margin < c0b or c0a - margin > c1b)


INTERRUPTED_GAP_MAX = 6  # max gap (pixels) between two same-color blobs for them to be
                          # considered "the same object, interrupted" -- see
                          # _detect_interrupted_pairs


def _detect_interrupted_pairs(blobs: list[dict]) -> list[dict]:
    """Pure geometry, no vision/LLM needed: two blobs of the SAME color that sit
    almost end-to-end (small gap, aligned on one axis) with a THIRD,
    differently-colored blob sitting in that gap are very likely ONE object
    interrupted by the other (e.g. a bar with a position marker on it), not two
    independent objects. This is exactly vc33's real structure (confirmed
    2026-09-04 by direct inspection: a horizontal gray bar cut in two by a
    purple position marker sitting between the halves) -- Gemma and Qwen2.5-VL
    both got this wrong when asked to infer it visually (see
    llm_relay_agent_experiments memory), even with blob ids drawn on the image.
    Since it's exact geometry (bounding boxes the code already has), there's no
    reason to make a vision model guess it -- compute and state it as fact.
    Returns a list of {blob_a, blob_b, interrupter, gap_size} dicts, indices
    into the input `blobs` list."""
    results = []
    by_color: dict[int, list[int]] = {}
    for i, b in enumerate(blobs):
        by_color.setdefault(b["color"], []).append(i)

    for color, idxs in by_color.items():
        if len(idxs) < 2:
            continue
        for x in range(len(idxs)):
            for y in range(x + 1, len(idxs)):
                ia, ib = idxs[x], idxs[y]
                ar0, ar1, ac0, ac1 = blobs[ia]["bbox"]
                br0, br1, bc0, bc1 = blobs[ib]["bbox"]

                row_overlap = min(ar1, br1) - max(ar0, br0)
                col_overlap = min(ac1, bc1) - max(ac0, bc0)

                if row_overlap >= 0 and (ac1 < bc0 or bc1 < ac0):
                    # same row band, separated along columns -- gap is between them
                    gap_lo, gap_hi = (ac1, bc0) if ac1 < bc0 else (bc1, ac0)
                    band_lo, band_hi = max(ar0, br0), min(ar1, br1)
                    axis = "col"
                elif col_overlap >= 0 and (ar1 < br0 or br1 < ar0):
                    # same column band, separated along rows -- gap is between them
                    gap_lo, gap_hi = (ar1, br0) if ar1 < br0 else (br1, ar0)
                    band_lo, band_hi = max(ac0, bc0), min(ac1, bc1)
                    axis = "row"
                else:
                    continue  # not aligned end-to-end on either axis

                gap_size = gap_hi - gap_lo
                if gap_size <= 0 or gap_size > INTERRUPTED_GAP_MAX:
                    continue

                for k, c in enumerate(blobs):
                    if c["color"] == color:
                        continue
                    cr0, cr1, cc0, cc1 = c["bbox"]
                    if axis == "col":
                        in_gap = cc0 <= gap_hi and cc1 >= gap_lo and cr0 <= band_hi and cr1 >= band_lo
                    else:
                        in_gap = cr0 <= gap_hi and cr1 >= gap_lo and cc0 <= band_hi and cc1 >= band_lo
                    if in_gap:
                        results.append({"blob_a": ia, "blob_b": ib, "interrupter": k, "gap_size": gap_size})
                        break
    return results


def _structural_fact_lines(blobs: list[dict]) -> list[str]:
    """Renders _detect_interrupted_pairs's output as prompt-ready fact lines,
    shared by ToolsAgent and VisionToolsAgent so both state this as a GIVEN
    fact rather than asking a brain/vision model to infer it (see
    _detect_interrupted_pairs's docstring)."""
    lines = []
    for pair in _detect_interrupted_pairs(blobs):
        lines.append(
            f"- blob_{pair['blob_a']} and blob_{pair['blob_b']} (same color) are very likely ONE "
            f"object, split by blob_{pair['interrupter']} sitting in the small gap ({pair['gap_size']}px) "
            f"between them -- e.g. a bar/track with a position marker on it. Computed exactly from "
            f"positions, not a guess.")
    return lines


SHIFT_THRESHOLD = 2.5  # minimum centroid shift (pixels) to count as "actually moved", filters
                        # out sub-pixel labeling-order noise from colors shared by static decor


def _raw_mask_centroid(grid: np.ndarray, colors) -> tuple[float, float] | None:
    """Centroid over ALL pixels of these color(s), ignoring connected-component
    boundaries -- robust against a color being split into several tiny
    disconnected regions (e.g. a diagonal 2-pixel shape isn't 4-connected),
    which otherwise makes component-indexed matching pick inconsistent
    sub-regions between frames and look like spurious movement."""
    if isinstance(colors, int):
        colors = (colors,)
    ys, xs = np.where(np.isin(grid, list(colors)))
    if len(ys) == 0:
        return None
    return (float(ys.mean()), float(xs.mean()))


def _raw_mask_bbox(grid: np.ndarray, colors) -> tuple[int, int, int, int] | None:
    if isinstance(colors, int):
        colors = (colors,)
    ys, xs = np.where(np.isin(grid, list(colors)))
    if len(ys) == 0:
        return None
    return (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))


WINDOW_MARGIN = 12  # search radius (px) around the last known self bbox


def _local_mask_bbox(grid: np.ndarray, colors, anchor_bbox: tuple | None) -> tuple[int, int, int, int] | None:
    """Same as _raw_mask_bbox, but if an anchor is given, restrict the search to
    a window around it -- avoids a color shared with unrelated static decor
    elsewhere on the grid diluting/corrupting self's own position estimate."""
    if anchor_bbox is None:
        return _raw_mask_bbox(grid, colors)
    if isinstance(colors, int):
        colors = (colors,)
    r0, r1, c0, c1 = anchor_bbox
    wr0, wr1 = max(0, r0 - WINDOW_MARGIN), min(grid.shape[0], r1 + WINDOW_MARGIN + 1)
    wc0, wc1 = max(0, c0 - WINDOW_MARGIN), min(grid.shape[1], c1 + WINDOW_MARGIN + 1)
    window = grid[wr0:wr1, wc0:wc1]
    ys, xs = np.where(np.isin(window, list(colors)))
    if len(ys) == 0:
        return _raw_mask_bbox(grid, colors)  # lost it locally -- fall back to a global search
    return (int(ys.min()) + wr0, int(ys.max()) + wr0, int(xs.min()) + wc0, int(xs.max()) + wc0)


def _bbox_centroid(bbox: tuple) -> tuple[float, float]:
    r0, r1, c0, c1 = bbox
    return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)


def _closest_blob(blobs: list[dict], pos: tuple[float, float]) -> dict | None:
    if not blobs:
        return None
    return min(blobs, key=lambda b: abs(b["centroid"][0] - pos[0]) + abs(b["centroid"][1] - pos[1]))


# ============================================================================
# FONCTION DE DÉPLACEMENT PRINCIPALE (partagée par TOUS les agents, y compris
# les variantes "unified" -- UnifiedVisionAgent n'a pas sa propre logique de
# déplacement, elle utilise celle-ci via héritage, voir VisionToolsAgent plus
# bas dans src/llm_tools_vision_agent.py).
# ============================================================================
def _bfs_path(grid: np.ndarray, start: tuple[int, int], target_bbox: tuple,
              blocked_values: set[int], deltas: dict[str, tuple[int, int]]) -> list[GameAction] | None:
    """BFS (parcours en largeur) dans l'espace des pixels bruts, en utilisant
    UNIQUEMENT les déplacements ("deltas") appris jusqu'ici pour chaque
    action -- pas de règle codée en dur, seulement ce qui a été observé
    empiriquement (voir _update_from_last_transition, qui remplit `deltas`).
    Retourne la liste d'actions à exécuter pour atteindre `target_bbox`, ou
    None si aucun chemin n'est trouvé avec les lois de mouvement connues."""
    if not deltas:
        return None  # aucune loi de mouvement apprise encore -- impossible de planifier

    def overlaps_target(pos):
        # vrai si la position `pos` touche (avec une marge de 2px) la bbox cible --
        # c'est la règle "arrivé" utilisée PARTOUT dans le projet (voir aussi
        # pending_arrival_check plus bas, qui réutilise exactement cette même règle)
        r, c = pos
        r0, r1, c0, c1 = target_bbox
        return r0 - 2 <= r <= r1 + 2 and c0 - 2 <= c <= c1 + 2

    def blocked_at(pos):
        # vrai si la position est hors grille OU sur une couleur déjà identifiée
        # comme un mur (blocked_values, appris par essai -- une action qui ne
        # bouge pas self quand elle touche cette couleur)
        r, c = pos
        if r < 0 or c < 0 or r >= grid.shape[0] or c >= grid.shape[1]:
            return True
        return int(grid[r, c]) in blocked_values

    visited = {start}
    # named bfs_queue, not queue -- `queue` (the stdlib module) is imported at
    # module level for `_query_brain_bounded`'s timeout mechanism; a same-named
    # local here is harmless today (function-scoped) but a landmine for a future
    # edit that tries to use the module inside this function
    bfs_queue = deque([(start, [])])
    while bfs_queue:
        pos, path = bfs_queue.popleft()
        if overlaps_target(pos):
            return path  # trouvé ! on renvoie la séquence d'actions pour y arriver
        if len(path) >= 60:
            continue  # limite de profondeur -- évite une explosion combinatoire
        # on essaie chaque action dont on connaît l'effet (delta = (dr, dc))
        for action_name, (dr, dc) in deltas.items():
            if action_name not in ACTION_NAME_TO_ENUM:
                continue  # safety net: deltas should never contain ACTION6 (see
                          # _update_from_last_transition's guards), but skip
                          # defensively rather than KeyError if it ever does
            new_pos = (pos[0] + dr, pos[1] + dc)
            if new_pos in visited or blocked_at(new_pos):
                continue
            visited.add(new_pos)
            bfs_queue.append((new_pos, path + [ACTION_NAME_TO_ENUM[action_name]]))
    return None  # aucun chemin trouvé avec les lois de mouvement actuelles


def _query_brain(prompt: str, system_prompt: str = BRAIN_SYSTEM_PROMPT) -> str:
    if _ollama_available():
        import requests
        resp = requests.post(OLLAMA_URL, json={
            "model": BRAIN_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.4},
        }, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    llm = _get_local_llama()
    # one Llama instance is shared across every game's thread (see the note by
    # _llama_lock's definition) -- serialize inference calls too, not just init
    with _llama_lock:
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=60, temperature=0.4,  # reply is 1 short sentence + 1 id line, see prompt
        )
    return resp["choices"][0]["message"]["content"]


_pending_brain_threads: list[threading.Thread] = []
_pending_brain_threads_lock = threading.Lock()
DRAIN_TIMEOUT_S = 120  # see _drain_pending_brain_threads: bounded grace period at
                        # process exit, not indefinite


def _query_brain_bounded(prompt: str, system_prompt: str = BRAIN_SYSTEM_PROMPT) -> str:
    """Same as `_query_brain`, but bounded by BRAIN_CALL_TIMEOUT_S -- see the
    comment on BRAIN_CALL_TIMEOUT_S's definition for why an unbounded call is a
    whole-submission hang risk, not just a slow one, and why this uses a raw
    daemon thread instead of a ThreadPoolExecutor. The thread is also registered
    in `_pending_brain_threads` so `_drain_pending_brain_threads` (an atexit
    hook) can give it a chance to finish cleanly before interpreter shutdown --
    see that function's docstring, this was found to matter empirically, not
    theoretically. `system_prompt` is overridable so variants (e.g.
    llm_tools_vision_agent.VisionToolsAgent) can reuse this same timeout/drain
    machinery with their own prompt instead of duplicating it."""
    outcome: queue.Queue = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            outcome.put(("ok", _query_brain(prompt, system_prompt)))
        except Exception as e:  # noqa: BLE001 -- forwarded to the caller below
            outcome.put(("err", e))

    t = threading.Thread(target=worker, daemon=True)
    with _pending_brain_threads_lock:
        _pending_brain_threads.append(t)
    t.start()
    try:
        status, payload = outcome.get(timeout=BRAIN_CALL_TIMEOUT_S)
    except queue.Empty:
        raise TimeoutError(f"brain call exceeded {BRAIN_CALL_TIMEOUT_S}s") from None
    if status == "err":
        raise payload
    return payload


@atexit.register
def _drain_pending_brain_threads() -> None:
    """Give any brain-call threads a game already gave up waiting on (timed out
    in `_query_brain_bounded`, but not actually stopped -- Python cannot forcibly
    kill a thread) a bounded grace period to finish naturally before the
    interpreter starts tearing down.
    EMPIRICALLY REQUIRED, not defensive paranoia (see feedback_verify_before_asserting):
    a local test emulating Swarm.main() (3 concurrent games sharing this module's
    locked Llama singleton, forced onto the real CPU llama-cpp-python backend)
    reproducibly SEGFAULTED the whole Python process on exit -- twice, same
    scenario -- whenever a brain call was still actively running in native code
    (past its 25s timeout, abandoned by its caller) at the moment the process
    tried to exit. CPython does not guarantee it's safe for a daemon thread to be
    mid-call into a C extension (reacquiring the GIL to marshal a result) while
    the interpreter is finalizing. This hook only runs after `Swarm.main()` has
    already returned (i.e. every one of the ~110 per-game threads already
    finished its own full loop, independent of any brain-call timeout -- that
    per-game bound is unaffected), so a bounded wait here cannot reintroduce
    swarm.py's original unbounded-`t.join()` hang risk; it only closes the
    exit-time crash window for calls that were merely slow, not truly infinite."""
    deadline = time.time() + DRAIN_TIMEOUT_S
    with _pending_brain_threads_lock:
        threads = list(_pending_brain_threads)
    for t in threads:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        if t.is_alive():
            t.join(timeout=remaining)


class ToolsAgent(Agent):
    """Code-grounded self/wall/path tools + Qwen only for goal selection."""

    MAX_ACTIONS = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.self_colors: set[int] = set()  # the color used to TRACK position (kept to one, for stability)
        self.attached_colors: set[int] = set()  # other colors rigidly glued to self (own body parts)
        self.self_bbox: tuple[int, int, int, int] | None = None  # last known position, for local tracking
        self.prev_touched_keys: set[tuple] = set()  # landmarks already overlapping last turn (avoid re-logging)
        self.action_deltas: dict[str, tuple[int, int]] = {}
        self.pending_deltas: dict[str, tuple[int, int]] = {}  # unconfirmed first-time deltas, see _update_from_last_transition
        self.pending_self_candidates: dict[tuple[int, str], tuple] = {}  # unconfirmed self-bootstrap
                                                                          # guesses, keyed by (color,
                                                                          # action) -- see
                                                                          # _update_from_last_transition.
                                                                          # A dict, not one scalar slot:
                                                                          # round-robin bootstrap cycles
                                                                          # through a DIFFERENT action
                                                                          # every turn, so a single-slot
                                                                          # "last candidate" would get
                                                                          # overwritten before the same
                                                                          # (color, action) pair was ever
                                                                          # seen twice -- confirmed
                                                                          # empirically (2026-09-04) this
                                                                          # exact bug silently broke ls20
                                                                          # self-identification entirely.
        self.tried_actions: set[str] = set()  # exploration state -- separate from action_deltas confirmation
        self.blocked_values: set[int] = set()
        self.confirmed_bar_colors: set[int] = set()  # permanent once seen fragmenting past
                                                       # MAX_FRAGMENTS_PER_COLOR -- see
                                                       # _update_from_last_transition
        self.effects_log: list[dict] = []
        self.visited_blob_keys: set[tuple[int, int, int]] = set()  # (color, bbox_r0, bbox_c0), coarse
        self.blob_attempt_count: dict[tuple, int] = {}  # bounds retries on an unreachable target, see below
        self.current_goal_key: tuple | None = None
        self.current_goal_bbox: tuple | None = None  # target bbox for the in-flight plan, kept
                                                       # alongside current_goal_key so arrival can
                                                       # be verified once the plan is exhausted --
                                                       # see pending_arrival_check
        self.pending_arrival_check: tuple | None = None  # set when a BFS plan's last action was
                                                           # just taken -- resolved at the TOP of
                                                           # the next call, once self_bbox reflects
                                                           # that action's real effect (see
                                                           # _update_from_last_transition timing)
        self.pending_arrival_action: str | None = None  # the action that produced the arrival being
                                                          # checked -- see extra_step_budget below
        self.extra_step_budget = 0  # 2026-09-05: many ARC mechanics need walking PAST a marker's
                                     # edge, not just touching it (e.g. ls20's own documented
                                     # "touch the cross, then walk one tile further into the exit
                                     # doorway" two-step mechanic) -- when an arrival is confirmed
                                     # but no level transition follows, keep repeating the SAME
                                     # action a few more times before giving up and picking a brand
                                     # new target from scratch. Bounded (EXTRA_STEP_BUDGET) so a
                                     # genuine dead end can't loop forever.
        self.extra_step_action: str | None = None
        self.current_path: list[GameAction] = []
        self.recent_action_log: deque[tuple[str, tuple | None]] = deque(maxlen=STUCK_WINDOW)
        self.action_diff_stats: dict[str, dict] = {}  # per-action (n_tries, n_zero, total_diff) --
                                                        # lets the brain see "this action changes
                                                        # ~N px every time" without re-deriving it
                                                        # from scratch each consult
        self.goal_fail_count = 0
        self.brain_call_count = 0  # running total across the whole game -- see MAX_BRAIN_CALLS_TOTAL
        self.brain_calls_this_level = 0  # resets on level_changed -- see MAX_BRAIN_CALLS
        self.brain_notes = ""  # persistent hypothesis the brain carries forward across its own
                                # calls, updated in-place each consult instead of being rebuilt
                                # from scratch -- see NOTES_MAX_CHARS and _consult_brain
        self.direct_action_repeat_name: str | None = None  # see DIRECT_ACTION_REPEAT
        self.direct_action_repeat_remaining = 0
        self.state_graph = StateGraph()  # for games where self is never identified -- see
                                          # state_graph.py and the "not self.self_colors" branch
                                          # of _choose_action_impl
        self.prev_state_sig: tuple | None = None
        self.state_graph_path: list[str] = []
        self.prev_grid: np.ndarray | None = None
        self.prev_action_name: str | None = None
        self.prev_self_pos: tuple[float, float] | None = None
        self.prev_levels_completed: int | None = None
        self.last_raw_response = ""
        self.calibration_actions: list[str] = []  # fixed once, from turn 1's legal_names -- see
                                                     # the calibration block in _choose_action_impl
        self.calibration_step = 0
        self.calibration_done = False
        self.calibration_reset_pending = False  # True right after issuing calibration's own
                                                  # internal RESET, waiting for that frame to land

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        try:
            return latest_frame.state is GameState.WIN
        except Exception as e:
            # any exception here would otherwise propagate straight out of the
            # ARC-AGI-3-Agents framework's main loop (agents/agent.py's Agent.main
            # has no try/except around is_done/choose_action at all) -- treat an
            # unreadable state as "done" rather than risk it repeating forever
            print(f"[ToolsAgent] is_done crashed, treating as done: {e!r}")
            return True

    def _update_from_last_transition(self, grid: np.ndarray) -> None:
        """Pure code: figure out what moved, what blocked, what effects occurred.

        REPÈRE FR -- fonction appelée à CHAQUE tour, avant de décider la
        prochaine action. C'est ici que sont apprises les "lois de mouvement"
        (action_deltas : quelle action déplace le joueur de combien de
        lignes/colonnes) utilisées ensuite par _bfs_path pour planifier un
        chemin. Deux branches principales plus bas :
          1. `if self.self_colors:` -- le joueur est DÉJÀ identifié -> on
             mesure juste son déplacement réel et on met à jour/valide la loi
             de mouvement de la dernière action.
          2. `elif self.prev_action_name != "ACTION6":` -- le joueur n'est
             PAS encore identifié (phase "bootstrap") -> on cherche quelle
             couleur a le plus bougé pour deviner qui est "moi"."""
        if self.prev_grid is None or self.prev_action_name is None:
            return

        # per-action diff-magnitude stats, tracked regardless of whether self has
        # been identified -- gives the brain a cheap, code-computed signal ("this
        # action changes ~N px every time" vs "does literally nothing") instead of
        # having to re-guess an action's role from scratch each consult. See
        # _movement_summary, where this surfaces for actions with no confirmed
        # movement delta.
        diff_count = int((self.prev_grid != grid).sum())
        stats = self.action_diff_stats.setdefault(
            self.prev_action_name, {"n_tries": 0, "n_zero": 0, "total_diff": 0})
        stats["n_tries"] += 1
        stats["total_diff"] += diff_count
        if diff_count == 0:
            stats["n_zero"] += 1

        # persistent bar/gauge memory: once a color is EVER seen fragmenting into
        # more than MAX_FRAGMENTS_PER_COLOR same-color components, remember it as
        # bar-like for the rest of the game, not just the current frame. Fixes a
        # known gap (see llm_relay_agent_experiments memory, 2026-09-01 entry): the
        # raw per-frame fragment-count check alone stops excluding a bar once it
        # depletes below the threshold, so its last few remaining segments start
        # showing up as individual "landmarks" -- confirmed as the dominant real
        # cause of ls20 stuck-loop episodes via a 10-game instrumented trace
        # (2026-09-04, see STUCK_WINDOW's comment). Computed unconditionally (not
        # gated on self_colors) so this also helps games where self is never
        # identified (cd82/ka59/vc33-style) and effects-log/bar detection would
        # otherwise never run at all. Position-independent by construction -- no
        # assumption about WHERE a bar sits on the grid, only that a color's own
        # fragment count once exceeded the threshold.
        bar_counts_now = _bar_fragment_counts(grid, _bulk_colors(grid))
        self.confirmed_bar_colors |= set(bar_counts_now.keys())

        # 2026-09-05: geometric HUD signal (edge-pinned + extreme aspect ratio),
        # complementary to the fragment-count check above -- catches a SOLID
        # (non-fragmented) draining bar the fragment-count check structurally
        # cannot see. See _edge_aspect_bar_colors's docstring for the real
        # instrumented case that motivated this.
        self.confirmed_bar_colors |= _edge_aspect_bar_colors(grid, _bulk_colors(grid))

        # A second signal (pixel-count shrink over time) was tried here and REVERTED
        # (2026-09-04): ls20's real resource bar OSCILLATES (drains on steps, regrows
        # on pickups -- see the "resource_bar_grew" effect below), so tracking "ever
        # dropped below its historical max" wrongly flagged 5 different colors as
        # bar-like after just 2 turns each (including likely-legitimate landmark
        # colors), and crashed ls20's win rate from ~70% to 0/10 in a real trace.
        # Caught before it shipped -- see feedback_verify_before_asserting. The
        # fragment-count signal above is the one confirmed safe (7/10, no regression)
        # and worth keeping; a real fix for ls20's specific solid-bar-that-erodes case
        # needs a smarter signal (e.g. "shrinks immediately after every ACTION1-4,
        # never after other actions") that hasn't been built yet.

        bulk = _bulk_colors(self.prev_grid)
        teleported = False

        if self.self_colors:
            # REPÈRE FR -- BRANCHE 1 : joueur déjà identifié (self.self_colors
            # non vide). On mesure son déplacement réel depuis la dernière
            # frame et on confirme/apprend la loi de mouvement de l'action
            # jouée (self.action_deltas[nom_action] = (delta_ligne, delta_colonne)).
            # track position via a LOCAL window around the last known bbox, not a
            # global color mask -- a self color can be reused by unrelated static
            # decor elsewhere on the grid, which would dilute/corrupt the estimate.
            # self_colors is kept to the single bootstrapped color deliberately --
            # tracking stays clean; a separate attached_colors set (computed once,
            # right after bootstrap) handles excluding self's own other-colored
            # body parts from the landmark list without touching this math.
            prev_bbox = self.self_bbox or _raw_mask_bbox(self.prev_grid, self.self_colors)
            cur_bbox = _local_mask_bbox(grid, self.self_colors, prev_bbox)
            # _local_mask_bbox falls back to a whole-grid search when self_colors
            # vanishes from the local window (e.g. self got hidden/recolored by a
            # non-movement action). That global match can span several unrelated
            # same-colored sprites at once -- a diffuse/oversized bbox is a sign we
            # actually lost track, not a real position, so skip this turn rather
            # than trust a merged centroid.
            if prev_bbox and cur_bbox:
                prev_span = (prev_bbox[1] - prev_bbox[0] + 1) + (prev_bbox[3] - prev_bbox[2] + 1)
                cur_span = (cur_bbox[1] - cur_bbox[0] + 1) + (cur_bbox[3] - cur_bbox[2] + 1)
                if cur_span > max(6, prev_span * 3):
                    cur_bbox = None
            if prev_bbox and cur_bbox:
                p0, p1 = _bbox_centroid(prev_bbox), _bbox_centroid(cur_bbox)
                delta = (round(p1[0] - p0[0]), round(p1[1] - p0[1]))
                known_delta = self.action_deltas.get(self.prev_action_name)
                if known_delta is not None and delta != known_delta and delta != (0, 0):
                    # this action's delta was already confirmed before, and this
                    # transition doesn't match it -- a movement mechanic doesn't
                    # change mid-game, so self's tracked position just jumped in a
                    # way this action's confirmed law doesn't explain (soft
                    # respawn, forced reposition, etc). Same underlying situation
                    # as the character-swap case below (self's understanding of
                    # its own position no longer matches reality) -- treat it the
                    # same way: trust the new position, but also drop any
                    # in-flight plan computed for the now-stale position, so the
                    # next turn re-plans instead of blindly finishing a path that
                    # no longer starts where it thinks it does. Don't run effects
                    # detection on this jump (not a deliberate move).
                    self.self_bbox = cur_bbox
                    self.current_path = []
                    self.current_goal_key = None
                    self.goal_fail_count = 0
                    teleported = True
                elif delta != (0, 0):
                    self.self_bbox = cur_bbox
                    # ACTION6 ("click") targets arbitrary coordinates each turn -- unlike
                    # ACTION1-5/7 it has no single fixed compass-direction "delta" to
                    # learn at all, so never record one into action_deltas/pending_deltas
                    # (see the bootstrap branch below for the matching guard). Confirmed
                    # empirically (2026-09-04): on s5i5 (a pure-click game), a click's
                    # incidental large centroid shift got bootstrapped as if it were a
                    # movement law, which later crashed _bfs_path with
                    # KeyError('ACTION6') the moment it tried ACTION_NAME_TO_ENUM[...]
                    # (ACTION6 only exists in the separate _CLICK_ACTION_NAME_TO_ENUM
                    # table, since it's not a BFS-representable direction).
                    if known_delta is None and self.prev_action_name != "ACTION6":
                        # don't trust a NEW action's very first observed delta outright --
                        # a one-off jump (character-swap, screen transition) would get
                        # permanently baked in as if it were this action's fixed movement
                        # law, corrupting blocked-value inference and BFS pathfinding
                        # downstream. Require the same delta to repeat once before trusting it.
                        if self.pending_deltas.get(self.prev_action_name) == delta:
                            self.action_deltas[self.prev_action_name] = delta
                            del self.pending_deltas[self.prev_action_name]
                        else:
                            self.pending_deltas[self.prev_action_name] = delta
                    # else: delta reconfirms an already-known law, nothing new to record
                elif self.prev_action_name in self.action_deltas:
                    dr, dc = self.action_deltas[self.prev_action_name]
                    # self_colors didn't move as this action's already-confirmed delta
                    # predicts -- before concluding "wall", check whether some OTHER
                    # color just moved by exactly that delta instead. If so, the game
                    # handed control to a different sprite (e.g. a character-swap
                    # action), and self identity must re-lock onto it. This is generic:
                    # no assumption about which action causes the handoff, only that
                    # "the thing that now moves the way self used to" is self.
                    reassigned = False
                    exclude = bulk | self.self_colors | self.attached_colors
                    for color in [int(c) for c in np.unique(self.prev_grid) if int(c) not in exclude]:
                        alt_p0 = _raw_mask_centroid(self.prev_grid, color)
                        alt_p1 = _raw_mask_centroid(grid, color)
                        if alt_p0 is None or alt_p1 is None:
                            continue
                        alt_delta = (round(alt_p1[0] - alt_p0[0]), round(alt_p1[1] - alt_p0[1]))
                        if alt_delta == (dr, dc):
                            self.self_colors = {color}
                            self.self_bbox = _raw_mask_bbox(grid, self.self_colors)
                            self.attached_colors = set()
                            for b in _find_blobs(grid, bulk | self.self_colors):
                                if _bbox_overlaps_or_adjacent(self.self_bbox, b["bbox"], margin=1):
                                    self.attached_colors.add(b["color"])
                            self.current_path = []
                            self.current_goal_key = None
                            self.goal_fail_count = 0
                            reassigned = True
                            break
                    if not reassigned:
                        dest = (int(round(p0[0] + dr)), int(round(p0[1] + dc)))
                        if 0 <= dest[0] < self.prev_grid.shape[0] and 0 <= dest[1] < self.prev_grid.shape[1]:
                            self.blocked_values.add(int(self.prev_grid[dest]))
        elif self.prev_action_name != "ACTION6":
            # REPÈRE FR -- BRANCHE 2 : phase "bootstrap", personne n'est encore
            # identifié comme "moi". On regarde TOUTES les couleurs de la
            # grille et on prend celle qui a le plus bougé entre les deux
            # dernières frames -- si ce même (couleur, action, déplacement) se
            # répète une 2e fois, on la déclare "moi" (self.self_colors).
            # bootstrap: among non-bulk colors present in both frames, pick whichever
            # shows the LARGEST raw-mask centroid shift, requiring it to clear a
            # threshold well above the sub-pixel noise a shared/split color can produce.
            # Never bootstrap self-identity off an ACTION6 click (see the matching
            # guard above): a click can move/affect some OTHER object at the clicked
            # coordinates, and its shift has no reason to represent a rigid "self"
            # avatar the rest of this class's BFS/delta model assumes -- confirmed
            # empirically this was exactly the root cause of the ACTION6 KeyError
            # crash on s5i5 (a pure-click game, no rigid self-avatar even exists there).
            candidates = [int(c) for c in np.unique(self.prev_grid) if int(c) not in bulk]
            best_color, best_shift, best_delta = None, SHIFT_THRESHOLD, None
            for color in candidates:
                p0 = _raw_mask_centroid(self.prev_grid, color)
                p1 = _raw_mask_centroid(grid, color)
                if p0 is None or p1 is None:
                    continue
                shift = abs(p1[0] - p0[0]) + abs(p1[1] - p0[1])
                if shift > best_shift:
                    best_color = color
                    best_shift = shift
                    best_delta = (round(p1[0] - p0[0]), round(p1[1] - p0[1]))
            if best_color is not None:
                # don't commit to a self-identity on the FIRST large shift seen --
                # require the same (color, action, delta) combination to repeat once
                # before trusting it, matching the standard already used for a NEW
                # action's delta once self is already known (pending_deltas below).
                # Added 2026-09-04 after confirming empirically that giving movement
                # actions a chance to run during cd82/ka59's bootstrap (see the
                # non_click_names change in _choose_action_impl) let a single
                # non-Cartesian repaint (a rotating selector, a mass-push shove) get
                # misidentified as "self moving" -- the tell was wildly inconsistent
                # delta magnitudes across actions (e.g. ka59: ACTION1 (14,22) vs
                # ACTION2-4 around (2,0)), not something a single rigid sprite would
                # ever produce.
                cand_key = (best_color, self.prev_action_name)
                if self.pending_self_candidates.get(cand_key) == best_delta:
                    self.self_colors = {best_color}
                    self.self_bbox = _raw_mask_bbox(grid, self.self_colors)
                    self.action_deltas[self.prev_action_name] = best_delta
                    self.pending_self_candidates = {}
                    # any color forming a blob immediately touching self RIGHT NOW is
                    # almost certainly another part of self's own sprite (rigidly
                    # glued, e.g. a two-tone body) -- exclude it from ever being a
                    # landmark, without folding it into position-tracking (keeps
                    # that math clean)
                    for b in _find_blobs(grid, bulk | self.self_colors):
                        if _bbox_overlaps_or_adjacent(self.self_bbox, b["bbox"], margin=1):
                            self.attached_colors.add(b["color"])
                else:
                    self.pending_self_candidates[cand_key] = best_delta

        # effects: did self just touch/overlap a landmark? log any OTHER region that
        # changed, but only on the turn a landmark is NEWLY touched -- otherwise a
        # persistent/blinking visual near a landmark self is lingering next to would
        # get re-logged as a fresh "effect" every single turn
        if self.self_colors and not teleported:
            exclude = bulk | self.self_colors | self.attached_colors | self.confirmed_bar_colors
            cur_blobs = _find_blobs(grid, exclude)
            self_bbox = self.self_bbox or _raw_mask_bbox(grid, self.self_colors)

            # bar/gauge-like colors (excluded from the landmark list) are tracked
            # separately by fragment count -- if one grows right when a landmark is
            # newly touched, that's a specific, labeled "grants more time/moves"
            # association instead of a vague pixel-diff blob (e.g. a "coin" pickup
            # that refills a step budget bar)
            prev_bar_counts = _bar_fragment_counts(self.prev_grid, bulk | self.self_colors | self.attached_colors)
            cur_bar_counts = _bar_fragment_counts(grid, bulk | self.self_colors | self.attached_colors)
            grown_bars = {c: (prev_bar_counts.get(c, 0), n) for c, n in cur_bar_counts.items()
                          if n > prev_bar_counts.get(c, 0)}

            touched_now = set()
            if self_bbox:
                for b in cur_blobs:
                    if _bbox_overlaps_or_adjacent(self_bbox, b["bbox"]):
                        key = (b["color"], b["bbox"][0], b["bbox"][2])
                        touched_now.add(key)
                        if key in self.prev_touched_keys:
                            continue  # already touching last turn, not a new event
                        if grown_bars:
                            for bar_color, (before, after) in grown_bars.items():
                                entry = {"touched_color": b["color"], "touched_pos": b["centroid"],
                                          "effect_type": "resource_bar_grew", "bar_color": bar_color,
                                          "fragments_before": before, "fragments_after": after}
                                if entry not in self.effects_log:
                                    self.effects_log.append(entry)
                        diff_mask = (self.prev_grid != grid)
                        r0, r1, c0, c1 = self_bbox
                        diff_mask[max(0, r0 - 3):r1 + 4, max(0, c0 - 3):c1 + 4] = False  # exclude self's own footprint
                        if diff_mask.any():
                            ys, xs = np.where(diff_mask)
                            region = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))
                            entry = {"touched_color": b["color"], "touched_pos": b["centroid"], "effect_region": region}
                            if entry not in self.effects_log:
                                self.effects_log.append(entry)
            self.prev_touched_keys = touched_now

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Thin wrapper: the real logic (`_choose_action_impl`) is a fairly
        involved pipeline (blob/self tracking, BFS, brain queries) touching a
        real Kaggle-hidden game on every one of its ~80 turns per game, across
        ~110 concurrently-run games on the real competition rerun -- an
        untested edge case on any single hidden grid (unusual shape, a color
        pattern our bootstrap/BFS code doesn't expect, etc.) would otherwise
        propagate straight out of this method and out of the ARC-AGI-3-Agents
        framework's main loop uncaught (agents/agent.py's Agent.main has no
        try/except around choose_action at all). Catch broadly and fall back
        to a safe, cheap action instead of ever raising -- see
        feedback_verify_before_asserting: two real Phase B ERRORs so far with
        no traceback surfaced, this is the generic hardening asked for in
        response, not a guess at one specific bug."""
        try:
            return self._choose_action_impl(frames, latest_frame)
        except Exception as e:
            print(f"[ToolsAgent] choose_action crashed: {e!r}")
            traceback.print_exc()
            if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
                return GameAction.RESET
            # `latest_frame.available_actions` holds raw ints (GameAction.value),
            # not GameAction objects -- must convert via the same lookup tables
            # used everywhere else. A prior version of this fallback returned the
            # raw int directly: harmless-looking, but the framework's own
            # take_action()/do_action_request() immediately does
            # `action.action_data.model_dump()`, which raises AttributeError on a
            # plain int. That would have propagated straight out of this
            # except-block's caller (Agent.main() has no try/except of its own),
            # silently killing that one game's thread -- defeating the entire
            # point of this fallback existing. Caught by a Kaggle-safety code
            # audit, not by running it (see feedback_verify_before_asserting: this
            # specific path was never actually exercised, since _choose_action_impl
            # has never yet raised in any local test).
            legal = [ACTION_NAME_TO_ENUM[ACTION_NAMES[a]]
                     for a in latest_frame.available_actions if a in ACTION_NAMES]
            if not legal:
                return GameAction.RESET
            return legal[self.action_counter % len(legal)]

    def _movement_summary(self, legal_names: list[str]) -> str:
        lines = []
        for name in legal_names:
            if name in self.action_deltas:
                dr, dc = self.action_deltas[name]
                lines.append(f"- {name}: CONFIRMED to move you by (row {dr:+d}, col {dc:+d})")
                continue
            stats = self.action_diff_stats.get(name)
            if stats and stats["n_tries"] > 0:
                if stats["n_zero"] == stats["n_tries"]:
                    lines.append(f"- {name}: tried {stats['n_tries']}x, ZERO visible change on the "
                                  f"grid every time (likely inert or blocked from here)")
                else:
                    avg = stats["total_diff"] / stats["n_tries"]
                    lines.append(f"- {name}: tried {stats['n_tries']}x, no confirmed self-movement "
                                  f"but changes ~{avg:.0f} pixels on the grid EVERY time (likely a "
                                  f"real non-spatial effect -- selector/button/toggle, not blocked)")
            elif name in self.tried_actions:
                lines.append(f"- {name}: tried, no confirmed movement (may be non-spatial -- "
                              f"a button/toggle/selector rather than movement)")
        return "\n".join(lines) if lines else "(no actions tried yet)"

    def _periodic_due(self) -> bool:
        # ported from llm_tools_vision_agent.py's "shared" mode -- see MODEL_TRIGGER_INTERVAL
        return (self.action_counter > 0 and self.action_counter % MODEL_TRIGGER_INTERVAL == 0
                and self.brain_calls_this_level < MAX_BRAIN_CALLS
                and self.brain_call_count < MAX_BRAIN_CALLS_TOTAL)

    def _consult_brain(self, blobs: list[dict], legal_names: list[str],
                        self_pos: tuple[float, float] | None) -> tuple[dict | None, str | None]:
        """Ask Qwen to pick a landmark or suggest a direct action. Returns
        (target_blob_or_None, direct_action_name_or_None). Works whether or not
        `self` has been identified yet (self_pos may be None) -- see
        MODEL_TRIGGER_INTERVAL's comment for why this must not assume self is known."""
        id_map = {}
        blob_lines = []
        for i, b in enumerate(blobs):
            bid = f"blob_{i}"
            id_map[bid] = b
            key = (b["color"], b["bbox"][0], b["bbox"][2])
            if key in self.visited_blob_keys:
                status = "visited"
            elif self.blob_attempt_count.get(key, 0) > 0:
                status = "in-progress"
            else:
                status = "unvisited"
            blob_lines.append(f"{bid}: color={b['color']} pos={b['centroid']} size={b['size']} ({status})")
        blob_lines = blob_lines or ["(none detected)"]
        structural_lines = _structural_fact_lines(blobs)

        effects_lines = []
        for e in self.effects_log:
            if e.get("effect_type") == "resource_bar_grew":
                effects_lines.append(
                    f"- touching color {e['touched_color']} at {e['touched_pos']} GREW a resource/time "
                    f"bar (color {e['bar_color']}) from {e['fragments_before']} to {e['fragments_after']} "
                    f"segments -- this likely grants more moves/time, worth prioritizing if you have "
                    f"other unvisited landmarks like this one")
            else:
                effects_lines.append(
                    f"- touching color {e['touched_color']} at {e['touched_pos']} changed region "
                    f"rows{e['effect_region'][0]}-{e['effect_region'][1]} cols{e['effect_region'][2]}-{e['effect_region'][3]}")
        effects_lines = effects_lines or ["(none discovered yet)"]

        position_line = (f"Your position: {self_pos}" if self_pos is not None else
                          "Your position: NOT YET FOUND -- no single object has been reliably "
                          "tracked as \"you\" despite trying multiple actions (see movement laws "
                          "below). \"Move toward a landmark\" is likely not applicable here.")

        prompt = (
            f"{position_line}\n\n"
            f"Your notes from earlier turns:\n{self.brain_notes or '(none yet -- this is your first consult this game)'}\n\n"
            f"Movement laws discovered so far:\n{self._movement_summary(legal_names)}\n\n"
            f"Landmarks visible now:\n" + "\n".join(blob_lines) + "\n\n" +
            (f"Structural facts computed exactly from positions (not guesses):\n"
             + "\n".join(structural_lines) + "\n\n" if structural_lines else "") +
            f"Effects discovered so far:\n" + "\n".join(effects_lines) + "\n\n"
            f"Available actions: {', '.join(legal_names)}\n"
            f"Which landmark should you move toward next, or which action should you try directly?"
        )
        # count the ATTEMPT, not just a successful reply -- see the identical note
        # further down about why (bounds total wall-clock spent on brain calls)
        self.brain_call_count += 1
        self.brain_calls_this_level += 1
        try:
            reply = _query_brain_bounded(prompt)
        except Exception as e:
            reply = ""
            print(f"[ToolsAgent] brain query failed/timed out: {e!r}")
        self.last_raw_response = reply
        self._update_brain_notes(reply)
        chosen = None
        for line in reversed(reply.strip().splitlines()):
            line = line.strip()
            if line in id_map or line in legal_names:
                chosen = line
                break
        if chosen in id_map:
            return id_map[chosen], None
        if chosen in legal_names:
            return None, chosen
        return None, None

    def _update_brain_notes(self, reply: str) -> None:
        """Pull the "NOTES:" line out of a brain reply and carry it forward as
        `self.brain_notes`, fed back into the next prompt (see _consult_brain) --
        this is the whole mechanism: a short persistent hypothesis the brain
        writes for itself instead of every call reasoning from a blank slate.
        Deliberately tolerant of a malformed/missing NOTES line (older prompt
        format, or the model just not following instructions) -- falls back to
        keeping whatever notes were already there rather than erasing them."""
        for line in reply.strip().splitlines():
            line = line.strip()
            if line.upper().startswith("NOTES:"):
                text = line[len("NOTES:"):].strip()
                if text and text.lower() not in ("(unchanged)", "unchanged"):
                    self.brain_notes = text[:NOTES_MAX_CHARS]
                return

    def _choose_click_action(self, grid: np.ndarray, blobs: list[dict],
                              legal_names: list[str]) -> GameAction:
        """ACTION6 ("click") has completely different semantics from every
        other action here: there's no self-avatar to move, no BFS path to
        execute -- each turn IS just "click somewhere". Reuses the same
        blob-detection + brain consultation as the movement path, but the
        brain's chosen blob becomes a click TARGET directly instead of a BFS
        goal. Ported from llm_tools_vision_agent.py's already-validated
        implementation (tested standalone on vc33).
        NOTE, not yet fixed -- known issue before this can run multi-threaded:
        GameAction enum members are process-wide singletons, so `.set_data()`
        mutates shared state on the single `GameAction.ACTION6` object itself.
        Harmless for a single-threaded local test, but a real race condition
        once reused across Swarm's ~110 concurrent per-game threads on the
        real Kaggle submission -- two games choosing ACTION6 around the same
        time could interleave `.set_data()` calls and one could send the
        other's click coordinates. Needs a fix (e.g. constructing a fresh
        ComplexAction instead of mutating the singleton) before real
        multi-threaded use."""
        target = None
        budget_ok = (self.brain_calls_this_level < MAX_BRAIN_CALLS
                     and self.brain_call_count < MAX_BRAIN_CALLS_TOTAL)
        if blobs and budget_ok and (self._periodic_due() or not self.tried_actions):
            target, _ = self._consult_brain(blobs, legal_names, self_pos=None)
        if target is None and blobs:
            unvisited = [b for b in blobs
                         if (b["color"], b["bbox"][0], b["bbox"][2]) not in self.visited_blob_keys]
            target = unvisited[0] if unvisited else blobs[self.action_counter % len(blobs)]
        if target is not None:
            self.visited_blob_keys.add((target["color"], target["bbox"][0], target["bbox"][2]))
            cy, cx = target["centroid"]
        else:
            cy, cx = grid.shape[0] / 2, grid.shape[1] / 2  # no blobs at all -- click center as a fallback
        action = GameAction.ACTION6
        action.set_data({"x": int(round(cx)), "y": int(round(cy))})
        self.tried_actions.add("ACTION6")
        return action

    def _choose_action_impl(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.prev_grid = None
            self.prev_action_name = None
            self.self_bbox = None  # stale post-reset; force a fresh local/global search next turn
            self.current_path = []
            self.current_goal_key = None
            self.current_goal_bbox = None
            self.pending_arrival_check = None
            self.pending_arrival_action = None
            self.extra_step_budget = 0
            self.extra_step_action = None
            self.recent_action_log.clear()
            self.direct_action_repeat_remaining = 0  # don't carry a pre-reset commitment across
            self.prev_state_sig = None  # stale post-reset; state_graph itself is NOT cleared --
            self.state_graph_path = []  # learned edges/goals stay valid across a reset, only the
                                         # in-flight plan and "last known position" reset
            return GameAction.RESET

        legal = [a for a in latest_frame.available_actions if a in _CLICK_ACTION_NAMES]
        if not legal:
            return GameAction.RESET
        legal_names = [_CLICK_ACTION_NAMES[a] for a in legal]

        grid = _grid_of(latest_frame)
        # a level transition swaps the entire layout -- comparing against prev_grid
        # here would misread "a whole new level's HUD appeared" as "something you
        # touched changed", so skip the diff-based update for this one turn
        level_changed = (self.prev_levels_completed is not None and
                          latest_frame.levels_completed != self.prev_levels_completed)
        if level_changed:
            self.prev_grid = None
            # a new level is a genuinely new spatial puzzle -- confirmed empirically
            # (2026-09-03) that ls20 gets permanently stuck at level 1 even given 1000
            # actions once the flat per-GAME MAX_BRAIN_CALLS budget is spent, because
            # the deterministic fallback alone can't reason about an unfamiliar new
            # layout. Give each level its own fresh budget instead of one shared
            # per-game pool. brain_call_count (the running total) is kept unchanged
            # for logging/telemetry and as the overall Kaggle-latency safety net below.
            self.brain_calls_this_level = 0
            # a level change moves self to a brand-new layout -- any in-flight
            # arrival check or same-action streak was measuring the OLD level and
            # would be comparing apples to oranges against the new one
            self.pending_arrival_check = None
            self.pending_arrival_action = None
            self.extra_step_budget = 0
            self.extra_step_action = None
            self.recent_action_log.clear()
        self.prev_levels_completed = latest_frame.levels_completed
        self._update_from_last_transition(grid)

        # Calibration phase (2026-09-06, user-requested): on a genuinely fresh
        # game, deterministically try every non-click action TWICE before doing
        # anything goal-directed, then RESET back to a clean start -- replaces
        # discovering movement laws opportunistically (whichever action the old
        # round-robin happened to try first) with a guaranteed, complete pass.
        # By the time calibration ends, action_deltas holds every action's real
        # effect, and a game with NO rigid movement avatar at all is known for
        # certain (not just inferred slowly) before any real, goal-directed
        # action is spent -- see llm_tools_agent's "not self.self_colors"
        # bootstrap branch below, which now starts with tried_actions already
        # fully populated in that case and routes straight to click/direct-
        # action mode instead of re-discovering the same dead end.
        # Verified via arcengine's own source (ARCBaseGame.handle_reset):
        # `elif self._action_count == 0 or self._state == GameState.WIN:
        # full_reset() else: level_reset()` -- a RESET here does NOT lose
        # levels_completed/level progress (level_reset only, since action_count
        # is already > 0 by calibration's end and the game isn't won -- `main()`
        # would already have stopped calling choose_action at all if it were,
        # via is_done(), so latest_frame.state is never WIN here). Only a
        # full_reset (gated on action_count==0) would lose progress, which
        # never applies once calibration has taken at least one action.
        if not self.calibration_done:
            if not self.calibration_actions:
                # first-ever turn: fix the calibration list once, from whatever
                # non-click actions are legal right now. Empty (click-only game,
                # e.g. vc33/s5i5) -- nothing to calibrate, skip straight through
                # without wasting a RESET.
                self.calibration_actions = [n for n in legal_names if n != "ACTION6"]
            if not self.calibration_actions:
                self.calibration_done = True
            elif self.calibration_reset_pending:
                # calibration's own RESET just landed -- same stale-state
                # bookkeeping the top-of-function NOT_PLAYED/GAME_OVER branch
                # does (position/plan state is stale post-reset), but
                # self_colors/action_deltas learned during calibration are
                # deliberately KEPT, not cleared.
                self.prev_grid = None
                self.prev_action_name = None
                # Unlike the top-of-function NOT_PLAYED/GAME_OVER branch, self_bbox is
                # NOT nulled to None here -- that branch relies on prev_self_pos as a
                # fallback (see "self already known" branch below), which is safe there
                # because self was always found many real turns into an uninterrupted
                # playthrough BEFORE any reset could fire, so prev_self_pos already held
                # a valid value. Calibration breaks that assumption: self can be found
                # and immediately followed by calibration's own RESET with ZERO real
                # turns in between, so prev_self_pos is still its __init__ default of
                # None -- confirmed the hard way via a real Kaggle Phase A run
                # (2026-09-07): `_closest_blob(unvisited, self_pos)` crashed with
                # "'NoneType' object is not subscriptable" on the very first
                # post-calibration goal-directed turn. Recompute a fresh bbox directly
                # from the just-arrived post-reset grid instead of leaving both
                # self_bbox AND prev_self_pos empty.
                self.self_bbox = _raw_mask_bbox(grid, self.self_colors) if self.self_colors else None
                self.prev_self_pos = _bbox_centroid(self.self_bbox) if self.self_bbox else None
                self.current_path = []
                self.current_goal_key = None
                self.current_goal_bbox = None
                self.pending_arrival_check = None
                self.pending_arrival_action = None
                self.extra_step_budget = 0
                self.extra_step_action = None
                self.recent_action_log.clear()
                self.direct_action_repeat_remaining = 0
                self.prev_state_sig = None
                self.state_graph_path = []
                self.calibration_reset_pending = False
                self.calibration_done = True
                # fall through to the normal logic below for THIS turn's real choice
            elif self.calibration_step < 2 * len(self.calibration_actions):
                chosen_name = self.calibration_actions[self.calibration_step % len(self.calibration_actions)]
                self.calibration_step += 1
                self.tried_actions.add(chosen_name)
                self.prev_grid = grid
                self.prev_action_name = chosen_name
                return _CLICK_ACTION_NAME_TO_ENUM[chosen_name]
            else:
                self.calibration_reset_pending = True
                self.prev_grid = grid
                self.prev_action_name = None  # RESET isn't a movement action -- nothing to diff it against
                return GameAction.RESET

        if not level_changed:
            if self.pending_arrival_check is not None:
                # resolve the arrival check deferred from the LAST turn (see
                # current_goal_bbox/pending_arrival_check comments in __init__) --
                # self.self_bbox now reflects that final action's real effect, so
                # this is the first point this can be checked honestly. Confirmed
                # via a real playthrough this was previously NOT checked at all:
                # the old code declared "arrived" the instant current_path emptied,
                # even if a stale plan (computed before a wall got confirmed) never
                # actually got self there -- silently marking an unreached target
                # visited and moving on, instead of retrying or picking elsewhere.
                #
                # Uses the exact same point-vs-bbox/margin-2 rule as _bfs_path's own
                # overlaps_target -- NOT a bbox-vs-bbox comparison (tried first,
                # empirically broken: confirmed via instrumented real ls20 play that
                # it almost NEVER succeeded, tanking win rate from ~3/4 to ~1/7-1/10).
                # self.self_bbox only spans self's tracked color, which can be a
                # strict sub-region of the full multi-color sprite (e.g. ls20's
                # two-tone body) offset from where BFS's own centroid-based search
                # considered itself "arrived" -- a bbox-edge comparison is stricter
                # than the single-point check BFS itself already succeeded by, so it
                # was rejecting real arrivals BFS had already correctly achieved.
                arrived = False
                if self.self_bbox is not None:
                    r, c = _bbox_centroid(self.self_bbox)
                    tr0, tr1, tc0, tc1 = self.pending_arrival_check
                    arrived = tr0 - 2 <= r <= tr1 + 2 and tc0 - 2 <= c <= tc1 + 2
                if arrived:
                    self.goal_fail_count = 0
                    if self.current_goal_key is not None:
                        self.visited_blob_keys.add(self.current_goal_key)
                    # 2026-09-05: arrival confirmed but the level didn't transition --
                    # many ARC mechanics need self to walk PAST a marker's edge, not
                    # just touch its bounding box (see EXTRA_STEP_BUDGET's comment).
                    # Give continuing in the same direction a bounded number of tries
                    # BEFORE falling through to picking a brand new target from
                    # scratch -- generic ("keep going the way you were already
                    # heading"), no assumption about which game or which marker.
                    if not level_changed and self.pending_arrival_action is not None:
                        self.extra_step_budget = EXTRA_STEP_BUDGET
                        self.extra_step_action = self.pending_arrival_action
                else:
                    self.goal_fail_count += 1
                self.pending_arrival_check = None
                self.pending_arrival_action = None

            if self.self_colors and self.prev_action_name is not None:
                self.recent_action_log.append((self.prev_action_name, self.self_bbox))
                if (len(self.recent_action_log) == self.recent_action_log.maxlen
                        and len({e[1] for e in self.recent_action_log}) <= STUCK_MAX_DISTINCT_BBOX):
                    # position-based stuck detection -- see STUCK_WINDOW's comment for why
                    # this replaced an action-identity-based check. Force a fresh plan next
                    # turn instead of continuing to grind on a desynced/blocked/re-failing path.
                    # Also count this as a failed ATTEMPT on whatever goal was active -- the
                    # 10-game trace (see STUCK_WINDOW's comment) found 7/8 stuck episodes were
                    # re-attempts of an already-failing goal, so crediting this toward the
                    # existing blob_attempt_count>=3 give-up threshold (same mechanism used at
                    # selection time) makes a persistently-blocked target get excluded sooner,
                    # instead of only a fresh selection-time pick counting as an attempt.
                    if self.current_goal_key is not None:
                        self.blob_attempt_count[self.current_goal_key] = (
                            self.blob_attempt_count.get(self.current_goal_key, 0) + 1)
                        if self.blob_attempt_count[self.current_goal_key] >= 3:
                            self.visited_blob_keys.add(self.current_goal_key)
                    self.current_path = []
                    self.current_goal_key = None
                    self.current_goal_bbox = None
                    self.goal_fail_count = max(self.goal_fail_count, 1)
                    self.recent_action_log.clear()

        bulk = _bulk_colors(grid)
        blobs = _find_blobs(grid, bulk | self.self_colors | self.attached_colors | self.confirmed_bar_colors)

        # bootstrap phase: self not identified yet -- cycle through actions to seed movement data
        if not self.self_colors:
            # generic clustered state-graph (see state_graph.py): observe every
            # transition regardless of which sub-branch below ends up choosing the
            # action, and if a path to a previously-discovered goal state (a
            # levels_completed increase) is known, follow it -- takes priority
            # over everything else here since it's the only option backed by an
            # actual observed win, not a guess. A no-op until the FIRST real win
            # on this game (find_path returns None with an empty goal set), so
            # this is purely additive: it cannot make an already-tried game worse,
            # only give it a shot at reusing a win once one has ever happened.
            sig = state_signature(blobs)
            if self.prev_state_sig is not None and self.prev_action_name is not None:
                self.state_graph.record_transition(self.prev_state_sig, self.prev_action_name, sig)
            if level_changed:
                self.state_graph.mark_goal(sig)
                self.state_graph_path = []  # a level transition invalidates any in-flight plan
            self.prev_state_sig = sig

            if not self.state_graph_path:
                self.state_graph_path = self.state_graph.find_path(sig) or []
            if self.state_graph_path:
                chosen_name = self.state_graph_path.pop(0)
                if chosen_name in legal_names:
                    self.tried_actions.add(chosen_name)
                    self.prev_grid = grid
                    self.prev_action_name = chosen_name
                    return _CLICK_ACTION_NAME_TO_ENUM[chosen_name]
                self.state_graph_path = []  # stale (action no longer legal) -- replan next time

            non_click_names = [n for n in legal_names if n != "ACTION6"]
            non_click_tried = all(n in self.tried_actions for n in non_click_names)
            if "ACTION6" in legal_names and (not non_click_names or non_click_tried):
                # pure-click game (no movement actions at all, e.g. vc33) OR every
                # non-click action already tried at least once without self ever
                # being found -- route to the dedicated click handler. Changed
                # 2026-09-04 from an unconditional check: cd82/ka59 mix ACTION6
                # with real movement actions (ACTION1-5, confirmed via
                # available_actions inspection) that drive the actual dial/
                # mass-push mechanic -- the old unconditional-click-first order
                # meant those actions were NEVER tried during bootstrap, so
                # action_deltas/the state graph could never observe their real
                # effect. vc33 (no non-click actions at all) is unaffected:
                # non_click_names is empty there, so this still routes to click
                # immediately, identical to the old behavior.
                action = self._choose_click_action(grid, blobs, legal_names)
                self.prev_grid = grid
                self.prev_action_name = action.name
                return action
            if self.direct_action_repeat_remaining > 0 and self.direct_action_repeat_name in legal_names:
                # mid-commitment to a brain-suggested action -- see DIRECT_ACTION_REPEAT.
                # Skip round-robin/brain-consult entirely this turn, just repeat it.
                self.direct_action_repeat_remaining -= 1
                chosen_name = self.direct_action_repeat_name
                self.tried_actions.add(chosen_name)
                self.prev_grid = grid
                self.prev_action_name = chosen_name
                return _CLICK_ACTION_NAME_TO_ENUM[chosen_name]
            if self._periodic_due():
                # periodic check-in even during bootstrap -- see MODEL_TRIGGER_INTERVAL: a
                # game where self is never identified (ka59/vc33-style mechanics) would
                # otherwise NEVER reach the brain at all, confirmed empirically 2026-09-03
                _, direct_action_name = self._consult_brain(blobs, legal_names, self_pos=None)
                if direct_action_name is not None:
                    self.tried_actions.add(direct_action_name)
                    self.direct_action_repeat_name = direct_action_name
                    self.direct_action_repeat_remaining = DIRECT_ACTION_REPEAT - 1
                    self.prev_grid = grid
                    self.prev_action_name = direct_action_name
                    return _CLICK_ACTION_NAME_TO_ENUM[direct_action_name]
            untried = [a for a in legal_names if a not in self.tried_actions]
            chosen_name = untried[0] if untried else legal_names[self.action_counter % len(legal_names)]
            action = _CLICK_ACTION_NAME_TO_ENUM[chosen_name]
            self.tried_actions.add(chosen_name)
            self.prev_grid = grid
            self.prev_action_name = chosen_name
            return action

        self_pos = _bbox_centroid(self.self_bbox) if self.self_bbox else self.prev_self_pos
        self.prev_self_pos = self_pos
        self_pos_int = (int(round(self_pos[0])), int(round(self_pos[1]))) if self_pos else (0, 0)

        # 2026-09-05: "walk past the marker" retry -- see extra_step_budget's
        # comment. Takes priority over picking a brand new goal, but only for a
        # bounded number of tries, and only while there's no other in-flight
        # plan already running (current_path empty -- an active plan toward
        # something else should never get interrupted by this).
        if not self.current_path and self.extra_step_budget > 0 and self.extra_step_action in legal_names:
            self.extra_step_budget -= 1
            self.tried_actions.add(self.extra_step_action)
            self.prev_grid = grid
            self.prev_action_name = self.extra_step_action
            return _CLICK_ACTION_NAME_TO_ENUM[self.extra_step_action]

        # need a new goal?
        if not self.current_path:
            # only ask Qwen on a FRESH need (goal_fail_count==0: either the very
            # first goal, or the last one was actually reached) and under the
            # hard per-level call cap -- OR periodically regardless (see
            # MODEL_TRIGGER_INTERVAL). A failed/exhausted path (goal_fail_count>0)
            # re-enters this block on literally the next turn since current_path
            # stays empty -- calling Qwen again there is what drove call counts
            # to ~98/game before an earlier fix (measured; see
            # feedback_verify_before_asserting). Retries instead pick
            # deterministically (free, instant), preferring an unvisited landmark,
            # same as Qwen would tend to anyway.
            # NOTE: deliberately NOT gated on `if blobs:` -- empty blobs (e.g. some
            # games' pushable objects fragmenting past MAX_FRAGMENTS_PER_COLOR) is
            # exactly one of the stuck states the periodic trigger exists to break
            # out of, confirmed when this same gating bug was first found and fixed
            # in llm_tools_vision_agent.py.
            periodic_due = self._periodic_due()
            fresh_goal_due = (bool(blobs) and self.goal_fail_count == 0
                               and self.brain_calls_this_level < MAX_BRAIN_CALLS
                               and self.brain_call_count < MAX_BRAIN_CALLS_TOTAL)
            ask_brain = fresh_goal_due or periodic_due
            target = None
            direct_action_name = None
            if ask_brain:
                target, direct_action_name = self._consult_brain(blobs, legal_names, self_pos)

            if direct_action_name is not None:
                # brain judged the mechanic isn't spatial pathing -- try its
                # suggested action directly instead of routing through BFS
                self.tried_actions.add(direct_action_name)
                self.prev_grid = grid
                self.prev_action_name = direct_action_name
                return _CLICK_ACTION_NAME_TO_ENUM[direct_action_name]

            if target is None and blobs:
                unvisited = [b for b in blobs
                             if (b["color"], b["bbox"][0], b["bbox"][2]) not in self.visited_blob_keys]
                target = _closest_blob(unvisited, self_pos) or _closest_blob(blobs, self_pos)
            if target is not None:
                self.current_goal_key = (target["color"], target["bbox"][0], target["bbox"][2])
                self.current_goal_bbox = target["bbox"]  # see pending_arrival_check
                # NOT marked visited here anymore (was: immediately on selection, even
                # before confirming BFS found a path or self ever arrived) -- confirmed
                # via a real playthrough GIF (2026-09-03) that this caused the agent to
                # abandon a pushable object it had already moved most of the way to a
                # goal, since re-selecting the SAME object (now at a shifted position,
                # so a "new" blob key) still lost to the deterministic "prefer
                # unvisited" bias once ANY blob sharing that identity had been touched.
                # Now only marked visited on confirmed arrival, below, where
                # goal_fail_count resets to 0. Bounded retry safety net:
                # blob_attempt_count caps repeated selection of a target BFS can't
                # actually reach at 3 tries before giving up on it, so this cannot
                # regress into an infinite retry loop on something unreachable.
                self.blob_attempt_count[self.current_goal_key] = (
                    self.blob_attempt_count.get(self.current_goal_key, 0) + 1)
                if self.blob_attempt_count[self.current_goal_key] >= 3:
                    self.visited_blob_keys.add(self.current_goal_key)
                path = _bfs_path(grid, self_pos_int, target["bbox"], self.blocked_values, self.action_deltas)
                self.current_path = path or []
            if not self.current_path:
                # no plan available -- fall back to an unexplored action to keep learning
                untried = [a for a in legal_names if a not in self.tried_actions]
                chosen_name = untried[0] if untried else legal_names[self.action_counter % len(legal_names)]
                self.tried_actions.add(chosen_name)
                self.prev_grid = grid
                self.prev_action_name = chosen_name
                self.goal_fail_count += 1
                return _CLICK_ACTION_NAME_TO_ENUM[chosen_name]

        action = self.current_path.pop(0)
        self.prev_grid = grid
        self.prev_action_name = action.name
        if not self.current_path and self.current_goal_bbox is not None:
            # defer the actual arrival verdict to the TOP of the next call, once
            # self_bbox reflects this action's real effect -- see
            # pending_arrival_check's comment in __init__ and the resolution logic
            # in _choose_action_impl right after _update_from_last_transition
            self.pending_arrival_check = self.current_goal_bbox
            self.pending_arrival_action = action.name  # see extra_step_budget
        return action
