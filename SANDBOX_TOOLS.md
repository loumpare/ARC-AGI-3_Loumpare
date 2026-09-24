# Duck Harness Python sandbox — full toolbox

Everything exposed to the model inside the `python` tool call (Duck Harness / ARC3-Inference), native + our additions. The sandbox resets between calls: nothing persists except what's re-injected in `runtime_globals` each turn.

## Native state objects

| Name | Type | Description |
|---|---|---|
| `current_frame` | `FrameView` | Current board state: `.ascii`, `.segmentation`, `.step`, `.level`, `.shape` |
| `previous_frame` | `FrameView` | Board state just before the last real action |
| `latest_frame` | `FrameView` | Alias of `current_frame` |
| `history` | `list[HistoryEntryView]` | Chronological action/frame log; each entry has `.action`, `.frame` |
| `transitions` | `list[TransitionView]` | Real action transitions only; each has `.action`, `.before_frame`, `.after_frame`, `.result` |
| `last_transition` | `TransitionView` \| `None` | `transitions[-1]` |
| `last_action` | `str` \| `None` | Name of the most recent real action |
| `last_action_frame` | `FrameView` | Post-action frame for `last_action` |
| `last_action_result` | `dict` | Raw result of the last `action(...)` call (`board_changed`, `done`, `level_completed`, `game_over`, `reward`, `valid_actions`, ...) |
| `valid_actions` | `list[str]` | Actions currently legal |
| `action(actions)` | function | Executes one or more real environment actions; refreshes all state above |

## Native `current_frame.segmentation` (connected-component analysis)

Returns `{"nodes": [...], "adjacency_list": [...]}`. Each node:

| Field | Meaning |
|---|---|
| `id` | Index, ordered top-most-left-most |
| `color` | ARC color character |
| `hash` | Translation-invariant signature (color + normalized shape) — same object across frames gets the same hash regardless of position |
| `pixels` | Cell count |
| `boundary` | Clockwise outer-perimeter corners, `[[row, col], ...]` |
| `children` | IDs of objects fully enclosed by this one |

`adjacency_list`: `[i, j]` pairs of node IDs that share a 4-connected edge.

## Sandbox restrictions

- Modules: `bisect, collections, copy, fractions, functools, heapq, itertools, json, math, operator, random, re, statistics, string` only — no `numpy`, no `PIL`.
- Builtins: restricted safe subset (no `open`, `eval`, `exec`, `input`, network).
- CPU timeout + file/descriptor limits per call.

## Our 3 added tools (2026-09-11, ablation checkpoints in progress)

Injected via monkeypatch into the sandbox bootstrap — not a fork of the vendored harness. Each callable **both** as a free function and as a bound method on any `FrameView` (added after local testing showed the model naturally tries the method form).

| Tool | Signature | What it does | Why it helps |
|---|---|---|---|
| **`diff_frames`** | `diff_frames(before, after)` → `{"moved": [...], "appeared": [...], "disappeared": [...]}` | Object-level diff between two frames, matched by `hash` (translation-invariant). Each entry has `color`, `pixels`, position(s). | The sandbox has no memory between calls, so the model was re-deriving "what changed" by hand every single turn — burning tokens/time and contributing to the context-overflow bug we hit on the 27B run. One call replaces that boilerplate. |
| **`find_shared_palette_regions`** | `find_shared_palette_regions(frame)` → `[{"region_a": [ids], "region_b": [ids], "shared_colors": [...]}, ...]` | Clusters segmentation nodes into spatially separate regions (union-find, proximity ≤ 6px), reports region pairs sharing ≥2 colors. | Surfaces the "legend / workbench" pattern common in ARC-AGI-3: one region encodes a target configuration, a separate region must be made to match it. Ported from the user's own play heuristic (`_detect_shared_palette_regions` in `src/llm_tools_agent.py`), already confirmed to sharpen reasoning on `cd82`, `sk48`, `tr87` in an earlier session. |
| **`detect_symmetry`** | `detect_symmetry(frame)` → per symmetry type: `{"mismatch_fraction": float, "diff_cells": [[row, col], ...]}` for horizontal mirror, vertical mirror, 180° rotation, and (if square) diagonal transpose | Checks the ascii grid against each symmetry and returns exactly which cells break it. | A near-zero mismatch with only a few `diff_cells` often pinpoints the single controllable object or target cell directly, instead of the model scanning the whole board by eye. |

### Calling convention

Both forms work identically (added for `find_shared_palette_regions` and `detect_symmetry`; `diff_frames` stays free-function-only since it takes two frames, not one):

```python
regions = find_shared_palette_regions(current_frame)
regions = current_frame.find_shared_palette_regions()   # equivalent

sym = detect_symmetry(current_frame)
sym = current_frame.detect_symmetry()                    # equivalent

changes = diff_frames(previous_frame, current_frame)      # free function only
```

### System prompt addendum (per tool, only when enabled)

Each checkpoint condition enables exactly **one** of the three tools (isolated ablation), plus an explicit "do NOT hand-roll this by hand" instruction and a usage example — added after local testing showed a documentation-only mention wasn't enough to get the model to actually call the tool instead of reimplementing it inline.

## Known limitations / open findings

- Local smoke testing (`qwen2.5vl:7b`, `gemma4:e4b` via Ollama) showed small/mid models often reimplement the tool's logic by hand instead of calling it, even when documented — the explicit "do NOT hand-roll" + example fix measurably increased real invocation.
- `detect_symmetry`'s `diff_cells` can be dominated by a HUD/timer border row rather than the meaningful anomaly, on boards that have one — not yet filtered out.
- First real Kaggle test of `find_shared_palette_regions` (v2, pre-context-strengthening) showed the tool was **never invoked** across 9 games / 133 code blocks — that run is uninformative about the tool's usefulness, not evidence against it. The context-strengthened version is queued for retest.
