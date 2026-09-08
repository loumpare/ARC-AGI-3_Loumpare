"""REPL-tool agent, adapted from Tufa Labs' "Duck Harness" (winning solution
to ARC-AGI-3 Milestone 1, github.com/Tufalabs/duck-harness, blog writeup at
tufalabs.ai/research/duck-harness/). See DESIGN_LOG.md 2026-09-08 for the
research trail that led here.

Core idea (theirs, not ours): give the LLM exactly ONE tool -- a Python REPL
-- with the current game state exposed as inspectable variables
(`current_frame`, `history`, `valid_actions`, ...) plus an `action(...)`
callable, instead of a menu of hand-built domain tools (pathfinder, blob
picker, etc. like src/llm_tools_agent.py). Their own finding, stated
explicitly in the writeup: hand-crafted tools HURT relative to letting the
model write its own inspection/decision code. This module is a direct test
of that claim against our stack (local Ollama qwen3.8, not their Qwen 3.6
27B FP8), reusing our own already-validated segmentation code
(llm_tools_agent._find_unfiltered_blobs/_bulk_colors) as the `.segmentation`
view instead of duck-harness's own segmenter -- same idea, our
implementation.

Deliberate simplifications vs. the original (noted so nobody mistakes this
for a faithful reproduction):
  - Their harness controls its own env-stepping loop and can call
    `action(...)` several times per LLM turn, each one stepping the REAL
    environment immediately and refreshing the runtime variables in place
    for the rest of that same code snippet. Our project's Agent contract
    (agents/agent.py Agent.main()) calls choose_action() exactly once per
    real environment step, so `action(...)` here only QUEUES actions; the
    first is executed immediately (this choose_action() call's return
    value), the rest execute automatically on subsequent choose_action()
    calls with NO further LLM call in between -- close in spirit (the model
    can still batch a confident multi-step plan) but not identical (no
    mid-snippet re-observation of a real env step).
  - No subprocess/Docker sandbox isolation -- exec() runs in-process with a
    reduced builtins/globals set (blocks file/network/process access) but
    is not adversarially hardened. Acceptable for a local research
    prototype running our own generated code, not for untrusted input.
  - No vision-language "attached image" turn by default (kept text-only via
    `.ascii` + `.segmentation` for a first test); can be added the same way
    llm_vlm_agent.py does if the text-only version shows a real signal
    worth amplifying.

No game-specific code anywhere (see feedback_no_game_hacking) -- the prompt
text below never references a game_id or a specific mechanic.
"""
from __future__ import annotations

import re
import string
from typing import Any

import numpy as np
import requests
from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent
from llm_tools_agent import (
    ACTION_NAME_TO_ENUM,
    ACTION_NAMES,
    ACTION_SPACE,
    _bulk_colors,
    _find_unfiltered_blobs,
)

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3.8:latest"
MAX_ACTIONS = 150
MAX_BRAIN_CALLS_TOTAL = 40   # separate LLM call budget, independent of MAX_ACTIONS -- most
                             # actions come from queued multi-action plans, not fresh calls
HISTORY_WINDOW = 6           # how many past (action, frame) entries the prompt shows
PYTHON_CALL_TIMEOUT_S = 10   # soft guard: if exec() runs long we still return, just log it
MAX_QUEUED_ACTIONS = 8       # cap how many actions one code snippet can queue blind

# 0-15 ARC color id -> a single readable ASCII character, same spirit as Duck
# Harness's ARC_COLOR_LEGEND (letters instead of raw digits so an LLM reading
# `.ascii` doesn't confuse a color id with a row/col number in its own reasoning).
_COLOR_CHARS = string.ascii_uppercase[:16]  # 'A'..'P' for color ids 0..15
ARC_COLOR_LEGEND = ", ".join(f"{c}={i}" for i, c in enumerate(_COLOR_CHARS))

_CLICK_NAMES = dict(ACTION_NAMES)
_CLICK_NAMES[GameAction.ACTION6.value] = "ACTION6"
_CLICK_NAME_TO_ENUM = dict(ACTION_NAME_TO_ENUM)
_CLICK_NAME_TO_ENUM["ACTION6"] = GameAction.ACTION6

def _safe_builtins() -> dict:
    import builtins as _b
    allowed = [
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
        "int", "len", "list", "map", "max", "min", "next", "print", "range",
        "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
        "isinstance", "type", "Exception", "ValueError", "TypeError",
        "RuntimeError", "frozenset", "divmod", "abs",
    ]
    return {name: getattr(_b, name) for name in allowed if hasattr(_b, name)}


def _grid_ascii(grid: np.ndarray) -> str:
    return "\n".join("".join(_COLOR_CHARS[int(v)] for v in row) for row in grid)


def _shape_hash(color: int, ys: np.ndarray, xs: np.ndarray) -> int:
    y0, x0 = int(ys.min()), int(xs.min())
    coords = tuple(sorted(zip((ys - y0).tolist(), (xs - x0).tolist())))
    return hash((color, coords))


def _segmentation(grid: np.ndarray) -> dict:
    """Same connected-component extraction already validated in
    llm_tools_agent.py (_find_unfiltered_blobs / _bulk_colors), reshaped into
    Duck Harness's node/adjacency view: id, color, hash (position-independent
    shape signature -- same hash means same object regardless of where it
    moved), pixel count, bbox, and an edge-adjacency list between nodes."""
    bulk = _bulk_colors(grid)
    raw = _find_unfiltered_blobs(grid, exclude_colors=bulk)
    nodes = []
    for i, b in enumerate(raw):
        y0, y1, x0, x1 = b["bbox"]
        mask = grid == b["color"]
        ys, xs = np.where(mask)
        # keep only the pixels belonging to THIS connected component's bbox region
        in_bbox = (ys >= y0) & (ys <= y1) & (xs >= x0) & (xs <= x1)
        ys, xs = ys[in_bbox], xs[in_bbox]
        nodes.append({
            "id": i,
            "color": b["color"],
            "hash": _shape_hash(b["color"], ys, xs),
            "pixels": b["size"],
            "bbox": [y0, x0, y1, x1],  # [row_min, col_min, row_max, col_max]
        })
    adjacency = []
    for i in range(len(nodes)):
        y0i, x0i, y1i, x1i = nodes[i]["bbox"]
        for j in range(i + 1, len(nodes)):
            y0j, x0j, y1j, x1j = nodes[j]["bbox"]
            touch = (y0i <= y1j + 1 and y1i >= y0j - 1 and x0i <= x1j + 1 and x1i >= x0j - 1)
            if touch:
                adjacency.append([i, j])
    return {"nodes": nodes, "adjacency_list": adjacency, "background_colors": sorted(bulk)}


class FrameView:
    def __init__(self, grid: np.ndarray, step: int, level: int):
        self._grid = grid
        self.step = step
        self.level = level
        self.shape = tuple(grid.shape)
        self._seg = None

    @property
    def ascii(self) -> str:
        return _grid_ascii(self._grid)

    @property
    def segmentation(self) -> dict:
        if self._seg is None:
            self._seg = _segmentation(self._grid)
        return self._seg

    def __repr__(self) -> str:
        return f"FrameView(level={self.level}, step={self.step}, shape={self.shape[0]}x{self.shape[1]})"


class HistoryEntry:
    def __init__(self, action: str, frame: FrameView):
        self.action = action
        self.frame = frame

    def __repr__(self) -> str:
        return f"HistoryEntry(action={self.action!r}, frame={self.frame})"


SYSTEM_PROMPT = f"""\
You are playing an abstract grid-based puzzle game you have never seen before, \
through a single Python tool. There is no rulebook -- infer everything from \
how the board changes after you act.

Runtime variables available in your code:
- `current_frame`: the latest board. `.ascii` is the grid as text (one \
character per cell, legend: {ARC_COLOR_LEGEND}). `.segmentation` returns \
{{'nodes': [...], 'adjacency_list': [...], 'background_colors': [...]}} -- \
each node is one connected same-color shape with id, color, hash (same hash \
= same shape regardless of position -- use it to track an object across \
frames), pixels (cell count), bbox=[row_min,col_min,row_max,col_max]. \
`background_colors` are colors covering >10% of the grid -- usually \
floor/wall, not an interactive object.
- `previous_frame`: the frame before the most recent action, or None.
- `history`: list of the last {HISTORY_WINDOW} HistoryEntry(action, frame).
- `valid_actions`: list of legal action name strings for this turn.
- `action(actions)`: queue one or more actions to actually execute, e.g. \
`action(["ACTION3"])` or `action([{{"action": "ACTION6", "row": 4, "col": 7}}])`. \
Row is vertical, col is horizontal. Only call this once per code snippet. \
Queuing more than one action means the rest execute automatically on later \
turns WITHOUT you seeing the intermediate frames or writing more code -- only \
batch actions you are confident about; if you need to re-observe after each \
step, queue exactly one.

Rules:
- Prefer `.segmentation` over reading `.ascii` cell by cell; use `.ascii` only \
to check a small specific region.
- A long thin strip of repeated small blocks flush against one edge of the \
grid is usually a HUD/timer/progress bar, not a set of clickable objects -- \
do not treat it as the puzzle.
- Once you understand the goal, prefer an explicit search (BFS/shortest-path) \
over guessing when a target position is involved.
- Your code has no state from previous calls -- recompute anything you need. \
Only stdlib names are available (no imports); a small numpy-free helper set \
is preloaded (no numpy, no file/network access).
- Do not print full grids. Print short, decision-relevant summaries.
- Respond with EXACTLY one fenced python code block and nothing else outside it:
```python
# your code here, ending with a call to action(...)
```
"""


def _extract_code(text: str) -> str | None:
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else None


def _query_llm(prompt: str, model_name: str = MODEL_NAME) -> str:
    resp = requests.post(OLLAMA_URL, json={
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.4},
    }, timeout=180)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


class ReplToolsAgent(Agent):
    """Duck-Harness-style single-Python-tool agent. See module docstring."""

    MAX_ACTIONS = MAX_ACTIONS
    MODEL_NAME = MODEL_NAME

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._history: list[HistoryEntry] = []
        self._prev_frame: FrameView | None = None
        self._queue: list[dict] = []
        self._brain_calls = 0
        self.last_raw_response = ""
        self.last_code = ""
        self.last_error = ""

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._history = []
            self._prev_frame = None
            self._queue = []
            return GameAction.RESET

        grid = np.array(latest_frame.frame[0], dtype=int)
        cur = FrameView(grid, step=self.action_counter, level=latest_frame.levels_completed)

        legal = [a for a in latest_frame.available_actions if a in _CLICK_NAMES]
        if not legal:
            return GameAction.RESET
        legal_names = [_CLICK_NAMES[a] for a in legal]

        if self._queue:
            queued = self._queue.pop(0)
            return self._to_game_action(queued, legal_names, cur)

        if self._brain_calls >= MAX_BRAIN_CALLS_TOTAL:
            # budget exhausted -- fall back to trying the first legal action rather
            # than crashing the run; still records history for post-hoc analysis
            act = self._to_game_action({"action": legal_names[0]}, legal_names, cur)
            self._record(legal_names[0], cur)
            return act

        prompt = self._build_prompt(cur, legal_names)
        try:
            reply = _query_llm(prompt, self.MODEL_NAME)
        except Exception as e:
            reply = ""
            print(f"[ReplToolsAgent] Ollama query failed: {e}")
        self.last_raw_response = reply
        self._brain_calls += 1

        code = _extract_code(reply) or ""
        self.last_code = code
        queued_actions = self._exec_code(code, cur, legal_names)

        if not queued_actions:
            # no valid action(...) call parsed -- fall back to first legal action
            queued_actions = [{"action": legal_names[0]}]

        self._queue = queued_actions[1:MAX_QUEUED_ACTIONS]
        first = queued_actions[0]
        self._record(first.get("action", legal_names[0]), cur)
        return self._to_game_action(first, legal_names, cur)

    def _record(self, action_name: str, cur: FrameView) -> None:
        self._history.append(HistoryEntry(action_name, cur))
        self._history = self._history[-HISTORY_WINDOW:]
        self._prev_frame = cur

    def _to_game_action(self, spec: dict, legal_names: list[str], cur: FrameView) -> GameAction:
        name = spec.get("action", legal_names[0])
        if name not in legal_names:
            name = legal_names[0]
        action = _CLICK_NAME_TO_ENUM[name]
        if name == "ACTION6":
            row = int(spec.get("row", cur.shape[0] // 2))
            col = int(spec.get("col", cur.shape[1] // 2))
            action.set_data({"x": col, "y": row})
        action.reasoning = (self.last_raw_response or "")[:300]
        return action

    def _build_prompt(self, cur: FrameView, legal_names: list[str]) -> str:
        hist_lines = []
        for h in self._history[-HISTORY_WINDOW:]:
            hist_lines.append(f"  action={h.action!r} -> level={h.frame.level} step={h.frame.step}")
        parts = [
            f"Current level: {cur.level}, step: {cur.step}.",
            f"Grid shape: {cur.shape[0]}x{cur.shape[1]}.",
            f"Valid actions this turn: {legal_names}.",
            "Recent history:" + ("\n" + "\n".join(hist_lines) if hist_lines else " (none yet)"),
        ]
        if self.last_error:
            # feed the exec failure back so the model can fix its own mistake instead of
            # repeating it blind -- without this it never learns its bbox/segmentation
            # format assumptions were wrong (confirmed empirically: repeated identical
            # "'int' object is not subscriptable" across consecutive calls with no feedback)
            parts.append(
                f"\nYour PREVIOUS code snippet raised an error and took no action: "
                f"{self.last_error}\nPrevious code was:\n```python\n{self.last_code}\n```\n"
                f"Fix this in your new code."
            )
        parts += ["", "Write your python code now."]
        return "\n".join(parts)

    def _exec_code(self, code: str, cur: FrameView, legal_names: list[str]) -> list[dict]:
        queued: list[dict] = []

        def _action(actions):
            if isinstance(actions, (str, dict)):
                actions = [actions]
            for a in actions[:MAX_QUEUED_ACTIONS]:
                if isinstance(a, str):
                    queued.append({"action": a})
                elif isinstance(a, dict):
                    queued.append(dict(a))

        sandbox_globals = {
            "__builtins__": _safe_builtins(),
            "current_frame": cur,
            "previous_frame": self._prev_frame,
            "history": list(self._history),
            "valid_actions": list(legal_names),
            "action": _action,
        }
        try:
            exec(compile(code, "<repl_agent>", "exec"), sandbox_globals, {})
            self.last_error = ""
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            print(f"[ReplToolsAgent] code exec failed: {self.last_error}")
        return queued
