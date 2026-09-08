"""Executable-world-model agent, adapted from "Executable World Models for
ARC-AGI-3 in the Era of Coding Agents" (Sergey Rodionov, arXiv:2605.05138,
AGI-2026; code at github.com/astroseger/arc-3-agents-baseline1). See
DESIGN_LOG.md 2026-09-08 for the research trail.

Core idea (theirs): instead of an LLM choosing actions directly from raw
observations every turn, have it MAINTAIN AN EXECUTABLE PYTHON MODEL of the
game's transition dynamics, VERIFY that model against recorded observations
(does the code actually reproduce what really happened?), and PLAN by
simulating candidate actions through the model before spending a real
environment step on them. Their full system runs Codex CLI + GPT-5.5 with
three separate files (engine/state-io/planner) and a replay+planner verifier
stack, reaching 58.12% mean RHAE on 25 public games (15/25 fully solved).

This is a DELIBERATELY SCALED-DOWN adaptation, not a reproduction -- we have
no Codex CLI / GPT-5.5-class access, only local Ollama (qwen3.8). Honest
differences from the paper, so nobody mistakes this for the real thing:
  - ONE function, not three files: `predict_next_ascii(ascii_grid, action)
    -> ascii_grid`, over the same letter-coded text grid the REPL agent uses
    (src/llm_repl_agent.py), not a rich object-level simulation.
  - Verification is exact-string-match rate over recorded transitions, not
    their fixed-interface replay-verifier tooling.
  - Planning is 1-ply lookahead (simulate every legal action once, prefer
    the one predicted to change the state the most / least like anything
    already seen) -- not a real search/planner module.
  - Refinement is budget-capped (MAX_REFINE_CALLS) local LLM calls, not an
    unbounded Codex agent session with its own tool-calling loop.
The refinement/verify/plan loop structure is the thing being tested, not
raw win-rate parity with a GPT-5.5-class model -- expect much lower model
fidelity from an ~8B local model and report that honestly.

No game-specific code anywhere (see feedback_no_game_hacking).
"""
from __future__ import annotations

import re
import signal
from typing import Any

import numpy as np
import requests
from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent
from llm_repl_agent import _CLICK_NAME_TO_ENUM, _CLICK_NAMES, _grid_ascii

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5-coder:7b"  # switched from qwen3.8 (27B general) 2026-09-08 after 4/4
                                  # refinement calls timed out at 240s locally with the full
                                  # two-full-grid prompt -- this task is pure code generation, a
                                  # smaller code-specialized model should both respond faster
                                  # (7B vs 27B) and be a better fit for the task itself; not yet
                                  # re-verified, see DESIGN_LOG.md 2026-09-08
MAX_ACTIONS = 150
MAX_REFINE_CALLS = 6         # LLM calls spent maintaining/refining the world-model code -- kept
                              # small (vs. the paper's unbounded Codex session) because a single
                              # refinement call with full 64x64 grids in the prompt measured
                              # 180s+ locally (qwen3.8 on this machine, 2026-09-08 smoke test),
                              # against a GPT-5.5/Codex-CLI cloud setup the paper's own budget
                              # doesn't need to economize this hard
REFINE_EVERY = 10            # attempt a refinement every N real actions (paper: "periodically
                              # asked to refactor" -- ours is action-count-paced instead of
                              # Codex-session-turn-paced)
TRANSITION_LOG_CAP = 40      # bounded transition history kept for verification/refinement
MAX_SHOWN_TRANSITIONS = 2    # how many FULL before/after grids go in the refinement prompt
                              # (context budget -- a 64x64 ascii grid is already ~4KB of text,
                              # trimmed from 4 to 2 for local-model latency, see MAX_REFINE_CALLS)
OLLAMA_TIMEOUT_S = 240        # generous local-inference timeout for the refinement call
                              # specifically (large prompt) -- caught gracefully either way,
                              # see _refine_world_model's try/except

_INITIAL_WORLD_MODEL = '''\
def predict_next_ascii(ascii_grid, action):
    """Predict the grid (as the same newline-delimited letter-coded ascii
    text) after taking `action` from `ascii_grid`. `action` is one of the
    legal action name strings (e.g. "ACTION1") or a dict like
    {"action": "ACTION6", "row": r, "col": c} for a click.
    Starting stub: identity (predicts nothing changes) -- must be refined.
    """
    return ascii_grid
'''

WORLD_MODEL_SYSTEM_PROMPT = """\
You are maintaining an executable Python world model for an unfamiliar \
grid-based puzzle game. You do not choose actions -- your only job is to \
write and refine ONE function:

def predict_next_ascii(ascii_grid, action):
    # ascii_grid: current board as newline-delimited text, one character per \
cell (A=color 0, B=color 1, ... letters are just color ids)
    # action: a legal action name string (e.g. "ACTION1"), or a dict like \
{"action": "ACTION6", "row": r, "col": c} for a click action
    # returns: predicted next ascii_grid, SAME format (newline-delimited, \
same character legend)
    ...

You will be shown REAL recorded (before_grid, action, after_grid) examples \
-- your job is to make this function actually reproduce them, as exactly as \
possible. You will also be told your CURRENT accuracy (fraction of recorded \
transitions your function predicts exactly right) so you know whether your \
last change helped or hurt.

Rules:
- Only stdlib is available at runtime (no imports needed/allowed for this \
task -- plain string/list operations on the grid text are enough).
- The function must be self-contained and deterministic.
- Do not hard-code a full transition table of memorized grids as your only \
logic -- infer a general rule (movement direction, what a click does, what \
toggles, what's a fixed wall/background) that would generalize to unseen \
positions, but you are allowed a fallback branch for cases you don't yet \
understand.
- Respond with EXACTLY one fenced python code block containing the complete \
new function definition (nothing else outside the block):
```python
def predict_next_ascii(ascii_grid, action):
    ...
```
"""


def _extract_code(text: str) -> str | None:
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else None


def _query_llm(system: str, prompt: str, model_name: str = MODEL_NAME) -> str:
    resp = requests.post(OLLAMA_URL, json={
        "model": model_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.3},
    }, timeout=OLLAMA_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _safe_builtins() -> dict:
    import builtins as _b
    allowed = [
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
        "int", "len", "list", "map", "max", "min", "next", "range",
        "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
        "isinstance", "type", "Exception", "ValueError", "TypeError",
        "RuntimeError", "frozenset", "divmod",
    ]
    return {name: getattr(_b, name) for name in allowed if hasattr(_b, name)}


def _compile_model(code: str) -> tuple[Any, str]:
    """Exec the candidate world-model code in a restricted namespace, return
    (predict_fn or None, error_str)."""
    ns: dict = {}
    try:
        exec(compile(code, "<world_model>", "exec"), {"__builtins__": _safe_builtins()}, ns)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    fn = ns.get("predict_next_ascii")
    if fn is None:
        return None, "no predict_next_ascii function defined"
    return fn, ""


PREDICT_TIMEOUT_S = 2  # LLM-generated code could contain an accidental infinite loop (e.g. a
                        # malformed scan over the grid text); this runs many times per turn
                        # (once per legal action for planning, plus once per logged transition
                        # for verification), so a hang here would silently freeze the whole
                        # agent -- unlike the one-off compile check, this guards every call


class _PredictTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _PredictTimeout()


def _run_predict(fn, ascii_grid: str, action) -> tuple[str | None, str]:
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(PREDICT_TIMEOUT_S)
    try:
        out = fn(ascii_grid, action)
        if not isinstance(out, str):
            return None, f"predict_next_ascii returned {type(out).__name__}, expected str"
        return out, ""
    except _PredictTimeout:
        return None, f"predict_next_ascii did not return within {PREDICT_TIMEOUT_S}s (likely infinite loop)"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _match_score(pred: str, actual: str) -> float:
    """Fraction of matching characters, position-wise, over the longer of
    the two strings -- graded signal (a small local model will rarely get an
    EXACT match early on; this makes partial progress visible instead of an
    all-or-nothing 0/1) alongside the strict exact-match rate reported
    separately for the headline number."""
    if pred == actual:
        return 1.0
    n = max(len(pred), len(actual), 1)
    matches = sum(1 for a, b in zip(pred, actual) if a == b)
    return matches / n


class WorldModelAgent(Agent):
    """Executable-world-model agent. See module docstring."""

    MAX_ACTIONS = MAX_ACTIONS
    MODEL_NAME = MODEL_NAME

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.world_model_code = _INITIAL_WORLD_MODEL
        self.predict_fn, _ = _compile_model(self.world_model_code)
        self.transitions: list[tuple[str, Any, str]] = []  # (before_ascii, action, after_ascii)
        self.refine_calls = 0
        self.last_refine_error = ""
        self.exact_match_rate = 0.0
        self.soft_match_rate = 0.0
        self._prev_ascii: str | None = None
        self._prev_action = None
        self._round_robin_i = 0

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._prev_ascii = None
            self._prev_action = None
            return GameAction.RESET

        grid = np.array(latest_frame.frame[0], dtype=int)
        ascii_grid = _grid_ascii(grid)

        # record the transition caused by the LAST real action, if any
        if self._prev_ascii is not None and self._prev_action is not None:
            self.transitions.append((self._prev_ascii, self._prev_action, ascii_grid))
            self.transitions = self.transitions[-TRANSITION_LOG_CAP:]

        legal = [a for a in latest_frame.available_actions if a in _CLICK_NAMES]
        if not legal:
            return GameAction.RESET
        legal_names = [_CLICK_NAMES[a] for a in legal]

        due_refine = (self.action_counter > 0 and self.action_counter % REFINE_EVERY == 0
                      and self.refine_calls < MAX_REFINE_CALLS and self.transitions)
        if due_refine:
            self._refine_world_model()

        self._recompute_accuracy()

        action_spec = self._plan_one_action(ascii_grid, legal_names)
        action = self._to_game_action(action_spec, legal_names, grid.shape)

        self._prev_ascii = ascii_grid
        self._prev_action = action_spec
        return action

    def _plan_one_action(self, ascii_grid: str, legal_names: list[str]) -> Any:
        """1-ply lookahead: simulate every legal action through the current
        world model, prefer the one predicted to differ most from the
        current state (novelty proxy -- avoid actions the model believes are
        no-ops) among actions not already confirmed-no-op in the real
        transition log. Falls back to round-robin while the model is still
        just the identity stub (predicting zero change for everything would
        otherwise tie every action at score 0)."""
        if self.predict_fn is None or self.exact_match_rate == 0.0 and self.soft_match_rate < 0.99:
            # model not yet informative (still identity-like) -- explore round-robin
            # over actions not yet confirmed as a real no-op in the transition log
            confirmed_noop = self._confirmed_noop_actions()
            candidates = [n for n in legal_names if n not in confirmed_noop] or legal_names
            name = candidates[self._round_robin_i % len(candidates)]
            self._round_robin_i += 1
            return self._name_to_spec(name, ascii_grid)

        best_name, best_score = None, -1.0
        for name in legal_names:
            spec = self._name_to_spec(name, ascii_grid)
            pred, err = _run_predict(self.predict_fn, ascii_grid, spec)
            if err or pred is None:
                continue
            score = 1.0 - _match_score(pred, ascii_grid)  # higher = predicted to change more
            if score > best_score:
                best_name, best_score = name, score
        if best_name is None:
            best_name = legal_names[self._round_robin_i % len(legal_names)]
            self._round_robin_i += 1
        return self._name_to_spec(best_name, ascii_grid)

    def _confirmed_noop_actions(self) -> set[str]:
        noop = set()
        seen = {}
        for before, act, after in self.transitions:
            name = act if isinstance(act, str) else act.get("action")
            seen.setdefault(name, []).append(before == after)
        for name, results in seen.items():
            if results and all(results) and len(results) >= 2:
                noop.add(name)
        return noop

    def _name_to_spec(self, name: str, ascii_grid: str) -> Any:
        if name != "ACTION6":
            return name
        rows = ascii_grid.split("\n")
        return {"action": "ACTION6", "row": len(rows) // 2, "col": len(rows[0]) // 2 if rows else 0}

    def _to_game_action(self, spec: Any, legal_names: list[str], shape: tuple[int, int]) -> GameAction:
        name = spec if isinstance(spec, str) else spec.get("action", legal_names[0])
        if name not in legal_names:
            name = legal_names[0]
            spec = name
        action = _CLICK_NAME_TO_ENUM[name]
        if name == "ACTION6":
            row = int(spec.get("row", shape[0] // 2)) if isinstance(spec, dict) else shape[0] // 2
            col = int(spec.get("col", shape[1] // 2)) if isinstance(spec, dict) else shape[1] // 2
            action.set_data({"x": col, "y": row})
        return action

    def _recompute_accuracy(self) -> None:
        if not self.transitions or self.predict_fn is None:
            self.exact_match_rate = 0.0
            self.soft_match_rate = 0.0
            return
        exact, soft, n = 0, 0.0, 0
        for before, act, after in self.transitions:
            pred, err = _run_predict(self.predict_fn, before, act)
            if err or pred is None:
                n += 1
                continue
            if pred == after:
                exact += 1
            soft += _match_score(pred, after)
            n += 1
        self.exact_match_rate = exact / n if n else 0.0
        self.soft_match_rate = soft / n if n else 0.0

    def _refine_world_model(self) -> None:
        self.refine_calls += 1
        sample = self.transitions[-MAX_SHOWN_TRANSITIONS:]
        example_blocks = []
        for i, (before, act, after) in enumerate(sample):
            example_blocks.append(
                f"Example {i}: action={act!r}\nBEFORE:\n{before}\nAFTER:\n{after}\n"
            )
        prompt_parts = [
            f"Current function:\n```python\n{self.world_model_code}```\n",
            f"Current accuracy over {len(self.transitions)} recorded transitions: "
            f"{self.exact_match_rate:.0%} exact match, {self.soft_match_rate:.0%} character-level match.",
        ]
        if self.last_refine_error:
            prompt_parts.append(f"Your last submitted version FAILED to run: {self.last_refine_error}")
        prompt_parts.append("\nRecorded examples:\n" + "\n".join(example_blocks))
        prompt_parts.append("Write an improved predict_next_ascii now.")
        prompt = "\n".join(prompt_parts)

        try:
            reply = _query_llm(WORLD_MODEL_SYSTEM_PROMPT, prompt, self.MODEL_NAME)
        except Exception as e:
            print(f"[WorldModelAgent] Ollama query failed: {e}")
            return

        code = _extract_code(reply)
        if not code:
            self.last_refine_error = "no python code block in response"
            return

        fn, err = _compile_model(code)
        if err:
            self.last_refine_error = err
            print(f"[WorldModelAgent] refined model failed to compile: {err}")
            return

        # accept only if it does not make exact-match accuracy on the recorded
        # log worse than the current model -- a cheap regression guard, same
        # spirit as the paper's replay verifier gating acceptance
        old_fn = self.predict_fn
        old_exact = self.exact_match_rate
        self.predict_fn = fn
        self._recompute_accuracy()
        new_exact = self.exact_match_rate
        if new_exact < old_exact and old_fn is not None:
            self.predict_fn = old_fn
            self._recompute_accuracy()
            self.last_refine_error = (
                f"rejected: new version scored {new_exact:.0%} exact match "
                f"vs previous {old_exact:.0%} on the same transitions"
            )
            return

        self.world_model_code = code
        self.last_refine_error = ""
