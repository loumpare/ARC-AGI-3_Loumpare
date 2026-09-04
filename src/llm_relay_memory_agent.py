"""Comparison agent (2026-08-31), memory variant of src/llm_relay_agent.py.

Adds two things the plain relay was missing:
  1. Change memory: "eyes" (gemma4:e4b) is shown its own previous description
     plus the action just taken, and must explicitly say what changed (which
     element, in which direction, color, shape) instead of describing the
     grid from scratch every time.
  2. Goal memory: "brain" (qwen2.5:7b) maintains a one-line GOAL/hypothesis
     that persists across turns (and across episode resets, so lessons
     learned aren't thrown away when the game restarts).

A third idea discussed with the user -- the brain actively querying the eyes
with a follow-up question before deciding -- is deliberately NOT implemented
here yet (would triple model calls per action); left for a v2 if this first
step shows a measurable effect.

Generic prompts only -- no per-game logic (see feedback_no_game_hacking).
Implements the same `agents.agent.Agent` contract so it can be tested
through the real ARC-AGI-3-Agents framework.
"""
from __future__ import annotations

import base64
import io
import re
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

OLLAMA_URL = "http://localhost:11434/api/chat"
EYES_MODEL = "gemma4:e4b"
BRAIN_MODEL = "qwen2.5:7b"
UPSCALE_SIZE = 512

ACTION_SPACE = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION7]
ACTION_NAMES = {a.value: a.name for a in ACTION_SPACE}
ACTION_NAME_TO_ENUM = {a.name: a for a in ACTION_SPACE}

EYES_SYSTEM_PROMPT = """\
You are the perception module for a game-playing agent. You are shown an \
image of a grid-based puzzle game you have never seen before.

First, describe what you currently see: grid size, distinct colors/regions \
present, notable shapes or objects, and their approximate positions \
(e.g. top-left, center, along an edge).

Then, if you are given your own description from before the last action and \
the name of that action, compare the two carefully and state what changed: \
which element moved and in which direction (up/down/left/right, or none), \
whether any color, shape, or rotation changed, or whether nothing visibly \
changed. If this is the first turn, skip this part.

Be concise (4-6 sentences total) and purely descriptive -- do not guess what \
the game is about or what to do next.
"""

BRAIN_SYSTEM_PROMPT = """\
You are the decision module for a game-playing agent. Another module (the \
"eyes") has looked at the current game grid and describes it below, \
including what changed since the last action if applicable. You also carry \
a short-term memory of your current goal or hypothesis about the game from \
previous turns -- update it as you learn more, or keep it unchanged if it's \
still your best guess.

You do not know what any action does -- infer it from how the description \
changes after you act. Your goal is to reach a WIN state by completing the \
game's hidden objective. The game may also end in GAME_OVER, after which it \
resets and you try again with anything you learned (your goal memory \
carries over the reset).

Respond in exactly three parts, in this order:
1. A short reasoning (1-2 sentences) about what you observe and what you plan.
2. One line starting with "GOAL:" stating your current hypothesis or \
objective in a single sentence.
3. On the LAST line, write exactly one of the available action names and \
nothing else, e.g.:
ACTION3
"""


def _grid_to_image_b64(grid) -> str:
    arr = np.array(grid, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=UPSCALE_SIZE // 6)
    ax.imshow(arr, vmin=0, vmax=15, cmap="tab20", interpolation="nearest")
    ax.axis("off")
    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _history_to_text(frames: list[FrameData]) -> str:
    lines = []
    tail = frames[-4:]
    for i, f in enumerate(tail):
        tag = f"t-{len(tail) - i - 1}" if i < len(tail) - 1 else "now"
        lines.append(f"[{tag}] state={f.state.name if hasattr(f.state, 'name') else f.state} "
                     f"levels_completed={f.levels_completed}")
    return "\n".join(lines)


def _parse_action(text: str, legal_names: list[str]) -> str | None:
    for line in reversed(text.strip().splitlines()):
        line = line.strip().upper()
        if line.startswith("GOAL:"):
            continue
        for name in legal_names:
            if name in line:
                return name
    for name in legal_names:
        if re.search(rf"\b{name}\b", text.upper()):
            return name
    return None


def _parse_goal(text: str, previous_goal: str) -> str:
    for line in text.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("GOAL:"):
            return line.split(":", 1)[1].strip()
    return previous_goal


def _query_eyes(image_b64: str, prev_description: str, last_action_name: str | None) -> str:
    if prev_description and last_action_name:
        user_text = (
            f"Your previous description of the grid was:\n{prev_description}\n\n"
            f"You then took action {last_action_name}. Here is the resulting image. "
            f"Describe the current grid, then describe what changed."
        )
    else:
        user_text = "Describe this game grid image."
    resp = requests.post(OLLAMA_URL, json={
        "model": EYES_MODEL,
        "messages": [
            {"role": "system", "content": EYES_SYSTEM_PROMPT},
            {"role": "user", "content": user_text, "images": [image_b64]},
        ],
        "stream": False,
        "options": {"temperature": 0.2},
    }, timeout=180)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _query_brain(prompt: str) -> str:
    resp = requests.post(OLLAMA_URL, json={
        "model": BRAIN_MODEL,
        "messages": [
            {"role": "system", "content": BRAIN_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.4},
    }, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


class RelayMemoryAgent(Agent):
    """Relay agent + change memory (eyes) + persistent goal memory (brain)."""

    MAX_ACTIONS = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_description = ""
        self.last_action_name: str | None = None
        self.goal_memory = ""
        self.last_raw_response = ""

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.last_description = ""
            self.last_action_name = None
            return GameAction.RESET

        legal = [a for a in latest_frame.available_actions if a in ACTION_NAMES]
        if not legal:
            return GameAction.RESET
        legal_names = [ACTION_NAMES[a] for a in legal]

        image_b64 = _grid_to_image_b64(latest_frame.frame[0])
        try:
            description = _query_eyes(image_b64, self.last_description, self.last_action_name)
        except Exception as e:
            description = ""
            print(f"[RelayMemoryAgent] eyes query failed: {e}")

        goal_line = f"Current goal/hypothesis: {self.goal_memory}" if self.goal_memory else \
            "Current goal/hypothesis: none yet -- this is early in the game."
        prompt = (
            f"Eyes module's description (current grid + what changed):\n{description}\n\n"
            f"{goal_line}\n\n"
            f"Recent history:\n{_history_to_text(frames)}\n\n"
            f"Available actions: {', '.join(legal_names)}\n"
            f"Choose one action."
        )
        try:
            reply = _query_brain(prompt)
        except Exception as e:
            reply = ""
            print(f"[RelayMemoryAgent] brain query failed: {e}")
        self.last_raw_response = reply

        self.goal_memory = _parse_goal(reply, self.goal_memory)
        chosen = _parse_action(reply, legal_names)
        if chosen is None:
            chosen = legal_names[0]  # fallback: first legal action, no game-specific bias
        action = ACTION_NAME_TO_ENUM[chosen]
        action.reasoning = reply[:300]

        self.last_description = description
        self.last_action_name = chosen
        return action
