"""Comparison agent (2026-08-31), "self-model" variant of
src/llm_relay_active_agent.py -- adds a 4th brick the user asked for after
seeing the diagnosis on the previous run: the agent kept a stable GOAL of
"reach the small blue dot" for ~70 turns, but that blue dot almost certainly
IS the player's own on-screen marker (its position barely changed across
150 actions) -- the agent was chasing itself.

New in this variant:
  4. Self memory: "eyes" (gemma4:e4b) maintains a persistent hypothesis
     about which single visual element represents the agent itself -- the
     element whose position/appearance changes in response to actions,
     across many turns, as opposed to elements that stay fixed or animate
     on their own. Carried forward and refined turn to turn (SELF: line,
     same convention as GOAL:). The brain (qwen2.5:7b) is told explicitly
     not to set a goal that targets this element -- it already is it.

Keeps the three previous bricks unchanged:
  1. Change memory: eyes is shown its own previous description + the last
     action taken, and must say what changed.
  2. Goal memory: brain maintains a persistent GOAL/hypothesis line across
     turns and across episode resets.
  3. Active query: brain may ask eyes one clarifying question per turn
     before finalizing (capped at one round-trip).

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

Finally, maintain a hypothesis about which single element in the image \
represents the agent itself -- the entity being controlled by the actions. \
This is normally the element whose position or appearance changes in \
direct response to actions, tracked consistently across multiple turns, as \
opposed to elements that stay fixed or animate on their own. If you were \
given a previous self-hypothesis below, check whether it still holds given \
what just changed: keep it if it still fits, or revise it if the evidence \
points elsewhere. State it as exactly one line starting with "SELF:", e.g.:
SELF: the small blue cross-shaped marker near the center of the grid

Be concise (5-7 sentences total, plus the SELF line) and purely descriptive \
-- do not guess what the game's objective is.
"""

EYES_FOLLOWUP_SYSTEM_PROMPT = """\
You are the perception module for a game-playing agent, looking at the same \
image you just described. The decision module has a specific follow-up \
question about something it needs clarified before it can decide. Answer \
concisely (1-2 sentences), staying purely descriptive about the image -- do \
not guess what the game is about or what to do.
"""

BRAIN_SYSTEM_PROMPT = """\
You are the decision module for a game-playing agent. Another module (the \
"eyes") has looked at the current game grid and describes it below, \
including what changed since the last action, and its current hypothesis \
about which element represents yourself. You also carry a short-term \
memory of your current goal from previous turns -- update it as you learn \
more, or keep it unchanged if it's still your best guess.

Important: the element identified as "yourself" is NOT a target -- you \
already are it. Never set a GOAL that means reaching, collecting, or \
interacting with your own marker; use its position only as a reference \
point for where you currently are. Set goals about the OTHER elements in \
the description instead.

You do not know what any action does -- infer it from how the description \
changes after you act. Your goal is to reach a WIN state by completing the \
game's hidden objective. The game may also end in GAME_OVER, after which it \
resets and you try again with anything you learned (your goal memory \
carries over the reset).

You may respond in one of two modes:

MODE A -- ask a clarifying question first: if the eyes' description leaves \
something important ambiguous that would change your decision, respond \
with EXACTLY one line:
QUERY: <your specific question about the image>
and nothing else. Only use this when it would genuinely change your choice \
of action -- most turns you should have enough information already.

MODE B -- finalize your decision: respond in exactly three parts, in this \
order:
1. A short reasoning (1-2 sentences) about what you observe and what you plan.
2. One line starting with "GOAL:" stating your current hypothesis or \
objective in a single sentence (never your own marker -- see above).
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
        if line.startswith("GOAL:") or line.startswith("QUERY:") or line.startswith("SELF:"):
            continue
        for name in legal_names:
            if name in line:
                return name
    for name in legal_names:
        if re.search(rf"\b{name}\b", text.upper()):
            return name
    return None


def _parse_labeled_line(text: str, label: str, previous: str) -> str:
    for line in text.strip().splitlines():
        line = line.strip()
        if line.upper().startswith(label):
            return line.split(":", 1)[1].strip()
    return previous


def _parse_query(text: str) -> str | None:
    stripped = text.strip()
    if stripped.upper().startswith("QUERY:"):
        return stripped.split(":", 1)[1].strip()
    return None


def _query_eyes(image_b64: str, prev_description: str, last_action_name: str | None,
                 self_memory: str) -> str:
    parts = []
    if prev_description and last_action_name:
        parts.append(
            f"Your previous description of the grid was:\n{prev_description}\n\n"
            f"You then took action {last_action_name}. Here is the resulting image. "
            f"Describe the current grid, then describe what changed."
        )
    else:
        parts.append("Describe this game grid image.")
    if self_memory:
        parts.append(f"\nYour previous self-hypothesis was: {self_memory}\n"
                      f"Check if it still holds, and restate or revise it.")
    user_text = "\n".join(parts)
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


def _query_eyes_followup(image_b64: str, question: str) -> str:
    resp = requests.post(OLLAMA_URL, json={
        "model": EYES_MODEL,
        "messages": [
            {"role": "system", "content": EYES_FOLLOWUP_SYSTEM_PROMPT},
            {"role": "user", "content": question, "images": [image_b64]},
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


class RelaySelfAgent(Agent):
    """Relay agent + change memory + goal memory + active query + self memory."""

    MAX_ACTIONS = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_description = ""
        self.last_action_name: str | None = None
        self.goal_memory = ""
        self.self_memory = ""
        self.last_raw_response = ""
        self.last_query: str | None = None
        self.last_query_answer: str | None = None

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
            eyes_reply = _query_eyes(image_b64, self.last_description, self.last_action_name,
                                      self.self_memory)
        except Exception as e:
            eyes_reply = ""
            print(f"[RelaySelfAgent] eyes query failed: {e}")

        self.self_memory = _parse_labeled_line(eyes_reply, "SELF:", self.self_memory)
        description = eyes_reply

        self.last_query = None
        self.last_query_answer = None

        def build_prompt(desc: str) -> str:
            goal_line = f"Current goal: {self.goal_memory}" if self.goal_memory else \
                "Current goal: none yet -- this is early in the game."
            self_line = f"Self hypothesis (this is you, never a target): {self.self_memory}" if \
                self.self_memory else "Self hypothesis: not yet determined."
            return (
                f"Eyes module's description (current grid, what changed, self-hypothesis):\n{desc}\n\n"
                f"{self_line}\n"
                f"{goal_line}\n\n"
                f"Recent history:\n{_history_to_text(frames)}\n\n"
                f"Available actions: {', '.join(legal_names)}\n"
                f"Choose one action."
            )

        try:
            reply = _query_brain(build_prompt(description))
        except Exception as e:
            reply = ""
            print(f"[RelaySelfAgent] brain query failed: {e}")

        query = _parse_query(reply)
        if query:
            self.last_query = query
            try:
                answer = _query_eyes_followup(image_b64, query)
            except Exception as e:
                answer = ""
                print(f"[RelaySelfAgent] eyes followup failed: {e}")
            self.last_query_answer = answer
            description = f"{description}\n\nFollow-up question: {query}\nAnswer: {answer}"
            forced_prompt = build_prompt(description) + (
                "\n\n(You already asked one clarifying question, answered above. "
                "You must finalize your decision now -- respond in MODE B.)"
            )
            try:
                reply = _query_brain(forced_prompt)
            except Exception as e:
                reply = ""
                print(f"[RelaySelfAgent] brain finalize query failed: {e}")

        self.last_raw_response = reply
        self.goal_memory = _parse_labeled_line(reply, "GOAL:", self.goal_memory)
        chosen = _parse_action(reply, legal_names)
        if chosen is None:
            chosen = legal_names[0]  # fallback: first legal action, no game-specific bias
        action = ACTION_NAME_TO_ENUM[chosen]
        action.reasoning = reply[:300]

        self.last_description = eyes_reply
        self.last_action_name = chosen
        return action
