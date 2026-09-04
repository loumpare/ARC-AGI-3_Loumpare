"""Comparison agent (2026-08-31), "eyes+brain" relay variant: two frozen
pretrained models chained the way our trained Eyes/Brain/Policy pipeline is
structured, but with zero training -- pure prompt relay.

Stage 1 ("Eyes"): gemma4:e4b (VLM) looks at the rendered grid image and
produces a structured text description (shapes, colors, layout, positions).
Stage 2 ("Brain"): qwen2.5:7b (LLM, text-only) reads that description plus
recent history and picks an action.

This is the "relay" (variant A) discussed with the user, as opposed to a
real embedding-level splice with a trained adapter (variant B, not
implemented -- would need local HF weights instead of Ollama's text-only
API, plus an actual GRPO training loop on the adapter).

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
image of a grid-based puzzle game you have never seen before. Describe what \
you see in plain text so a second module (which cannot see the image) can \
reason about it. Cover: grid size, the distinct colors/regions present, any \
notable shapes or objects, their approximate positions (e.g. top-left, \
center, along an edge), and anything that looks like a boundary, marker, or \
highlighted cell. Be concise (3-5 sentences) and purely descriptive -- do \
not guess what the game is about or what to do.
"""

BRAIN_SYSTEM_PROMPT = """\
You are the decision module for a game-playing agent. Another module (the \
"eyes") has already looked at the current game grid and given you a text \
description of it below. You do not know what any action does -- you must \
infer it from how the description changes after you act. Your goal is to \
reach a WIN state by completing the game's hidden objective. The game may \
also end in GAME_OVER, after which it resets and you try again with \
anything you learned.

Respond with a short reasoning (1-2 sentences) about what you observe and \
what you plan to try, then on the LAST line write exactly one of the \
available action names and nothing else, e.g.:
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
        for name in legal_names:
            if name in line:
                return name
    for name in legal_names:
        if re.search(rf"\b{name}\b", text.upper()):
            return name
    return None


def _query_eyes(image_b64: str) -> str:
    resp = requests.post(OLLAMA_URL, json={
        "model": EYES_MODEL,
        "messages": [
            {"role": "system", "content": EYES_SYSTEM_PROMPT},
            {"role": "user", "content": "Describe this game grid image.", "images": [image_b64]},
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


class RelayAgent(Agent):
    """Two frozen models chained eyes->brain, no training. See module docstring."""

    MAX_ACTIONS = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_description = ""
        self.last_raw_response = ""

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        legal = [a for a in latest_frame.available_actions if a in ACTION_NAMES]
        if not legal:
            return GameAction.RESET
        legal_names = [ACTION_NAMES[a] for a in legal]

        image_b64 = _grid_to_image_b64(latest_frame.frame[0])
        try:
            description = _query_eyes(image_b64)
        except Exception as e:
            description = ""
            print(f"[RelayAgent] eyes query failed: {e}")
        self.last_description = description

        prompt = (
            f"Eyes module's description of the current grid:\n{description}\n\n"
            f"Recent history:\n{_history_to_text(frames)}\n\n"
            f"Available actions: {', '.join(legal_names)}\n"
            f"Choose one action."
        )
        try:
            reply = _query_brain(prompt)
        except Exception as e:
            reply = ""
            print(f"[RelayAgent] brain query failed: {e}")
        self.last_raw_response = reply

        chosen = _parse_action(reply, legal_names)
        if chosen is None:
            chosen = legal_names[0]  # fallback: first legal action, no game-specific bias
        action = ACTION_NAME_TO_ENUM[chosen]
        action.reasoning = reply[:300]
        return action
