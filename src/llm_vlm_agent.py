"""Comparison agent (2026-08-31), vision variant of src/llm_prompt_agent.py:
renders the grid as an actual image and asks a vision-language model
(gemma4:e4b via local Ollama -- gemma4:31b was tried first but doesn't fit
in 24GB with context/vision overhead, forcing 85% CPU offload and making it
impractically slow) to look at it directly, instead of dumping the grid as
a wall of text numbers. No training at all -- pure prompting.

Generic prompt only -- no per-game logic (see feedback_no_game_hacking).
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
MODEL_NAME = "gemma4:e4b"
UPSCALE_SIZE = 512  # 64x64 grid rendered at low res is too small for a VLM to resolve

ACTION_SPACE = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION7]
ACTION_NAMES = {a.value: a.name for a in ACTION_SPACE}
ACTION_NAME_TO_ENUM = {a.name: a for a in ACTION_SPACE}

SYSTEM_PROMPT = """\
You are playing an abstract grid-based puzzle game you have never seen before.
Each turn you are shown an image of the current game grid -- a small grid of \
colored cells -- and a list of action names you may take. You do not know \
what any action does -- you must infer it from how the image changes after \
you act. Your goal is to reach a WIN state by completing the game's hidden \
objective. The game may also end in GAME_OVER, after which it resets and you \
try again with anything you learned.

Look carefully at the shapes, colors, and layout in the image before deciding.

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


def _query_vlm(prompt: str, image_b64: str, model_name: str = MODEL_NAME) -> str:
    resp = requests.post(OLLAMA_URL, json={
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt, "images": [image_b64]},
        ],
        "stream": False,
        "options": {"temperature": 0.4},
    }, timeout=180)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


class VLMAgent(Agent):
    """Direct vision-prompting comparison agent, no training. See module docstring."""

    MAX_ACTIONS = 80
    MODEL_NAME = MODEL_NAME  # class attribute so a subclass/test harness can point this at a
                             # different local Ollama model (e.g. qwen3.8) without editing the
                             # module default other agents/scripts might still rely on

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
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
        prompt = (
            f"Recent history:\n{_history_to_text(frames)}\n\n"
            f"Available actions: {', '.join(legal_names)}\n"
            f"Look at the image and choose one action."
        )
        try:
            reply = _query_vlm(prompt, image_b64, self.MODEL_NAME)
        except Exception as e:
            reply = ""
            print(f"[VLMAgent] Ollama query failed: {e}")
        self.last_raw_response = reply

        chosen = _parse_action(reply, legal_names)
        if chosen is None:
            chosen = legal_names[0]  # fallback: first legal action, no game-specific bias
        action = ACTION_NAME_TO_ENUM[chosen]
        action.reasoning = reply[:300]
        return action
