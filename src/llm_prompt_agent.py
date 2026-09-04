"""Comparison agent (2026-08-31): no training at all -- prompts a general-
purpose pretrained LLM (Qwen2.5-7B-Instruct via local Ollama) directly with
a text description of the grid and asks it to choose an action. Tests
whether a capable off-the-shelf model's own reasoning beats our from-scratch
GRPO-trained policy (which never found a win on ls20) without any RL.

Generic prompt only -- no per-game logic (see feedback_no_game_hacking).
Implements the same `agents.agent.Agent` contract as `src/my_agent.py` so it
can be tested through the real ARC-AGI-3-Agents framework.
"""
from __future__ import annotations

import re
from typing import Any

import requests
from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:7b"

ACTION_SPACE = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION7]
ACTION_NAMES = {a.value: a.name for a in ACTION_SPACE}
ACTION_NAME_TO_ENUM = {a.name: a for a in ACTION_SPACE}

SYSTEM_PROMPT = """\
You are playing an abstract grid-based puzzle game you have never seen before.
Each turn you are shown a grid of numbers (0-15, each number is a distinct \
color/state) and a list of action names you may take. You do not know what \
any action does -- you must infer it from how the grid changes after you act. \
Your goal is to reach a WIN state by completing the game's hidden objective. \
The game may also end in GAME_OVER, after which it resets and you try again \
with anything you learned.

Respond with a short reasoning (1-2 sentences) about what you observe and \
what you plan to try, then on the LAST line write exactly one of the \
available action names and nothing else, e.g.:
ACTION3
"""


def _grid_to_text(grid) -> str:
    return "\n".join(" ".join(str(v) for v in row) for row in grid)


def _history_to_text(frames: list[FrameData]) -> str:
    lines = []
    for i, f in enumerate(frames[-4:]):
        tag = f"t-{len(frames[-4:]) - i - 1}" if i < len(frames[-4:]) - 1 else "now"
        lines.append(f"[{tag}] state={f.state.name if hasattr(f.state, 'name') else f.state} "
                     f"levels_completed={f.levels_completed}")
    return "\n".join(lines)


def _parse_action(text: str, legal_names: list[str]) -> str | None:
    for line in reversed(text.strip().splitlines()):
        line = line.strip().upper()
        for name in legal_names:
            if name in line:
                return name
    # fallback: search whole text
    for name in legal_names:
        if re.search(rf"\b{name}\b", text.upper()):
            return name
    return None


def _query_llm(prompt: str) -> str:
    resp = requests.post(OLLAMA_URL, json={
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.4},
    }, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


class LLMAgent(Agent):
    """Direct-prompting comparison agent, no training. See module docstring."""

    MAX_ACTIONS = 80

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

        prompt = (
            f"Grid ({len(latest_frame.frame[0])}x{len(latest_frame.frame[0][0])}):\n"
            f"{_grid_to_text(latest_frame.frame[0])}\n\n"
            f"Recent history:\n{_history_to_text(frames)}\n\n"
            f"Available actions: {', '.join(legal_names)}\n"
            f"Choose one action."
        )
        try:
            reply = _query_llm(prompt)
        except Exception as e:
            reply = ""
            print(f"[LLMAgent] Ollama query failed: {e}")
        self.last_raw_response = reply

        chosen = _parse_action(reply, legal_names)
        if chosen is None:
            chosen = legal_names[0]  # fallback: first legal action, no game-specific bias
        action = ACTION_NAME_TO_ENUM[chosen]
        action.reasoning = reply[:300]
        return action
