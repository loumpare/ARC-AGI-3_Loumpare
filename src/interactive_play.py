"""Lets a HUMAN play a local ARC-AGI-3 game turn by turn through the
conversation, to capture how a person actually approaches an unfamiliar
puzzle (hypothesis -> test -> confirm -> execute -> verify) -- see
llm_relay_agent_experiments memory, 2026-09-07: every automated experiment
today points at "exploiting a confirmed direction" as the real gap, and the
user wants to compare against their own strategy.

No persistent process needed: each call replays the full action history
from a fresh RESET (these environments are deterministic and replays are
cheap at this scale), applies the new action, saves the resulting frame as
a PNG, and appends to the log -- so each turn is just one CLI invocation.

Usage:
    python3 src/interactive_play.py ls20 START       # reset, show initial frame
    python3 src/interactive_play.py ls20 ACTION1      # apply, show result
    python3 src/interactive_play.py ls20 ACTION3
    python3 src/interactive_play.py ls20 RESET        # in-game reset (real RESET action)
    python3 src/interactive_play.py ls20 RESTART      # wipe the log, start over from scratch
"""
from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "data", "ARC-AGI-3-Agents"))
os.environ.setdefault("OPERATION_MODE", "offline")
os.environ.setdefault("ENVIRONMENTS_DIR", os.path.join(REPO, "data", "environment_files"))

import numpy as np  # noqa: E402
from arcengine import GameAction  # noqa: E402
import arc_agi  # noqa: E402

from gif_utils import grid_to_image  # noqa: E402

LOG_DIR = os.path.join(REPO, "results", "interactive_play")
NAME_TO_ACTION = {a.name: a for a in [
    GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4,
    GameAction.ACTION5, GameAction.ACTION7, GameAction.RESET,
]}


def _log_path(game_id: str) -> str:
    return os.path.join(LOG_DIR, f"{game_id}_actions.json")


def _frame_path(game_id: str) -> str:
    return os.path.join(LOG_DIR, f"{game_id}_frame.png")


def _load_log(game_id: str) -> list[str]:
    path = _log_path(game_id)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def _save_log(game_id: str, actions: list[str]) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(_log_path(game_id), "w") as f:
        json.dump(actions, f, indent=2)


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python3 src/interactive_play.py <game_id> <ACTION_NAME|START|RESTART>")
        sys.exit(1)
    game_id, action_name = sys.argv[1], sys.argv[2].upper()

    if action_name == "RESTART":
        _save_log(game_id, [])
        print(f"Log wiped for {game_id}. Run again with START to begin.")
        return

    history = _load_log(game_id)
    if action_name != "START":
        if action_name not in NAME_TO_ACTION:
            print(f"Unknown action {action_name!r}. Legal: {sorted(NAME_TO_ACTION)} (or START/RESTART)")
            sys.exit(1)
        history.append(action_name)

    arcade = arc_agi.Arcade()
    env = arcade.make(game_id)
    env.reset()
    for name in history:
        env.step(NAME_TO_ACTION[name])
    latest = env.observation_space

    grid = np.array(latest.frame[0], dtype=int)
    os.makedirs(LOG_DIR, exist_ok=True)
    grid_to_image(grid).save(_frame_path(game_id))
    _save_log(game_id, history)

    legal_names = sorted(NAME_TO_ACTION[n].name for n in NAME_TO_ACTION
                          if NAME_TO_ACTION[n].value in latest.available_actions)
    print(f"game={game_id} action_taken={action_name} step={len(history)}")
    print(f"state={latest.state.name} levels_completed={latest.levels_completed}/{latest.win_levels}")
    print(f"legal_actions_now={legal_names}")
    print(f"frame saved to: {_frame_path(game_id)}")


if __name__ == "__main__":
    main()
