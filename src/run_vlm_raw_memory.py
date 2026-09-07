"""Same raw-qwen experiment as run_vlm_raw.py, but with ONE piece of memory
added: each turn, the model is shown the PREVIOUS frame (before its last
action) alongside the CURRENT frame (after it), labeled, plus which action
was taken between them -- a minimal, single-step before/after comparison,
not the full persistent-notes/action_deltas machinery the code-scaffolded
agents use. Everything else (no BFS, no self-ID, no landmarks) stays exactly
as bare as run_vlm_raw.py's VLMAgent.

Usage:
    python3 src/run_vlm_raw_memory.py --games ls20 --actions 80 --out-dir results/vlm_raw_memory_qwen
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "data", "ARC-AGI-3-Agents"))
os.environ.setdefault("OPERATION_MODE", "offline")
os.environ.setdefault("ENVIRONMENTS_DIR", os.path.join(REPO, "data", "environment_files"))

import numpy as np  # noqa: E402
import requests  # noqa: E402
from arcengine import FrameData, GameAction, GameState  # noqa: E402
import arc_agi  # noqa: E402

from agents.agent import Agent  # noqa: E402
from gif_utils import grid_to_image  # noqa: E402
from llm_vlm_agent import (  # noqa: E402
    ACTION_NAME_TO_ENUM,
    ACTION_NAMES,
    OLLAMA_URL,
    _grid_to_image_b64,
    _parse_action,
)

MODEL_NAME = "qwen3.8:latest"

SYSTEM_PROMPT = """\
You are playing an abstract grid-based puzzle game you have never seen before.
Each turn you are shown TWO images: the grid BEFORE your last action, and the \
grid AFTER it (the current grid) -- compare them to see exactly what your \
last action changed, if anything. You do not know what any action does in \
general -- you must infer it turn by turn from these before/after \
comparisons. Your goal is to reach a WIN state by completing the game's \
hidden objective. The game may also end in GAME_OVER, after which it resets \
and you try again with anything you learned.

Look carefully at what moved, appeared, disappeared, or changed color between \
the two images before deciding your next action.

Respond with a short reasoning (1-2 sentences) about what changed and what \
you plan to try, then on the LAST line write exactly one of the available \
action names and nothing else, e.g.:
ACTION3
"""


class QwenVLMMemoryAgent(Agent):
    """Copy of VLMAgent's choose_action, extended to show the previous frame
    + the action taken alongside the current frame (see module docstring) --
    a full method override rather than a thin subclass hook, since VLMAgent's
    choose_action isn't split into a wrapper+impl the way the ToolsAgent
    family is."""

    MAX_ACTIONS = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_raw_response = ""
        self.prev_image_b64: str | None = None
        self.prev_action_taken: str | None = None

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.prev_image_b64 = None
            self.prev_action_taken = None
            return GameAction.RESET

        legal = [a for a in latest_frame.available_actions if a in ACTION_NAMES]
        if not legal:
            return GameAction.RESET
        legal_names = [ACTION_NAMES[a] for a in legal]

        current_image_b64 = _grid_to_image_b64(latest_frame.frame[0])

        if self.prev_image_b64 is not None:
            images = [self.prev_image_b64, current_image_b64]
            prompt = (
                f"Image 1 = the grid BEFORE your last action.\n"
                f"Image 2 = the grid AFTER your last action (current) -- you took "
                f"{self.prev_action_taken} between them.\n\n"
                f"Available actions: {', '.join(legal_names)}\n"
                f"Compare the two images, then choose one action."
            )
        else:
            images = [current_image_b64]
            prompt = (
                f"This is the very first frame -- no previous action to compare against yet.\n\n"
                f"Available actions: {', '.join(legal_names)}\n"
                f"Look at the image and choose one action."
            )

        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt, "images": images},
                ],
                "stream": False,
                "options": {"temperature": 0.4},
            }, timeout=180)
            resp.raise_for_status()
            reply = resp.json()["message"]["content"]
        except Exception as e:
            reply = ""
            print(f"[QwenVLMMemoryAgent] Ollama query failed: {e}")
        self.last_raw_response = reply

        chosen = _parse_action(reply, legal_names)
        if chosen is None:
            chosen = legal_names[0]
        action = ACTION_NAME_TO_ENUM[chosen]
        action.reasoning = reply[:300]

        self.prev_image_b64 = current_image_b64
        self.prev_action_taken = chosen
        return action


def run_and_record(agent, out_gif_path: str, max_actions: int) -> dict:
    agent.MAX_ACTIONS = max_actions
    frames_imgs = []
    agent.timer = time.time()
    latest = agent._convert_raw_frame_data(agent.arc_env.observation_space)
    frames_imgs.append(grid_to_image(np.array(latest.frame[0], dtype=int)))

    while not agent.is_done(agent.frames, latest) and agent.action_counter <= agent.MAX_ACTIONS:
        action = agent.choose_action(agent.frames, latest)
        frame = agent.take_action(action)
        if frame:
            agent.append_frame(frame)
            latest = frame
            frames_imgs.append(grid_to_image(np.array(latest.frame[0], dtype=int)))
        agent.action_counter += 1

    agent.cleanup()
    from pathlib import Path
    Path(out_gif_path).parent.mkdir(parents=True, exist_ok=True)
    frames_imgs[0].save(out_gif_path, save_all=True, append_images=frames_imgs[1:],
                         duration=120, loop=0, optimize=True)
    return dict(actions=agent.action_counter, levels=agent.frames[-1].levels_completed,
                state=agent.frames[-1].state.name, n_frames=len(frames_imgs), gif_path=out_gif_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", required=True)
    ap.add_argument("--actions", type=int, default=80)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    arcade = arc_agi.Arcade()
    results = []
    for game_id in args.games:
        print(f"=== {game_id} ===", flush=True)
        env = arcade.make(game_id)
        env.reset()
        agent = QwenVLMMemoryAgent(card_id="vlm-mem", game_id=game_id, agent_name="vlm-mem",
                                    ROOT_URL="http://offline", record=False, arc_env=env, tags=[])
        gif_path = os.path.join(args.out_dir, f"{game_id}.gif")
        t0 = time.time()
        summary = run_and_record(agent, gif_path, args.actions)
        summary["elapsed_s"] = round(time.time() - t0, 1)
        summary["game_id"] = game_id
        summary["last_reasoning"] = agent.last_raw_response[:500]
        results.append(summary)
        print(f"  levels={summary['levels']} state={summary['state']} "
              f"({summary['elapsed_s']}s, {summary['actions']} actions)", flush=True)

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    n_lvl = sum(1 for r in results if r["levels"] > 0)
    print(f"\nTOTAL: levels_completed>0 in {n_lvl}/{len(results)}")


if __name__ == "__main__":
    main()
