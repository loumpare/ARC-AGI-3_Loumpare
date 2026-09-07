"""Post-hoc reflection test: play a game normally with UnifiedVisionAgent,
then -- SEPARATELY, after the fact -- show the model the whole recorded
trajectory (sampled frames + the full action log) and ask it to infer what
the game's objective actually was, in hindsight.

Purpose: the live brain only ever sees the CURRENT frame plus a short text
summary, and is budget-capped (MAX_BRAIN_CALLS_GAME, see
llm_tools_vision_agent.py) -- this checks whether the SAME model, given the
full episode at once with no turn budget pressure, reasons about the goal
any better. Not used during actual gameplay; a diagnostic/reporting tool
only.

Usage:
    python3 src/replay_reflection.py --games ls20 cd82 re86 sk48 tr87 \
        --actions 150 --out-dir results/reflection_20260907
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "data", "ARC-AGI-3-Agents"))
os.environ.setdefault("OPERATION_MODE", "offline")
os.environ.setdefault("ENVIRONMENTS_DIR", os.path.join(REPO, "data", "environment_files"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import arc_agi  # noqa: E402

from gif_utils import ARC_COLORS, UPSCALE, grid_to_image  # noqa: E402
from llm_unified_vision_agent import UnifiedVisionAgent, OLLAMA_URL  # noqa: E402

REFLECTION_MODEL = "qwen3.8:latest"
REFLECTION_TIMEOUT_S = 300
N_SAMPLED_FRAMES = 12

REFLECTION_SYSTEM_PROMPT = """\
You are analyzing a recording of an AI agent playing an unfamiliar ARC-AGI-3 \
puzzle game, AFTER the fact. You did not play it yourself -- you are shown a \
sampled sequence of frames from across the whole episode (in chronological \
order, each labeled with its step number) and the complete list of actions \
taken every single step. Colors are drawn from a fixed 16-color palette; the \
grid is otherwise abstract (no text, no icons with inherent meaning).

Your job: figure out, in hindsight, what this game's actual objective and \
mechanic most likely were. You have information the live agent never had -- \
the ENTIRE trajectory, not just one frame at a time -- so look for patterns \
across the whole sequence: what changed together, what stayed constant, \
whether the same action ever produced different effects depending on \
context, whether progress correlates with a specific region/object/color.

Respond in exactly three parts:
1. GOAL: one or two concrete sentences on what you believe the objective was.
2. MECHANIC: one or two sentences on what mechanic/rule seems to govern how \
actions affect the grid (e.g. rigid avatar movement, a toggle/cycle, a \
matching puzzle, something else).
3. STRATEGY: if you had known this from the start, what should the agent \
have actually done differently?
"""


def run_and_record_with_actions(agent, out_gif_path: str, max_actions: int) -> dict:
    """Same loop as gif_utils.run_and_record, but also logs the action name
    taken at every step (needed for the reflection prompt's action log)."""
    from arcengine import FrameData  # noqa: F401

    agent.MAX_ACTIONS = max_actions
    frames_imgs = []
    actions_taken = []
    agent.timer = time.time()
    latest = agent._convert_raw_frame_data(agent.arc_env.observation_space)
    frames_imgs.append(grid_to_image(np.array(latest.frame[0], dtype=int)))

    while not agent.is_done(agent.frames, latest) and agent.action_counter <= agent.MAX_ACTIONS:
        action = agent.choose_action(agent.frames, latest)
        actions_taken.append(action.name)
        frame = agent.take_action(action)
        if frame:
            agent.append_frame(frame)
            latest = frame
            frames_imgs.append(grid_to_image(np.array(latest.frame[0], dtype=int)))
        agent.action_counter += 1

    agent.cleanup()

    from pathlib import Path
    Path(out_gif_path).parent.mkdir(parents=True, exist_ok=True)
    frames_imgs[0].save(
        out_gif_path, save_all=True, append_images=frames_imgs[1:],
        duration=120, loop=0, optimize=True,
    )

    return dict(
        frames_imgs=frames_imgs,
        actions_taken=actions_taken,
        levels_completed=int(agent.frames[-1].levels_completed),
        win_levels=int(agent.frames[-1].win_levels),
        state=agent.frames[-1].state.name,
        gif_path=out_gif_path,
        brain_notes=getattr(agent, "brain_notes", ""),
    )


def _img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def sample_frame_indices(n_frames: int, n_sample: int) -> list[int]:
    if n_frames <= n_sample:
        return list(range(n_frames))
    idxs = sorted(set(round(i * (n_frames - 1) / (n_sample - 1)) for i in range(n_sample)))
    return idxs


def build_action_log_text(actions_taken: list[str]) -> str:
    lines = [f"step {i+1}: {a}" for i, a in enumerate(actions_taken)]
    return "\n".join(lines)


def query_reflection(frames_imgs: list[Image.Image], actions_taken: list[str]) -> str:
    import requests

    idxs = sample_frame_indices(len(frames_imgs), N_SAMPLED_FRAMES)
    images_b64 = [_img_to_b64(frames_imgs[i]) for i in idxs]
    frame_label_lines = [f"Image {k+1} corresponds to the grid right after step {idx} "
                          f"(0 = initial frame, before any action)." for k, idx in enumerate(idxs)]
    user_prompt = (
        f"You are shown {len(idxs)} sampled frames from a {len(frames_imgs)-1}-step episode, "
        f"in this order:\n" + "\n".join(frame_label_lines) + "\n\n"
        f"Full action log (every single step, not just the sampled ones):\n"
        f"{build_action_log_text(actions_taken)}\n\n"
        f"Based on ALL of this, answer as instructed (GOAL / MECHANIC / STRATEGY)."
    )
    resp = requests.post(OLLAMA_URL, json={
        "model": REFLECTION_MODEL,
        "messages": [
            {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt, "images": images_b64},
        ],
        "stream": False,
        "options": {"temperature": 0.3},
    }, timeout=REFLECTION_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", required=True)
    ap.add_argument("--actions", type=int, default=150)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    arcade = arc_agi.Arcade()
    report_entries = []

    for game_id in args.games:
        print(f"=== {game_id}: playing ===", flush=True)
        env = arcade.make(game_id)
        env.reset()
        agent = UnifiedVisionAgent(card_id="reflection", game_id=game_id, agent_name="reflection",
                                    ROOT_URL="http://offline", record=False, arc_env=env, tags=[])
        gif_path = os.path.join(args.out_dir, f"{game_id}.gif")
        t0 = time.time()
        result = run_and_record_with_actions(agent, gif_path, args.actions)
        play_elapsed = time.time() - t0
        print(f"  played: levels_completed={result['levels_completed']} state={result['state']} "
              f"({play_elapsed:.1f}s)", flush=True)

        print(f"=== {game_id}: reflecting ===", flush=True)
        t0 = time.time()
        try:
            reflection = query_reflection(result["frames_imgs"], result["actions_taken"])
        except Exception as e:
            reflection = f"(reflection query failed: {e!r})"
        reflect_elapsed = time.time() - t0
        print(f"  reflection done ({reflect_elapsed:.1f}s)", flush=True)

        report_entries.append({
            "game_id": game_id,
            "gif_path": gif_path,
            "levels_completed": result["levels_completed"],
            "win_levels": result["win_levels"],
            "state": result["state"],
            "live_brain_notes": result["brain_notes"],
            "reflection": reflection,
            "play_elapsed_s": round(play_elapsed, 1),
            "reflect_elapsed_s": round(reflect_elapsed, 1),
        })
        with open(os.path.join(args.out_dir, "report.json"), "w") as f:
            json.dump(report_entries, f, indent=2)

    # markdown report
    md_lines = [f"# Reflection report ({len(args.games)} games, {args.actions} actions each)\n"]
    for e in report_entries:
        md_lines.append(f"## {e['game_id']}")
        md_lines.append(f"- levels_completed: {e['levels_completed']} / win_levels: {e['win_levels']} "
                         f"({e['state']})")
        md_lines.append(f"- GIF: `{e['gif_path']}`")
        md_lines.append(f"\n**Live in-game notes (turn-by-turn, budget-capped):**\n\n{e['live_brain_notes'] or '(none)'}\n")
        md_lines.append(f"**Retrospective reflection (full trajectory, no budget pressure):**\n\n{e['reflection']}\n")
        md_lines.append("---\n")
    with open(os.path.join(args.out_dir, "report.md"), "w") as f:
        f.write("\n".join(md_lines))
    print(f"\nReport written to {os.path.join(args.out_dir, 'report.md')}")


if __name__ == "__main__":
    main()
