"""Approach #1 (LoRA distillation) prep, step 1/2: generate a supervised
(image, target_description) dataset for fine-tuning a vision-language model's
scene-description behavior, with ZERO human labeling and ZERO LLM calls --
every target description is templated directly from ToolsAgent's own
code-computed facts (blob positions/sizes/colors, and the structural
interrupted-pair detection in llm_tools_agent._detect_interrupted_pairs).

Motivation (see llm_relay_agent_experiments memory, 2026-09-04): Gemma and
Qwen2.5-VL both hallucinate spatial relationships when asked to describe an
annotated ARC-AGI-3 frame from scratch (e.g. calling two halves of one bar
"a cross intersection"). Rather than trusting either model's own guess as a
label, this generates guaranteed-correct targets from the exact geometry the
agent's code already has -- turning a hard perception problem into a cheap,
unlimited-supply supervised-learning problem.

Usage:
    python3 generate_vision_distillation_dataset.py --games ls20 cd82 ka59 re86 vc33 \
        --steps-per-game 150 --sample-every 5 --out-dir <dataset dir>

Not yet run at scale -- see train_vision_lora.py for the training script this
dataset feeds into. Both are ready to launch, not launched yet (explicit user
request: prepare only, run later).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "data", "ARC-AGI-3-Agents"))
os.environ.setdefault("OPERATION_MODE", "offline")
os.environ.setdefault("ENVIRONMENTS_DIR", os.path.join(REPO, "data", "environment_files"))

import numpy as np  # noqa: E402

import arc_agi  # noqa: E402
from llm_tools_agent import (  # noqa: E402
    ToolsAgent,
    _bulk_colors,
    _detect_interrupted_pairs,
    _find_blobs,
)
from llm_tools_vision_agent import _grid_to_image_b64  # noqa: E402
import base64  # noqa: E402
import io  # noqa: E402
from PIL import Image  # noqa: E402

ALL_GAMES = ["ls20", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52", "lp85", "m0r0",
             "r11l", "re86", "s5i5", "sb26", "sc25", "sk48", "sp80", "su15", "tn36", "tr87",
             "tu93", "vc33", "wa30", "ar25", "bp35"]


def render_gold_description(blobs: list[dict]) -> str:
    """Templates a guaranteed-correct description from code facts alone --
    this is the supervised target the LoRA fine-tune will teach a VLM to
    reproduce. Deliberately factual/plain rather than stylistically rich (the
    code can't know what something visually LOOKS like, only where/how big it
    is and how it relates to other objects) -- the point of this dataset is to
    fix relational/spatial mistakes, not to teach flavor text."""
    if not blobs:
        return "No distinct objects were detected in this frame."
    pairs = _detect_interrupted_pairs(blobs)
    described = set()
    sentences = []
    for pair in pairs:
        a, b, itr = pair["blob_a"], pair["blob_b"], pair["interrupter"]
        sentences.append(
            f"blob_{a} and blob_{b} are actually the SAME object (color {blobs[a]['color']}), "
            f"split into two visible pieces by blob_{itr} (color {blobs[itr]['color']}) sitting "
            f"in the small gap between them -- likely a track, bar, or slider with a position "
            f"marker on it, not two separate objects.")
        described.update([a, b, itr])
    for i, b in enumerate(blobs):
        if i in described:
            continue
        r, c = b["centroid"]
        sentences.append(
            f"blob_{i} is a color {b['color']} object of size {b['size']} pixels, "
            f"located at approximately row {r:.0f}, column {c:.0f}.")
    return " ".join(sentences)


def collect_dataset(games: list[str], steps_per_game: int, sample_every: int, out_dir: str) -> None:
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.jsonl")

    arcade = arc_agi.Arcade()
    n_examples = 0
    with open(manifest_path, "w") as manifest:
        for game_id in games:
            env = arcade.make(game_id)
            env.reset()
            agent = ToolsAgent(card_id="distill", game_id=game_id, agent_name="distill",
                                ROOT_URL="http://offline", record=False, arc_env=env, tags=[])
            agent.MAX_ACTIONS = steps_per_game

            latest = agent._convert_raw_frame_data(agent.arc_env.observation_space)
            for step in range(steps_per_game):
                grid = np.array(latest.frame[0], dtype=int)
                if step % sample_every == 0:
                    bulk = _bulk_colors(grid)
                    blobs = _find_blobs(grid, bulk | agent.self_colors | agent.attached_colors
                                         | agent.confirmed_bar_colors)
                    if blobs:
                        target_text = render_gold_description(blobs)
                        image_b64 = _grid_to_image_b64(grid, blobs)
                        img_path = os.path.join(img_dir, f"{game_id}_{step:04d}.png")
                        with open(img_path, "wb") as f:
                            f.write(base64.b64decode(image_b64))
                        manifest.write(json.dumps({
                            "game_id": game_id, "step": step,
                            "image_path": os.path.relpath(img_path, out_dir),
                            "n_blobs": len(blobs),
                            "target_text": target_text,
                        }) + "\n")
                        n_examples += 1

                action = agent.choose_action(agent.frames, latest)
                new_frame = agent.take_action(action)
                if new_frame:
                    agent.append_frame(new_frame)
                    latest = new_frame
                agent.action_counter += 1

            print(f"{game_id}: done")

    print(f"\n{n_examples} (image, target_text) examples written to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--games", nargs="+", default=["ls20", "cd82", "ka59", "re86", "vc33"])
    p.add_argument("--steps-per-game", type=int, default=150)
    p.add_argument("--sample-every", type=int, default=5,
                    help="save one training example every N steps (avoids near-duplicate "
                         "consecutive frames dominating the dataset)")
    p.add_argument("--out-dir", default=os.path.join(REPO, "checkpoints", "vision_distillation_dataset"))
    args = p.parse_args()
    collect_dataset(args.games, args.steps_per_game, args.sample_every, args.out_dir)
