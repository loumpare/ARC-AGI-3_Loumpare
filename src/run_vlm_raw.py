"""Runs the raw VLMAgent (no code-grounded tools at all -- just an image +
short history + available actions, every single turn) with qwen3.8 instead
of its original gemma4:e4b, for direct comparison against the heavily
code-scaffolded ToolsAgent/VisionToolsAgent/UnifiedVisionAgent family.

Usage:
    python3 src/run_vlm_raw.py --games ls20 cd82 tr87 --actions 80 --out-dir results/vlm_raw_qwen
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "data", "ARC-AGI-3-Agents"))
os.environ.setdefault("OPERATION_MODE", "offline")
os.environ.setdefault("ENVIRONMENTS_DIR", os.path.join(REPO, "data", "environment_files"))

import arc_agi  # noqa: E402
from gif_utils import run_and_record  # noqa: E402
from llm_vlm_agent import VLMAgent  # noqa: E402


class QwenVLMAgent(VLMAgent):
    MODEL_NAME = "qwen3.8:latest"


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
        agent = QwenVLMAgent(card_id="vlm-raw", game_id=game_id, agent_name="vlm-raw",
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
