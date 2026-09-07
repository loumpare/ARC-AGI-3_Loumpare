"""Runs a given agent class over all (or a subset of) the 25 local public
games and reports per-game outcome: self_found, levels_completed, win,
action_counter, and any uncaught exception -- the same shape of information
past sessions kept re-deriving ad hoc in throwaway scratchpad scripts.

Usage:
    python3 src/benchmark_agent.py --agent unified --games ls20 cd82 --actions 150
    python3 src/benchmark_agent.py --agent tools --out results/tools_25game.json
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

import numpy as np  # noqa: E402
import arc_agi  # noqa: E402

ALL_GAMES = ["ls20", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52", "lp85", "m0r0",
             "r11l", "re86", "s5i5", "sb26", "sc25", "sk48", "sp80", "su15", "tn36", "tr87",
             "tu93", "vc33", "wa30", "ar25", "bp35"]


def _agent_class(name: str):
    if name == "tools":
        from llm_tools_agent import ToolsAgent
        return ToolsAgent
    if name == "vision":
        from llm_tools_vision_agent import VisionToolsAgent
        return VisionToolsAgent
    if name == "unified":
        from llm_unified_vision_agent import UnifiedVisionAgent
        return UnifiedVisionAgent
    raise ValueError(f"unknown agent {name!r}")


def run_one(agent_cls, game_id: str, max_actions: int, arcade) -> dict:
    env = arcade.make(game_id)
    env.reset()
    agent = agent_cls(card_id="benchmark", game_id=game_id, agent_name="benchmark",
                       ROOT_URL="http://offline", record=False, arc_env=env, tags=[])
    agent.MAX_ACTIONS = max_actions
    t0 = time.time()
    result = {"game_id": game_id}
    try:
        agent.main()
        latest = agent.frames[-1] if agent.frames else None
        result.update({
            "self_found": bool(agent.self_colors),
            "self_colors": sorted(agent.self_colors),
            "levels_completed": int(latest.levels_completed) if latest else None,
            "win_levels": int(latest.win_levels) if latest else None,
            "won": bool(latest and latest.levels_completed >= latest.win_levels and latest.win_levels > 0),
            "state": str(latest.state) if latest else None,
            "action_counter": agent.action_counter,
            "brain_call_count": getattr(agent, "brain_call_count", None),
            "brain_notes": getattr(agent, "brain_notes", ""),
            "current_goal_key": str(getattr(agent, "current_goal_key", None)),
            "error": None,
        })
    except Exception as e:
        result.update({"self_found": None, "levels_completed": None, "won": False,
                        "action_counter": getattr(agent, "action_counter", None), "error": repr(e)})
    result["elapsed_s"] = round(time.time() - t0, 1)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["tools", "vision", "unified"], default="unified")
    ap.add_argument("--games", nargs="*", default=ALL_GAMES)
    ap.add_argument("--actions", type=int, default=150)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    agent_cls = _agent_class(args.agent)
    arcade = arc_agi.Arcade()

    results = []
    for game_id in args.games:
        print(f"=== {game_id} ===", flush=True)
        r = run_one(agent_cls, game_id, args.actions, arcade)
        results.append(r)
        print(f"  self_found={r['self_found']} levels_completed={r.get('levels_completed')} "
              f"won={r['won']} error={r.get('error')} ({r['elapsed_s']}s)", flush=True)

    n_self = sum(1 for r in results if r.get("self_found"))
    n_win = sum(1 for r in results if r.get("won"))
    print(f"\nTOTAL: self_found {n_self}/{len(results)}, won {n_win}/{len(results)}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
