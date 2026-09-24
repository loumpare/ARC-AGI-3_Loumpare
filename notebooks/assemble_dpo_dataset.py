"""assemble_dpo_dataset.py — Step C of the DPO fine-tuning plan.

Converts the raw local_dpo_dataset_YYYY-MM-DD.jsonl (messages + chosen + rejected)
into trl.DPOTrainer conversational format and filters by token budget.

Input:  results/local_dpo_dataset_2026-09-21.jsonl
Output: results/dpo_train_dataset_2026-09-21/ (HuggingFace datasets format)
        results/dpo_train_stats_2026-09-21.json

Format (conversational, what trl.DPOTrainer accepts with dataset_type="conversational"):
  {"prompt": [{"role": ..., "content": ...}, ...],
   "chosen": [{"role": "assistant", "content": "..."}],
   "rejected": [{"role": "assistant", "content": "..."}]}

The system prompt is VERY long (~3k tokens). We truncate it to a compact header
that preserves the core tool interface and critical rules, reducing per-example
token count from ~8k to ~3k so examples fit in a 4096-token context.

Usage: python3 notebooks/assemble_dpo_dataset.py [--max-tokens 4096]
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

# Compact system prompt: preserves tool interface, key rules, drops tutorial text.
# Rationale: the EWM run transcripts use a long (12-16k char) system prompt.
# Fine-tuning the model on examples where the system prompt is truncated vs. real
# deployment (also truncated by context limit) is acceptable — both see the same
# short version at inference time when training examples are long.
COMPACT_SYSTEM_TEMPLATE = """\
You are a coding agent solving a grid-based puzzle game. Solve multi-level puzzles by calling the `python` tool to inspect game state and execute actions.

Runtime globals in every python call: `current_frame` (.ascii, .segmentation, .step, .level, .shape), `previous_frame`, `history`, `transitions`, `last_transition`, `valid_actions`, `last_action_result`. Call `action(list)` to execute actions (e.g. `action(['LEFT'])` or `action([{'action': 'MOUSE', 'row': R, 'col': C}])`).

Key rules:
- Use current_frame.segmentation as primary view (objects, colors, adjacency, containment, hashes).
- After action(), last_action_result has 'board_changed', 'reward', 'level_completed', 'done'.
- Before repeating an action, check last_action_result['board_changed'] -- if False, the action had no effect.
- If an action had no effect last time, do NOT repeat it without a reason. Instead: check valid_actions, probe adjacent objects, try a different approach, or verify your hypothesis first with a Python inspection call.
- Optimize for fewest actions while staying reliable. Stop acting if 'done' or 'game_over' is True.
- For MOUSE: pass row and col integer fields.
"""


def count_tokens_approx(text: str) -> int:
    """Rough token estimate: 4 chars/token (conservative)."""
    return len(text) // 4


def load_raw(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def build_conversational_record(raw: dict, max_tokens: int) -> dict | None:
    """Convert raw record to trl conversational format.

    Swaps system prompt to compact version, drops history turns if needed
    to stay within token budget, validates chosen != rejected.
    """
    if raw.get("chosen") in (None, "[DRY RUN]", ""):
        return None
    if raw.get("chosen") == raw.get("rejected"):
        return None

    # Build messages with compact system prompt
    messages = []
    orig_messages = raw["messages"]

    # Add compact system prompt
    messages.append({"role": "system", "content": COMPACT_SYSTEM_TEMPLATE})

    # Add prior conversation turns (skip any system messages from orig)
    for msg in orig_messages:
        if msg["role"] != "system":
            messages.append(msg)

    # Check total token count; drop oldest history pairs if needed
    chosen = raw["chosen"]
    rejected = raw["rejected"]
    total_chars = sum(len(m["content"]) for m in messages) + len(chosen) + len(rejected)
    budget_chars = max_tokens * 4

    # Drop oldest (user, assistant) pairs from history (keep system + latest user)
    while total_chars > budget_chars and len(messages) > 2:
        # Find first non-system user message to drop (+ following assistant if any)
        for i, m in enumerate(messages[1:], 1):
            if m["role"] == "user" and i < len(messages) - 1:
                # Drop this user + next assistant if present
                if messages[i + 1]["role"] == "assistant":
                    messages = messages[:i] + messages[i + 2:]
                else:
                    messages = messages[:i] + messages[i + 1:]
                total_chars = sum(len(m["content"]) for m in messages) + len(chosen) + len(rejected)
                break
        else:
            break  # nothing left to drop

    if sum(len(m["content"]) for m in messages) // 4 > max_tokens:
        return None  # still too long after trimming

    return {
        "game_id": raw.get("game_id", ""),
        "run": raw.get("run", ""),
        "analysis_step": raw.get("analysis_step", 0),
        "prompt": messages,
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--input", default=f"results/local_dpo_dataset_{date.today()}.jsonl")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}")
        return

    raw_records = load_raw(input_path)
    print(f"Loaded {len(raw_records)} raw records")

    converted = []
    skipped_no_chosen = 0
    skipped_too_long = 0
    skipped_duplicate = 0

    for r in raw_records:
        if r.get("chosen") in (None, "[DRY RUN]", ""):
            skipped_no_chosen += 1
            continue
        if r["chosen"] == r["rejected"]:
            skipped_duplicate += 1
            continue
        rec = build_conversational_record(r, args.max_tokens)
        if rec is None:
            skipped_too_long += 1
            continue
        converted.append(rec)

    print(f"Converted: {len(converted)}")
    print(f"Skipped (no chosen): {skipped_no_chosen}")
    print(f"Skipped (too long): {skipped_too_long}")
    print(f"Skipped (chosen==rejected): {skipped_duplicate}")

    # Token length stats
    lengths = [
        sum(len(m["content"]) for m in r["prompt"]) + len(r["chosen"][0]["content"]) + len(r["rejected"][0]["content"])
        for r in converted
    ]
    if lengths:
        avg = sum(lengths) / len(lengths)
        mx = max(lengths)
        print(f"Avg chars/example: {avg:.0f} (~{avg//4:.0f} tokens), max: {mx} (~{mx//4:.0f} tokens)")

    out_dir = Path(f"results/dpo_train_dataset_{date.today()}")
    out_dir.mkdir(exist_ok=True)

    # Save as JSONL (trl accepts this directly)
    out_path = out_dir / "train.jsonl"
    with out_path.open("w") as f:
        for r in converted:
            f.write(json.dumps(r) + "\n")
    print(f"Written: {out_path}")

    # Stats
    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps({
        "total_pairs": len(converted),
        "skipped_no_chosen": skipped_no_chosen,
        "skipped_too_long": skipped_too_long,
        "avg_chars": sum(lengths) / len(lengths) if lengths else 0,
        "max_chars": max(lengths) if lengths else 0,
    }, indent=2))
    print(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()
