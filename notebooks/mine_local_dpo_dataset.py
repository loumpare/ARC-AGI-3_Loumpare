"""mine_local_dpo_dataset.py — Step B of the DPO fine-tuning plan.

Mines "no_effect" turns from local game transcripts (ministral-3:14b runs),
reconstructs the full prompt context for each, generates a synthetic
verification-focused "chosen" response via Ollama, and saves the result
in trl.DPOTrainer conversational format.

Why local transcripts instead of Kaggle Qwen3.6-27B ones:
- Kaggle transcripts from v6/v7/v8 were in /tmp, lost after reboot.
- We're fine-tuning ministral-3:14b (not Qwen), so local transcripts
  better match the target model's own distribution.
- EWM-run no_effect turns don't reference EWM-specific tools (verified).

Output: results/local_dpo_dataset_YYYY-MM-DD.jsonl
  Each record: {"game_id", "run", "analysis_step", "action_num",
                "messages": [system, user, ...], "chosen": str, "rejected": str}

Usage: python3 notebooks/mine_local_dpo_dataset.py [--dry-run] [--max-turns N]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import date
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RUNS = {
    "local_ewm_1pass": "results/local_ewm_1pass_20260916",
    "local_ewm_4pass": "results/local_ewm_4pass_20260916",
}

# Only the last N analysis-step turns of conversation history to include in
# prompt (avoids OOM; keeps context focused on recent behaviour).
HISTORY_WINDOW = 3

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
OLLAMA_MODEL = "ministral-3:14b"

# Addendum appended to the USER message for synthetic chosen generation.
# It must not introduce tool names or concepts absent in the baseline system.
VERIFICATION_ADDENDUM = """
IMPORTANT FOR THIS TURN: Before calling action(), write a brief Python probe
that checks whether the proposed action will actually change the board state.
For example:
  - Compare current_frame.ascii or segmentation to what you expect after the
    action, using history[-1] or previous_frame as reference.
  - If you cannot confidently predict the effect, call action() with one test
    action, observe the result, and only continue if the board changed.
Do NOT repeat an action that had no visible effect on the previous turn."""

OUTPUT_DIR = Path("results")


# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------
TURN_HEADER_RE = re.compile(
    r"^--- analysis_step=(\d+) \| action=(\d+) \| [\d:]+ \| tool-agent ---$",
    re.MULTILINE,
)

def _split_sections(text: str) -> dict[str, list[str]]:
    """Return {section_name: [content, ...]} for all sections in a turn text."""
    parts = re.split(r"\n\[([A-Z][A-Z ]+[A-Z])\]\n", "\n" + text)
    result: dict[str, list[str]] = {}
    for i in range(1, len(parts), 2):
        name = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""
        result.setdefault(name, []).append(content.strip())
    return result


def parse_turns(transcript_text: str) -> list[dict]:
    """Parse transcript into list of turn dicts.

    Each dict: {analysis_step, action_num, system_prompt?, user_prompts[], assistants[]}
    system_prompt is only present on the first turn.
    """
    headers = list(TURN_HEADER_RE.finditer(transcript_text))
    turns = []
    for i, m in enumerate(headers):
        step = int(m.group(1))
        action_num = int(m.group(2))
        turn_start = m.end()
        turn_end = headers[i + 1].start() if i + 1 < len(headers) else len(transcript_text)
        turn_text = transcript_text[turn_start:turn_end]
        sections = _split_sections(turn_text)
        turns.append({
            "analysis_step": step,
            "action_num": action_num,
            "system_prompt": sections.get("SYSTEM PROMPT", [None])[0],
            "user_prompts": sections.get("USER PROMPT", []),
            "assistants": sections.get("ASSISTANT", []),
        })
    return turns


# ---------------------------------------------------------------------------
# Events loading
# ---------------------------------------------------------------------------

def load_outcomes(events_path: Path) -> dict[int, dict]:
    events = [json.loads(l) for l in events_path.read_text().splitlines() if l.strip()]
    events.sort(key=lambda e: e.get("action_num", 0))
    outcomes: dict[int, dict] = {}
    prev_board = None
    for e in events:
        a_num = e.get("action_num")
        if a_num is None:
            continue
        board = e.get("board_ascii", "")
        changed = prev_board is not None and board != prev_board
        prev_board = board
        ex = outcomes.get(a_num, {"reward": 0.0, "board_changed": False})
        outcomes[a_num] = {
            "reward": max(ex["reward"], e.get("reward", 0.0) or 0.0),
            "board_changed": ex["board_changed"] or changed,
        }
    return outcomes


# ---------------------------------------------------------------------------
# Prompt reconstruction
# ---------------------------------------------------------------------------

def build_messages(turns: list[dict], anchor_idx: int, history_window: int) -> list[dict]:
    """Build OpenAI-format messages list for the anchor turn.

    Includes: system prompt + (at most history_window) prior turns + current
    turn's first USER message.
    """
    # System prompt from first turn
    system_prompt = None
    for t in turns:
        if t["system_prompt"]:
            system_prompt = t["system_prompt"]
            break

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # History: prior turns (up to window), only first user/assistant pair each
    prior_turns = turns[:anchor_idx]
    windowed = prior_turns[-history_window:] if len(prior_turns) > history_window else prior_turns

    for t in windowed:
        if t["user_prompts"]:
            messages.append({"role": "user", "content": t["user_prompts"][0]})
        if t["assistants"]:
            messages.append({"role": "assistant", "content": t["assistants"][0]})

    # Anchor turn's USER message (first one)
    anchor = turns[anchor_idx]
    if anchor["user_prompts"]:
        messages.append({"role": "user", "content": anchor["user_prompts"][0]})

    return messages


# ---------------------------------------------------------------------------
# Synthetic chosen generation
# ---------------------------------------------------------------------------

def generate_chosen(messages: list[dict], temperature: float = 0.7) -> str | None:
    """Call Ollama ministral-3:14b with the anchor prompt + verification addendum."""
    # Inject addendum into last user message
    augmented = list(messages)
    if augmented and augmented[-1]["role"] == "user":
        augmented[-1] = {
            "role": "user",
            "content": augmented[-1]["content"] + "\n" + VERIFICATION_ADDENDUM,
        }

    payload = {
        "model": OLLAMA_MODEL,
        "messages": augmented,
        "temperature": temperature,
        "max_tokens": 1024,
        "stream": False,
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  [WARN] Ollama call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main mining loop
# ---------------------------------------------------------------------------

def mine_run(run_name: str, run_dir: str, dry_run: bool, max_turns: int) -> list[dict]:
    records = []
    for txt_path in sorted(Path(run_dir + "/transcripts").glob("*.txt")):
        game_id = txt_path.name.split("-")[0]
        events_path = Path(run_dir + "/artifacts/" + txt_path.stem + "_events.jsonl")
        if not events_path.exists():
            continue

        outcomes = load_outcomes(events_path)
        text = txt_path.read_text(errors="ignore")
        turns = parse_turns(text)
        if not turns:
            continue

        prev_action_num = 0
        for idx, turn in enumerate(turns):
            action_num = turn["action_num"]

            # Label the turn
            batch = [outcomes[a] for a in range(prev_action_num + 1, action_num + 1) if a in outcomes]
            prev_action_num = action_num
            if not batch:
                continue

            max_reward = max(o["reward"] for o in batch)
            any_changed = any(o["board_changed"] for o in batch)

            if any_changed or max_reward > 0:
                continue  # only want no_effect turns

            # Must have an assistant response to use as rejected
            if not turn["assistants"]:
                continue

            rejected = turn["assistants"][0]

            # Build prompt messages
            messages = build_messages(turns, idx, HISTORY_WINDOW)
            if len(messages) < 2:
                continue  # need at least system + user

            if dry_run:
                records.append({
                    "game_id": game_id,
                    "run": run_name,
                    "analysis_step": turn["analysis_step"],
                    "action_num": action_num,
                    "messages": messages,
                    "chosen": "[DRY RUN]",
                    "rejected": rejected,
                })
                continue

            # Generate synthetic chosen
            print(f"  Generating chosen for {game_id} step={turn['analysis_step']} ...", end=" ", flush=True)
            t0 = time.time()
            chosen = generate_chosen(messages)
            elapsed = time.time() - t0

            if chosen is None:
                print(f"FAILED ({elapsed:.1f}s)")
                continue
            print(f"OK ({elapsed:.1f}s, {len(chosen)} chars)")

            records.append({
                "game_id": game_id,
                "run": run_name,
                "analysis_step": turn["analysis_step"],
                "action_num": action_num,
                "messages": messages,
                "chosen": chosen,
                "rejected": rejected,
            })

            if max_turns and len(records) >= max_turns:
                return records

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Parse only, no Ollama calls")
    parser.add_argument("--max-turns", type=int, default=0, help="Stop after N turns (0=all)")
    args = parser.parse_args()

    all_records = []
    for run_name, run_dir in RUNS.items():
        print(f"\n=== {run_name} ===")
        recs = mine_run(run_name, run_dir, args.dry_run, args.max_turns)
        print(f"  {len(recs)} no_effect turns collected")
        all_records.extend(recs)
        if args.max_turns and len(all_records) >= args.max_turns:
            break

    print(f"\nTOTAL: {len(all_records)} DPO pairs")

    out_path = OUTPUT_DIR / f"local_dpo_dataset_{date.today()}.jsonl"
    with out_path.open("w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
