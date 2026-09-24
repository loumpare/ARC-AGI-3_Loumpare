"""v2: label each transcript turn using REAL structured per-action ground
truth from events.jsonl (board_ascii, reward, level) instead of the coarse
"was the level eventually completed" heuristic from v1, which mislabeled
genuinely good reasoning as "stalled" whenever the level ran out of time.

Per-action label:
  - "progress": reward > 0 for this action (real scoring progress: a level
    advanced) OR the board actually changed vs the previous frame.
  - "no_effect": action executed but board_ascii identical to before (a
    wasted/uninformative action -- exactly the failure mode found in sb26).

A turn's THINKING is paired with the outcome of the action(s) it led to.
"""
import json
import re
from pathlib import Path

RUNS = {
    "baseline": "/tmp/baseline-v6-output",
    "D_both": "/tmp/variant-D_both-output",
    "F_timeout600": "/tmp/variant-F_timeout600-output",
}

TURN_HEADER_RE = re.compile(
    r"^--- analysis_step=(\d+) \| action=(\d+) \| [\d:]+ \| tool-agent ---$", re.MULTILINE
)
THINKING_RE = re.compile(r"\[THINKING\]\n(.*?)(?=\n\[|\Z)", re.DOTALL)
TOOLCALL_CODE_RE = re.compile(r"<parameter=code>\n(.*?)\n</parameter>", re.DOTALL)


def load_action_outcomes(events_path: Path) -> dict[int, dict]:
    """Map action_num -> {reward, board_changed, level} using real events.jsonl.

    Multiple events can share the same action_num (multi-frame animations for
    one real action) -- aggregate with max/any instead of last-write-wins,
    which was silently zeroing out real rewards recorded on an earlier frame
    of the same action_num.
    """
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
        existing = outcomes.get(a_num, {"reward": 0.0, "board_changed": False, "level": None})
        outcomes[a_num] = {
            "reward": max(existing["reward"], e.get("reward", 0.0) or 0.0),
            "board_changed": existing["board_changed"] or changed,
            "level": e.get("level") if e.get("level") is not None else existing["level"],
        }
    return outcomes


def mine_run(run_name: str, run_dir: str) -> list[dict]:
    records = []
    for txt_path in sorted(Path(f"{run_dir}/transcripts").glob("*.txt")):
        game_id = txt_path.name.split("-")[0]
        events_path = Path(f"{run_dir}/artifacts/{txt_path.stem}_events.jsonl")
        if not events_path.exists():
            continue
        outcomes = load_action_outcomes(events_path)

        text = txt_path.read_text(errors="ignore")
        headers = list(TURN_HEADER_RE.finditer(text))
        prev_action_num = 0
        for i, m in enumerate(headers):
            step_num, action_num = int(m.group(1)), int(m.group(2))
            turn_start = m.end()
            turn_end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
            turn_text = text[turn_start:turn_end]

            thinking_matches = THINKING_RE.findall(turn_text)
            if not thinking_matches:
                prev_action_num = action_num
                continue
            thinking = "\n---\n".join(t.strip() for t in thinking_matches)
            code_matches = TOOLCALL_CODE_RE.findall(turn_text)
            code = "\n---\n".join(c.strip() for c in code_matches)

            # A turn's action(...) call can batch multiple real actions before
            # returning control -- aggregate outcomes over the WHOLE range
            # since the previous turn's action_num, not just the last one.
            batch_outcomes = [
                outcomes[a] for a in range(prev_action_num + 1, action_num + 1) if a in outcomes
            ]
            prev_action_num = action_num
            if not batch_outcomes:
                continue
            max_reward = max(o["reward"] for o in batch_outcomes)
            any_changed = any(o["board_changed"] for o in batch_outcomes)
            final_level = batch_outcomes[-1]["level"]

            if max_reward > 0:
                label = "progress"
            elif any_changed:
                label = "board_changed_no_reward"
            else:
                label = "no_effect"

            records.append({
                "run": run_name,
                "game_id": game_id,
                "analysis_step": step_num,
                "action_num": action_num,
                "batch_size": len(batch_outcomes),
                "reward": max_reward,
                "level": final_level,
                "label": label,
                "thinking": thinking,
                "code": code,
            })
    return records


def main() -> None:
    all_records = []
    for run_name, run_dir in RUNS.items():
        recs = mine_run(run_name, run_dir)
        print(f"{run_name}: {len(recs)} turns mined")
        all_records.extend(recs)

    from collections import Counter
    counts = Counter(r["label"] for r in all_records)
    print(f"\nTOTAL: {len(all_records)} turns -> {dict(counts)}")

    out_path = Path("/tmp/reasoning_dataset_v2.jsonl")
    with out_path.open("w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
