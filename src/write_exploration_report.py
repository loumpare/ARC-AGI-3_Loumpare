"""Waits for the 2026-08-29 exploration grid (curiosity/archive/self-imitation,
15 runs across two launch batches -- see project_state.md memory) to finish,
then compiles a markdown report. Meant to run standalone/detached so it
survives after the interactive session that launched the grid ends.
"""
import json
import time
from collections import Counter
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
LOG_DIR = RESULTS_DIR / "logs"
OUT_PATH = RESULTS_DIR / "exploration_grid_report.md"

RUNS = {
    "archive_p0.3": "explore_archive_p0.3_20260829b",
    "archive_p0.6": "explore_archive_p0.6_20260829b",
    "archive_p0.9": "explore_archive_p0.9_20260829b",
    "replay_0.1": "explore_replay_0.1_20260829b",
    "replay_0.5": "explore_replay_0.5_20260829b",
    "replay_1.0": "explore_replay_1.0_20260829b",
    "combo_archive_replay": "explore_combo_archive_replay_20260829b",
    "curiosity_0.1": "explore_curiosity_0.1_20260829c",
    "curiosity_0.5": "explore_curiosity_0.5_20260829c",
    "curiosity_1.0": "explore_curiosity_1.0_20260829c",
    "combo_curiosity_archive": "explore_combo_curiosity_archive_20260829c",
    "combo_curiosity_replay": "explore_combo_curiosity_replay_20260829c",
    "combo_triple": "explore_combo_triple_20260829c",
    "combo_triple_cot": "explore_combo_triple_cot_20260829c",
    "combo_triple_imagination": "explore_combo_triple_imagination_20260829c",
}

BASELINE_REF = "final reward -1.793 (500 epochs) / -1.660 (2000 epochs, prior plateau) -- no mechanism enabled"


def is_done(run_name):
    log_path = LOG_DIR / f"{run_name}.log"
    if not log_path.exists():
        return False
    try:
        tail = log_path.read_text()[-2000:]
    except Exception:
        return False
    return "DONE in" in tail


def wait_for_all(poll_seconds=120):
    pending = set(RUNS.values())
    while pending:
        done_now = {r for r in pending if is_done(r)}
        pending -= done_now
        if pending:
            time.sleep(poll_seconds)
    return


def load_result(run_name):
    fp = RESULTS_DIR / f"{run_name}.json"
    if not fp.exists():
        return None
    try:
        return json.loads(fp.read_text())
    except Exception:
        return None


def build_report():
    rows = []
    for label, run_name in RUNS.items():
        d = load_result(run_name)
        if d is None:
            rows.append((label, run_name, None))
            continue
        outcome_log = d.get("outcome_log", [])
        wins = Counter(outcome_log).get("win", 0)
        skip_pct = 100 * d.get("skipped_updates", 0) / max(d.get("step", 1), 1)
        reward_hist = d.get("reward_history", [])
        final_reward = sum(reward_hist[-20:]) / len(reward_hist[-20:]) if reward_hist else None
        rows.append((label, run_name, dict(
            episodes=len(outcome_log),
            wins=wins,
            best=d.get("best_reward"),
            final_reward=final_reward,
            skip_pct=skip_pct,
            archive_final_size=d.get("archive_final_size", 0),
            nan_skips=d.get("nan_skips", 0),
            nan_frames=d.get("nan_frames_seen", 0),
        )))
    return rows


def render_markdown(rows):
    lines = []
    lines.append("# Grille d'exploration (curiosité / archive / self-imitation) -- 2026-08-29")
    lines.append("")
    lines.append(f"Référence sans mécanisme : {BASELINE_REF}")
    lines.append("")
    lines.append("| Config | Épisodes | Wins | Best | Reward final | Skip % | Archive finale | NaN |")
    lines.append("|---|---|---|---|---|---|---|---|")
    total_wins = 0
    for label, run_name, r in rows:
        if r is None:
            lines.append(f"| {label} | -- | -- | -- | -- | -- | -- | résultat manquant ({run_name}) |")
            continue
        total_wins += r["wins"]
        nan_flag = "oui" if (r["nan_skips"] or r["nan_frames"]) else "non"
        lines.append(
            f"| {label} | {r['episodes']} | {r['wins']} | {r['best']:.3f} | "
            f"{r['final_reward']:.3f} | {r['skip_pct']:.1f}% | {r['archive_final_size']} | {nan_flag} |"
        )
    lines.append("")
    lines.append(f"**Total wins sur la grille : {total_wins}**")
    lines.append("")

    valid = [(l, r) for l, _, r in rows if r is not None]
    if valid:
        best_by_final = max(valid, key=lambda x: x[1]["final_reward"])
        best_by_best = max(valid, key=lambda x: x[1]["best"])
        lines.append(f"- Meilleur reward final (moyenne des 20 derniers steps) : **{best_by_final[0]}** ({best_by_final[1]['final_reward']:.3f})")
        lines.append(f"- Meilleur `best_reward` isolé : **{best_by_best[0]}** ({best_by_best[1]['best']:.3f})")
        any_win = [l for l, r in valid if r["wins"] > 0]
        if any_win:
            lines.append(f"- Configs ayant trouvé au moins une victoire : {', '.join(any_win)}")
        else:
            lines.append("- Aucune config n'a trouvé de victoire.")
    lines.append("")
    lines.append("_Généré automatiquement par `src/write_exploration_report.py` une fois les 15 runs terminés._")
    return "\n".join(lines)


def main():
    wait_for_all(poll_seconds=120)
    rows = build_report()
    OUT_PATH.write_text(render_markdown(rows))
    print(f"Report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
