#!/usr/bin/env python3
"""Per-game blocker diagnostics + run-to-run variance for ARC-AGI-3 benchmark runs.

Usage:
  python scripts/blocker_variance_report.py RUN_DIR [RUN_DIR ...] [--out report_dir]

A RUN_DIR is a local benchmark job dir (benchmark.json + artifacts/*_events.jsonl)
or a downloaded Kaggle kernel output (artifacts/*_events.jsonl, optional
*_requests.jsonl). Pass several dirs of the same config to get variance.
Writes <out>/blockers.csv, <out>/variance.csv, <out>/report.md and prints report.md.
"""
import argparse, csv, glob, json, math, os, random, re, statistics as st
from collections import Counter

GAME_RE = re.compile(r"([a-z0-9]+)-[0-9a-f]+_p(\d+)_events\.jsonl$")


def b(v):
    return str(v).lower() == "true"


def load_events(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_bench(run_dir):
    p = os.path.join(run_dir, "benchmark.json")
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))
    return {r["game_id"].split("-")[0]: r for r in d.get("game_runs", [])}


def request_stats(run_dir, gid):
    """Optional Kaggle-style *_requests.jsonl: finish_reason / token usage per LLM call."""
    files = glob.glob(os.path.join(run_dir, f"{gid}-*_p*_requests.jsonl"))
    if not files:
        return {}
    fr, n, ptok, ctok, rtok, empty = Counter(), 0, 0, 0, 0, 0
    for fp in files:
        with open(fp) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("event") != "response":
                    continue
                n += 1
                fr[r.get("finish_reason")] += 1
                u = r.get("usage") or {}
                ptok += u.get("prompt_tokens", 0)
                ctok += u.get("completion_tokens", 0)
                rtok += u.get("reasoning_tokens", 0)
                if u.get("completion_tokens", 1) == 0:
                    empty += 1
    return {
        "llm_calls": n, "finish_tool_calls": fr.get("tool_calls", 0),
        "finish_stop": fr.get("stop", 0), "finish_length": fr.get("length", 0),
        "empty_completions": empty, "prompt_tokens": ptok,
        "completion_tokens": ctok, "reasoning_tokens": rtok,
    }


def transcript_stats(events):
    err = tmo = 0
    for e in events:
        if e.get("type") == "analysis":
            t = e.get("transcript") or ""
            err += len(re.findall(r"Traceback|Error:", t))
            tmo += len(re.findall(r"[Tt]imed? ?out|timeout", t))
    return err, tmo


def game_metrics(events):
    acts = [e for e in events if e.get("type") == "action"]
    analyses = [e for e in events if e.get("type") == "analysis"]
    n = len(acts)
    changed = [b(e.get("board_changed")) for e in acts]
    sigs = [(e.get("action_name"), e.get("action_display")) for e in acts]
    streak = best = 0
    for c in changed:
        streak = 0 if c else streak + 1
        best = max(best, streak)
    first_change = next((i + 1 for i, c in enumerate(changed) if c), None)
    rep = sum(1 for i in range(1, n) if sigs[i] == sigs[i - 1])
    # last 25 actions with zero board change = still blocked at the end
    tail_dead = n >= 25 and not any(changed[-25:])
    lvl_actions, cur, lvls = [], 0, 0
    for e in acts:
        cur += 1
        if b(e.get("level_completed")):
            lvls += 1
            lvl_actions.append(cur)
            cur = 0
    batched = sum(1 for e in acts if int(e.get("batch_size") or 1) > 1)
    seen = {json.dumps(next((e["board"] for e in events if e.get("type") == "initial" and "board" in e), None))}
    revisits = 0
    for e in acts:
        h = json.dumps(e.get("board"))
        revisits += h in seen
        seen.add(h)
    return {
        "state_revisit_rate": round(revisits / n, 3) if n else 0.0,
        "actions": n,
        "analysis_steps": len(analyses),
        "board_changed_rate": round(sum(changed) / n, 3) if n else 0.0,
        "longest_no_change_streak": best,
        "first_change_action": first_change,
        "repeat_action_rate": round(rep / (n - 1), 3) if n > 1 else 0.0,
        "distinct_actions": len(set(sigs)),
        "levels_from_events": lvls,
        "actions_per_won_level": lvl_actions,
        "game_over_events": sum(b(e.get("game_over")) for e in acts),
        "batched_actions": batched,
        "tail_dead_25": tail_dead,
    }


def classify(m, bench, req):
    """Heuristic primary blocker label; thresholds are deliberately simple."""
    tags = []
    if m["actions"] == 0:
        return "no_actions_taken"
    if m["levels_from_events"] > 0 and m["levels_from_events"] >= (bench.get("number_of_levels") or 99):
        return "solved"
    if m["board_changed_rate"] == 0:
        tags.append("zero_effect")
    elif m["board_changed_rate"] < 0.1:
        tags.append("mostly_no_effect")
    if m["longest_no_change_streak"] >= 40:
        tags.append("long_stall")
    if m["repeat_action_rate"] > 0.5 and m["board_changed_rate"] < 0.5:
        tags.append("action_loop")
    if m["state_revisit_rate"] > 0.5:
        tags.append("state_cycling")
    if m["tail_dead_25"] and m["levels_from_events"] > 0:
        tags.append("stuck_after_progress")
    if m["game_over_events"] >= 3:
        tags.append("dies_repeatedly")
    if req.get("finish_length", 0) > 0:
        tags.append("truncated_outputs")
    if req.get("empty_completions", 0) > 0:
        tags.append("empty_responses")
    if m["actions"] < 20 and not m["levels_from_events"]:
        tags.append("under_explored")
    if bench.get("final_wallclock_seconds") and bench["final_wallclock_seconds"] >= 1190 and m["actions"] < 120:
        tags.append("time_capped")
    if not tags:
        tags.append("explored_no_progress" if not m["levels_from_events"] else "partial_progress")
    return "+".join(tags)


def analyse_run(run_dir):
    bench = load_bench(run_dir)
    rows = {}
    for ep in sorted(glob.glob(os.path.join(run_dir, "artifacts", "*_events.jsonl"))):
        mt = GAME_RE.search(os.path.basename(ep))
        if not mt:
            continue
        gid = mt.group(1)
        ev = load_events(ep)
        m = game_metrics(ev)
        br = bench.get(gid, {})
        req = request_stats(run_dir, gid)
        err, tmo = transcript_stats(ev)
        score = br.get("final_score")
        if score is None:
            score = float("nan")
        row = {
            "game": gid, "run": os.path.basename(os.path.normpath(run_dir)),
            "score": score,
            "levels": br.get("levels_completed", m["levels_from_events"]),
            "n_levels": br.get("number_of_levels"),
            "state": br.get("state"),
            "wallclock_s": round(br.get("final_wallclock_seconds") or 0, 1),
            "transcript_errors": err, "transcript_timeouts": tmo,
            **{k: v for k, v in m.items()}, **req,
        }
        row["blocker"] = classify(m, br, req)
        rows[gid] = row
    return rows


def fmt(x):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"


def variance_tables(runs):
    """runs: list of {game: row}. Returns per-game stats + run-level stats."""
    games = sorted({g for r in runs for g in r})
    per_game = []
    for g in games:
        sc = [r[g]["score"] for r in runs if g in r and not math.isnan(r[g]["score"])]
        lv = [r[g]["levels"] for r in runs if g in r and r[g]["levels"] is not None]
        per_game.append({
            "game": g, "n_runs": len(sc),
            "mean": st.mean(sc) if sc else float("nan"),
            "std": st.stdev(sc) if len(sc) > 1 else float("nan"),
            "min": min(sc) if sc else float("nan"),
            "max": max(sc) if sc else float("nan"),
            "runs_scoring": sum(1 for s in sc if s > 0),
            "levels_per_run": lv,
        })
    run_means = []
    for r in runs:
        sc = [x["score"] for x in r.values() if not math.isnan(x["score"])]
        run_means.append(st.mean(sc) if sc else float("nan"))
    return per_game, run_means


def bootstrap_ci(values, iters=5000, seed=0):
    vals = [v for v in values if not math.isnan(v)]
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rnd = random.Random(seed)
    ms = sorted(st.mean(rnd.choices(vals, k=len(vals))) for _ in range(iters))
    return ms[int(0.025 * iters)], ms[int(0.975 * iters)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", default="results/blocker_variance_report")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = [analyse_run(d) for d in a.run_dirs]
    all_rows = [r for run in runs for r in run.values()]

    cols = ["run", "game", "score", "levels", "n_levels", "state", "actions", "analysis_steps",
            "board_changed_rate", "longest_no_change_streak", "first_change_action",
            "repeat_action_rate", "state_revisit_rate", "distinct_actions", "game_over_events", "batched_actions",
            "tail_dead_25", "wallclock_s", "transcript_errors", "transcript_timeouts",
            "llm_calls", "finish_length", "empty_completions", "completion_tokens",
            "reasoning_tokens", "actions_per_won_level", "blocker"]
    with open(os.path.join(a.out, "blockers.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    per_game, run_means = variance_tables(runs)
    with open(os.path.join(a.out, "variance.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_game[0].keys()))
        w.writeheader()
        w.writerows(per_game)

    L = []
    L.append(f"# Blocker + variance report ({len(runs)} run(s))\n")
    L.append("## Run-level scores\n")
    for d, m, run in zip(a.run_dirs, run_means, runs):
        won = sum(1 for r in run.values() if r["score"] and r["score"] > 0)
        L.append(f"- `{os.path.basename(os.path.normpath(d))}`: mean {fmt(m)}, "
                 f"games scoring {won}/{len(run)}")
    valid = [m for m in run_means if not math.isnan(m)]
    if len(valid) > 1:
        lo, hi = bootstrap_ci(valid)
        L.append(f"\nAcross runs: mean of means {fmt(st.mean(valid))}, std {fmt(st.stdev(valid))}, "
                 f"CV {fmt(st.stdev(valid) / st.mean(valid)) if st.mean(valid) else 'n/a'}, "
                 f"bootstrap 95% CI [{fmt(lo)}, {fmt(hi)}] (only {len(valid)} runs: treat as indicative).")
    # between-game vs within-game variance (one-way decomposition)
    if len(runs) > 1:
        within = [pg["std"] ** 2 for pg in per_game if not math.isnan(pg["std"])]
        gm = [pg["mean"] for pg in per_game if not math.isnan(pg["mean"])]
        if within and len(gm) > 1:
            L.append(f"Mean within-game variance {fmt(st.mean(within))} vs between-game variance "
                     f"of means {fmt(st.variance(gm))} (high within/between => results are noise-dominated).")
    L.append("\n## Per-game variance across runs\n")
    L.append("| game | mean | std | min | max | runs>0 | levels/run |")
    L.append("|---|---|---|---|---|---|---|")
    for pg in sorted(per_game, key=lambda x: -(x["mean"] if not math.isnan(x["mean"]) else -1)):
        L.append(f"| {pg['game']} | {fmt(pg['mean'])} | {fmt(pg['std'])} | {fmt(pg['min'])} | "
                 f"{fmt(pg['max'])} | {pg['runs_scoring']}/{pg['n_runs']} | {pg['levels_per_run']} |")
    L.append("\n## Blocker labels (per run x game)\n")
    tag_count = Counter(t for r in all_rows for t in r["blocker"].split("+"))
    L.append("Tag frequency: " + ", ".join(f"{t}={c}" for t, c in tag_count.most_common()))
    L.append("\n| game | " + " | ".join(os.path.basename(os.path.normpath(d))[-28:] for d in a.run_dirs) + " |")
    L.append("|---|" + "---|" * len(a.run_dirs))
    for pg in per_game:
        cells = []
        for run in runs:
            r = run.get(pg["game"])
            cells.append("-" if not r else f"{r['blocker']} (chg {r['board_changed_rate']}, streak {r['longest_no_change_streak']}, rev {r['state_revisit_rate']})")
        L.append(f"| {pg['game']} | " + " | ".join(cells) + " |")
    L.append("\nLegend: zero_effect = no action ever changed the board; long_stall = >=40 consecutive "
             "no-change actions; action_loop = >50% repeated identical actions with <50% board change; state_cycling = >50% of resulting boards already seen earlier in the game; stuck_after_progress = won a level "
             "then last 25 actions dead; truncated_outputs/empty_responses need *_requests.jsonl (Kaggle output); "
             "time_capped = 20-min wallclock cap cut the game before 120 actions.")
    rep = "\n".join(L)
    open(os.path.join(a.out, "report.md"), "w").write(rep)
    print(rep)
    print(f"\nWritten to {a.out}/ (blockers.csv, variance.csv, report.md)")


if __name__ == "__main__":
    main()
