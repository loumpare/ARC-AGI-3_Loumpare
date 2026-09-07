#!/bin/bash
# Unattended Phase A->B flow for 2026-09-06, kernel v19 (already pushed and
# running when this was launched -- see llm_relay_agent_experiments memory
# for today's fixes: edge+aspect-ratio HUD bar detector + post-arrival
# extra-step retry, 0c2d554). User explicitly authorized sending Phase B
# automatically the moment Phase A confirms COMPLETE, while asleep --
# but Phase B is skipped entirely if Phase A errors (no point spending
# quota on a build Phase A itself flagged as broken).
set -uo pipefail
KDIR="/home/lavolpe/Bureau/Kaggle/ARC-AGI-3/notebooks/submission_kernel_tools"
KERNEL="loumitrmas/arc-agi-3-tools-submission"
COMPETITION="arc-prize-2026-arc-agi-3"
VERSION=19
LOG="$KDIR/resubmit_20260906.log"
RESULT="$KDIR/resubmit_20260906.RESULT"

exec > >(tee -a "$LOG") 2>&1
cd "$KDIR"

echo "=== $(date -Is) : waiting for kernel v$VERSION Phase A to reach a terminal state ==="
STATUS=""
for i in $(seq 1 720); do
  STATUS=$(kaggle kernels status "$KERNEL" 2>&1)
  echo "[$i] $STATUS"
  if echo "$STATUS" | grep -qiE "complete|error"; then
    break
  fi
  sleep 30
done

if ! echo "$STATUS" | grep -qi "complete"; then
  echo "RESULT: PHASE_A_FAILED_OR_TIMED_OUT status=$STATUS" | tee "$RESULT"
  exit 1
fi

echo "--- Phase A COMPLETE. Downloading output to confirm submission.parquet exists ---"
kaggle kernels output "$KERNEL" -p /tmp/resubmit_20260906_output 2>&1
if [ ! -f /tmp/resubmit_20260906_output/submission.parquet ]; then
  echo "RESULT: PHASE_A_FAILED (no submission.parquet in kernel output)" | tee "$RESULT"
  exit 1
fi

echo "--- Phase B: submitting to competition (consumes today's quota) ---"
SUBMIT_OUT=$(kaggle competitions submit -c "$COMPETITION" \
  -f submission.parquet \
  -k "$KERNEL" -v "$VERSION" \
  -m "ToolsAgent v6: fixed a real VisionToolsAgent regression (missing 2026-09-04 arrival-check/anti-loop-watchdog port, commit 13324f2) plus a residual HUD-bar loop found via a 4-config vision-model comparison -- an edge-pinned/extreme-aspect-ratio geometric detector catches a solid (non-fragmenting) draining bar the fragment-count heuristic missed, and a bounded post-arrival extra-step retry handles ARC mechanics that need walking past a marker's edge, not just touching it (commit 0c2d554). Verified 6/6 wins on ls20 locally (up from 0/4 pre-fix) on both ToolsAgent and VisionToolsAgent, no regression on cd82/ka59/vc33/re86." 2>&1)
echo "$SUBMIT_OUT"

echo "--- Phase B: polling competition submission status ---"
BEFORE_SUBS=$(kaggle competitions submissions -c "$COMPETITION" 2>&1)
BEFORE_TOP_REF=$(echo "$BEFORE_SUBS" | awk 'NR==3{print $1}')
FINAL=""
NEW_REF=""
for i in $(seq 1 240); do
  SUBS=$(kaggle competitions submissions -c "$COMPETITION" 2>&1)
  TOP_REF=$(echo "$SUBS" | awk 'NR==3{print $1}')
  echo "[$i] top_ref=$TOP_REF"
  if [ "$TOP_REF" != "$BEFORE_TOP_REF" ]; then
    TOP_STATUS=$(echo "$SUBS" | awk 'NR==3' | grep -oE "SubmissionStatus\.[A-Z]+")
    if echo "$TOP_STATUS" | grep -qiE "complete|error"; then
      FINAL="$TOP_STATUS"
      NEW_REF="$TOP_REF"
      break
    fi
  fi
  sleep 30
done

if [ -n "$FINAL" ]; then
  echo "RESULT: PHASE_B_DONE status=$FINAL ref=$NEW_REF kernel_version=$VERSION" | tee "$RESULT"
else
  echo "RESULT: PHASE_B_TIMEOUT (no terminal status after 2h) kernel_version=$VERSION" | tee "$RESULT"
fi
echo "=== $(date -Is) : finished ==="
