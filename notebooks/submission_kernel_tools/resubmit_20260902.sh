#!/bin/bash
# One-shot scheduled resubmission of ToolsAgent, scheduled via cron for 2026-09-02 02:10 CEST
# (00:10 UTC, ~10min after Kaggle's daily quota resets at 00:00 UTC). Runs Phase A (free
# validation, rebuilds+pushes the kernel to catch any build regression) and, only if that
# succeeds, Phase B (the real quota-consuming competition submission), then polls both to
# a terminal state. Self-removes its own crontab line on exit so it never fires again.
set -uo pipefail

REPO_DIR="/home/lavolpe/Bureau/Kaggle/ARC-AGI-3"
KDIR="$REPO_DIR/notebooks/submission_kernel_tools"
KERNEL="loumitrmas/arc-agi-3-tools-submission"
COMPETITION="arc-prize-2026-arc-agi-3"
LOG="$KDIR/resubmit_20260902.log"
RESULT="$KDIR/resubmit_20260902.RESULT"

exec > >(tee -a "$LOG") 2>&1

# self-remove this cron entry so it only ever fires once
(crontab -l 2>/dev/null | grep -v "resubmit_20260902.sh") | crontab -

echo "=== $(date -Is) : resubmission run starting ==="

echo "--- current submissions (before) ---"
BEFORE_SUBS=$(kaggle competitions submissions -c "$COMPETITION" 2>&1)
echo "$BEFORE_SUBS"
BEFORE_TOP_REF=$(echo "$BEFORE_SUBS" | awk 'NR==3{print $1}')
echo "Most recent submission ref before this run: $BEFORE_TOP_REF"

cd "$KDIR" || { echo "RESULT: FAILED (cd $KDIR)" | tee "$RESULT"; exit 1; }

echo "--- Phase A: rebuild notebook from latest my_agent_tools.py ---"
python3 build_notebook.py 2>&1

echo "--- Phase A: kaggle kernels push ---"
PUSH_OUT=$(kaggle kernels push -p . 2>&1)
echo "$PUSH_OUT"
VERSION=$(echo "$PUSH_OUT" | grep -oE "Kernel version [0-9]+" | grep -oE "[0-9]+")
if [ -z "$VERSION" ]; then
  echo "RESULT: FAILED (could not parse pushed kernel version from push output)" | tee "$RESULT"
  exit 1
fi
echo "Pushed kernel version: $VERSION"

echo "--- Phase A: polling kernel status (up to 20 min) ---"
STATUS=""
for i in $(seq 1 80); do
  STATUS=$(kaggle kernels status "$KERNEL" 2>&1)
  echo "[$i] $STATUS"
  if echo "$STATUS" | grep -qiE "complete|error"; then
    break
  fi
  sleep 15
done

if ! echo "$STATUS" | grep -qi "complete"; then
  echo "RESULT: PHASE_A_FAILED status=$STATUS" | tee "$RESULT"
  exit 1
fi

echo "--- Phase A passed. Downloading output to sanity-check submission.parquet exists ---"
kaggle kernels output "$KERNEL" -p /tmp/resubmit_20260902_output 2>&1
if [ ! -f /tmp/resubmit_20260902_output/submission.parquet ]; then
  echo "RESULT: PHASE_A_FAILED (no submission.parquet in kernel output)" | tee "$RESULT"
  exit 1
fi

echo "--- Phase B: submitting to competition (consumes today's quota) ---"
# NOTE: -f must be the BARE output filename ("submission.parquet"), not a local path --
# for code competitions the API looks it up by name from the given kernel version's own
# output server-side. A full local path here caused a 400 Bad Request on the first attempt
# today (2026-09-02 10:20 run, kernel v9) -- verified by cross-checking `kaggle competitions
# submit --help` ("the name of the output file produced by a kernel, for code competitions")
# against the working command documented in the 2026-08-31 memory. No quota was consumed by
# that failed attempt (confirmed via `kaggle competitions submission-limits`, remaining=1).
SUBMIT_OUT=$(kaggle competitions submit -c "$COMPETITION" \
  -f submission.parquet \
  -k "$KERNEL" -v "$VERSION" \
  -m "ToolsAgent v2: fixed concurrency lock, brain-call cap, re86 character-switch identity confirmation, ls20 exploration-loop regression -- automated resubmission after 2026-09-01 SubmissionStatus.ERROR" 2>&1)
echo "$SUBMIT_OUT"

echo "--- Phase B: polling competition submission status (up to 30 min) ---"
# Only trust a terminal status once the top submission ref actually differs from the one
# seen before this run -- otherwise a failed/rejected submit call (e.g. today's earlier 400)
# would make this loop misreport YESTERDAY's stale top-row status as if it were fresh.
FINAL=""
NEW_REF=""
for i in $(seq 1 60); do
  SUBS=$(kaggle competitions submissions -c "$COMPETITION" 2>&1)
  echo "[$i]"
  echo "$SUBS"
  TOP_REF=$(echo "$SUBS" | awk 'NR==3{print $1}')
  if [ "$TOP_REF" = "$BEFORE_TOP_REF" ]; then
    echo "(top ref unchanged: $TOP_REF -- no new submission registered yet)"
  else
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
elif [ -n "$SUBMIT_OUT" ] && echo "$SUBMIT_OUT" | grep -qiE "error|bad request"; then
  echo "RESULT: PHASE_B_SUBMIT_REJECTED kernel_version=$VERSION detail=$SUBMIT_OUT" | tee "$RESULT"
else
  echo "RESULT: PHASE_B_TIMEOUT (no new submission ref registered after 30min) kernel_version=$VERSION" | tee "$RESULT"
fi

echo "=== $(date -Is) : resubmission run finished ==="
