#!/bin/bash
# Scheduled Phase B submission for the new UnifiedVisionAgent+GPU baseline,
# authorized by the user for 2026-09-07 02:05 local time (just after the
# daily quota resets at UTC midnight -- today's quota was already spent by
# the plain-ToolsAgent resubmit earlier on 2026-09-06). Kernel v2 already
# has a confirmed-COMPLETE Phase A run (25/25 public games, 0 crashes, see
# llm_relay_agent_experiments memory) -- this script re-confirms that before
# spending the quota, rather than assuming it's still true.
set -uo pipefail
KDIR="/home/lavolpe/Bureau/Kaggle/ARC-AGI-3/notebooks/submission_kernel_vision"
KERNEL="loumitrmas/arc-agi-3-vision-submission"
COMPETITION="arc-prize-2026-arc-agi-3"
VERSION=4
LOG="$KDIR/resubmit_20260907.log"
RESULT="$KDIR/resubmit_20260907.RESULT"

exec > >(tee -a "$LOG") 2>&1
cd "$KDIR"

echo "=== $(date -Is) : confirming kernel v$VERSION is still COMPLETE ==="
STATUS=$(kaggle kernels status "$KERNEL" 2>&1)
echo "$STATUS"
if ! echo "$STATUS" | grep -qi "complete"; then
  echo "RESULT: PHASE_A_NOT_COMPLETE status=$STATUS" | tee "$RESULT"
  exit 1
fi

echo "--- downloading output to confirm submission.parquet exists ---"
rm -rf /tmp/resubmit_20260907_output
kaggle kernels output "$KERNEL" -p /tmp/resubmit_20260907_output 2>&1
if [ ! -f /tmp/resubmit_20260907_output/submission.parquet ]; then
  echo "RESULT: NO_PARQUET (submission.parquet missing from kernel output)" | tee "$RESULT"
  exit 1
fi

echo "--- checking today's submission quota before spending it ---"
LIMITS=$(kaggle competitions submission-limits -c "$COMPETITION" 2>&1)
echo "$LIMITS"
if echo "$LIMITS" | grep -qi "Remaining today: 0"; then
  echo "RESULT: QUOTA_ALREADY_SPENT" | tee "$RESULT"
  exit 1
fi

echo "--- Phase B: submitting to competition (consumes today's quota) ---"
SUBMIT_OUT=$(kaggle competitions submit -c "$COMPETITION" \
  -f submission.parquet \
  -k "$KERNEL" -v "$VERSION" \
  -m "UnifiedVisionAgent v2 (GPU + calibration): ONE qwen3.8-27B multimodal call does both scene perception and decision per consult (src/llm_unified_vision_agent.py), fully GPU-offloaded on the competition's RTX PRO 6000 via our own CUDA-compiled llama-cpp-python wheel + our own qwen3.8 weights. Adds a pre-play calibration phase (try every non-click action twice, then reset, before any goal-directed play -- action_deltas fully known upfront instead of discovered opportunistically) -- local 150-action benchmark: self_found 13/25 (up from 5/25), 3/25 levels won including a NEW win (sp80) not present in the prior baseline. Real Kaggle Phase A (kernel v4, after fixing a caught-but-real self_bbox/self_pos None bug found in v3's own Phase A run): 25/25 public games completed with zero uncaught exceptions/errors, supports_gpu_offload confirmed True, 2/25 levels won (lp85, ls20) under the validation cell's tighter 60-action budget." 2>&1)
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
