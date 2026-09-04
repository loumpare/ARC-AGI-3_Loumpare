#!/bin/bash
# Poll submission 55958137 to a terminal state using --csv + python's csv module for
# exact field extraction (avoids the "status text appears inside description" false-positive
# that bit the resubmit script's naive whole-line grep twice already).
REF=55958137
COMPETITION=arc-prize-2026-arc-agi-3
RESULT_FILE="/home/lavolpe/Bureau/Kaggle/ARC-AGI-3/notebooks/submission_kernel_tools/poll_${REF}.RESULT"

for i in $(seq 1 120); do
  STATUS=$(kaggle competitions submissions -c "$COMPETITION" --csv 2>/dev/null | python3 -c "
import csv, sys
for row in csv.DictReader(sys.stdin):
    if row['ref'] == '$REF':
        print(row['status'])
        break
")
  echo "[$i] $(date -Is) ref=$REF status=$STATUS"
  if echo "$STATUS" | grep -qiE "^SubmissionStatus\.(COMPLETE|ERROR)$"; then
    echo "RESULT: $STATUS" | tee "$RESULT_FILE"
    exit 0
  fi
  sleep 30
done
echo "RESULT: TIMEOUT (still not terminal after 60min)" | tee "$RESULT_FILE"
