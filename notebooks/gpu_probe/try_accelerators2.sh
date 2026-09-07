#!/bin/bash
CANDIDATES=(
  "GPU_RTX_PRO_6000X1"
  "RTXPRO6000"
  "GPU_1XRTXPRO6000"
  "RTX_PRO_6000"
)
for ACC in "${CANDIDATES[@]}"; do
  echo "=== trying --accelerator $ACC ==="
  kaggle kernels push -p . --accelerator "$ACC" 2>&1
  for i in $(seq 1 40); do
    STATUS=$(kaggle kernels status loumitrmas/arc-agi-3-gpu-probe 2>&1)
    if echo "$STATUS" | grep -qiE "complete|error"; then break; fi
    sleep 15
  done
  echo "$STATUS"
  kaggle kernels output loumitrmas/arc-agi-3-gpu-probe -p /tmp/gpu_probe_batch2_$ACC 2>&1 | tail -1
  grep -oE "Tesla [A-Za-z0-9-]+|RTX [A-Za-z0-9 ]+|Quadro [A-Za-z0-9]+|PRO 6000|Pro 6000" /tmp/gpu_probe_batch2_$ACC/arc-agi-3-gpu-probe.log 2>/dev/null | sort -u
  echo "=== done with $ACC ==="
  echo
done
