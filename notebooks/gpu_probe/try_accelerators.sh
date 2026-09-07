#!/bin/bash
# Tries a batch of plausible --accelerator string candidates against the
# lightweight gpu_probe kernel, checking nvidia-smi's reported GPU name each
# time. Empirical, since the Kaggle SDK doesn't expose the valid enum names
# locally (confirmed via source read: "allowed names are in an enum not
# currently included in kagglesdk").
CANDIDATES=(
  "GPU_RTX6000PROX1"
  "RTX6000PRO"
  "GPU_1XRTX6000PRO"
  "PREFERRED_GPU"
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
  kaggle kernels output loumitrmas/arc-agi-3-gpu-probe -p /tmp/gpu_probe_batch_$ACC 2>&1 | tail -1
  grep -A2 "GPU  Name" /tmp/gpu_probe_batch_$ACC/arc-agi-3-gpu-probe.log 2>/dev/null || \
    grep "Tesla\|RTX\|A100\|H100\|V100\|Quadro" /tmp/gpu_probe_batch_$ACC/arc-agi-3-gpu-probe.log 2>/dev/null | head -3
  echo "=== done with $ACC ==="
  echo
done
