#!/bin/bash
# Batch 1: math pieces vs baseline, 1500 steps (~24.6M tokens), 13M params, 3 seeds
cd "$(dirname "$0")"
PY=../.venv/Scripts/python.exe
VARIANTS=("baseline" "attn=sigmoid" "attn=softmax_learned_temp" "qk_norm=false" "norm=dyt" "act=geglu" "act=relu2" "pos=nope" "opt=lion" "opt=muon")
for seed in 0 1 2; do
  for v in "${VARIANTS[@]}"; do
    name="${v//=/-}"
    if [ "$v" = "baseline" ]; then args=""; else args="$v"; fi
    $PY -u train.py "$name" $args seed=$seed 2>&1 | grep -E "RESULT|Error" | tee -a ../results/batch1.log
  done
done
echo BATCH1_DONE | tee -a ../results/batch1.log
