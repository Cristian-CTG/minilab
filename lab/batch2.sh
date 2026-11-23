#!/bin/bash
# Batch 2: missing seed + combo of winners (Muon + ReLU2), 3 seeds
cd "$(dirname "$0")"; PY=../.venv/Scripts/python.exe; mkdir -p ../results/logs
run() { name=$1; shift; log="../results/logs/b2_${name}_$(date +%H%M%S).log"
  for try in 1 2 3; do $PY -u train.py "$name" "$@" > "$log" 2>&1; rc=$?; [ $rc -ne 127 ] && break; sleep 5; done
  echo "$name $* exit=$rc $(grep RESULT "$log" | cut -c1-110)" | tee -a ../results/batch2.log; }
run attn-softmax_learned_temp attn=softmax_learned_temp seed=2
for s in 0 1 2; do run muon-relu2 opt=muon act=relu2 seed=$s; done
echo BATCH2_DONE
