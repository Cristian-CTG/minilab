#!/bin/bash
# Batch 4 (starts after batch3): Phase 2 internal language + Phase 3 latent reasoning
cd "$(dirname "$0")"; PY=../.venv/Scripts/python.exe; mkdir -p ../results/logs
until [ "$(grep -c . ../results/batch3.log 2>/dev/null)" -ge 8 ]; do sleep 60; done
run() { script=$1; name=$2; shift 2; log="../results/logs/b4_${name}_$(date +%H%M%S).log"
  for try in 1 2 3; do $PY -u $script "$name" "$@" > "$log" 2>&1; rc=$?; [ $rc -ne 127 ] && break; sleep 5; done
  echo "$name $* exit=$rc $(grep RESULT "$log" | cut -c1-140)" | tee -a ../results/batch4.log; }
for s in 0 1; do for v in es en es_norm bytes; do run train.py lang-$v data=p2_$v opt=muon seed=$s; done; done
for s in 0 1; do for md in direct cot pause latent; do run latent.py reason-$md mode=$md seed=$s; done; done
echo BATCH4_DONE
