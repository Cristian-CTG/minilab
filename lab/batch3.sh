#!/bin/bash
# Batch 3: escalado. Baseline (AdamW) vs Muon a ~29M y ~62M, 2 semillas
cd "$(dirname "$0")"; PY=../.venv/Scripts/python.exe; mkdir -p ../results/logs
run() { name=$1; shift; log="../results/logs/b3_${name}_$(date +%H%M%S).log"
  for try in 1 2 3; do $PY -u train.py "$name" "$@" > "$log" 2>&1; rc=$?; [ $rc -ne 127 ] && break; sleep 5; done
  echo "$name $* exit=$rc $(grep RESULT "$log" | cut -c1-120)" | tee -a ../results/batch3.log; }
M29="d=512 layers=8 heads=8"; M62="d=768 layers=8 heads=12 kv_heads=4"
for s in 0 1; do
  run base-29M $M29 seed=$s; run muon-29M $M29 opt=muon seed=$s
done
for s in 0 1; do
  run base-62M $M62 seed=$s; run muon-62M $M62 opt=muon seed=$s
done
echo BATCH3_DONE
