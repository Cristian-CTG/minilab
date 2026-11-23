#!/bin/bash
# Experiment queue with pause/resume.
#   bash lab/queue.sh          start or resume (finished runs are skipped)
#   bash lab/pause.sh          pause: the current run checkpoints and the queue stops
cd "$(dirname "$0")"; PY=../.venv/Scripts/python.exe; R=../results; mkdir -p $R/logs; touch $R/done.txt
rm -f $R/PAUSE
run() { key="$*"; grep -qxF "$key" $R/done.txt && return
  [ -e $R/PAUSE ] && { echo QUEUE_PAUSED; exit 0; }
  script=$1; name=$2; shift 2; log="$R/logs/q_${name}_$(date +%m%d%H%M%S).log"
  for try in 1 2 3; do $PY -u $script "$name" "$@" > "$log" 2>&1; rc=$?; [ $rc -ne 127 ] && break; sleep 5; done
  [ $rc -eq 3 ] && { echo "QUEUE_PAUSED during $name"; exit 0; }
  echo "$name $* exit=$rc $(grep RESULT "$log" | cut -c1-140)" | tee -a $R/queue.log
  [ $rc -eq 0 ] && echo "$key" >> $R/done.txt; }

# Batch 3: scaling, AdamW baseline vs Muon
M29="d=512 layers=8 heads=8"; M62="d=768 layers=8 heads=12 kv_heads=4"
for s in 0 1; do run train.py base-29M $M29 seed=$s; run train.py muon-29M $M29 opt=muon seed=$s; done
for s in 0 1; do run train.py base-62M $M62 seed=$s; run train.py muon-62M $M62 opt=muon seed=$s; done
# Batch 4: internal language (Phase 2) and latent reasoning (Phase 3)
for s in 0 1; do for v in es en es_norm bytes; do run train.py lang-$v data=p2_$v opt=muon seed=$s; done; done
for s in 0 1; do for md in direct cot pause latent; do run latent.py reason-$md mode=$md seed=$s; done; done
echo QUEUE_DONE
