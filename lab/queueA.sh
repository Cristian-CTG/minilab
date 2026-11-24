#!/bin/bash
# Etapa A (plan .hermes/plans): techo del modelo de 13M.  Pausar: bash lab/pause.sh   Reanudar: bash lab/queueA.sh
cd "$(dirname "$0")"; PY=../.venv/Scripts/python.exe; R=../results; mkdir -p $R/logs; touch $R/done.txt
rm -f $R/PAUSE
run() { key="$*"; grep -qxF "$key" $R/done.txt && return
  [ -e $R/PAUSE ] && { echo QUEUE_PAUSED; exit 0; }
  script=$1; name=$2; shift 2; log="$R/logs/A_${name}_$(date +%m%d%H%M%S).log"
  for try in 1 2 3; do $PY -u $script "$name" "$@" > "$log" 2>&1; rc=$?; [ $rc -ne 127 ] && break; sleep 5; done
  [ $rc -eq 3 ] && { echo "QUEUE_PAUSED during $name"; exit 0; }
  echo "$name $* exit=$rc $(grep RESULT "$log" | cut -c1-140)" | tee -a $R/queueA.log
  [ $rc -eq 0 ] && echo "$key" >> $R/done.txt; }

# A1: tasa de aprendizaje justa (3e-3 y 0.02 ya existen: baseline y opt-muon)
for s in 0 1; do
  for lr in 1e-3 2e-3 6e-3; do run train.py adamw-lr$lr lr=$lr seed=$s; done
  for lr in 0.005 0.01 0.04; do run train.py muon-lr$lr opt=muon lr_muon=$lr seed=$s; done
done
# A2: repetir los mal configurados
for lr in 3e-5 1e-4 1e-3; do run train.py lion-lr$lr opt=lion lr_lion=$lr seed=0; done
run train.py dyt-a1 norm=dyt dyt_a0=1.0 seed=0
run train.py dyt-a2 norm=dyt dyt_a0=2.0 seed=0
run train.py dyt-lr1e-3 norm=dyt lr=1e-3 seed=0
echo QUEUE_A12_DONE
