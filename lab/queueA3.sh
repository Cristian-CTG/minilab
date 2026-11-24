#!/bin/bash
# Etapa A3: piezas nuevas sobre Muon (base = opt-muon, lr_muon 0.02).  Pausar: bash lab/pause.sh   Reanudar: bash lab/queueA3.sh
cd "$(dirname "$0")"; PY=../.venv/Scripts/python.exe; R=../results; mkdir -p $R/logs; touch $R/done.txt
rm -f $R/PAUSE
run() { key="$*"; grep -qxF "$key" $R/done.txt && return
  [ -e $R/PAUSE ] && { echo QUEUE_PAUSED; exit 0; }
  script=$1; name=$2; shift 2; log="$R/logs/A3_${name}_$(date +%m%d%H%M%S).log"
  for try in 1 2 3; do $PY -u $script "$name" "$@" > "$log" 2>&1; rc=$?; [ $rc -ne 127 ] && break; sleep 5; done
  [ $rc -eq 3 ] && { echo "QUEUE_PAUSED during $name"; exit 0; }
  echo "$name $* exit=$rc $(grep RESULT "$log" | cut -c1-140)" | tee -a $R/queueA3.log
  [ $rc -eq 0 ] && echo "$key" >> $R/done.txt; }

# Cierre de A1: AdamW por debajo de 1e-3 y ReLU2 contra AdamW bien ajustado
for s in 0 1; do run train.py adamw-lr5e-4 lr=5e-4 seed=$s; run train.py relu2-lr1e-3 act=relu2 lr=1e-3 seed=$s; done
# Piezas nuevas, una a una, 3 semillas
for s in 0 1 2; do
  run train.py m-theta1e4 opt=muon rope_theta=1e4 seed=$s
  run train.py m-theta1e5 opt=muon rope_theta=1e5 seed=$s
  run train.py m-diff opt=muon attn=diff seed=$s
  run train.py m-zloss opt=muon zloss=1e-4 seed=$s
  run train.py m-mtp opt=muon mtp=true seed=$s
  run train.py m-untied opt=muon tie=false seed=$s
  run train.py m-ffn2 opt=muon ffn_mult=2 seed=$s
  run train.py m-ffn4 opt=muon ffn_mult=4 seed=$s
  run train.py m-kv1 opt=muon kv_heads=1 seed=$s
  run train.py m-kv6 opt=muon kv_heads=6 seed=$s
done
echo QUEUE_A3_DONE
