#!/bin/bash
# Colab: escalado justo (2 tokens por parametro), AdamW lr 1e-3 vs Muon.  Uso: AMP=fp16 bash lab/queue_colab.sh
# Reanudable: si Colab corta la sesion, vuelve a ejecutar la misma celda (checkpoint cada 250 pasos en Drive).
cd "$(dirname "$0")"; PY=${PY:-python}; AMP=${AMP:-bf16}; R=../results; mkdir -p $R/logs; touch $R/done.txt
rm -f $R/PAUSE
run() { key="$*"; grep -qxF "$key" $R/done.txt && return
  name=$1; shift; log="$R/logs/C_${name}_$(date +%m%d%H%M%S).log"
  $PY -u train.py "$name" "$@" amp=$AMP ckpt_every=250 eval_every=500 2>&1 | tee "$log" | grep --line-buffered "step\|RESULT\|Error\|resumed"; rc=${PIPESTATUS[0]}
  echo "$name $* exit=$rc" >> $R/queue_colab.log
  [ $rc -eq 0 ] && echo "$key" >> $R/done.txt; }
M13="steps=1615"; M29="d=512 layers=8 heads=8 steps=3540"; M62="d=768 layers=8 heads=12 kv_heads=4 steps=7610"
for s in 0 1; do
  for size in 13 29 62; do v="M$size"
    run c-adamw-${size}M ${!v} lr=1e-3 seed=$s
    run c-muon-${size}M ${!v} opt=muon seed=$s
  done
done
echo QUEUE_COLAB_DONE
