#!/bin/bash
# Re-run the 8 batch-1 runs that died silently; keep full logs to see why
cd "$(dirname "$0")"
PY=../.venv/Scripts/python.exe
mkdir -p ../results/logs
while read -r v s; do
  name="${v//=/-}"; args="$v"; [ "$v" = "baseline" ] && args=""
  $PY -u train.py "$name" $args seed=$s > "../results/logs/${name}_s$s.log" 2>&1
  echo "$name s$s exit=$? $(grep RESULT ../results/logs/${name}_s$s.log | cut -c1-120)" | tee -a ../results/rerun.log
done <<'EOF'
attn=softmax_learned_temp 1
norm=dyt 1
act=relu2 1
opt=lion 1
opt=muon 1
baseline 2
attn=sigmoid 2
attn=softmax_learned_temp 2
EOF
echo RERUN_DONE
