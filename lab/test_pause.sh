#!/bin/bash
# Self-test for pause/resume: start a short run, pause it, resume it
cd "$(dirname "$0")"; PY=../.venv/Scripts/python.exe; R=../results; rm -f $R/PAUSE
( sleep 25; touch $R/PAUSE ) &
$PY -u train.py ckpttest steps=300 eval_every=300 2>&1 | tail -2; echo "exit=${PIPESTATUS[0]}"; wait
ls $R/ckpt; rm -f $R/PAUSE
$PY -u train.py ckpttest steps=300 eval_every=300 2>&1 | tail -3
ls $R/ckpt; grep -v '"ckpttest"' $R/runs.jsonl > $R/r.tmp && mv $R/r.tmp $R/runs.jsonl
