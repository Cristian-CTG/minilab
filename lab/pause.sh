#!/bin/bash
# Pause the queue: the running train.py saves a checkpoint within ~50 steps and exits; resume with: bash lab/queue.sh
touch "$(dirname "$0")/../results/PAUSE"; echo "Pausa pedida. Espera el mensaje QUEUE_PAUSED (1-3 min)."
