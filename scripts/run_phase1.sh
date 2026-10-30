#!/usr/bin/env bash
# Phase 1: supervised baselines, joint training, and the three SSL pretraining
# runs (each followed by linear evaluation and fine-tuning of its backbone).
# One GPU per detached GNU Screen session; logs go to logs/<session>.screen.log.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
PY=.venv/bin/python

launch() {
    local session=$1 gpu=$2 commands=$3
    screen -L -Logfile "logs/${session}.screen.log" -dmS "$session" \
        bash -lc "cd '$PWD' && export CUDA_VISIBLE_DEVICES=$gpu && $commands"
    echo "started $session on GPU $gpu"
}

# SSL pretraining -> linear evaluation -> fine-tuning, all on one GPU.
ssl_chain() {
    local ssl_config=$1 ssl_run=$2
    echo "$PY -u pretrain.py --config configs/$ssl_config \
&& $PY -u train.py --config configs/linear_eval.yaml --run-name linear_${ssl_run} --init-checkpoint runs/${ssl_run}/last.pt \
&& $PY -u train.py --config configs/finetune.yaml --run-name ft_${ssl_run} --init-checkpoint runs/${ssl_run}/last.pt"
}

launch a2_sup_baseline   0 "$PY -u train.py --config configs/sup_baseline.yaml"
launch a2_sup_randaug    1 "$PY -u train.py --config configs/sup_randaug_mix.yaml"
launch a2_joint_control  2 "$PY -u train.py --config configs/joint_control.yaml"
launch a2_joint_simclr   3 "$PY -u train.py --config configs/joint_simclr.yaml"
launch a2_simclr         4 "$(ssl_chain ssl_simclr.yaml ssl_simclr_r18_e500)"
launch a2_rotation       5 "$(ssl_chain ssl_rotation.yaml ssl_rotation_r18_e150)"
launch a2_simclr_nocolor 6 "$(ssl_chain ssl_simclr_nocolor.yaml ssl_simclr_nocolor_r18_e500)"
launch a2_sup_strong     7 "$PY -u train.py --config configs/sup_strong.yaml"
screen -ls || true
