#!/usr/bin/env bash
# Phase 2, after phase 1: seed variance, label efficiency, SSL-epoch and crop
# ablations, and SimCLR initialization combined with strong augmentation or
# with the joint loss. Requires runs/ssl_simclr_r18_e500 from run_phase1.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
PY=.venv/bin/python
SIMCLR=runs/ssl_simclr_r18_e500

launch() {
    local session=$1 gpu=$2 commands=$3
    screen -L -Logfile "logs/${session}.screen.log" -dmS "$session" \
        bash -lc "cd '$PWD' && export CUDA_VISIBLE_DEVICES=$gpu && $commands"
    echo "started $session on GPU $gpu"
}

train() { echo "$PY -u train.py $*"; }
ft() { local name=$1 checkpoint=$2; shift 2; train --config configs/finetune.yaml --run-name "$name" --init-checkpoint "$checkpoint" "$@"; }
linear() { local name=$1 checkpoint=$2; shift 2; train --config configs/linear_eval.yaml --run-name "$name" --init-checkpoint "$checkpoint" "$@"; }

# SimCLR initialization + strong regularization, and SSL-epoch checkpoints.
launch a2_ft_randaug 0 "$(train --config configs/ft_randaug_mix.yaml --init-checkpoint $SIMCLR/last.pt) \
&& $(train --config configs/ft_randaug_mix.yaml --init-checkpoint $SIMCLR/last.pt --seed 1 --run-name ft_randaug_mix_seed1) \
&& $(ft ft_ssl_simclr_r18_e500_at300 $SIMCLR/epoch_0300.pt)"

# SimCLR pretraining followed by joint CE + NT-Xent fine-tuning.
launch a2_ft_joint 1 "$(train --config configs/ft_joint.yaml --init-checkpoint $SIMCLR/last.pt) \
&& $(ft ft_ssl_simclr_r18_e500_at100 $SIMCLR/epoch_0100.pt) \
&& $(ft ft_ssl_simclr_r18_e500_at200 $SIMCLR/epoch_0200.pt)"

# Longer SimCLR pretraining (1,000 epochs).
launch a2_simclr_e1000 2 "$PY -u pretrain.py --config configs/ssl_simclr.yaml --run-name ssl_simclr_r18_e1000 \
--set train.epochs=1000 --set 'train.save_epochs=[500]' \
&& $(linear linear_ssl_simclr_r18_e1000 runs/ssl_simclr_r18_e1000/last.pt) \
&& $(ft ft_ssl_simclr_r18_e1000 runs/ssl_simclr_r18_e1000/last.pt)"

# Crop-scale ablation: SimCLR's default (0.08, 1) instead of (0.2, 1).
launch a2_simclr_crop008 3 "$PY -u pretrain.py --config configs/ssl_simclr.yaml --run-name ssl_simclr_crop008_r18_e500 \
--set ssl.augmentation.crop_min=0.08 --set 'train.save_epochs=[]' \
&& $(linear linear_ssl_simclr_crop008_r18_e500 runs/ssl_simclr_crop008_r18_e500/last.pt) \
&& $(ft ft_ssl_simclr_crop008_r18_e500 runs/ssl_simclr_crop008_r18_e500/last.pt)"

# Seed variance of the supervised baselines (fixed validation split).
launch a2_seeds_sup 4 "$(train --config configs/sup_baseline.yaml --seed 1 --run-name sup_baseline_seed1) \
&& $(train --config configs/sup_baseline.yaml --seed 2 --run-name sup_baseline_seed2) \
&& $(train --config configs/sup_randaug_mix.yaml --seed 1 --run-name sup_randaug_mix_seed1)"

# Seed variance of SimCLR fine-tuning (same pretrained backbone, new fine-tuning seed).
launch a2_seeds_ft 5 "$(ft ft_ssl_simclr_r18_e500_seed1 $SIMCLR/last.pt --seed 1) \
&& $(ft ft_ssl_simclr_r18_e500_seed2 $SIMCLR/last.pt --seed 2) \
&& $(train --config configs/sup_strong.yaml --seed 1 --run-name sup_strong_seed1)"

# Label efficiency: 10% and 25% of the labels (stratified), batch 64 for every arm.
LOW="--set data.batch_size=64"
launch a2_labels_a 6 "$(train --config configs/sup_baseline.yaml --run-name sup_baseline_labels10 $LOW --set data.label_fraction=0.1) \
&& $(ft ft_ssl_simclr_r18_e500_labels10 $SIMCLR/last.pt $LOW --set data.label_fraction=0.1) \
&& $(linear linear_ssl_simclr_r18_e500_labels10 $SIMCLR/last.pt $LOW --set data.label_fraction=0.1) \
&& $(train --config configs/sup_baseline.yaml --run-name sup_baseline_labels25 $LOW --set data.label_fraction=0.25)"
launch a2_labels_b 7 "$(ft ft_ssl_simclr_r18_e500_labels25 $SIMCLR/last.pt $LOW --set data.label_fraction=0.25) \
&& $(linear linear_ssl_simclr_r18_e500_labels25 $SIMCLR/last.pt $LOW --set data.label_fraction=0.25) \
&& $(train --config configs/sup_baseline.yaml --run-name sup_baseline_labels50 $LOW --set data.label_fraction=0.5) \
&& $(ft ft_ssl_simclr_r18_e500_labels50 $SIMCLR/last.pt $LOW --set data.label_fraction=0.5)"
screen -ls || true
