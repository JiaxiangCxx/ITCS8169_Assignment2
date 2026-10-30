#!/usr/bin/env bash
# Phase 3: combine the best SSL setting (SimCLR, crop scale 0.08) with the best
# fine-tuning recipe (RandAugment + MixUp/CutMix), with three seeds; test SSL on
# all 2,400 unlabeled training images; and compare ResNet-18 with ResNet-50 at a
# matched SimCLR batch of 128 (ResNet-50 does not fit 256 x 2 views in 16 GB).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
PY=.venv/bin/python
CROP=runs/ssl_simclr_crop008_r18_e500
CROP_ALL=runs/ssl_simclr_crop008_all_r18_e500
E500=runs/ssl_simclr_r18_e500

launch() {
    local session=$1 gpu=$2 commands=$3
    screen -L -Logfile "logs/${session}.screen.log" -dmS "$session" \
        bash -lc "cd '$PWD' && export CUDA_VISIBLE_DEVICES=$gpu && $commands"
    echo "started $session on GPU $gpu"
}

train() { echo "$PY -u train.py $*"; }
pretrain() { echo "$PY -u pretrain.py --config configs/ssl_simclr.yaml --set ssl.augmentation.crop_min=0.08 --set 'train.save_epochs=[]' $*"; }
strong_ft() { local name=$1 checkpoint=$2; shift 2; train --config configs/ft_randaug_mix.yaml --run-name "$name" --init-checkpoint "$checkpoint" "$@"; }
ft() { local name=$1 checkpoint=$2; shift 2; train --config configs/finetune.yaml --run-name "$name" --init-checkpoint "$checkpoint" "$@"; }
linear() { local name=$1 checkpoint=$2; shift 2; train --config configs/linear_eval.yaml --run-name "$name" --init-checkpoint "$checkpoint" "$@"; }
wait_for() { echo "until [ -f $1/run_summary.json ]; do sleep 60; done"; }

# SimCLR (crop 0.08) on all 2,400 training images, labels unused, then strong fine-tuning on the 1,920 split.
launch a2_crop_all 0 "$(pretrain --run-name ssl_simclr_crop008_all_r18_e500 --set data.ssl_images=all_train) \
&& $(strong_ft ft_randaug_mix_crop008_all $CROP_ALL/last.pt) \
&& $(strong_ft ft_randaug_mix_crop008_all_seed1 $CROP_ALL/last.pt --seed 1)"
launch a2_crop_all_b 7 "$(wait_for $CROP_ALL) && $(strong_ft ft_randaug_mix_crop008_all_seed2 $CROP_ALL/last.pt --seed 2)"

# SimCLR (crop 0.08, 1,920 images) + strong fine-tuning, three seeds.
launch a2_crop_strong 1 "$(strong_ft ft_randaug_mix_crop008 $CROP/last.pt) \
&& $(strong_ft ft_randaug_mix_crop008_seed1 $CROP/last.pt --seed 1) \
&& $(strong_ft ft_randaug_mix_crop008_seed2 $CROP/last.pt --seed 2)"

# Third seeds for the phase-2 comparison of strong fine-tuning with and without SSL.
launch a2_seed2_strong 3 "$(strong_ft ft_randaug_mix_seed2 $E500/last.pt --seed 2) \
&& $(train --config configs/sup_randaug_mix.yaml --seed 2 --run-name sup_randaug_mix_seed2)"

# Backbone study at a matched SimCLR batch of 128 (learning rate scaled to 0.15).
SMALL="--set data.batch_size=128 --set train.learning_rate=0.15"
launch a2_r50_simclr 4 "$(pretrain --run-name ssl_simclr_crop008_r50_b128_e500 --set model.name=resnet50 $SMALL) \
&& $(linear linear_ssl_simclr_crop008_r50_b128_e500 runs/ssl_simclr_crop008_r50_b128_e500/last.pt --set model.name=resnet50) \
&& $(ft ft_ssl_simclr_crop008_r50_b128_e500 runs/ssl_simclr_crop008_r50_b128_e500/last.pt --set model.name=resnet50)"
launch a2_r18_simclr_b128 6 "$(pretrain --run-name ssl_simclr_crop008_r18_b128_e500 $SMALL) \
&& $(linear linear_ssl_simclr_crop008_r18_b128_e500 runs/ssl_simclr_crop008_r18_b128_e500/last.pt) \
&& $(ft ft_ssl_simclr_crop008_r18_b128_e500 runs/ssl_simclr_crop008_r18_b128_e500/last.pt)"
launch a2_r50_sup 5 "$(train --config configs/sup_baseline.yaml --run-name sup_baseline_r50 --set model.name=resnet50)"
screen -ls || true
