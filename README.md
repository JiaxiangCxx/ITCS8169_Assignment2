# ITCS 8169 Assignment 2 - Self-Supervised Visual Representation Learning

PyTorch code for 16-class scene classification (the 15-Scene categories plus
Flower) from 2,400 training images, using self-supervised representation
learning. **No externally pretrained weights or external data are used**: every
network is randomly initialized (`weights=None`) and trained only on the
provided training images. Model selection uses only the fixed, class-stratified
validation split from Assignment 1; the labeled test set is evaluated once per
run, for the checkpoint selected on validation.

The two-page report is [`report/report.pdf`](report/report.pdf) (source:
[`report/report.tex`](report/report.tex)). How AI tools were used is
documented in [`AI_USAGE.md`](AI_USAGE.md).

## Final result

| | |
|---|---|
| Model | Ensemble (averaged softmax) of 3 ResNet-18s. Each is SimCLR-pretrained from random initialization on the 2,400 training images (labels unused, crop scale 0.08-1, 500 epochs) and fine-tuned on the 1,920-image split with RandAugment, random erasing and MixUp/CutMix (seeds 0, 1, 2). No test-time augmentation. |
| Test accuracy (`test`, 400 labeled images) | **88.25%** (353/400) |
| Validation accuracy (fixed 480-image split) | 90.00% (432/480); the single models get 89.38-89.58% |
| Single-model test accuracy | 87.75 / 85.50 / 86.50% (mean 86.58 +- 1.13) |
| Configuration | [`configs/best_pretrain.yaml`](configs/best_pretrain.yaml) then [`configs/best_finetune.yaml`](configs/best_finetune.yaml) (`--seed 0/1/2`) |
| Checkpoints | `checkpoints/ft_randaug_mix_crop008_all_seed{0,1,2}.pt`, 42.7 MiB each, Git LFS |
| Training cost | 25 min SimCLR pretraining + 11 min fine-tuning per seed, one Quadro RTX 5000 |

The ensemble and the decision not to use flip test-time augmentation were chosen
on validation (flip TTA lowered the ensemble from 90.0% to 89.4%) before the test
set was evaluated. Caveat: the SSL stage of this recipe saw the 480 validation
images without labels, so its validation accuracy is slightly optimistic; the
same recipe with SSL on the 1,920 training-split images only gets 89.03% mean
validation and 86.17% mean test accuracy.

Per-class test accuracy and the confusion matrix are in
[`runs/final_test_evaluation/`](runs/final_test_evaluation/). Reproduce the
evaluation with:

```bash
python evaluate.py --data-root dataset --split test --output-dir runs/final_test_evaluation \
  --checkpoint checkpoints/ft_randaug_mix_crop008_all_seed0.pt \
               checkpoints/ft_randaug_mix_crop008_all_seed1.pt \
               checkpoints/ft_randaug_mix_crop008_all_seed2.pt
```

Expected output: `accuracy 0.8825, correct 353, total 400`.

| Checkpoint | SHA-256 |
|---|---|
| `ft_randaug_mix_crop008_all_seed0.pt` | `d9cac89611c9ea5504df29e8d7c84d45d0ee4828a9926f0eb7e30bc8a4d37a6a` |
| `ft_randaug_mix_crop008_all_seed1.pt` | `10b840fc3f1f25914310efff47f213cca05372301995bb5e70867d03dacf1c97` |
| `ft_randaug_mix_crop008_all_seed2.pt` | `bae300c6cd8fec9b4c1e4d92958d5024130d4ba2000700bd7d2d818834d30952` |

### Retrain the final model

```bash
CUDA_VISIBLE_DEVICES=0 python -u pretrain.py --config configs/best_pretrain.yaml
for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0 python -u train.py --config configs/best_finetune.yaml \
    --seed $seed --run-name ft_randaug_mix_crop008_all_seed$seed
done
for seed in 0 1 2; do
  python scripts/export_checkpoint.py --checkpoint runs/ft_randaug_mix_crop008_all_seed$seed/best.pt \
    --output checkpoints/ft_randaug_mix_crop008_all_seed$seed.pt
done
```

On the same GPU type this reproduced the seed-0 member bit for bit: rerunning
`configs/best_pretrain.yaml` and then `configs/best_finetune.yaml`
(`runs/repro_ssl_simclr_crop008_all_r18_e500`, `runs/repro_ft_randaug_mix_crop008_all`)
gave identical per-epoch SSL losses, k-NN and linear-probe accuracies, an
identical fine-tuning history, the same best epoch (170, 89.38% validation,
87.75% test) and identical weights. Cloning this repository and running the
evaluation command above reproduces 88.25%.

## 1. Environment

```bash
cd /projects/cci-abc/HPDiL/jzhang84/assignment2
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
```

The cci-abc base environment provides CUDA-enabled PyTorch 2.8.0 and
torchvision 0.23.0 (Python 3.13.9); `--system-site-packages` reuses them. Every
run records its exact versions, GPU and seed in `runs/<run>/environment.json`.
All experiments ran on one Quadro RTX 5000 (16 GB) per run.

The released checkpoints are stored with Git LFS. After cloning:

```bash
git lfs install
git lfs pull
```

## 2. Data

The same data as Assignment 1:

```bash
python scripts/download_data.py --destination dataset \
  --staging /tmp/${USER}_assignment2_data --workers 12
python scripts/check_data.py --data-root dataset
```

Expected counts: `train` 2,400 images in 16 classes (150 each), `test` 400
labeled images (25 each), `test2` 400 unlabeled images. Google Drive
rate-limits anonymous per-file downloads after a few hundred files; the script
resumes where it stopped, or download the folder as a zip in a browser and
unzip it into `dataset/`.

`splits/seed0_val20.json` is the Assignment 1 split (1,920 training and 480
validation images, 30 per class). It stores a fingerprint of the file list, and
loading fails if the data differ.

## 3. Method overview

| Script | Purpose |
|---|---|
| `pretrain.py` | Self-supervised pretraining: SimCLR (NT-Xent between two augmented views) or RotNet (predict 0/90/180/270 degree rotation). Labels are never used for the gradient; a weighted k-NN and an L-BFGS linear probe on frozen features monitor the validation split every 25 epochs. |
| `train.py` | Every classification run: supervised from scratch (`model.init: scratch`), fine-tuning an SSL backbone (`model.init: ssl`), linear evaluation (`model.freeze_backbone: true`), and joint CE + NT-Xent training (a `joint:` section; `joint.weight: 0` is the matched control). Selects the best epoch on validation, then evaluates the test set once. |
| `evaluate.py` | Accuracy, per-class accuracy and confusion matrix for one checkpoint or an ensemble, with optional flip test-time augmentation. |
| `src/gpu_augment.py` | SimCLR view generation (random resized crop, flip, color jitter, grayscale, blur) on the GPU for a whole batch; the per-image CPU pipeline cost ~50 ms per view. |

SimCLR views are generated from a 256x256 center region of each image (shorter
side resized to 256). Fifteen classes are grayscale and Flower is the only color
class, so color jitter mostly acts as brightness/contrast jitter here; removing
it (`ssl_simclr_nocolor`) lets instance discrimination succeed on intensity
statistics and the features become much worse.

## 4. Reproduce the experiments

Each phase script launches one detached GNU Screen session per GPU and writes
`logs/<session>.screen.log`:

```bash
bash scripts/run_phase1.sh   # baselines, joint training, SimCLR / RotNet / no-color SimCLR
bash scripts/run_phase2.sh   # seeds, label efficiency, SSL-epoch and crop ablations
bash scripts/run_phase3.sh   # best recipe (3 seeds), SSL on all images, ResNet-50
python summarize_runs.py --runs-root runs
python scripts/make_figures.py --runs-root runs --output report/figures/analysis.pdf
```

A single run is, for example:

```bash
CUDA_VISIBLE_DEVICES=0 python -u pretrain.py --config configs/ssl_simclr.yaml \
  --run-name ssl_simclr_crop008_r18_e500 --set ssl.augmentation.crop_min=0.08
CUDA_VISIBLE_DEVICES=0 python -u train.py --config configs/ft_randaug_mix.yaml \
  --run-name ft_randaug_mix_crop008 --init-checkpoint runs/ssl_simclr_crop008_r18_e500/last.pt
```

`--set section.key=value` overrides any configuration value; the resolved
configuration of every run is saved as `runs/<run>/config.yaml`. `train.py`
and `pretrain.py` refuse to overwrite a non-empty run directory.

## 5. Experiments

Every run uses `splits/seed0_val20.json`. Each run directory under `runs/` holds
the resolved `config.yaml`, `environment.json`, per-epoch `history.csv` and
`events.jsonl`, and `run_summary.json` (checkpoints are not committed);
`runs/experiment_summary.csv` collects all of them. Validation is the best epoch
on the 480-image split; test is that checkpoint on the 400 test images.

**Main comparison** (ResNet-18, 200 epochs, AdamW, mean +- standard deviation
over seeds; the gap is clean training-split accuracy minus validation accuracy):

| Recipe | Seeds | Val. (%) | Test (%) | Gap |
|---|---|---|---|---|
| Supervised, random init, basic aug. (`sup_baseline`) | 3 | 83.40 +- 0.32 | 81.25 +- 0.90 | 16.4 |
| Supervised, Assignment 1 strong aug. (`sup_strong`) | 2 | 83.75 +- 0.59 | 84.12 +- 0.18 | 16.2 |
| Supervised, RandAugment + MixUp/CutMix (`sup_randaug_mix`) | 3 | 87.29 +- 0.36 | 84.58 +- 1.38 | 12.7 |
| SimCLR (crop 0.2) + fine-tune, basic aug. (`ft_ssl_simclr_r18_e500`) | 3 | 87.78 +- 0.52 | 82.83 +- 0.38 | 12.2 |
| SimCLR (crop 0.2) + fine-tune, RandAug + Mix (`ft_randaug_mix`) | 3 | 88.19 +- 0.79 | 87.67 +- 0.88 | 11.8 |
| SimCLR (crop 0.08) + fine-tune, RandAug + Mix (`ft_randaug_mix_crop008`) | 3 | 89.03 +- 0.12 | 86.17 +- 0.63 | 10.9 |
| SimCLR (crop 0.08, all 2,400 images) + fine-tune, RandAug + Mix | 3 | 89.51 +- 0.12 | 86.58 +- 1.13 | 10.5 |
| Final: ensemble of the 3 runs above | - | 90.00 | **88.25** | - |

**SSL objectives and ablations** (one seed; k-NN and linear probe are on frozen
features at the end of pretraining; linear eval trains only a linear head):

| SSL run | k-NN val. | Linear probe val. | Linear eval val. / test | Fine-tune val. / test |
|---|---|---|---|---|
| SimCLR, crop 0.2, 500 ep. | 81.25 | 80.00 | 81.25 / 77.00 | 88.33 / 83.25 |
| SimCLR, crop 0.2, 1,000 ep. | 79.38 | 79.58 | 80.62 / 79.00 | 88.12 / 84.00 |
| SimCLR, crop 0.08, 500 ep. | 81.67 | 82.29 | 84.38 / 81.50 | 87.50 / 84.75 |
| SimCLR, no color jitter / grayscale | 68.96 | 69.79 | 71.04 / 69.75 | 85.00 / 80.50 |
| RotNet (rotation prediction), 150 ep. | 58.96 | 66.46 | 69.17 / 65.25 | 85.00 / 80.25 |
| Joint CE + 0.5 NT-Xent from scratch (control: weight 0) | - | - | - | 85.83 / 81.50 (86.25 / 81.00) |
| SimCLR, then joint CE + NT-Xent fine-tuning | - | - | - | 86.88 / 81.75 |

**Label efficiency** (test accuracy, 10/25/50/100% of the 1,920 labels, batch 64
below 100%, one seed below 100% and the 3-seed mean at 100%): supervised 62.00 / 75.50 / 78.00 / 81.25; SimCLR + fine-tune
69.50 / 77.00 / 81.50 / 82.83; SimCLR + linear 71.75 / 74.00 / - / 77.00.

**Backbone** (SimCLR crop 0.08, batch 128 for both, since ResNet-50 does not fit
256 x 2 views in 16 GB): supervised ResNet-50 from scratch 79.58 val / 81.25 test
(ResNet-18: 83.75 / 82.25); SimCLR + fine-tune ResNet-50 87.50 / 85.00, ResNet-18
88.12 / 83.25.

**SSL length** (fine-tuning the crop-0.2 SimCLR backbone saved at epoch
100 / 200 / 300 / 500): validation 84.79 / 87.50 / 87.08 / 88.33, test
82.00 / 81.50 / 84.25 / 83.25.

## Repository layout

```
pretrain.py           self-supervised pretraining (SimCLR, RotNet) with k-NN / linear-probe monitoring
train.py              supervised, fine-tuning, linear-evaluation and joint CE + SSL training
evaluate.py           test/validation evaluation, ensembles, flip TTA
summarize_runs.py     collects run summaries into runs/experiment_summary.csv
src/                  data pipeline, GPU augmentation, models, SSL losses, training utilities
configs/              one YAML file per experiment family
splits/               the Assignment 1 validation split
runs/                 lightweight artifacts of every run (config, history, summary; no checkpoints)
checkpoints/          released final checkpoints (Git LFS)
scripts/              data download/check, experiment launchers, figures, checkpoint export
report/               report source, figure and PDF
```
