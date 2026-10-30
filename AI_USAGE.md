# AI Usage

## Tools used
Claude Code (Anthropic).

## How AI assisted me
1. **Gathering ideas on which models to use.** Which self-supervised methods
   and backbones to try with only 2,400 images (SimCLR, RotNet, joint CE + SSL
   training, ResNet-18 vs. ResNet-50), and how to compare them fairly (matched
   controls, several seeds, label-efficiency runs).
2. **Running the experiments.** The assistant implemented the training code,
   launched the runs on cci-abc and collected the results. When per-image CPU
   augmentation made each SimCLR pretraining run take about 4 hours, it moved
   the augmentations to the GPU (about 20 minutes per run).
3. **Checking grammar and polishing the report.** The assistant helped write
   the report and README from the saved result files, then checked the grammar
   and polished the wording.

## Verification
Every reported number comes from `runs/*/run_summary.json` or
`runs/final_test_evaluation/metrics.json`. The final model (ensemble, no
test-time augmentation) was fixed on validation before the test set was
evaluated for it, and retraining the final recipe reproduced the original run
bit for bit.
