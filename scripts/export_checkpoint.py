#!/usr/bin/env python3
"""Export a training checkpoint as a small, weights-only release file.

Keeps what evaluate.py needs (classifier weights, configuration, class order)
and drops the joint-training projector, which is not used for classification.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="runs/<run>/best.pt")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    release = {
        "model_state": {
            key: value
            for key, value in checkpoint["model_state"].items()
            if not key.startswith("projector.")
        },
        "config": checkpoint["config"],
        "class_names": checkpoint["class_names"],
        "epoch": checkpoint["epoch"],
        "best_val_accuracy": checkpoint["best_val_accuracy"],
        "init_info": checkpoint.get("init_info", {}),
        "source_checkpoint": str(args.checkpoint),
        "torch_version": checkpoint.get("torch_version"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(release, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"Wrote {output} ({output.stat().st_size / 2**20:.1f} MiB), SHA-256 {digest}")


if __name__ == "__main__":
    main()
