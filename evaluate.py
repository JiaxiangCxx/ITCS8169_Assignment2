#!/usr/bin/env python3
"""Evaluate one checkpoint (or an ensemble of checkpoints) on a labeled split."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from src.data import load_training_data, make_labeled_loader, make_loader, build_eval_transform
from src.models import build_classifier
from src.utils import resolve_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate selected checkpoints")
    parser.add_argument("--checkpoint", required=True, nargs="+", help="One or more checkpoints; several are ensembled")
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--split", default="test", help="'test', or 'val' for the fixed validation split")
    parser.add_argument("--tta", action="store_true", help="Average predictions with the horizontally flipped image")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device")
    return parser.parse_args()


def load_model(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    # The joint-training projector is not needed for classification.
    state = {key: value for key, value in checkpoint["model_state"].items() if not key.startswith("projector.")}
    model = build_classifier(config["model"], num_classes=len(checkpoint["class_names"]))
    model.load_state_dict(state)
    return model.to(device).eval(), config, list(checkpoint["class_names"]), checkpoint


@torch.inference_mode()
def predict_probabilities(model, loader, device, tta: bool) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities, labels = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images).float()
        batch_probabilities = logits.softmax(dim=1)
        if tta:
            flipped = model(torch.flip(images, dims=(3,))).float().softmax(dim=1)
            batch_probabilities = (batch_probabilities + flipped) / 2
        probabilities.append(batch_probabilities.cpu())
        labels.append(targets)
    return torch.cat(probabilities), torch.cat(labels)


def save_confusion_plot(matrix: np.ndarray, class_names: list[str], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; skipping confusion-matrix plot")
        return
    figure, axis = plt.subplots(figsize=(10, 9))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(range(len(class_names)), labels=class_names, rotation=60, ha="right")
    axis.set_yticks(range(len(class_names)), labels=class_names)
    axis.set(xlabel="Predicted class", ylabel="True class", title="Confusion matrix")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    ensemble_probabilities = None
    labels = None
    class_names = None
    for path in args.checkpoint:
        model, config, checkpoint_classes, _ = load_model(path, device)
        data_config = dict(config["data"], root=args.data_root)
        if args.split == "val":
            training = load_training_data(data_config, seed=int(config["experiment"].get("seed", 0)))
            loader = make_loader(
                training.data, training.val_indices, build_eval_transform(data_config), data_config, shuffle=False
            )
            dataset_classes = training.data.class_names
        else:
            loader, dataset_classes = make_labeled_loader(data_config, split=args.split)
        if dataset_classes != checkpoint_classes:
            raise RuntimeError(f"Checkpoint class order {checkpoint_classes} does not match dataset {dataset_classes}")
        if class_names is not None and checkpoint_classes != class_names:
            raise RuntimeError("Ensembled checkpoints disagree on the class order")
        class_names = checkpoint_classes
        probabilities, split_labels = predict_probabilities(model, loader, device, args.tta)
        if labels is not None and not torch.equal(labels, split_labels):
            raise RuntimeError("Ensembled checkpoints saw the images in a different order")
        labels = split_labels
        ensemble_probabilities = probabilities if ensemble_probabilities is None else ensemble_probabilities + probabilities

    predictions = ensemble_probabilities.argmax(dim=1)
    num_classes = len(class_names)
    confusion = torch.bincount(labels * num_classes + predictions, minlength=num_classes**2)
    confusion = confusion.reshape(num_classes, num_classes).numpy()
    correct = int(np.trace(confusion))
    total = int(confusion.sum())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / "confusion_matrix.csv", confusion, delimiter=",", fmt="%d")
    per_class_rows = [
        {
            "class": name,
            "correct": int(confusion[index, index]),
            "total": int(confusion[index].sum()),
            "accuracy": float(confusion[index, index] / max(1, confusion[index].sum())),
        }
        for index, name in enumerate(class_names)
    ]
    with (output_dir / "per_class_accuracy.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_class_rows[0]))
        writer.writeheader()
        writer.writerows(per_class_rows)

    metrics = {
        "checkpoints": [str(Path(path).resolve()) for path in args.checkpoint],
        "split": args.split,
        "tta": args.tta,
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
    }
    save_json(metrics, output_dir / "metrics.json")
    save_confusion_plot(confusion, class_names, output_dir / "confusion_matrix.png")
    print(metrics)


if __name__ == "__main__":
    main()
