#!/usr/bin/env python3
"""Supervised training, SSL fine-tuning, linear evaluation, and joint CE + SimCLR training.

One script covers every classification run so that the compared methods share
the data split, evaluation, optimizer code and model selection rule:

* model.init: scratch            -> supervised baseline (random initialization)
* model.init: ssl                -> fine-tune a pretrain.py backbone
* model.freeze_backbone: true    -> linear evaluation of a frozen backbone
* joint section present          -> CE on two SSL views + joint.weight * NT-Xent
                                    (joint.weight: 0 is the augmentation-only control)

The model is selected on the fixed validation split only; the labeled test set
is evaluated once at the end for the selected checkpoint.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.transforms import v2

from src.data import (
    build_eval_transform,
    build_train_transform,
    load_training_data,
    make_labeled_loader,
    make_loader,
    stratified_subset,
)
from src.gpu_augment import GPUMultiViewLoader, GPUViewAugmenter, square_image_tensor
from src.models import build_classifier, load_ssl_backbone, parameter_counts, set_backbone_trainable
from src.ssl import nt_xent
from src.training import WarmupCosineSchedule, build_optimizer, environment_info, split_decay_groups
from src.utils import (
    append_jsonl,
    apply_overrides,
    atomic_torch_save,
    load_config,
    plot_history,
    resolve_device,
    save_json,
    save_yaml,
    set_seed,
    write_history_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or fine-tune a classifier for Assignment 2")
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    parser.add_argument("--run-name", help="Override experiment.name")
    parser.add_argument("--data-root", help="Override data.root")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--device", help="For example cuda, cuda:0, or cpu")
    parser.add_argument(
        "--seed",
        type=int,
        help="Override experiment.seed; the validation split stays fixed",
    )
    parser.add_argument("--init-checkpoint", help="Override model.init_checkpoint")
    parser.add_argument("--epochs", type=int, help="Override train.epochs (smoke tests)")
    parser.add_argument(
        "--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
        help="Override any configuration value, e.g. --set data.label_fraction=0.1",
    )
    return parser.parse_args()


@torch.inference_mode()
def evaluate(model, loader, criterion, device, amp_enabled) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
        total_loss += criterion(logits.float(), labels).item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += labels.size(0)
    return total_loss / total_examples, total_correct / total_examples


def build_mixer(train_config: dict, num_classes: int):
    """Batch-level MixUp/CutMix; returns None when both are disabled."""
    mixers = []
    if float(train_config.get("mixup_alpha", 0.0)) > 0:
        mixers.append(v2.MixUp(alpha=float(train_config["mixup_alpha"]), num_classes=num_classes))
    if float(train_config.get("cutmix_alpha", 0.0)) > 0:
        mixers.append(v2.CutMix(alpha=float(train_config["cutmix_alpha"]), num_classes=num_classes))
    if not mixers:
        return None
    return v2.RandomChoice(mixers) if len(mixers) > 1 else mixers[0]


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    schedule,
    scaler,
    device,
    amp_enabled,
    joint_config: dict | None,
    mixer,
    mix_prob: float,
    freeze_backbone: bool,
) -> dict:
    model.train()
    if freeze_backbone:
        # Linear evaluation: keep the frozen backbone's BatchNorm statistics fixed.
        model.backbone.eval()
    totals = {"loss": 0.0, "ce": 0.0, "ssl": 0.0, "ssl_accuracy": 0.0, "correct": 0, "examples": 0}
    batches = 0
    for inputs, labels in loader:
        schedule.step()
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if joint_config is not None:
            images = torch.cat(inputs).to(device, non_blocking=True, memory_format=torch.channels_last)
            targets = torch.cat([labels, labels])
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits, projections = model.forward_with_projection(images)
            ce_loss = criterion(logits.float(), targets)
            z1, z2 = projections.float().chunk(2)
            ssl_loss, ssl_accuracy = nt_xent(z1, z2, float(joint_config.get("temperature", 0.2)))
            loss = ce_loss + float(joint_config["weight"]) * ssl_loss
            totals["ssl"] += ssl_loss.item()
            totals["ssl_accuracy"] += ssl_accuracy.item()
            hard_targets = targets
        else:
            images = inputs.to(device, non_blocking=True)
            targets = labels
            if mixer is not None and torch.rand(()) < mix_prob:
                images, targets = mixer(images, labels)
            images = images.contiguous(memory_format=torch.channels_last)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
            ce_loss = criterion(logits.float(), targets)
            loss = ce_loss
            hard_targets = labels
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        totals["loss"] += loss.item()
        totals["ce"] += ce_loss.item()
        totals["correct"] += (logits.argmax(dim=1) == hard_targets).sum().item()
        totals["examples"] += hard_targets.size(0)
        batches += 1
    return {
        "train_loss": totals["loss"] / batches,
        "train_ce_loss": totals["ce"] / batches,
        "train_ssl_loss": totals["ssl"] / batches if joint_config is not None else None,
        "train_ssl_accuracy": totals["ssl_accuracy"] / batches if joint_config is not None else None,
        "train_accuracy": totals["correct"] / totals["examples"],
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.run_name:
        config["experiment"]["name"] = args.run_name
    if args.data_root:
        config["data"]["root"] = args.data_root
    if args.seed is not None:
        config["data"].setdefault("split_seed", int(config["experiment"].get("seed", 0)))
        config["experiment"]["seed"] = args.seed
    if args.init_checkpoint:
        config["model"]["init_checkpoint"] = args.init_checkpoint
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    apply_overrides(config, args.set)

    run_name = str(config["experiment"]["name"])
    run_dir = Path(args.runs_root) / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory is not empty: {run_dir}. Use --run-name to avoid overwriting."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, run_dir / "config.yaml")

    seed = int(config["experiment"].get("seed", 0))
    set_seed(seed, deterministic=bool(config["train"].get("deterministic", True)))
    device = resolve_device(args.device)
    amp_enabled = bool(config["train"].get("amp", True)) and device.type == "cuda"

    data_config = config["data"]
    model_config = config["model"]
    train_config = config["train"]
    joint_config = config.get("joint")
    training = load_training_data(data_config, seed=seed)
    class_names = training.data.class_names
    num_classes = len(class_names)

    # train_on_all refits a fixed recipe on all 2,400 images; there is then no
    # held-out validation set, so the last epoch is the selected model.
    train_on_all = bool(data_config.get("train_on_all", False))
    train_indices = training.train_indices + (training.val_indices if train_on_all else [])
    label_fraction = float(data_config.get("label_fraction", 1.0))
    train_indices = stratified_subset(
        train_indices, training.data.targets, label_fraction, seed=int(data_config.get("label_subset_seed", 0))
    )

    if joint_config is not None:
        if build_mixer(train_config, num_classes) is not None:
            raise ValueError("MixUp/CutMix is not supported together with joint training")
        # Two SimCLR views per image, generated on the GPU (see src/gpu_augment.py).
        train_loader = GPUMultiViewLoader(
            square_image_tensor(training.data, train_indices, int(data_config.get("cache_size", 256))).to(device),
            torch.tensor([training.data.targets[index] for index in train_indices], device=device),
            GPUViewAugmenter(int(data_config["image_size"]), data_config.get("ssl_augmentation", {}), device, seed),
            batch_size=int(data_config["batch_size"]),
            num_views=2,
            seed=seed,
        )
    else:
        train_loader = make_loader(
            training.data, train_indices, build_train_transform(data_config), data_config,
            shuffle=True, seed=seed, drop_last=True,
        )
    val_loader = make_loader(
        training.data, training.val_indices, build_eval_transform(data_config), data_config, shuffle=False
    )

    model = build_classifier(model_config, num_classes, with_projector=joint_config is not None)
    init = str(model_config.get("init", "scratch"))
    init_info: dict = {"init": init}
    if init == "ssl":
        init_info.update(load_ssl_backbone(model, model_config["init_checkpoint"]))
    elif init != "scratch":
        raise ValueError("model.init must be 'scratch' or 'ssl'")
    freeze_backbone = bool(model_config.get("freeze_backbone", False))
    if freeze_backbone:
        set_backbone_trainable(model, trainable=False)
    model = model.to(device, memory_format=torch.channels_last)

    backbone_lr_scale = float(train_config.get("backbone_lr_scale", 1.0))
    groups = split_decay_groups(model.backbone, backbone_lr_scale, prefix="backbone_")
    groups += split_decay_groups(model.head, 1.0, prefix="head_")
    if model.projector is not None:
        groups += split_decay_groups(model.projector, 1.0, prefix="projector_")
    groups = [group for group in groups if group["params"]]
    optimizer = build_optimizer(groups, train_config)
    schedule = WarmupCosineSchedule(optimizer, train_config, steps_per_epoch=len(train_loader))
    criterion = nn.CrossEntropyLoss(label_smoothing=float(train_config.get("label_smoothing", 0.0)))
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    mixer = build_mixer(train_config, num_classes)
    mix_prob = float(train_config.get("mix_prob", 1.0))

    total_parameters, trainable_parameters = parameter_counts(model)
    environment = {
        **environment_info(device, amp_enabled),
        **init_info,
        "class_names": class_names,
        "parameter_count": total_parameters,
        "trainable_parameter_count": trainable_parameters,
        "labeled_train_images": len(train_indices),
        "train_on_all": train_on_all,
        **training.split_info,
    }
    save_json(environment, run_dir / "environment.json")
    print(environment)

    epochs = int(train_config["epochs"])
    patience = int(train_config.get("early_stopping_patience", 0))
    best_val_accuracy = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict] = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, schedule, scaler, device, amp_enabled,
            joint_config, mixer, mix_prob, freeze_backbone,
        )
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device, amp_enabled)

        improved = train_on_all or val_accuracy > best_val_accuracy
        if improved:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        row = {
            "epoch": epoch,
            **train_metrics,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": schedule.learning_rate(schedule.step_count - 1),
            "epoch_seconds": time.time() - epoch_start,
        }
        history.append(row)
        append_jsonl(row, run_dir / "events.jsonl")
        write_history_csv(history, run_dir / "history.csv")

        payload = {
            "epoch": epoch,
            "best_val_accuracy": best_val_accuracy,
            "model_state": model.state_dict(),
            "config": config,
            "class_names": class_names,
            "split_info": training.split_info,
            "init_info": init_info,
            "torch_version": torch.__version__,
            "torchvision_version": __import__("torchvision").__version__,
        }
        atomic_torch_save(payload, run_dir / "last.pt")
        if improved:
            atomic_torch_save(payload, run_dir / "best.pt")

        message = (
            f"Epoch {epoch:03d}/{epochs} | train loss {row['train_loss']:.4f} "
            f"acc {row['train_accuracy']:.4f}"
        )
        if joint_config is not None:
            message += f" (ssl {row['train_ssl_loss']:.3f})"
        message += (
            f" | val loss {val_loss:.4f} acc {val_accuracy:.4f} | "
            f"best {best_val_accuracy:.4f} (epoch {best_epoch}) | {row['epoch_seconds']:.1f}s"
        )
        print(message, flush=True)

        if patience > 0 and epochs_without_improvement >= patience:
            print(f"Early stopping after {patience} epochs without improvement")
            break

    # Evaluate the selected checkpoint: clean (non-augmented) training-split
    # accuracy for the generalization gap, then the labeled test set once.
    selected = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(selected["model_state"])
    eval_criterion = nn.CrossEntropyLoss()
    clean_train_loader = make_loader(
        training.data, train_indices, build_eval_transform(data_config), data_config, shuffle=False
    )
    _, clean_train_accuracy = evaluate(model, clean_train_loader, eval_criterion, device, amp_enabled)
    test_accuracy = None
    if bool(train_config.get("evaluate_test", True)):
        test_loader, test_classes = make_labeled_loader(data_config, str(data_config.get("test_dir", "test")))
        if test_classes != class_names:
            raise RuntimeError(f"Test class order {test_classes} does not match training {class_names}")
        _, test_accuracy = evaluate(model, test_loader, eval_criterion, device, amp_enabled)

    summary = {
        "experiment": run_name,
        "stage": "classify",
        "model": model_config["name"],
        **init_info,
        "freeze_backbone": freeze_backbone,
        "joint_weight": float(joint_config["weight"]) if joint_config is not None else None,
        "augmentation": "ssl_two_view" if joint_config is not None else data_config.get("augmentation"),
        "label_fraction": label_fraction,
        "labeled_train_images": len(train_indices),
        "train_on_all": train_on_all,
        "image_size": int(data_config["image_size"]),
        "seed": seed,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_accuracy": None if train_on_all else best_val_accuracy,
        "clean_train_accuracy": clean_train_accuracy,
        "generalization_gap": None if train_on_all else clean_train_accuracy - best_val_accuracy,
        "test_accuracy": test_accuracy,
        "elapsed_seconds": time.time() - start_time,
        **training.split_info,
    }
    save_json(summary, run_dir / "run_summary.json")
    plot_history(history, run_dir / "curves.png")
    print(
        f"Completed {run_name}: best validation accuracy {best_val_accuracy:.4f} "
        f"(epoch {best_epoch}), test accuracy {test_accuracy}"
    )
    print(f"Artifacts: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
