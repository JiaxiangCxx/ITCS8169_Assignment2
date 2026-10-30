#!/usr/bin/env python3
"""Self-supervised pretraining (SimCLR or RotNet) on the training images only.

Labels are never used for the gradient. They are used only by the k-NN and
linear-probe monitors, which measure how linearly separable / clustered the
frozen backbone features are on the fixed validation split.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data import build_eval_transform, load_training_data, make_loader
from src.gpu_augment import GPUMultiViewLoader, GPUViewAugmenter, square_image_tensor
from src.models import SSLNetwork, parameter_counts
from src.ssl import extract_features, knn_accuracy, linear_probe_accuracy, nt_xent, rotate_batch
from src.training import WarmupCosineSchedule, build_optimizer, environment_info, split_decay_groups
from src.utils import (
    append_jsonl,
    apply_overrides,
    atomic_torch_save,
    load_config,
    resolve_device,
    save_json,
    save_yaml,
    set_seed,
    write_history_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-supervised pretraining for Assignment 2")
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    parser.add_argument("--run-name", help="Override experiment.name")
    parser.add_argument("--data-root", help="Override data.root")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--device", help="For example cuda, cuda:0, or cpu")
    parser.add_argument("--seed", type=int, help="Override experiment.seed")
    parser.add_argument("--epochs", type=int, help="Override train.epochs (smoke tests)")
    parser.add_argument(
        "--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
        help="Override any configuration value, e.g. --set ssl.augmentation.crop_min=0.08",
    )
    return parser.parse_args()


def monitor(model, train_loader, val_loader, num_classes, device, amp_enabled) -> dict:
    train_features, train_labels = extract_features(model.backbone, train_loader, device, amp_enabled)
    val_features, val_labels = extract_features(model.backbone, val_loader, device, amp_enabled)
    return {
        "knn_val_accuracy": knn_accuracy(
            train_features, train_labels, val_features, val_labels, num_classes
        ),
        "linear_probe_val_accuracy": linear_probe_accuracy(
            train_features, train_labels, val_features, val_labels, num_classes
        ),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config, sections=("experiment", "data", "model", "ssl", "train"))
    if args.run_name:
        config["experiment"]["name"] = args.run_name
    if args.data_root:
        config["data"]["root"] = args.data_root
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
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
    ssl_config = config["ssl"]
    method = str(ssl_config["method"])
    training = load_training_data(data_config, seed=seed)
    num_classes = len(training.data.class_names)

    # By default the SSL objective sees only the 1,920 training-split images, so
    # the validation images stay completely unseen until evaluation.
    ssl_images = str(data_config.get("ssl_images", "train_split"))
    if ssl_images == "train_split":
        ssl_indices = training.train_indices
    elif ssl_images == "all_train":
        ssl_indices = training.train_indices + training.val_indices
    else:
        raise ValueError("data.ssl_images must be 'train_split' or 'all_train'")

    # Views are generated on the GPU from a uint8 copy of the images (~0.4 GB).
    ssl_loader = GPUMultiViewLoader(
        square_image_tensor(training.data, ssl_indices, int(data_config.get("cache_size", 256))).to(device),
        torch.tensor([training.data.targets[index] for index in ssl_indices], device=device),
        GPUViewAugmenter(int(data_config["image_size"]), ssl_config.get("augmentation", {}), device, seed),
        batch_size=int(data_config["batch_size"]),
        num_views=2 if method == "simclr" else 1,
        seed=seed,
    )
    eval_transform = build_eval_transform(data_config)
    monitor_train_loader = make_loader(
        training.data, training.train_indices, eval_transform, data_config, shuffle=False
    )
    monitor_val_loader = make_loader(
        training.data, training.val_indices, eval_transform, data_config, shuffle=False
    )

    model = SSLNetwork(
        str(config["model"]["name"]), method, config["model"].get("projector")
    ).to(device, memory_format=torch.channels_last)
    optimizer = build_optimizer(split_decay_groups(model), config["train"])
    schedule = WarmupCosineSchedule(optimizer, config["train"], steps_per_epoch=len(ssl_loader))
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    temperature = float(ssl_config.get("temperature", 0.2))

    environment = {
        **environment_info(device, amp_enabled),
        "method": method,
        "ssl_images": ssl_images,
        "ssl_image_count": len(ssl_indices),
        "steps_per_epoch": len(ssl_loader),
        "parameter_count": parameter_counts(model)[0],
        "class_names": training.data.class_names,
        **training.split_info,
    }
    save_json(environment, run_dir / "environment.json")
    print(environment)

    epochs = int(config["train"]["epochs"])
    monitor_every = int(config["train"].get("monitor_every", 25))
    save_epochs = {int(epoch) for epoch in config["train"].get("save_epochs", [])}
    history: list[dict] = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        total_loss = total_accuracy = 0.0
        batches = 0
        for inputs, _ in ssl_loader:
            lr = schedule.step()
            if method == "simclr":
                images = torch.cat(inputs).contiguous(memory_format=torch.channels_last)
            else:
                images, rotation_labels = rotate_batch(inputs)
                images = images.contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(images)
            if method == "simclr":
                z1, z2 = outputs.float().chunk(2)
                loss, accuracy = nt_xent(z1, z2, temperature)
            else:
                logits = outputs.float()
                loss = F.cross_entropy(logits, rotation_labels)
                accuracy = (logits.argmax(dim=1) == rotation_labels).float().mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            batches += 1

        row = {
            "epoch": epoch,
            "ssl_loss": total_loss / batches,
            "ssl_accuracy": total_accuracy / batches,
            "learning_rate": lr,
            "epoch_seconds": time.time() - epoch_start,
            "knn_val_accuracy": None,
            "linear_probe_val_accuracy": None,
        }
        monitored = epoch == 1 or epoch % monitor_every == 0 or epoch == epochs
        if monitored:
            row.update(
                monitor(model, monitor_train_loader, monitor_val_loader, num_classes, device, amp_enabled)
            )
        history.append(row)
        append_jsonl(row, run_dir / "events.jsonl")
        write_history_csv(history, run_dir / "history.csv")

        payload = {
            "epoch": epoch,
            "backbone_state": model.backbone.state_dict(),
            "model_state": model.state_dict(),
            "config": config,
            "class_names": training.data.class_names,
            "torch_version": torch.__version__,
        }
        if monitored:
            atomic_torch_save(payload, run_dir / "last.pt")
        if epoch in save_epochs:
            atomic_torch_save(payload, run_dir / f"epoch_{epoch:04d}.pt")

        message = (
            f"Epoch {epoch:04d}/{epochs} | ssl loss {row['ssl_loss']:.4f} "
            f"acc {row['ssl_accuracy']:.4f} | lr {lr:.4g} | {row['epoch_seconds']:.1f}s"
        )
        if monitored:
            message += (
                f" | kNN {row['knn_val_accuracy']:.4f} "
                f"linear {row['linear_probe_val_accuracy']:.4f}"
            )
        print(message, flush=True)

    final = history[-1]
    summary = {
        "experiment": run_name,
        "stage": "pretrain",
        "method": method,
        "model": config["model"]["name"],
        "seed": seed,
        "epochs_completed": len(history),
        "final_ssl_loss": final["ssl_loss"],
        "final_ssl_accuracy": final["ssl_accuracy"],
        "knn_val_accuracy": final["knn_val_accuracy"],
        "linear_probe_val_accuracy": final["linear_probe_val_accuracy"],
        "elapsed_seconds": time.time() - start_time,
        "ssl_images": ssl_images,
        **training.split_info,
    }
    save_json(summary, run_dir / "run_summary.json")
    print(f"Completed {run_name}: kNN {final['knn_val_accuracy']:.4f}, "
          f"linear probe {final['linear_probe_val_accuracy']:.4f}")
    print(f"Artifacts: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
