from __future__ import annotations

import math
import platform
import sys

import torch
import torch.nn as nn


def build_optimizer(parameter_groups: list[dict], train_config: dict) -> torch.optim.Optimizer:
    """Each group may carry an `lr_scale`; the per-step schedule multiplies it in."""
    base_lr = float(train_config["learning_rate"])
    for group in parameter_groups:
        group.setdefault("lr_scale", 1.0)
        group["lr"] = base_lr * group["lr_scale"]
    name = str(train_config.get("optimizer", "adamw")).lower()
    weight_decay = float(train_config.get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(parameter_groups, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            parameter_groups,
            momentum=float(train_config.get("momentum", 0.9)),
            nesterov=True,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def split_decay_groups(model: nn.Module, lr_scale: float = 1.0, prefix: str = "") -> list[dict]:
    """Exclude BatchNorm parameters and biases from weight decay."""
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if parameter.ndim <= 1 else decay).append(parameter)
    return [
        {"params": decay, "lr_scale": lr_scale, "name": f"{prefix}decay"},
        {"params": no_decay, "lr_scale": lr_scale, "weight_decay": 0.0, "name": f"{prefix}no_decay"},
    ]


class WarmupCosineSchedule:
    """Per-iteration linear warmup followed by cosine decay to `minimum_learning_rate`."""

    def __init__(self, optimizer: torch.optim.Optimizer, train_config: dict, steps_per_epoch: int) -> None:
        self.optimizer = optimizer
        self.base_lr = float(train_config["learning_rate"])
        self.min_lr = float(train_config.get("minimum_learning_rate", 0.0))
        self.total_steps = int(train_config["epochs"]) * steps_per_epoch
        self.warmup_steps = int(float(train_config.get("warmup_epochs", 0)) * steps_per_epoch)
        self.step_count = 0

    def learning_rate(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * min(1.0, progress)))

    def step(self) -> float:
        lr = self.learning_rate(self.step_count)
        for group in self.optimizer.param_groups:
            group["lr"] = lr * group["lr_scale"]
        self.step_count += 1
        return lr

    def state_dict(self) -> dict:
        return {"step_count": self.step_count}


def environment_info(device: torch.device, amp_enabled: bool) -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "amp": amp_enabled,
    }
