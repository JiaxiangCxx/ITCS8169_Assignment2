from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


# Every network here is randomly initialized (weights=None). The assignment
# forbids externally pretrained weights, so there is deliberately no code path
# that downloads any.
SUPPORTED_BACKBONES = {"resnet18", "resnet34", "resnet50"}


def build_backbone(name: str) -> tuple[nn.Module, int]:
    if name not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported backbone {name!r}; choose one of {sorted(SUPPORTED_BACKBONES)}")
    model = models.get_model(name, weights=None)
    feature_dim = model.fc.in_features
    model.fc = nn.Identity()
    return model, feature_dim


def build_projector(
    in_dim: int,
    hidden_dim: int = 2048,
    out_dim: int = 128,
    num_layers: int = 2,
) -> nn.Sequential:
    """SimCLR projection MLP. The contrastive loss is applied to its output, so the
    invariances it enforces are absorbed here rather than in the backbone features."""
    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(num_layers - 1):
        layers.extend(
            [nn.Linear(dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)]
        )
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class SSLNetwork(nn.Module):
    """Backbone plus the pretext head used only during self-supervised pretraining."""

    def __init__(self, backbone_name: str, method: str, projector_config: dict | None = None) -> None:
        super().__init__()
        self.method = method
        self.backbone, feature_dim = build_backbone(backbone_name)
        if method == "simclr":
            self.head = build_projector(feature_dim, **(projector_config or {}))
        elif method == "rotation":
            self.head = nn.Linear(feature_dim, 4)
        else:
            raise ValueError(f"Unsupported SSL method: {method}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images))


class Classifier(nn.Module):
    """Backbone + linear classifier, with an optional projector for joint CE + SSL training."""

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        dropout: float = 0.0,
        projector_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.backbone, feature_dim = build_backbone(backbone_name)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(feature_dim, num_classes)
        self.projector = (
            build_projector(feature_dim, **projector_config) if projector_config is not None else None
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.backbone(images)))

    def forward_with_projection(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        return self.head(self.dropout(features)), self.projector(features)


def build_classifier(model_config: dict, num_classes: int, with_projector: bool = False) -> Classifier:
    projector_config = model_config.get("projector", {}) if with_projector else None
    return Classifier(
        str(model_config["name"]),
        num_classes=num_classes,
        dropout=float(model_config.get("dropout", 0.0)),
        projector_config=projector_config,
    )


def load_ssl_backbone(model: Classifier, checkpoint_path: str | Path) -> dict:
    """Initialize the classifier backbone from a pretrain.py checkpoint.

    For joint fine-tuning (the classifier has a projector) of a SimCLR
    backbone, the pretrained projection head is loaded too, so the contrastive
    objective continues from where pretraining stopped instead of restarting.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.backbone.load_state_dict(checkpoint["backbone_state"])
    method = checkpoint["config"]["ssl"]["method"]
    projector_loaded = False
    if model.projector is not None and method == "simclr":
        head_state = {
            key.removeprefix("head."): value
            for key, value in checkpoint["model_state"].items()
            if key.startswith("head.")
        }
        model.projector.load_state_dict(head_state)
        projector_loaded = True
    return {
        "init_checkpoint": str(checkpoint_path),
        "init_method": method,
        "init_epochs": checkpoint["epoch"],
        "init_projector": projector_loaded,
    }


def set_backbone_trainable(model: Classifier, trainable: bool) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = trainable


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable
