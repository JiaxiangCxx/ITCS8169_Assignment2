from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    """SimCLR's normalized-temperature cross entropy.

    Each of the 2N views must pick out the other view of its own image among
    the 2N - 1 candidates in the batch; the other 2N - 2 views act as
    negatives. Returns the loss and the fraction of views whose positive is
    the most similar candidate (an instance-discrimination accuracy).
    """
    batch_size = z1.size(0)
    z = F.normalize(torch.cat([z1, z2]).float(), dim=1)
    logits = z @ z.t() / temperature
    logits.fill_diagonal_(float("-inf"))
    targets = torch.cat(
        [torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)]
    ).to(z.device)
    loss = F.cross_entropy(logits, targets)
    accuracy = (logits.argmax(dim=1) == targets).float().mean()
    return loss, accuracy


def rotate_batch(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """RotNet pretext task: every image in all four 90-degree rotations, labeled 0-3."""
    rotated = torch.cat([torch.rot90(images, k, dims=(2, 3)) for k in range(4)])
    labels = torch.arange(4, device=images.device).repeat_interleave(images.size(0))
    return rotated, labels


@torch.inference_mode()
def extract_features(
    backbone: nn.Module,
    loader,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    was_training = backbone.training
    backbone.eval()
    features, labels = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            features.append(backbone(images).float())
        labels.append(targets.to(device))
    backbone.train(was_training)
    return torch.cat(features), torch.cat(labels)


@torch.inference_mode()
def knn_accuracy(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    query_features: torch.Tensor,
    query_labels: torch.Tensor,
    num_classes: int,
    k: int = 20,
    temperature: float = 0.07,
) -> float:
    """Weighted cosine k-NN classifier (Wu et al., 2018) used to monitor features without training a head."""
    train_features = F.normalize(train_features, dim=1)
    query_features = F.normalize(query_features, dim=1)
    similarity = query_features @ train_features.t()
    weights, neighbors = similarity.topk(k, dim=1)
    weights = (weights / temperature).exp()
    votes = torch.zeros(query_features.size(0), num_classes, device=query_features.device)
    votes.scatter_add_(1, train_labels[neighbors], weights)
    return (votes.argmax(dim=1) == query_labels).float().mean().item()


def linear_probe_accuracy(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    query_features: torch.Tensor,
    query_labels: torch.Tensor,
    num_classes: int,
    weight_decay: float = 1e-3,
    steps: int = 200,
) -> float:
    """Multinomial logistic regression on frozen, standardized features (full-batch L-BFGS).

    The hyperparameters are fixed for every checkpoint so that probe accuracies
    are comparable across SSL methods and epochs.
    """
    # Features come from inference mode; clone them into ordinary tensors so
    # autograd may save them for the backward pass.
    train_features, query_features = train_features.clone(), query_features.clone()
    train_labels = train_labels.clone()
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True) + 1e-6
    train_x = ((train_features - mean) / std).detach()
    query_x = ((query_features - mean) / std).detach()
    with torch.enable_grad():
        classifier = nn.Linear(train_x.size(1), num_classes).to(train_x.device)
        optimizer = torch.optim.LBFGS(
            classifier.parameters(), lr=1.0, max_iter=steps, history_size=20,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(classifier(train_x), train_labels)
            loss = loss + weight_decay * classifier.weight.pow(2).sum() / 2
            loss.backward()
            return loss

        optimizer.step(closure)
    with torch.no_grad():
        predictions = classifier(query_x).argmax(dim=1)
    return (predictions == query_labels).float().mean().item()
