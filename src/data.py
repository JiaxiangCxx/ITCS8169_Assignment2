from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .utils import save_json, seed_worker


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class ImageData:
    """All images of one split, decoded once and kept in memory as uint8 arrays.

    The training set is only 2,400 images of about 256 px, so caching it costs
    under 0.5 GB and removes JPEG decoding from the multi-view SSL pipeline.
    """

    root: Path
    images: list[np.ndarray]
    targets: list[int]
    class_names: list[str]
    relative_paths: list[str]


@dataclass
class TrainingData:
    data: ImageData
    train_indices: list[int]
    val_indices: list[int]
    split_info: dict


def _load_rgb(path: str, cache_size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if cache_size and min(image.size) != cache_size:
            image = transforms.functional.resize(
                image,
                cache_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            )
        return np.asarray(image).copy()


def load_image_folder(root: str | Path, cache_size: int = 256) -> ImageData:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    folder = datasets.ImageFolder(root)
    with ThreadPoolExecutor(max_workers=8) as pool:
        images = list(
            pool.map(lambda sample: _load_rgb(sample[0], cache_size), folder.samples)
        )
    return ImageData(
        root=root,
        images=images,
        targets=[int(target) for _, target in folder.samples],
        class_names=list(folder.classes),
        relative_paths=[str(Path(path).relative_to(root)) for path, _ in folder.samples],
    )


def _dataset_fingerprint(data: ImageData) -> str:
    return hashlib.sha256("\n".join(data.relative_paths).encode("utf-8")).hexdigest()


def _stratified_split(
    targets: list[int],
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        by_class[int(target)].append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for target in sorted(by_class):
        indices = by_class[target]
        rng.shuffle(indices)
        val_count = max(1, int(round(len(indices) * val_fraction)))
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def load_or_create_split(
    data: ImageData,
    split_file: str | Path,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    split_file = Path(split_file)
    fingerprint = _dataset_fingerprint(data)
    if split_file.exists():
        with split_file.open("r", encoding="utf-8") as stream:
            saved = json.load(stream)
        if saved["dataset_fingerprint"] != fingerprint:
            raise RuntimeError(
                f"Dataset contents changed after {split_file} was created; "
                "use a new split filename or inspect the data"
            )
        if int(saved["seed"]) != seed or float(saved["val_fraction"]) != val_fraction:
            raise RuntimeError(f"Split settings do not match existing file: {split_file}")
        return list(saved["train_indices"]), list(saved["val_indices"])

    train_indices, val_indices = _stratified_split(
        data.targets, val_fraction=val_fraction, seed=seed
    )
    payload = {
        "seed": seed,
        "val_fraction": val_fraction,
        "dataset_fingerprint": fingerprint,
        "dataset_size": len(data.images),
        "class_names": data.class_names,
        "train_indices": train_indices,
        "val_indices": val_indices,
    }
    save_json(payload, split_file)
    return train_indices, val_indices


def stratified_subset(
    indices: list[int],
    targets: list[int],
    fraction: float,
    seed: int,
) -> list[int]:
    """Keep `fraction` of the labeled images of every class (label-efficiency runs)."""
    if fraction >= 1.0:
        return list(indices)
    by_class: dict[int, list[int]] = defaultdict(list)
    for index in indices:
        by_class[targets[index]].append(index)
    rng = random.Random(seed)
    kept: list[int] = []
    for target in sorted(by_class):
        members = sorted(by_class[target])
        rng.shuffle(members)
        kept.extend(members[: max(1, int(round(len(members) * fraction)))])
    rng.shuffle(kept)
    return kept


def load_training_data(data_config: dict, seed: int) -> TrainingData:
    data_root = Path(data_config["root"])
    data = load_image_folder(
        data_root / str(data_config.get("train_dir", "train")),
        cache_size=int(data_config.get("cache_size", 256)),
    )
    # The split seed is separate from the training seed so that seed-variance
    # runs are evaluated on exactly the same validation images.
    train_indices, val_indices = load_or_create_split(
        data,
        split_file=data_config["split_file"],
        val_fraction=float(data_config["val_fraction"]),
        seed=int(data_config.get("split_seed", seed)),
    )
    split_info = {
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "split_file": str(data_config["split_file"]),
    }
    return TrainingData(data, train_indices, val_indices, split_info)


class CachedImageDataset(Dataset):
    def __init__(self, data: ImageData, indices: list[int], transform) -> None:
        self.data = data
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        index = self.indices[position]
        image = Image.fromarray(self.data.images[index])
        return self.transform(image), self.data.targets[index]


class MultiView:
    """Return several independently augmented views of the same image."""

    def __init__(self, view_transforms: list) -> None:
        self.view_transforms = view_transforms

    def __call__(self, image):
        return [transform(image) for transform in self.view_transforms]


def _interpolation(data_config: dict) -> transforms.InterpolationMode:
    name = str(data_config.get("interpolation", "bilinear")).upper()
    try:
        return transforms.InterpolationMode[name]
    except KeyError as error:
        raise ValueError(f"Unknown interpolation mode: {name}") from error


def _normalize() -> list:
    return [transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]


def build_eval_transform(data_config: dict) -> transforms.Compose:
    image_size = int(data_config["image_size"])
    resize_size = int(data_config.get("resize_size", math.ceil(image_size / 0.875)))
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=_interpolation(data_config)),
            transforms.CenterCrop(image_size),
            *_normalize(),
        ]
    )


def build_ssl_view_transform(image_size: int, augmentation: dict) -> transforms.Compose:
    """SimCLR-style view generator; every component can be switched off for ablations.

    Fifteen of the sixteen classes are grayscale scenes and only Flower is in
    color, so the color components are exposed separately: with them disabled
    the SSL objective can solve instance discrimination with color histograms.
    """
    crop_min = float(augmentation.get("crop_min", 0.2))
    jitter = float(augmentation.get("color_jitter", 0.4))
    operations: list = [
        transforms.RandomResizedCrop(
            image_size,
            scale=(crop_min, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(),
    ]
    jitter_p = float(augmentation.get("jitter_p", 0.8))
    if jitter > 0 and jitter_p > 0:
        operations.append(
            transforms.RandomApply(
                [transforms.ColorJitter(jitter, jitter, jitter, min(0.1, jitter / 4))],
                p=jitter_p,
            )
        )
    grayscale_p = float(augmentation.get("grayscale_p", 0.2))
    if grayscale_p > 0:
        operations.append(transforms.RandomGrayscale(p=grayscale_p))
    blur_p = float(augmentation.get("blur_p", 0.5))
    if blur_p > 0:
        kernel = 2 * int(image_size * 0.05) + 1
        operations.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel, sigma=(0.1, 2.0))], p=blur_p
            )
        )
    solarize_p = float(augmentation.get("solarize_p", 0.0))
    if solarize_p > 0:
        operations.append(transforms.RandomSolarize(128, p=solarize_p))
    return transforms.Compose(operations + _normalize())


def build_train_transform(data_config: dict) -> transforms.Compose:
    """Supervised augmentation profiles; `basic` and `strong` match Assignment 1."""
    image_size = int(data_config["image_size"])
    augmentation = str(data_config.get("augmentation", "basic"))
    interpolation = _interpolation(data_config)
    if augmentation == "ssl":
        return build_ssl_view_transform(image_size, data_config.get("ssl_augmentation", {}))
    if augmentation == "none":
        return build_eval_transform(data_config)
    if augmentation == "basic":
        operations = [
            transforms.RandomResizedCrop(
                image_size, scale=(0.75, 1.0), interpolation=interpolation
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    elif augmentation == "strong":
        operations = [
            transforms.RandomResizedCrop(
                image_size, scale=(0.60, 1.0), interpolation=interpolation
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.35, contrast=0.35, saturation=0.35, hue=0.08
                    )
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomRotation(degrees=10),
        ]
    elif augmentation == "randaugment":
        operations = [
            transforms.RandomResizedCrop(
                image_size, scale=(0.35, 1.0), interpolation=interpolation
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(
                num_ops=2, magnitude=int(data_config.get("randaugment_magnitude", 9))
            ),
        ]
    else:
        raise ValueError(f"Unknown augmentation profile: {augmentation}")
    operations.extend(_normalize())
    if float(data_config.get("random_erasing", 0.0)) > 0:
        operations.append(
            transforms.RandomErasing(p=float(data_config["random_erasing"]))
        )
    return transforms.Compose(operations)


def make_loader(
    data: ImageData,
    indices: list[int],
    transform,
    data_config: dict,
    shuffle: bool,
    seed: int = 0,
    drop_last: bool = False,
    batch_size: int | None = None,
) -> DataLoader:
    num_workers = int(data_config.get("num_workers", 6))
    return DataLoader(
        CachedImageDataset(data, indices, transform),
        batch_size=int(batch_size or data_config["batch_size"]),
        shuffle=shuffle,
        drop_last=drop_last,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
    )


def make_labeled_loader(
    data_config: dict,
    split: str,
) -> tuple[DataLoader, list[str]]:
    data = load_image_folder(
        Path(data_config["root"]) / split,
        cache_size=int(data_config.get("cache_size", 256)),
    )
    loader = make_loader(
        data,
        list(range(len(data.images))),
        build_eval_transform(data_config),
        data_config,
        shuffle=False,
    )
    return loader, data.class_names
