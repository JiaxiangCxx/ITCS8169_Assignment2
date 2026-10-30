#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Assignment 1 data")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    return parser.parse_args()


def list_images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> None:
    args = parse_args()
    result: dict = {"splits": {}}
    bad_images: list[str] = []
    for split in ("train", "test", "test2"):
        root = args.data_root / split
        if not root.is_dir():
            raise FileNotFoundError(root)
        paths = list_images(root)
        classes = Counter(path.parent.name for path in paths) if split != "test2" else {}
        result["splits"][split] = {
            "image_count": len(paths),
            "class_counts": dict(sorted(classes.items())),
        }
        for path in paths:
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as exc:
                bad_images.append(f"{path}: {exc}")

    expected = {"train": 2400, "test": 400, "test2": 400}
    observed = {
        split: information["image_count"]
        for split, information in result["splits"].items()
    }
    if observed != expected:
        raise RuntimeError(f"Expected {expected}, observed {observed}")
    train_classes = result["splits"]["train"]["class_counts"]
    test_classes = result["splits"]["test"]["class_counts"]
    if len(train_classes) != 16 or set(train_classes) != set(test_classes):
        raise RuntimeError("Train/test class directories do not match the expected 16 classes")
    if bad_images:
        raise RuntimeError("Unreadable images:\n" + "\n".join(bad_images))
    print(json.dumps(result, indent=2, sort_keys=True))
    print("Data validation passed")


if __name__ == "__main__":
    main()

