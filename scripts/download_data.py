#!/usr/bin/env python3
"""Download the public assignment data through a node-local staging directory."""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gdown
from PIL import Image


FOLDER_ID = "1NWC3TMsXSWN2TeoYMCjhf2N1b-WRDh-M"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=Path("data"))
    parser.add_argument(
        "--staging", type=Path, default=Path("/tmp/jzhang84_assignment1_data")
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--folder-id", default=FOLDER_ID)
    return parser.parse_args()


def image_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_layout(root: Path, verify_images: bool = True) -> dict:
    train_root = root / "train"
    test_root = root / "test"
    test2_root = root / "test2"
    train_classes = sorted(path.name for path in train_root.iterdir() if path.is_dir())
    test_classes = sorted(path.name for path in test_root.iterdir() if path.is_dir())
    if train_classes != test_classes or len(train_classes) != 16:
        raise RuntimeError(
            f"Expected the same 16 classes in train and test; "
            f"got train={train_classes}, test={test_classes}"
        )

    counts = {
        "train": len(image_paths(train_root)),
        "test": len(image_paths(test_root)),
        "test2": len(image_paths(test2_root)),
    }
    expected = {"train": 2400, "test": 400, "test2": 400}
    if counts != expected:
        raise RuntimeError(f"Image counts do not match: expected {expected}, got {counts}")

    class_counts = Counter(
        ("train", path.parent.name) for path in image_paths(train_root)
    )
    class_counts.update(("test", path.parent.name) for path in image_paths(test_root))
    if any(class_counts[("train", name)] != 150 for name in train_classes):
        raise RuntimeError(f"Training classes are not balanced at 150 images: {class_counts}")
    if any(class_counts[("test", name)] != 25 for name in test_classes):
        raise RuntimeError(f"Test classes are not balanced at 25 images: {class_counts}")

    if verify_images:
        for path in image_paths(root):
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as exc:
                raise RuntimeError(f"Unreadable image {path}: {exc}") from exc
    return {"counts": counts, "classes": train_classes}


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.workers > 32:
        raise ValueError("--workers must be between 1 and 32")
    args.staging.mkdir(parents=True, exist_ok=True)

    manifest = gdown.download_folder(
        id=args.folder_id,
        output=str(args.staging),
        quiet=True,
        use_cookies=False,
        skip_download=True,
    )
    discovered_entries = len(manifest)
    manifest = [
        item
        for item in manifest
        if Path(item.path).suffix.lower() in IMAGE_EXTENSIONS
    ]
    print(
        f"Discovered {discovered_entries} Drive entries; "
        f"downloading {len(manifest)} image files"
    )

    def fetch(item):
        destination = Path(item.local_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and destination.stat().st_size > 0:
            return "reused"
        result = gdown.download(
            id=item.id,
            output=str(destination),
            quiet=True,
            use_cookies=False,
            resume=True,
            retries=4,
        )
        if result is None or not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError(f"Download failed: {item.path}")
        return "downloaded"

    completed = 0
    reused = 0
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch, item): item for item in manifest}
        for future in as_completed(futures):
            item = futures[future]
            try:
                reused += future.result() == "reused"
            except Exception as exc:
                failures.append(f"{item.path}: {exc}")
            completed += 1
            if completed % 100 == 0 or completed == len(manifest):
                print(
                    f"Progress {completed}/{len(manifest)}; "
                    f"reused={reused}; failures={len(failures)}",
                    flush=True,
                )
    if failures:
        print("The following files failed; rerun the same command to resume:")
        print("\n".join(failures))
        raise SystemExit(1)

    summary = validate_layout(args.staging, verify_images=True)
    print(f"Staging validation passed: {summary}")
    args.destination.mkdir(parents=True, exist_ok=True)
    for child in args.staging.iterdir():
        destination = args.destination / child.name
        if child.is_dir():
            shutil.copytree(child, destination, dirs_exist_ok=True)
        elif child.is_file():
            shutil.copy2(child, destination)
    final_summary = validate_layout(args.destination, verify_images=True)
    print(f"Final data validation passed: {final_summary}")
    print(f"Dataset ready at {args.destination.resolve()}")


if __name__ == "__main__":
    main()
