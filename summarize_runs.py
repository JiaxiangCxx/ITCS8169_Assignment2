#!/usr/bin/env python3
"""Collect every runs/<run>/run_summary.json into one CSV and print a compact table."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output", default="runs/experiment_summary.csv")
    return parser.parse_args()


def percent(value) -> str:
    return "-" if value is None else f"{100 * value:.2f}"


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(Path(args.runs_root).glob("*/run_summary.json")):
        with path.open("r", encoding="utf-8") as stream:
            rows.append(json.load(stream))
    if not rows:
        print(f"No run summaries under {args.runs_root}")
        return

    fieldnames: list[str] = []
    for row in rows:
        fieldnames.extend(key for key in row if key not in fieldnames)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Pretraining runs (validation accuracy of frozen features, %)")
    print(f"{'run':44s} {'epochs':>6s} {'kNN':>7s} {'linear':>7s}")
    for row in rows:
        if row.get("stage") == "pretrain":
            print(
                f"{row['experiment']:44s} {row['epochs_completed']:6d} "
                f"{percent(row['knn_val_accuracy']):>7s} {percent(row['linear_probe_val_accuracy']):>7s}"
            )
    print()
    print("Classification runs (%)")
    print(f"{'run':44s} {'val':>7s} {'test':>7s} {'gap':>7s} {'best ep':>7s}")
    for row in rows:
        if row.get("stage") == "classify":
            print(
                f"{row['experiment']:44s} {percent(row['best_val_accuracy']):>7s} "
                f"{percent(row['test_accuracy']):>7s} {percent(row['generalization_gap']):>7s} "
                f"{row['best_epoch']:7d}"
            )
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
