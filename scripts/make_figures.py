#!/usr/bin/env python3
"""Draw the report figure from saved run histories and summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Colour encodes the method and is the same in every panel.
SUPERVISED = "#eb6834"
SIMCLR = "#2a78d6"
SIMCLR_VARIANT = "#8fb8ea"
SIMCLR_STRONG = "#0d3f80"
ROTATION = "#1baf7a"
LINEAR = "#7a5cc7"
REFERENCE = "#898781"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("report/figures/analysis.pdf"))
    return parser.parse_args()


def read_history(path: Path) -> dict[str, list[float]] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    columns: dict[str, list[float]] = {}
    for key in rows[0]:
        columns[key] = [float(row[key]) if row[key] not in ("", "None") else None for row in rows]
    return columns


def read_summary(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def style_axis(axis, title: str, xlabel: str, ylim: tuple[float, float]) -> None:
    axis.set_title(title, loc="left", fontsize=8.5, color=INK, pad=4)
    axis.set_xlabel(xlabel, fontsize=8, color=INK_SECONDARY, labelpad=1)
    axis.set_ylabel("Accuracy (%)", fontsize=8, color=INK_SECONDARY, labelpad=2)
    axis.set_ylim(*ylim)
    axis.grid(axis="y", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(AXIS)
        axis.spines[side].set_linewidth(0.6)
    axis.tick_params(colors=INK_SECONDARY, labelsize=7.5, length=2, width=0.6)


def plot_monitor(axis, runs_root: Path) -> None:
    """k-NN validation accuracy of the frozen backbone during SSL pretraining."""
    style_axis(axis, "(a) k-NN acc. of SSL features", "SSL epoch", (20, 90))
    handles = []
    for run, color, label, style in (
        ("ssl_simclr_crop008_r18_e500", SIMCLR_STRONG, "SimCLR, crop 0.08", "-"),
        ("ssl_simclr_r18_e500", SIMCLR, "SimCLR, crop 0.2", "-"),
        ("ssl_simclr_r18_e1000", SIMCLR, "crop 0.2, 1000 ep.", (0, (1, 1.2))),
        ("ssl_simclr_nocolor_r18_e500", SIMCLR_VARIANT, "crop 0.2, no color aug.", "-"),
        ("ssl_rotation_r18_e150", ROTATION, "RotNet", "-"),
    ):
        summary = read_summary(runs_root / run / "run_summary.json")
        history = read_history(runs_root / run / "history.csv")
        if summary is None or history is None:
            continue
        points = [
            (epoch, 100 * value)
            for epoch, value in zip(history["epoch"], history["knn_val_accuracy"])
            if value is not None
        ]
        axis.plot(*zip(*points), color=color, linewidth=1.2, linestyle=style)
        handles.append(Line2D([], [], color=color, linewidth=1.4, linestyle=style, label=label))
    axis.legend(handles=handles, loc="lower right", fontsize=6, frameon=False, handlelength=1.8)


def plot_overfitting(axis, runs_root: Path) -> None:
    """Training vs validation accuracy: random initialization vs SimCLR initialization."""
    style_axis(axis, "(b) Train (dashed) vs. val. acc.", "Epoch", (0, 104))
    for run, color in (("sup_baseline", SUPERVISED), ("ft_ssl_simclr_r18_e500", SIMCLR)):
        history = read_history(runs_root / run / "history.csv")
        if history is None:
            continue
        epochs = history["epoch"]
        axis.plot(epochs, [100 * v for v in history["val_accuracy"]], color=color, linewidth=1.3)
        axis.plot(epochs, [100 * v for v in history["train_accuracy"]], color=color,
                  linewidth=0.9, linestyle=(0, (3, 2)))
    axis.legend(
        handles=[
            Line2D([], [], color=SUPERVISED, linewidth=1.4, label="random init"),
            Line2D([], [], color=SIMCLR, linewidth=1.4, label="SimCLR init"),
        ],
        loc="lower right", fontsize=6.5, frameon=False,
    )


def plot_label_efficiency(axis, runs_root: Path) -> None:
    """Test accuracy as a function of the number of labeled training images per class."""
    style_axis(axis, "(c) Label efficiency (test)", "Labeled images per class", (20, 95))
    arms = (
        ("sup_baseline", SUPERVISED, "supervised, random init"),
        ("ft_ssl_simclr_r18_e500", SIMCLR, "SimCLR + fine-tune"),
        ("linear_ssl_simclr_r18_e500", LINEAR, "SimCLR + linear"),
    )
    for base, color, label in arms:
        points = []
        for suffix in ("_labels10", "_labels25", "_labels50", ""):
            summary = read_summary(runs_root / f"{base}{suffix}" / "run_summary.json")
            if summary is not None and summary.get("test_accuracy") is not None:
                points.append((summary["labeled_train_images"] / 16, 100 * summary["test_accuracy"]))
        if points:
            points.sort()
            axis.plot(*zip(*points), color=color, linewidth=1.4, marker="o", markersize=3, label=label)
    axis.set_xscale("log")
    axis.set_xticks([12, 30, 60, 120], labels=["12", "30", "60", "120"])
    axis.minorticks_off()
    axis.legend(loc="lower right", fontsize=6.5, frameon=False)


def main() -> None:
    args = parse_args()
    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    figure, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
    plot_monitor(axes[0], args.runs_root)
    plot_overfitting(axes[1], args.runs_root)
    plot_label_efficiency(axes[2], args.runs_root)
    figure.tight_layout(pad=0.3, w_pad=1.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)
    figure.savefig(args.output.with_suffix(".png"), dpi=200)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
