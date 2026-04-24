"""Task 1 — Dataset Collection & EDA.

Loads the Elliptic Bitcoin dataset, inspects class imbalance, missing values,
and temporal / feature distributions, and writes everything to naina/outputs/.

Run:
    python scripts/01_eda.py
    python scripts/01_eda.py --synthetic   # force the synthetic fallback
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from _dataset import load

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")


def _plot_class_distribution(classes: pd.DataFrame) -> None:
    counts = classes["class"].value_counts().rename({"1": "illicit", "2": "licit"})
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=list(counts.index), y=counts.values, hue=list(counts.index),
                 ax=ax, palette="Set2", legend=False)
    ax.set_title("Node class distribution (Elliptic Bitcoin)")
    ax.set_xlabel("class")
    ax.set_ylabel("# nodes")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_class_distribution.png", dpi=150)
    plt.close(fig)


def _plot_labeled_imbalance(classes: pd.DataFrame) -> None:
    labeled = classes[classes["label"] != -1]
    counts = labeled["label"].value_counts().rename({0: "licit (0)", 1: "illicit (1)"})
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=["#2ecc71", "#e74c3c"],
        startangle=90,
        wedgeprops=dict(edgecolor="white"),
    )
    ax.set_title("Fraud vs legit among *labeled* nodes")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_labeled_imbalance.png", dpi=150)
    plt.close(fig)


def _plot_time_distribution(features: pd.DataFrame, classes: pd.DataFrame) -> None:
    merged = features[["txId", "time_step"]].merge(classes[["txId", "label"]], on="txId")
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, color, name in [(0, "#2ecc71", "licit"), (1, "#e74c3c", "illicit")]:
        subset = merged[merged["label"] == label]
        if len(subset):
            sns.histplot(
                subset["time_step"],
                bins=int(merged["time_step"].max()),
                ax=ax,
                color=color,
                label=name,
                alpha=0.6,
                stat="count",
            )
    ax.set_title("Transactions over time — fraud vs legit")
    ax.set_xlabel("time step")
    ax.set_ylabel("# nodes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_time_distribution.png", dpi=150)
    plt.close(fig)


def _plot_missingness(features: pd.DataFrame) -> None:
    miss = features.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(8, 4))
    if miss.empty:
        ax.text(0.5, 0.5, "No missing values in any column", ha="center", va="center")
        ax.set_axis_off()
    else:
        sns.barplot(x=miss.values, y=miss.index, ax=ax, color="#3498db")
        ax.set_xlabel("fraction missing")
    ax.set_title("Missing-value fraction per column")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_missingness.png", dpi=150)
    plt.close(fig)


def _plot_feature_snapshot(features: pd.DataFrame, classes: pd.DataFrame) -> None:
    """Peek at the first 6 local features, split by label, to see separability."""
    merged = features.merge(classes[["txId", "label"]], on="txId")
    merged = merged[merged["label"] != -1]
    cols = [f"feat_{i}" for i in range(6)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, col in zip(axes.flat, cols):
        for label, color, name in [(0, "#2ecc71", "licit"), (1, "#e74c3c", "illicit")]:
            sns.kdeplot(
                merged.loc[merged["label"] == label, col],
                ax=ax,
                color=color,
                label=name,
                fill=True,
                alpha=0.4,
            )
        ax.set_title(col)
        ax.legend(fontsize=8)
    fig.suptitle("First 6 features — class-conditional KDE", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_feature_snapshot.png", dpi=150)
    plt.close(fig)


def _write_report(
    raw,
    counts: pd.Series,
    labeled_counts: pd.Series,
    pos_weight: float,
    n_missing_cells: int,
) -> None:
    total = len(raw.classes)
    report = [
        "# EDA Report — Elliptic Bitcoin Dataset",
        "",
        f"**Data source:** {'synthetic fallback' if raw.is_synthetic else 'real Elliptic CSVs'}",
        "",
        "## 1. Size",
        f"- Nodes (transactions): **{total:,}**",
        f"- Edges (bitcoin flow): **{len(raw.edges):,}**",
        f"- Feature columns per node: **{len(raw.feature_columns)}** "
        f"(94 local + 71 aggregated, per the Kaggle release)",
        f"- Time steps: **{raw.features['time_step'].nunique()}** "
        f"(each ≈ 2 weeks in the real dataset)",
        "",
        "## 2. Class distribution",
        f"- illicit (class=1): **{int(counts.get('1', 0)):,}**",
        f"- licit  (class=2): **{int(counts.get('2', 0)):,}**",
        f"- unknown         : **{int(counts.get('unknown', 0)):,}**",
        "",
        "Among labeled nodes only:",
        f"- licit: **{int(labeled_counts.get(0, 0)):,}**",
        f"- illicit: **{int(labeled_counts.get(1, 0)):,}** "
        f"({labeled_counts.get(1, 0) / max(labeled_counts.sum(), 1):.2%})",
        "",
        "> Severe imbalance → Janhavi should set "
        f"`pos_weight ≈ {pos_weight:.2f}` in BCEWithLogitsLoss.",
        "",
        "## 3. Missing values",
        f"- Missing cells across entire features table: **{n_missing_cells:,}**",
        "",
        "## 4. Plots written to `outputs/`",
        "- `eda_class_distribution.png`",
        "- `eda_labeled_imbalance.png`",
        "- `eda_time_distribution.png`",
        "- `eda_missingness.png`",
        "- `eda_feature_snapshot.png`",
        "",
        "## 5. Key takeaways",
        f"1. Illicit nodes are {labeled_counts.get(1, 0) / max(labeled_counts.sum(), 1):.1%} of labeled data — weighted loss is essential.",
        f"2. {int(counts.get('unknown', 0)) / max(total, 1):.1%} of nodes are unlabeled → train/val/test masks must skip them.",
        "3. Features are already anonymised + numerical; no categorical encoding needed.",
        "4. Time-step is a natural split axis (temporal holdout mimics real deployment).",
    ]
    (OUT_DIR / "eda_report.md").write_text("\n".join(report))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic fallback")
    args = parser.parse_args()

    raw = load(prefer_synthetic=args.synthetic)
    print(f"Loaded {'synthetic' if raw.is_synthetic else 'real'} dataset: "
          f"{len(raw.features):,} nodes · {len(raw.edges):,} edges")

    counts = raw.classes["class"].value_counts()
    labeled = raw.classes[raw.classes["label"] != -1]
    labeled_counts = labeled["label"].value_counts()
    n_licit = int(labeled_counts.get(0, 0))
    n_illicit = max(int(labeled_counts.get(1, 0)), 1)
    pos_weight = n_licit / n_illicit

    n_missing = int(raw.features.isna().sum().sum())

    _plot_class_distribution(raw.classes)
    _plot_labeled_imbalance(raw.classes)
    _plot_time_distribution(raw.features, raw.classes)
    _plot_missingness(raw.features)
    _plot_feature_snapshot(raw.features, raw.classes)
    _write_report(raw, counts, labeled_counts, pos_weight, n_missing)

    print(f"Wrote EDA report + plots to {OUT_DIR}")
    print(f"Recommended pos_weight for BCEWithLogitsLoss: {pos_weight:.2f}")


if __name__ == "__main__":
    main()
