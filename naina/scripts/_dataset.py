"""Shared Elliptic-Bitcoin dataset loader used by the EDA, graph, and feature scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FEATURES_CSV = DATA_DIR / "elliptic_txs_features.csv"
EDGES_CSV = DATA_DIR / "elliptic_txs_edgelist.csv"
CLASSES_CSV = DATA_DIR / "elliptic_txs_classes.csv"

NUM_LOCAL_FEATURES = 94
NUM_AGG_FEATURES = 71
NUM_FEATURES = NUM_LOCAL_FEATURES + NUM_AGG_FEATURES


@dataclass
class EllipticRaw:
    features: pd.DataFrame
    edges: pd.DataFrame
    classes: pd.DataFrame
    is_synthetic: bool

    @property
    def feature_columns(self) -> list[str]:
        return [c for c in self.features.columns if c.startswith("feat_")]


def real_dataset_available() -> bool:
    return FEATURES_CSV.exists() and EDGES_CSV.exists() and CLASSES_CSV.exists()


def load_real() -> EllipticRaw:
    feat_cols = ["txId", "time_step"] + [f"feat_{i}" for i in range(NUM_FEATURES)]

    # ---------------- LOAD FEATURES SAFELY ----------------
    features = pd.read_csv(
        FEATURES_CSV,
        header=None,
        names=feat_cols,
        sep=",",
        dtype=str
    )

    # Convert IDs safely
    features["txId"] = pd.to_numeric(features["txId"], errors="coerce")
    features["time_step"] = pd.to_numeric(features["time_step"], errors="coerce")

    # Convert feature columns safely
    for col in features.columns[2:]:
        features[col] = pd.to_numeric(features[col], errors="coerce")

    # Clean bad values
    features = features.dropna(subset=["txId"])
    features = features.fillna(0.0)

    # Final type casting
    features["txId"] = features["txId"].astype("int64")
    features["time_step"] = features["time_step"].astype("int32")
    features.iloc[:, 2:] = features.iloc[:, 2:].astype("float32")

    # ---------------- LOAD EDGES + CLASSES ----------------
    edges = pd.read_csv(EDGES_CSV)
    classes = pd.read_csv(CLASSES_CSV)

    # Keep only valid nodes (VERY IMPORTANT)
    valid_ids = set(features["txId"])

    edges = edges[
        edges["txId1"].isin(valid_ids) &
        edges["txId2"].isin(valid_ids)
    ]

    classes = classes[classes["txId"].isin(valid_ids)]

    # Label mapping
    label_map = {"1": 1, "2": 0, "unknown": -1}
    classes["label"] = classes["class"].astype(str).map(label_map).astype(int)

    return EllipticRaw(
        features=features,
        edges=edges,
        classes=classes,
        is_synthetic=False
    )


def load_synthetic(
    n_nodes: int = 2000,
    n_time_steps: int = 10,
    illicit_frac: float = 0.02,
    unknown_frac: float = 0.55,
    avg_degree: float = 2.5,
    seed: int = 7,
) -> EllipticRaw:

    rng = np.random.default_rng(seed)

    tx_ids = np.arange(1, n_nodes + 1)
    time_steps = rng.integers(1, n_time_steps + 1, size=n_nodes)
    X = rng.normal(size=(n_nodes, NUM_FEATURES)).astype(np.float32)

    # Slight signal for fraud
    n_illicit = max(2, int(n_nodes * illicit_frac))
    n_unknown = int(n_nodes * unknown_frac)

    labels = np.zeros(n_nodes, dtype=int)
    illicit_idx = rng.choice(n_nodes, size=n_illicit, replace=False)
    labels[illicit_idx] = 1

    remaining = np.setdiff1d(np.arange(n_nodes), illicit_idx)
    unknown_idx = rng.choice(remaining, size=n_unknown, replace=False)

    X[illicit_idx, 0] += 2.5
    X[illicit_idx, 1] -= 1.8

    feat_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(NUM_FEATURES)])
    feat_df.insert(0, "time_step", time_steps)
    feat_df.insert(0, "txId", tx_ids)

    # Build edges
    n_edges = int(n_nodes * avg_degree)
    src = rng.integers(1, n_nodes + 1, size=n_edges)
    dst = rng.integers(1, n_nodes + 1, size=n_edges)
    mask = src != dst

    edges = pd.DataFrame({
        "txId1": src[mask],
        "txId2": dst[mask]
    }).drop_duplicates()

    # Classes
    class_col = np.where(labels == 1, "1", "2").astype(object)
    class_col[unknown_idx] = "unknown"

    classes = pd.DataFrame({
        "txId": tx_ids,
        "class": class_col
    })

    label_map = {"1": 1, "2": 0, "unknown": -1}
    classes["label"] = classes["class"].astype(str).map(label_map).astype(int)

    return EllipticRaw(
        features=feat_df,
        edges=edges,
        classes=classes,
        is_synthetic=True
    )


def load(prefer_synthetic: bool = False) -> EllipticRaw:
    if prefer_synthetic or not real_dataset_available():
        return load_synthetic()
    return load_real()