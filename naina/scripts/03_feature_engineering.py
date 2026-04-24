"""Task 3 — Node Feature Engineering.

Takes the graph built in step 2 and assembles a clean feature matrix X:
- 166 anonymised Elliptic features (kept as-is, they are already numeric)
- 4 graph-structural features we compute ourselves:
    * in_degree
    * out_degree
    * degree_centrality
    * time_delta (time_step minus the earliest time_step a node's neighbours appear in)
- StandardScaler applied column-wise.

Finally we emit a PyTorch Geometric `Data` object with:
    data.x          [N, F]    float32
    data.edge_index [2, E]    long
    data.y          [N]       long   (0 / 1 / -1 unknown)
    data.train_mask, val_mask, test_mask  — temporal split on labeled nodes

Run:
    python scripts/03_feature_engineering.py
    python scripts/03_feature_engineering.py --synthetic
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from _dataset import load

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Temporal split points — matches the convention Elliptic papers use.
TRAIN_MAX_TIME_FRAC = 0.70
VAL_MAX_TIME_FRAC = 0.85


def load_or_build_graph(raw) -> nx.DiGraph:
    """Re-use the pickled graph from step 2 if present, otherwise build it now."""
    pkl = OUT_DIR / "graph.gpickle"
    if pkl.exists():
        with pkl.open("rb") as f:
            return pickle.load(f)

    # Local import keeps this script usable even if step 2 wasn't run.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_graph_module", Path(__file__).with_name("02_graph_construction.py")
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.build_graph(raw)


def compute_structural_features(G: nx.DiGraph) -> pd.DataFrame:
    """Per-node graph features. Uses the degree-based approximation for centrality
    (full betweenness would be prohibitive on 200k nodes)."""
    nodes = list(G.nodes)
    n = len(nodes)
    in_deg = np.array([G.in_degree(u) for u in nodes], dtype=np.float32)
    out_deg = np.array([G.out_degree(u) for u in nodes], dtype=np.float32)
    total_deg = in_deg + out_deg
    # degree_centrality = deg / (N-1) on the undirected-ish version
    deg_centrality = total_deg / max(n - 1, 1)

    # time_delta = this node's time_step minus the earliest time_step among its neighbours.
    # Proxy for "account age" — how long after first-seen activity this node appeared.
    time_by_node = {u: G.nodes[u]["time_step"] for u in nodes}
    time_delta = np.zeros(n, dtype=np.float32)
    for i, u in enumerate(nodes):
        neigh_times = [time_by_node[v] for v in G.predecessors(u)] + \
                      [time_by_node[v] for v in G.successors(u)]
        neigh_times = [t for t in neigh_times if t >= 0]
        if neigh_times:
            time_delta[i] = time_by_node[u] - min(neigh_times)

    return pd.DataFrame(
        {
            "txId": nodes,
            "in_degree": in_deg,
            "out_degree": out_deg,
            "degree_centrality": deg_centrality,
            "time_delta": time_delta,
        }
    )


def assemble_feature_matrix(raw, structural: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
    """Merge Elliptic features + structural features. Returns X (scaled) and node ordering."""
    feat_cols = raw.feature_columns
    df = raw.features[["txId", *feat_cols]].merge(structural, on="txId", how="left")
    df = df.fillna(0.0)
    df = df.sort_values("txId").reset_index(drop=True)

    X_cols = feat_cols + ["in_degree", "out_degree", "degree_centrality", "time_delta"]
    X = df[X_cols].to_numpy(dtype=np.float32)
    X = StandardScaler().fit_transform(X).astype(np.float32)
    return X, df["txId"].tolist()


def build_edge_index(G: nx.DiGraph, node_order: list[int]) -> torch.Tensor:
    idx = {tx: i for i, tx in enumerate(node_order)}
    src = []
    dst = []
    for u, v in G.edges:
        if u in idx and v in idx:
            src.append(idx[u])
            dst.append(idx[v])
    if not src:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def temporal_split(
    time_steps: np.ndarray, labels: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Temporal split on *labeled* nodes only. Unknown-label nodes go into none of the masks."""
    n = len(labels)
    labeled_mask = labels != -1

    unique_times = np.sort(np.unique(time_steps[labeled_mask]))
    if len(unique_times) == 0:
        z = torch.zeros(n, dtype=torch.bool)
        return z, z, z
    train_cut = unique_times[int(len(unique_times) * TRAIN_MAX_TIME_FRAC) - 1]
    val_cut = unique_times[int(len(unique_times) * VAL_MAX_TIME_FRAC) - 1]

    train_mask = labeled_mask & (time_steps <= train_cut)
    val_mask = labeled_mask & (time_steps > train_cut) & (time_steps <= val_cut)
    test_mask = labeled_mask & (time_steps > val_cut)
    return (
        torch.tensor(train_mask, dtype=torch.bool),
        torch.tensor(val_mask, dtype=torch.bool),
        torch.tensor(test_mask, dtype=torch.bool),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    raw = load(prefer_synthetic=args.synthetic)
    G = load_or_build_graph(raw)
    print(f"Graph: {G.number_of_nodes():,} nodes · {G.number_of_edges():,} edges")

    structural = compute_structural_features(G)
    X, node_order = assemble_feature_matrix(raw, structural)
    print(f"Feature matrix X: shape={X.shape}, dtype={X.dtype}")

    edge_index = build_edge_index(G, node_order)
    print(f"edge_index: shape={tuple(edge_index.shape)}")

    # Align labels + time steps to node_order.
    label_by_tx = dict(zip(raw.classes["txId"], raw.classes["label"]))
    time_by_tx = dict(zip(raw.features["txId"], raw.features["time_step"]))
    y = np.array([label_by_tx.get(tx, -1) for tx in node_order], dtype=np.int64)
    ts = np.array([time_by_tx.get(tx, -1) for tx in node_order], dtype=np.int64)

    train_mask, val_mask, test_mask = temporal_split(ts, y)
    print(
        f"Split sizes — train: {int(train_mask.sum()):,}, "
        f"val: {int(val_mask.sum()):,}, test: {int(test_mask.sum()):,}"
    )

    data = Data(
        x=torch.from_numpy(X),
        edge_index=edge_index,
        y=torch.from_numpy(y),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    # Attach a couple of convenience attributes Janhavi will want.
    data.time_step = torch.from_numpy(ts)
    data.num_classes = 2

    out = OUT_DIR / "pyg_data.pt"
    torch.save(data, out)
    print(f"Saved PyG Data to {out}")
    print(data)


if __name__ == "__main__":
    main()
