"""Task 2 — Graph Construction.

Builds a directed NetworkX graph from the Elliptic transaction edges, visualises a
fraud-centred subgraph, and pickles the graph for the feature-engineering step.

Nodes  = transactions (txId)
Edges  = directed bitcoin flow (txId1 -> txId2)
Labels = {0 licit, 1 illicit, -1 unknown} on each node

Run:
    python scripts/02_graph_construction.py
    python scripts/02_graph_construction.py --synthetic
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from _dataset import load

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_graph(raw) -> nx.DiGraph:
    """Assemble a NetworkX DiGraph with labels and time_step on each node."""
    G = nx.DiGraph()

    # Add every transaction as a node even if it has no edges — we need the full node set.
    label_by_tx = dict(zip(raw.classes["txId"], raw.classes["label"]))
    time_by_tx = dict(zip(raw.features["txId"], raw.features["time_step"]))
    for tx in raw.features["txId"]:
        G.add_node(
            int(tx),
            label=int(label_by_tx.get(tx, -1)),
            time_step=int(time_by_tx.get(tx, -1)),
        )

    edges = raw.edges.itertuples(index=False, name=None)
    G.add_edges_from((int(u), int(v)) for u, v in edges)
    return G


def print_summary(G: nx.DiGraph) -> dict:
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    degrees = np.fromiter((d for _, d in G.degree()), dtype=int)
    n_isolated = int((degrees == 0).sum())
    labels = np.fromiter((G.nodes[n]["label"] for n in G.nodes), dtype=int)

    stats = {
        "nodes": n_nodes,
        "edges": n_edges,
        "isolated_nodes": n_isolated,
        "mean_degree": float(degrees.mean()) if len(degrees) else 0.0,
        "max_degree": int(degrees.max()) if len(degrees) else 0,
        "n_illicit": int((labels == 1).sum()),
        "n_licit": int((labels == 0).sum()),
        "n_unknown": int((labels == -1).sum()),
    }
    print("Graph summary:")
    for k, v in stats.items():
        print(f"  {k:>15}: {v:,}" if isinstance(v, int) else f"  {k:>15}: {v:.3f}")
    return stats


def visualise_fraud_neighbourhood(G: nx.DiGraph, hops: int = 2, max_nodes: int = 150) -> None:
    """Pick a fraud node with highest degree and draw its k-hop neighbourhood."""
    fraud_nodes = [n for n, d in G.nodes(data=True) if d.get("label") == 1]
    if not fraud_nodes:
        print("No illicit nodes found — skipping subgraph visualisation")
        return

    # Use undirected view for neighbourhood expansion so we see both in- and out-flow.
    undirected = G.to_undirected(as_view=True)
    seed = max(fraud_nodes, key=lambda n: undirected.degree(n))

    nodes = {seed}
    frontier = {seed}
    for _ in range(hops):
        next_frontier = set()
        for u in frontier:
            next_frontier.update(undirected.neighbors(u))
        nodes |= next_frontier
        frontier = next_frontier
        if len(nodes) >= max_nodes:
            break

    # Cap size but always keep the seed so its position exists in the layout.
    capped = set(list(nodes - {seed})[: max_nodes - 1]) | {seed}
    sub = G.subgraph(capped).copy()

    colors = []
    for n in sub.nodes:
        lbl = sub.nodes[n]["label"]
        colors.append({1: "#e74c3c", 0: "#2ecc71", -1: "#95a5a6"}.get(lbl, "#95a5a6"))

    pos = nx.spring_layout(sub, seed=42, k=0.5 / np.sqrt(max(len(sub), 1)))
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_edges(sub, pos, ax=ax, alpha=0.3, arrows=True, arrowsize=8)
    nx.draw_networkx_nodes(sub, pos, ax=ax, node_color=colors, node_size=80, linewidths=0.5,
                            edgecolors="black")
    # Highlight the seed
    nx.draw_networkx_nodes(sub, pos, nodelist=[seed], node_color="#e74c3c", node_size=250,
                            edgecolors="black", linewidths=2, ax=ax)
    ax.set_title(f"{hops}-hop neighbourhood around fraud node {seed}\n"
                 f"red = illicit · green = licit · grey = unknown")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "graph_fraud_subgraph.png", dpi=150)
    plt.close(fig)
    print(f"Drew {hops}-hop subgraph around fraud node {seed} ({len(sub):,} nodes)")


def plot_degree_distribution(G: nx.DiGraph) -> None:
    degrees = np.fromiter((d for _, d in G.degree()), dtype=int)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(degrees, bins=50, log=True, color="#3498db", edgecolor="white")
    ax.set_xlabel("node degree")
    ax.set_ylabel("# nodes (log scale)")
    ax.set_title("Degree distribution — transaction graph")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "graph_degree_distribution.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    raw = load(prefer_synthetic=args.synthetic)
    print(f"Loaded {'synthetic' if raw.is_synthetic else 'real'} dataset")

    G = build_graph(raw)
    print_summary(G)
    plot_degree_distribution(G)
    visualise_fraud_neighbourhood(G)

    out_pickle = OUT_DIR / "graph.gpickle"
    with out_pickle.open("wb") as f:
        pickle.dump(G, f)
    print(f"Saved graph to {out_pickle}")


if __name__ == "__main__":
    main()
