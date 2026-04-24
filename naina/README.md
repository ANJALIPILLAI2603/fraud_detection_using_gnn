# Naina — Data & Graph Engineer

My part of the **Fraud Detection using GNN** mini project.

## Deliverables

| # | Task | Script | Output |
|---|------|--------|--------|
| 1 | Dataset Collection & EDA | `scripts/01_eda.py` | `outputs/eda_report.md` + 5 EDA plots |
| 2 | Graph Construction | `scripts/02_graph_construction.py` | `outputs/graph.gpickle`, `graph_fraud_subgraph.png`, `graph_degree_distribution.png` |
| 3 | Node Feature Engineering | `scripts/03_feature_engineering.py` | `outputs/pyg_data.pt` (PyG `Data` object) |

The final artifact `outputs/pyg_data.pt` is the single file I hand off to **Janhavi** for GNN training.

## Dataset

**Elliptic Bitcoin dataset** — real bitcoin transaction graph.
- 203,769 transaction nodes
- 234,355 directed edges (bitcoin flow)
- 165 anonymised features per node (94 local + 71 aggregated — Kaggle release)
- Labels: 1 = illicit (fraud), 2 = licit, unknown = unlabeled
- Severe class imbalance — illicit is 9.8% of *labeled* nodes, 2.2% of *all* nodes

Download from Kaggle: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
Place the three CSVs inside `naina/data/`:
- `elliptic_txs_features.csv`
- `elliptic_txs_edgelist.csv`
- `elliptic_txs_classes.csv`

## How to run

```bash
cd naina
pip install -r requirements.txt
python scripts/01_eda.py
python scripts/02_graph_construction.py
python scripts/03_feature_engineering.py
```

If you don't have the real dataset yet, every script accepts a `--synthetic` flag
that generates a tiny synthetic Elliptic-shaped dataset so the pipeline still runs
end-to-end for testing.

---

# 📦 Handoff to Janhavi

Everything Janhavi needs is in **`naina/outputs/`**. Two files matter:

1. `pyg_data.pt` — the PyG `Data` object (this is what you train on)
2. `eda_report.md` — dataset stats, class imbalance, recommended `pos_weight`

## 1. `pyg_data.pt` — the PyG `Data` object

Binary PyTorch file. Load it in one line:

```python
import torch
from torch_geometric.data import Data

data: Data = torch.load("naina/outputs/pyg_data.pt", weights_only=False)
print(data)
# Data(x=[203769, 169], edge_index=[2, 234355], y=[203769],
#      train_mask=[203769], val_mask=[203769], test_mask=[203769],
#      time_step=[203769], num_classes=2)
```

### What each attribute means

| Attribute | Shape | Dtype | Meaning |
|---|---|---|---|
| `data.x` | `[203769, 169]` | `float32` | Node feature matrix — 165 Elliptic features + 4 structural features (in_degree, out_degree, degree_centrality, time_delta). Already StandardScaler-normalised — **do not re-normalise**. |
| `data.edge_index` | `[2, 234355]` | `int64` | Directed edges in PyG format. Column `[u, v]` means bitcoin flow from node `u` → node `v`. |
| `data.y` | `[203769]` | `int64` | Label per node: `0` = licit, `1` = illicit, `-1` = unknown (unlabeled). |
| `data.train_mask` | `[203769]` | `bool` | `True` for the 29,894 nodes used for training (earliest 70% of timesteps, labeled only). |
| `data.val_mask` | `[203769]` | `bool` | `True` for the 7,829 validation nodes (next 15% of timesteps). |
| `data.test_mask` | `[203769]` | `bool` | `True` for the 8,841 test nodes (last 15% of timesteps). |
| `data.time_step` | `[203769]` | `int64` | Raw time step 1–49 per node. Useful if you want to redo the split with different cutoffs. |
| `data.num_classes` | — | `int` | `2` (binary classification). |

### Why temporal split (and not random)
Fraud detection in deployment means predicting on *future* transactions using models trained on *past* ones. Random splits leak future info and inflate metrics. The masks already enforce this: train → val → test are ordered by `time_step`.

### How to use it in training

```python
import torch
from torch.nn import BCEWithLogitsLoss
from torch_geometric.nn import SAGEConv

data = torch.load("naina/outputs/pyg_data.pt", weights_only=False)

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, 1)  # 1 logit for binary
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index).squeeze(-1)

model = GraphSAGE(in_dim=data.num_features)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# Recommended: see pos_weight from the EDA report (currently 9.25)
pos_weight = torch.tensor([9.25])
loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight)

for epoch in range(100):
    model.train()
    opt.zero_grad()
    logits = model(data.x, data.edge_index)
    loss = loss_fn(logits[data.train_mask], data.y[data.train_mask].float())
    loss.backward()
    opt.step()
```

### Gotchas

- **Don't train on `y == -1`.** 77% of nodes are unlabeled — the masks already exclude them, but if you ever roll your own indexing, filter `data.y != -1` first.
- **Don't re-normalise `data.x`.** StandardScaler already ran in step 3.
- **`data.y` is `int64`, but BCEWithLogitsLoss wants `float`.** Cast with `.float()` as shown above.
- **`weights_only=False`** is required on PyTorch ≥ 2.6 — PyG `Data` objects aren't in the safe-load allowlist.

## 2. `eda_report.md` — dataset summary

Plain markdown with all the stats you'd want to cite in the project report. Key numbers to look at:

- **`pos_weight ≈ 9.25`** — this is `num_licit / num_illicit` among labeled nodes. Plug it into `BCEWithLogitsLoss(pos_weight=...)` to handle class imbalance.
- **Class counts**: 4,545 illicit · 42,019 licit · 157,205 unknown.
- **165 feature columns** (the Kaggle CSV ships with 165, not 166 as the paper claims).
- **Time steps 1–49**, each ≈ 2 weeks of real bitcoin activity.

## 3. Plots you can reuse in the slide deck / report

All PNGs in `naina/outputs/`:

| File | What it shows | Good for |
|---|---|---|
| `eda_class_distribution.png` | Bar chart: illicit vs licit vs unknown | Intro slide "why this is hard" |
| `eda_labeled_imbalance.png` | Pie chart: fraud vs legit among labeled | Motivating weighted loss |
| `eda_time_distribution.png` | Nodes per time step, split by class | Showing temporal structure |
| `eda_missingness.png` | Missing-value fraction per column | Data-quality slide |
| `eda_feature_snapshot.png` | KDE of first 6 features, class-conditional | "Features are partially separable" |
| `graph_fraud_subgraph.png` | 2-hop neighbourhood around highest-degree fraud node | The "why GNN" slide — shows fraud clusters |
| `graph_degree_distribution.png` | Log-scale degree histogram | Graph-structure slide |

## Handoff checklist for Janhavi

- [ ] `git pull` to get the latest `outputs/`
- [ ] `pip install torch torch-geometric` (if not already)
- [ ] Load with `torch.load("naina/outputs/pyg_data.pt", weights_only=False)`
- [ ] Read `naina/outputs/eda_report.md` for `pos_weight` and class counts
- [ ] Build 2-layer GraphSAGE, use BCEWithLogitsLoss with `pos_weight=9.25`
- [ ] Train on `data.train_mask`, tune on `data.val_mask`, report on `data.test_mask`
- [ ] Save final model as `model.pt` for Anjali + Khushi
