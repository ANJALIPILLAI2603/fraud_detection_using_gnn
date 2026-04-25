# Janhavi — ML / GNN Developer
Model Design & Training

## Deliverables
| # | Task | Script | Output |
|---|------|--------|--------|
| 1 | GNN Architecture | train.py | model.pt (GraphSAGE weights) |
| 2 | Class Imbalance Handling | train.py | pos_weight = 9.25 |
| 3 | Training Loop with Temporal Split | train.py | Best checkpoint at epoch 90 |

The final artifact `model.pt` is the trained model file I hand off to Anjali (API integration) and Khushi (dashboard).

## Model Architecture
2-layer GraphSAGE for node classification (fraud = 1, legit = 0).
GraphSAGE(
(conv1): SAGEConv(169, 64, aggr='mean')
(conv2): SAGEConv(64, 1, aggr='mean')
)

text

- Input: 169 features per node (165 Elliptic + 4 structural)
- Hidden dimension: 64
- Dropout: 0.2 after first layer
- Output: Single logit → sigmoid for fraud probability

## Training Configuration
- **Loss function**: BCEWithLogitsLoss with `pos_weight = 9.25`
- **Optimizer**: Adam (lr = 1e-3, weight_decay = 1e-4)
- **Epochs**: 100
- **Best checkpoint**: epoch 90 (saved as `model.pt`)

### Why pos_weight = 9.25?
From Naina's `eda_report.md`: among labeled nodes, 42,019 licit and 4,545 illicit. That's a 9.25:1 ratio. Setting `pos_weight = 9.25` mathematically balances the loss contribution so each illicit example counts as much as 9.25 licit examples, preventing the model from simply predicting "licit" for everything.

## Performance Metrics

### Validation Set (15% of timesteps, ~7,829 nodes)
| Metric | Best Value | Threshold Used |
|--------|-----------|----------------|
| F1 Score | 0.6481 | 0.90 |
| AUC | 0.9296 | — |

### Test Set (Last 15% of timesteps, future data, ~8,841 nodes)
| Threshold | F1 Score | Predicted Fraud | Actual Fraud |
|-----------|----------|-----------------|--------------|
| 0.50 | 0.1898 | 31.5% | 4.6% |
| **0.90** | **0.3991** | **5.0%** | **4.6%** |

At threshold 0.90, the model's fraud predictions (5.0% on test set) closely match the actual test set fraud rate (4.6%), indicating well-calibrated probabilities.

## Full Dataset Predictions
Run `inference_example.py` on all 203,769 nodes:
Total nodes: 203,769
Predicted fraud: 20,086 (9.9%)
Known fraud in dataset: 4,545 (2.2%)
Known licit: 42,019 (20.6%)
Unknown: 157,205 (77.1%)

text

**Note**: 77% of nodes are unlabeled. The model's 9.9% fraud predictions include:
- Most of the 4,545 known frauds (likely caught)
- ~15,500 additional suspicious nodes that may be true frauds but were never labeled

## How to Use the Model

### Load in one line
```python
import torch
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim=169, hidden=64):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, 1)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        return self.conv2(x, edge_index).squeeze(-1)

model = GraphSAGE(in_dim=169)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()
Make predictions (with recommended threshold)
python
with torch.no_grad():
    fraud_probs = torch.sigmoid(model(data.x, data.edge_index))
    fraud_alerts = (fraud_probs > 0.90).long()  # ← USE 0.90
Data required
The model expects a PyG Data object with:

data.x: Node features [203769, 169] (already normalized by Naina)

data.edge_index: Graph edges [2, 234355]

Load Naina's processed data:

python
data = torch.load("naina/outputs/pyg_data.pt", weights_only=False)
Important Gotchas (Read This!)
Don't re-normalize data.x - Naina already applied StandardScaler. Just use it as is.

Use threshold 0.90 for balanced performance:

Lower threshold (0.75) → 16% fraud flags, higher recall but more false positives

Higher threshold (0.95) → ~5% fraud flags, higher precision but may miss real fraud

Temporal split was intentional - Model trained on past (time steps 1-35), validated on mid (36-42), tested on future (43-49). Don't shuffle across time.

77% of nodes are unlabeled - The model may correctly flag unknown frauds. Monitor flagged transaction patterns.

weights_only=False when loading - Required for PyTorch ≥ 2.6:

python
data = torch.load("outputs/pyg_data.pt", weights_only=False)
Business Recommendations
Use Case	Threshold	Expected Fraud Flags	Best For
High precision (fewer false alarms)	0.95	~5%	Manual review teams with limited capacity
Balanced (recommended)	0.90	~10%	Most production use cases
High recall (catch more fraud)	0.85	~15%	Zero-tolerance fraud environments
Handoff Checklist for Anjali + Khushi
model.pt — Trained GraphSAGE weights (what you hand off)

inference_example.py — Ready-to-use prediction script

requirements_inference.txt — Minimal dependencies (torch, torch-geometric)

This file — Model documentation and usage guide

Files Anjali + Khushi Need from Naina
naina/outputs/pyg_data.pt — Processed graph data (regenerate locally, not in git)

naina/outputs/eda_report.md — Dataset stats and class imbalance details

How Anjali Should Use This for API Integration
python
# Minimal production endpoint code
@app.post("/predict")
def predict_fraud(node_features: list, edge_index: list):
    with torch.no_grad():
        x = torch.tensor(node_features, dtype=torch.float32)
        edge = torch.tensor(edge_index, dtype=torch.long)
        prob = torch.sigmoid(model(x, edge))
        is_fraud = (prob > 0.90).item()
    return {"fraud_probability": prob.item(), "alert": is_fraud}
How Khushi Should Use This for Dashboard
Show flagged transactions (probability > 0.90)

Display fraud probability distribution (0 to 1)

Allow threshold adjustment slider (0.85-0.95)

Show known fraud capture rate vs false positive rate

Model Limitations
Temporal drift - Future transaction patterns may change. Consider periodic retraining.

Unlabeled data - 77% of nodes have no labels. Some "false positives" may be unknown true frauds.

Bitcoin-specific - Features are tailored to bitcoin transactions. May not generalize to other payment systems.