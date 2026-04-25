# save as tune_threshold.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import numpy as np
from sklearn.metrics import f1_score

# Define model architecture (must match training)
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim=169, hidden=64):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, 1)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        return self.conv2(x, edge_index).squeeze(-1)

# Load data
print("Loading data...")
data = torch.load("outputs/pyg_data.pt", weights_only=False)

# Load model
print("Loading model...")
model = GraphSAGE(in_dim=169)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

# Get validation probabilities
print("Making predictions...")
with torch.no_grad():
    val_probs = torch.sigmoid(model(data.x, data.edge_index)[data.val_mask])

# Find best threshold
val_labels = data.y[data.val_mask].cpu().numpy()
val_probs_np = val_probs.cpu().numpy()

thresholds = np.arange(0.1, 0.95, 0.05)
print("\nThreshold | Val F1 | Predicted Fraud %")
print("-" * 45)

best_f1 = 0
best_thresh = 0.5

for thresh in thresholds:
    preds = (val_probs_np > thresh).astype(int)
    f1 = f1_score(val_labels, preds)
    fraud_pct = preds.mean() * 100
    print(f"{thresh:.2f}       | {f1:.4f} | {fraud_pct:.1f}%")
    
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print(f"\n✅ Best threshold: {best_thresh:.2f} (F1 = {best_f1:.4f})")

# Also check test set with different thresholds
print("\n" + "="*50)
print("TEST SET RESULTS (using different thresholds)")
print("="*50)

with torch.no_grad():
    all_probs = torch.sigmoid(model(data.x, data.edge_index))
    test_probs = all_probs[data.test_mask]
    test_labels = data.y[data.test_mask].cpu().numpy()
    test_probs_np = test_probs.cpu().numpy()
    
    for thresh in [0.5, 0.65, best_thresh, 0.75, 0.80, 0.85, 0.90]:
        preds = (test_probs_np > thresh).astype(int)
        f1 = f1_score(test_labels, preds)
        fraud_pct = preds.mean() * 100
        actual_fraud_pct = (test_labels == 1).mean() * 100
        print(f"Threshold {thresh:.2f} | F1: {f1:.4f} | Predicted Fraud: {fraud_pct:.1f}% | Actual: {actual_fraud_pct:.1f}%")

print(f"\n💡 Recommendation: Use threshold = {best_thresh:.2f} for best F1 score")