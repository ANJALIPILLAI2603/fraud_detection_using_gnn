"""Fraud Detection Inference - Production Ready"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim=169, hidden=64):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, 1)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        return self.conv2(x, edge_index).squeeze(-1)

# Load
model = GraphSAGE(in_dim=169)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()
data = torch.load("outputs/pyg_data.pt", weights_only=False)

# Predict with OPTIMAL THRESHOLD
THRESHOLD = 0.90

with torch.no_grad():
    fraud_probs = torch.sigmoid(model(data.x, data.edge_index))
    fraud_predictions = (fraud_probs > THRESHOLD).long()

# Results
total = len(fraud_predictions)
fraud_count = (fraud_predictions == 1).sum().item()
known_fraud = (data.y == 1).sum().item()
known_licit = (data.y == 0).sum().item()

print("="*50)
print("FRAUD DETECTION RESULTS")
print("="*50)
print(f"Total nodes: {total:,}")
print(f"Predicted fraud: {fraud_count:,} ({100*fraud_count/total:.1f}%)")
print(f"Known fraud in dataset: {known_fraud:,} ({100*known_fraud/total:.1f}%)")
print(f"Known licit: {known_licit:,} ({100*known_licit/total:.1f}%)")
print(f"Unknown: {total - known_fraud - known_licit:,} ({100*(total - known_fraud - known_licit)/total:.1f}%)")
print(f"\n✅ Threshold used: {THRESHOLD}")
print("💡 Tip: Adjust threshold between 0.85-0.95 based on business needs")