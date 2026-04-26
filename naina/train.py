import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torch_geometric.nn import SAGEConv
from sklearn.metrics import f1_score, roc_auc_score

# Load the data (the file Naina prepared for you)
data = torch.load("outputs/pyg_data.pt", weights_only=False)

# Define the GraphSAGE model
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, 1)  # 1 logit for binary

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        return self.conv2(x, edge_index).squeeze(-1)

# Initialize model, optimizer, and loss
model = GraphSAGE(in_dim=data.num_features)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# IMPORTANT: Use pos_weight from eda_report.md to handle class imbalance [citation:7][citation:3]
# The report says pos_weight = num_licit / num_illicit = 42019 / 4545 ≈ 9.25
pos_weight = torch.tensor([9.25])  # Naina's EDA report tells you this exact number
loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight)

# Training loop
def train():
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)
    loss = loss_fn(logits[data.train_mask], data.y[data.train_mask].float())
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluation function
@torch.no_grad()
def evaluate(mask):
    model.eval()
    logits = model(data.x, data.edge_index)
    preds = (torch.sigmoid(logits[mask]) > 0.90).float()
    y_true = data.y[mask].cpu().numpy()
    y_pred = preds.cpu().numpy()
    return {
        'f1': f1_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, torch.sigmoid(logits[mask]).cpu().numpy())
    }

# Train for 100 epochs
print("Starting training...")
for epoch in range(100):
    loss = train()
    if epoch % 10 == 0:
        val_metrics = evaluate(data.val_mask)
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_metrics['auc']:.4f}")

# Final test evaluation
test_metrics = evaluate(data.test_mask)
print(f"\nTest Results - F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['auc']:.4f}")

# Save the model for Anjali and Khushi
torch.save(model.state_dict(), "model.pt")
print("Model saved as model.pt")