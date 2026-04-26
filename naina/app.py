import streamlit as st
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("💰 Bitcoin Fraud Detection using GNN")
st.markdown("""
This dashboard analyzes fraud detection using Graph Neural Networks.
It compares GNN with traditional ML models and visualizes transaction networks.
""")

# ---------------- MODEL ----------------
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim=169, hidden=64):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        return self.conv2(x, edge_index).squeeze(-1)

# ---------------- LOAD DATA ----------------
data = torch.load("outputs/pyg_data.pt", weights_only=False)

model = GraphSAGE(in_dim=169)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Controls")
threshold = st.sidebar.slider("Select Threshold", 0.5, 0.95, 0.90)

# =========================================================
# 🔥 1. NEW TRANSACTION PREDICTION (TOP - DEMO FRIENDLY)
# =========================================================
st.subheader("🔍 Predict New Transaction")

st.markdown("Enter transaction characteristics (simplified demo input)")

amount = st.number_input("💰 Transaction Amount (normalized)", value=0.0)
connections = st.number_input("🔗 Number of Connections", value=1.0)
time_feature = st.number_input("⏱️ Time Feature", value=0.0)

st.info("ℹ️ Simplified demo. Actual model uses 169 engineered features + graph context.")

if st.button("Predict Fraud"):

    new_node = data.x.mean(dim=0).cpu().numpy()
    new_node[0] = amount
    new_node[1] = connections
    new_node[2] = time_feature

    new_tensor = torch.tensor(new_node, dtype=torch.float).unsqueeze(0)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    with torch.no_grad():
        prob = torch.sigmoid(model(new_tensor, edge_index)).item()

    st.success(f"Fraud Probability: {prob:.4f}")

    if prob > threshold:
        st.error("⚠️ High Fraud Risk Transaction")
    elif prob > 0.5:
        st.warning("⚠️ Moderate Risk Transaction")
    else:
        st.success("✅ Likely Legitimate Transaction")

st.caption("Model outputs probability. Final decision depends on threshold.")

# =========================================================
# 📊 2. MODEL METRICS
# =========================================================
with torch.no_grad():
    probs = torch.sigmoid(model(data.x, data.edge_index))
    preds = (probs > threshold).long()

y_true = data.y.cpu().numpy()
y_pred = preds.cpu().numpy()
y_prob = probs.cpu().numpy()

mask = y_true != -1
y_true_f = y_true[mask]
y_pred_f = y_pred[mask]
y_prob_f = y_prob[mask]

total = len(y_pred)
fraud_count = (y_pred == 1).sum()
fraud_percent = (fraud_count / total) * 100
auc = roc_auc_score(y_true_f, y_prob_f)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", total)
col2.metric("Fraud Detected", int(fraud_count))
col3.metric("Fraud %", f"{fraud_percent:.2f}%")
col4.metric("AUC Score", f"{auc:.2f}")

# =========================================================
# 📉 3. CONFUSION MATRIX
# =========================================================
st.subheader("📉 Confusion Matrix")

cm = confusion_matrix(y_true_f, y_pred_f)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Licit", "Fraud"],
            yticklabels=["Licit", "Fraud"],
            ax=ax)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)

# =========================================================
# 📊 4. MODEL COMPARISON
# =========================================================
st.subheader("📊 Model Comparison")

comparison_data = {
    "Model": ["Logistic Regression", "Random Forest", "GNN"],
    "Precision (Fraud)": [0.67, 1.00, 0.14],
    "Recall (Fraud)": [0.25, 0.12, 0.07],
    "F1 Score": [0.36, 0.22, 0.10]
}

df = pd.DataFrame(comparison_data)
st.dataframe(df, use_container_width=True)

st.info("⚠️ Fraud detection is highly imbalanced. Recall is prioritized.")