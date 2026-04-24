Fraud Detection using Graph Neural Networks
Mini project · 4 members · equal workload division

PROJECT PIPELINE
📦 Dataset
→
🔧 Preprocessing
→
🕸 Graph Construction
→
🧠 GNN Model
→
📊 Evaluation
→
🖥 Demo / UI
TECH STACK
Python
PyTorch Geometric
NetworkX
scikit-learn
Pandas / NumPy
Matplotlib / Seaborn
Streamlit (demo)
Kaggle IEEE-CIS / Elliptic Dataset
TEAM TASKS
NA


Naina
Data & Graph Engineer
Dataset + Graph Construction
1
Dataset Collection & EDA
Download Elliptic Bitcoin or IEEE-CIS dataset. Analyse class imbalance (fraud vs non-fraud), plot distributions, check missing values.
pandas + seaborn, Kaggle/UCI
2
Graph Construction
Convert transactions into a graph: nodes = accounts/transactions, edges = money flow. Use NetworkX to build and visualise the graph structure.
NetworkX + PyTorch Geometric Data()
3
Node Feature Engineering
Create node features: transaction amount, time delta, degree centrality, account age. Normalise and prepare feature matrix X.
pandas + sklearn StandardScaler
JA


Janhavi
ML / GNN Developer
Model Design & Training
1
GNN Architecture
Build a 2–3 layer GCN or GraphSAGE model in PyTorch Geometric for node classification (fraud = 1, legit = 0).
PyG: GCNConv / SAGEConv + ReLU + Dropout
2
Handle Class Imbalance
Apply weighted loss or SMOTE to handle the heavy fraud/non-fraud imbalance. Use focal loss or pos_weight in BCELoss.
torch: BCEWithLogitsLoss(pos_weight=...)
3
Training Loop
Write train/val/test split, training loop with Adam optimizer, learning rate scheduler. Save best model checkpoint.
torch + PyG DataLoader
AN


Anjali
Evaluation & Analysis
Metrics + Comparison
1
Evaluation Metrics
Compute precision, recall, F1, AUC-ROC, confusion matrix. Focus on recall — catching fraud matters more than false alarms.
sklearn.metrics + matplotlib ROC curve
2
Baseline Comparison
Train a Random Forest and Logistic Regression on same features (no graph). Compare results to show GNN's advantage.
sklearn: RandomForestClassifier
3
Graph Visualisation
Visualise a subgraph of flagged fraud clusters. Highlight fraud nodes in red vs legit in blue. Show how GNN learns neighbourhood patterns.
NetworkX + matplotlib draw_networkx()
KH


Khushi
Demo & Presentation
UI + Report + Slides
1
Streamlit Demo App
Build a simple web app where user inputs transaction details → app predicts fraud probability using saved model. Show graph neighbourhood of that node.
streamlit run app.py · torch.load(model)
2
Project Report
Write abstract, related work, methodology, results, conclusion. Include GNN architecture diagram, results table, and ROC curve.
Google Docs / LaTeX · 6–8 pages
3
Presentation Slides
10–12 slide deck: problem, why graphs, architecture, results, live demo. Prepare each member's speaking points.
Canva / Google Slides · demo last slide


Here's the full breakdown! Let me walk you through the key decisions:
Why this dataset? The Elliptic Bitcoin dataset is perfect for this — it's a real transaction graph with 203k nodes and labeled fraud/legit nodes. Alternatively, the IEEE-CIS Kaggle dataset works great too. Both are free.

Why GNN specifically? Traditional ML (Random Forest etc.) treats each transaction independently. GNNs look at the neighbourhood — if your neighbours are fraudsters, you're suspicious too. That's the core insight your project should highlight, and Anjali's baseline comparison will demonstrate this beautifully.

Recommended GNN architecture for beginners: 2-layer GraphSAGE → it works well on large graphs without needing the full graph in memory. Use node classification with sigmoid output.
How to split the work in practice:

Naina finishes first (data + graph), passes the Data() object to Janhavi
Janhavi trains the model, saves it as model.pt, passes to Anjali and Khushi
Everyone works in the same GitHub repo with separate folders

Quick tip for Janhavi — the class imbalance is severe (~2% fraud). Using pos_weight = (num_legit / num_fraud) in BCEWithLogitsLoss makes a huge difference. Don't skip this.
Ask me if you want a starter code template for any of the four parts!