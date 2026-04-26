import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load data
data = torch.load("outputs/pyg_data.pt", weights_only=False)

X = data.x.cpu().numpy()
y = data.y.cpu().numpy()

# filter labeled only
mask = y != -1
X = X[mask]
y = y[mask]

# split using masks
train_mask = data.train_mask.cpu().numpy()[mask]
test_mask = data.test_mask.cpu().numpy()[mask]

X_train = X[train_mask]
y_train = y[train_mask]

X_test = X[test_mask]
y_test = y[test_mask]

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("\nLogistic Regression:")
print(classification_report(y_test, lr_pred))

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\nRandom Forest:")
print(classification_report(y_test, rf_pred))