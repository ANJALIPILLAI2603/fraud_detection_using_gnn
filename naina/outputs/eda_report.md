# EDA Report — Elliptic Bitcoin Dataset

**Data source:** real Elliptic CSVs

## 1. Size
- Nodes (transactions): **203,769**
- Edges (bitcoin flow): **234,355**
- Feature columns per node: **165** (94 local + 71 aggregated, per the Kaggle release)
- Time steps: **49** (each ≈ 2 weeks in the real dataset)

## 2. Class distribution
- illicit (class=1): **4,545**
- licit  (class=2): **42,019**
- unknown         : **157,205**

Among labeled nodes only:
- licit: **42,019**
- illicit: **4,545** (9.76%)

> Severe imbalance → Janhavi should set `pos_weight ≈ 9.25` in BCEWithLogitsLoss.

## 3. Missing values
- Missing cells across entire features table: **0**

## 4. Plots written to `outputs/`
- `eda_class_distribution.png`
- `eda_labeled_imbalance.png`
- `eda_time_distribution.png`
- `eda_missingness.png`
- `eda_feature_snapshot.png`

## 5. Key takeaways
1. Illicit nodes are 9.8% of labeled data — weighted loss is essential.
2. 77.1% of nodes are unlabeled → train/val/test masks must skip them.
3. Features are already anonymised + numerical; no categorical encoding needed.
4. Time-step is a natural split axis (temporal holdout mimics real deployment).