import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# -----------------------------------------------------------------------------
# Load & Filter
# -----------------------------------------------------------------------------
df = (
    pd.read_csv("Data-1.csv")
      .filter(
          [
              "AR", "BRCA1", "BRCA2", "CD82", "CDH1", "CHEK2", "EHBP1", "ELAC2",
              "EP300", "EPHB2", "EZH2", "FGFR2", "FGFR4", "GNMT", "HNF1B", "HOXB13",
              "IGF2", "ITGA6", "KLF6", "LRP2", "MAD1L1", "MED12", "MSMB", "MSR1",
              "MXI1", "NBN", "PCNT", "PLXNB1", "PTEN", "RNASEL", "SRD5A2", "STAT3",
              "TGFBR1", "WRN", "WT1", "ZFHX3",
              "age_at_initial_pathologic_diagnosis", "gleason_score", "sample_type_id",
          ],
          axis=1,
      )
)
print("Data shape:", df.shape)
print(df.head())


# -----------------------------------------------------------------------------
# Part 1: Raw Features → train/test split → Decision Tree → evaluate
# -----------------------------------------------------------------------------
X = df.drop("sample_type_id", axis=1)
y = df["sample_type_id"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

dt1 = DecisionTreeClassifier(random_state=42)
dt1.fit(X_train, y_train)

# visualize (max_depth=3 for readability)
plt.figure(figsize=(18, 10))
plot_tree(
    dt1,
    feature_names=X.columns,
    class_names=[str(c) for c in dt1.classes_],
    filled=True,
    max_depth=3,
)
plt.title("Part 1: Decision Tree (max_depth=3)")
plt.tight_layout()
plt.savefig("tree_part1.png")
plt.close()

y_pred1 = dt1.predict(X_test)
print("\n--- Part 1 Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred1))
print(classification_report(y_test, y_pred1))


# -----------------------------------------------------------------------------
# Part 2: normalize → select top 15 features → retrain & evaluate w/ threshold
# -----------------------------------------------------------------------------
# 1) normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2) select top 15 features
selector = SelectKBest(score_func=f_classif, k=15)
X_sel = selector.fit_transform(X_scaled, y)
selected_features = X.columns[selector.get_support()]
print("\nSelected top 15 features:\n", list(selected_features))

# 3) train/test split
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X_sel, y, test_size=0.2, random_state=42, stratify=y
)

# 4) train
dt2 = DecisionTreeClassifier(random_state=42)
dt2.fit(X2_train, y2_train)

# 5) visualize
plt.figure(figsize=(18, 10))
plot_tree(
    dt2,
    feature_names=list(selected_features),
    class_names=[str(c) for c in dt2.classes_],
    filled=True,
    max_depth=3,
)
plt.title("Part 2: Decision Tree on Top 15 Features (max_depth=3)")
plt.tight_layout()
plt.savefig("tree_part2.png")
plt.close()

# 6) predict probabilities & apply threshold
proba = dt2.predict_proba(X2_test)
# assume binary: positive class is classes_[1]
pos_label = dt2.classes_[1]
neg_label = dt2.classes_[0]
proba_pos = proba[:, 1]
y2_pred_thresh = np.where(proba_pos >= 0.42, pos_label, neg_label)

print("\n--- Part 2 Evaluation (threshold=0.42) ---")
print("Accuracy:", accuracy_score(y2_test, y2_pred_thresh))
print(classification_report(y2_test, y2_pred_thresh))

# 7) confusion matrix
cm = confusion_matrix(y2_test, y2_pred_thresh)
print("Confusion matrix:\n", cm)

# 8) ROC AUC
# still compute AUC on raw probabilities
# compute AUC
auc = roc_auc_score(y2_test, proba_pos)
print("ROC AUC:", auc)

# plot ROC curve from probabilities
RocCurveDisplay.from_predictions(y2_test, proba_pos, pos_label=dt2.classes_[1])
plt.title("ROC Curve (Part 2, threshold=0.42)")
plt.savefig("roc_part2.png", dpi=150, bbox_inches="tight")
plt.close()