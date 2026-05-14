"""Problem 2a: Classification rules for high-K vs Pb-Ba glass."""
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text

from data_loader import OXIDE_SHORT, load_and_merge

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

valid, oxide_cols = load_and_merge()

# ── Step 1: Training data = unweathered samples only ───────────────
train_data = valid[valid["表面风化"] == "无风化"].copy()
X_train = train_data[oxide_cols].values
y_train = (train_data["类型"] == "铅钡").astype(int).values  # 1=铅钡, 0=高钾

print("Step 1 — Training data (unweathered only):")
print(f"  高钾: {(y_train==0).sum()}, 铅钡: {(y_train==1).sum()}")

# ── Step 2: Single-oxide discrimination ────────────────────────────
print("\nStep 2 — Single-oxide discriminability:")
results_ox = []
for ox, short in zip(oxide_cols, OXIDE_SHORT):
    hk = X_train[y_train == 0, oxide_cols.index(ox)]
    pb = X_train[y_train == 1, oxide_cols.index(ox)]

    # Mann-Whitney
    u_stat, p_val = mannwhitneyu(hk, pb, alternative="two-sided")

    # Cohen's d
    d = (pb.mean() - hk.mean()) / np.sqrt((pb.var() + hk.var()) / 2) if (pb.var() + hk.var()) > 0 else 0

    # ROC-AUC (single feature)
    if len(np.unique(X_train[:, oxide_cols.index(ox)])) > 1:
        auc = roc_auc_score(y_train, X_train[:, oxide_cols.index(ox)])
    else:
        auc = 0.5

    results_ox.append({"oxide": short, "p": p_val, "cohens_d": abs(d), "auc": auc})

df_ox = pd.DataFrame(results_ox).sort_values("auc", ascending=False)
print(df_ox.to_string(index=False))
print(f"\nTop discriminators (AUC > 0.9): {len(df_ox[df_ox['auc'] > 0.9])}")

# Visualize top discriminators
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Plot 1: AUC bar
axes[0].barh(df_ox["oxide"], df_ox["auc"], color=["#2196F3" if a > 0.9 else "#BDBDBD" for a in df_ox["auc"]])
axes[0].axvline(x=0.9, color="red", linestyle="--", alpha=0.5)
axes[0].set_xlabel("ROC-AUC")
axes[0].set_title("单氧化物判别力 (ROC-AUC)")
axes[0].invert_yaxis()

pb_idx = oxide_cols.index([c for c in oxide_cols if "PbO" in c][0])
k_idx = oxide_cols.index([c for c in oxide_cols if "K2O" in c][0])
ba_idx = oxide_cols.index([c for c in oxide_cols if "BaO" in c][0])
pb_col = oxide_cols[pb_idx]
k_col = oxide_cols[k_idx]
ba_col = oxide_cols[ba_idx]

for label, color, marker in [("高钾", "#4CAF50", "o"), ("铅钡", "#FF5722", "s")]:
    mask = y_train == (1 if label == "铅钡" else 0)
    axes[1].scatter(X_train[mask, pb_idx], X_train[mask, k_idx],
                    c=color, label=label, alpha=0.7, s=60, edgecolors="white")
axes[1].set_xlabel("PbO (%)")
axes[1].set_ylabel("K2O (%)")
axes[1].set_title("PbO vs K2O (未风化样品)")
axes[1].legend()

# Plot 3: PbO vs BaO
for label, color, marker in [("高钾", "#4CAF50", "o"), ("铅钡", "#FF5722", "s")]:
    mask = y_train == (1 if label == "铅钡" else 0)
    axes[2].scatter(X_train[mask, pb_idx], X_train[mask, ba_idx],
                    c=color, label=label, alpha=0.7, s=60, edgecolors="white")
axes[2].set_xlabel("PbO (%)")
axes[2].set_ylabel("BaO (%)")
axes[2].set_title("PbO vs BaO (未风化样品)")
axes[2].legend()

plt.tight_layout()
plt.savefig("analysis/fig2a_discrimination.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: analysis/fig2a_discrimination.png")

# ── Step 3: LDA ────────────────────────────────────────────────────
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
lda_coef = pd.Series(lda.coef_[0], index=OXIDE_SHORT).sort_values(key=abs, ascending=False)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lda_cv = cross_val_score(lda, X_train, y_train, cv=cv, scoring="accuracy")

print("\nStep 3 — LDA:")
print(f"  CV accuracy: {lda_cv.mean():.3f} +/- {lda_cv.std():.3f}")
print(f"  Top coefficients:\n{lda_coef.head(6).to_string()}")

# ── Decision tree (simple rules) ───────────────────────────────────
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
dt_cv = cross_val_score(dt, X_train, y_train, cv=cv, scoring="accuracy")

print("\nDecision Tree:")
print(f"  CV accuracy: {dt_cv.mean():.3f} +/- {dt_cv.std():.3f}")
print(f"  Rules:\n{export_text(dt, feature_names=OXIDE_SHORT)}")

# ── Step 4: Apply to weathered samples ────────────────────────────
test_data = valid[valid["表面风化"] == "风化"].copy()
X_test = test_data[oxide_cols].values
y_test = (test_data["类型"] == "铅钡").astype(int).values

lda_pred = lda.predict(X_test)
dt_pred = dt.predict(X_test)

print("\nStep 4 — Classification of weathered samples:")
print(f"  LDA accuracy: {(lda_pred == y_test).mean():.3f}")
print(f"  DT accuracy: {(dt_pred == y_test).mean():.3f}")

# Which samples are misclassified?
misclass = test_data[lda_pred != y_test]
print(f"\n  LDA misclassifications ({len(misclass)}/{len(test_data)}):")
for i, idx in enumerate(misclass.index):
    row = misclass.iloc[i]
    true_type = row["类型"]
    pred_type = "铅钡" if lda_pred[list(test_data.index).index(idx)] == 1 else "高钾"
    k_val = row[k_col]
    pb_val = row[pb_col]
    print(f"    {idx}: true={true_type}, pred={pred_type}, K2O={k_val:.2f}%, PbO={pb_val:.2f}%")

# ── ROC curves ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
from sklearn.metrics import roc_curve

for ox, short in [("PbO", "PbO"), ("K2O", "K2O"), ("BaO", "BaO")]:
    idx = oxide_cols.index([c for c in oxide_cols if ox in c][0])
    fpr, tpr, _ = roc_curve(y_train, X_train[:, idx])
    ax.plot(fpr, tpr, label=f"{short} (AUC={roc_auc_score(y_train, X_train[:, idx]):.3f})")

lda_proba = lda.predict_proba(X_train)[:, 1]
fpr, tpr, _ = roc_curve(y_train, lda_proba)
ax.plot(fpr, tpr, "k--", linewidth=2, label=f"LDA (AUC={roc_auc_score(y_train, lda_proba):.3f})")

ax.plot([0, 1], [0, 1], "gray", linestyle=":", alpha=0.5)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC曲线: 高钾 vs 铅钡 分类")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("analysis/fig2a_roc.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: analysis/fig2a_roc.png")

# Save results
results = {
    "top_discriminators": df_ox.head(6).to_dict("records"),
    "lda_cv_accuracy": float(lda_cv.mean()),
    "lda_top_coefficients": {k: float(v) for k, v in lda_coef.head(6).items()},
    "dt_cv_accuracy": float(dt_cv.mean()),
    "weathered_lda_accuracy": float((lda_pred == y_test).mean()),
    "weathered_dt_accuracy": float((dt_pred == y_test).mean()),
}
with open("analysis/problem2a_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("Saved: analysis/problem2a_results.json")
