"""Problem 3: Classify unknown samples A1-A8 and sensitivity analysis."""
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

from data_loader import load_and_merge

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

valid, oxide_cols = load_and_merge()

# Train classifiers on unweathered samples
train = valid[valid["表面风化"] == "无风化"]
X_train = train[oxide_cols].values
y_train = (train["类型"] == "铅钡").astype(int).values

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# PbO/BaO thresholds from training data
ba_hk = X_train[y_train == 0, oxide_cols.index([c for c in oxide_cols if "BaO" in c][0])]
ba_pb = X_train[y_train == 1, oxide_cols.index([c for c in oxide_cols if "BaO" in c][0])]
pb_hk = X_train[y_train == 0, oxide_cols.index([c for c in oxide_cols if "PbO" in c][0])]
pb_pb = X_train[y_train == 1, oxide_cols.index([c for c in oxide_cols if "PbO" in c][0])]

# Optimal thresholds from training data
ba_threshold = (ba_hk.max() + ba_pb.min()) / 2
pb_threshold = (pb_hk.max() + pb_pb.min()) / 2
print(f"BaO threshold: {ba_threshold:.2f}% (高钾 max={ba_hk.max():.2f}, 铅钡 min={ba_pb.min():.2f})")
print(f"PbO threshold: {pb_threshold:.2f}% (高钾 max={pb_hk.max():.2f}, 铅钡 min={pb_pb.min():.2f})")

# ── Load unknown samples ───────────────────────────────────────────
df_unknown = pd.read_csv("data/表单3_未知类别.csv", skiprows=1)
print(f"\nUnknown samples: {len(df_unknown)}")
print(df_unknown.iloc[:, :3].to_string())

uk_oxide_cols = [df_unknown.columns[i] for i in range(2, 16)]
df_unknown[uk_oxide_cols] = df_unknown[uk_oxide_cols].fillna(0)
X_uk = df_unknown[uk_oxide_cols].values

# ── Classification by multiple methods ─────────────────────────────
sample_ids = df_unknown.iloc[:, 0].tolist()
weathering = df_unknown.iloc[:, 1].tolist()

results = []
for i, sid in enumerate(sample_ids):
    x = X_uk[i]
    ba_val = x[list(df_unknown.columns).index([c for c in uk_oxide_cols if "BaO" in c][0]) - 2]
    pb_val = x[list(df_unknown.columns).index([c for c in uk_oxide_cols if "PbO" in c][0]) - 2]
    k_val = x[list(df_unknown.columns).index([c for c in uk_oxide_cols if "K2O" in c][0]) - 2]
    si_val = x[list(df_unknown.columns).index([c for c in uk_oxide_cols if "SiO2" in c][0]) - 2]

    # Method 1: BaO rule
    ba_rule = "铅钡" if ba_val > ba_threshold else "高钾"

    # Method 2: PbO rule
    pb_rule = "铅钡" if pb_val > pb_threshold else "高钾"

    # Method 3: Decision tree
    dt_pred = "铅钡" if dt.predict([x])[0] == 1 else "高钾"

    # Method 4: LDA
    lda_pred = "铅钡" if lda.predict([x])[0] == 1 else "高钾"

    # Consensus
    votes = [ba_rule, pb_rule, dt_pred, lda_pred]
    pb_votes = sum(1 for v in votes if v == "铅钡")
    hk_votes = 4 - pb_votes
    consensus = "铅钡" if pb_votes >= 3 else ("高钾" if hk_votes >= 3 else "不确定")

    results.append({
        "样品": sid, "表面风化": weathering[i],
        "SiO2": si_val, "K2O": k_val, "PbO": pb_val, "BaO": ba_val,
        "BaO规则": ba_rule, "PbO规则": pb_rule,
        "决策树": dt_pred, "LDA": lda_pred,
        "铅钡票数": pb_votes, "共识": consensus,
    })

df_results = pd.DataFrame(results)
print("\n── Classification Results ──")
print(df_results[["样品", "表面风化", "PbO", "BaO", "BaO规则", "PbO规则", "决策树", "LDA", "共识"]].to_string(index=False))

# ── Sensitivity Analysis ───────────────────────────────────────────
print("\n── Sensitivity Analysis ──")

# 1. Threshold perturbation
print("\n1. BaO threshold perturbation (+-20%):")
for perturb in [0.8, 0.9, 1.0, 1.1, 1.2]:
    t = ba_threshold * perturb
    changed = 0
    for i, sid in enumerate(sample_ids):
        ba_val = X_uk[i, list(df_unknown.columns).index([c for c in uk_oxide_cols if "BaO" in c][0]) - 2]
        orig = "铅钡" if ba_val > ba_threshold else "高钾"
        new = "铅钡" if ba_val > t else "高钾"
        if orig != new:
            changed += 1
    pct = int(perturb * 100)
    print(f"  BaO > {t:.2f}% ({pct}%): {changed}/8 样品分类变化")

# 2. Weathering correction (apply 1c correction then re-classify)
print("\n2. Classification after weathering correction:")
# Recompute correction vectors (same logic as problem1c)
# Pb-Ba: paired correction from artifacts 49 and 50
pb_pairs_deltas = []
for art_id in ["49", "50"]:
    sub = valid[valid.index == art_id]
    pre_sample = sub[sub["采样类型"] == "未风化点"]
    post_sample = sub[sub["采样类型"] == "普通"]
    if len(pre_sample) == 1 and len(post_sample) == 1:
        pb_pairs_deltas.append(post_sample[oxide_cols].iloc[0].values - pre_sample[oxide_cols].iloc[0].values)
pb_delta = np.mean(pb_pairs_deltas, axis=0)

# High-K: group mean difference (weathered - unweathered)
hk_w = valid[(valid["类型"] == "高钾") & (valid["表面风化"] == "风化")]
hk_uw = valid[(valid["类型"] == "高钾") & (valid["表面风化"] == "无风化")]
hk_delta = hk_w[oxide_cols].mean().values - hk_uw[oxide_cols].mean().values

for i, sid in enumerate(sample_ids):
    x = X_uk[i]
    weath = weathering[i]

    if weath == "风化":
        # Try both corrections
        pre_pb = np.clip(x - pb_delta, 0, 100)
        pre_hk = np.clip(x - hk_delta, 0, 100)

        ba_pb_corrected = pre_pb[list(df_unknown.columns).index([c for c in uk_oxide_cols if "BaO" in c][0]) - 2]
        pb_pb_corrected = pre_pb[list(df_unknown.columns).index([c for c in uk_oxide_cols if "PbO" in c][0]) - 2]

        ba_hk_corrected = pre_hk[list(df_unknown.columns).index([c for c in uk_oxide_cols if "BaO" in c][0]) - 2]
        pb_hk_corrected = pre_hk[list(df_unknown.columns).index([c for c in uk_oxide_cols if "PbO" in c][0]) - 2]

        orig = df_results.loc[i, "共识"]
        print(f"  {sid} ({weath}): 原始共识={orig}")
        print(f"    Pb-Ba校正后: BaO={ba_pb_corrected:.2f}%, PbO={pb_pb_corrected:.2f}% → "
              f"{'铅钡' if ba_pb_corrected > ba_threshold else '高钾'}")
        print(f"    高钾校正后: BaO={ba_hk_corrected:.2f}%, PbO={pb_hk_corrected:.2f}% → "
              f"{'铅钡' if ba_hk_corrected > ba_threshold else '高钾'}")
    else:
        print(f"  {sid} ({weath}): 无需校正")

# 3. Multi-method agreement matrix
print("\n3. Multi-method agreement:")
methods = ["BaO规则", "PbO规则", "决策树", "LDA"]
agreement = np.zeros((4, 4), dtype=int)
for i in range(4):
    for j in range(4):
        agreement[i, j] = (df_results[methods[i]] == df_results[methods[j]]).sum()
print("  Pairwise agreement (out of 8):")
print("         " + " ".join(f"{m[:4]:>6}" for m in methods))
for i, m in enumerate(methods):
    print(f"  {m[:6]:<6} " + " ".join(f"{agreement[i,j]:>6}" for j in range(4)))

# ── Visualization ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: BaO vs PbO with training data and unknowns
ba_idx = oxide_cols.index([c for c in oxide_cols if "BaO" in c][0])
pb_idx = oxide_cols.index([c for c in oxide_cols if "PbO" in c][0])

# Training data
for label, color, marker in [("高钾", "#4CAF50", "o"), ("铅钡", "#FF5722", "s")]:
    mask = y_train == (1 if label == "铅钡" else 0)
    axes[0].scatter(X_train[mask, pb_idx], X_train[mask, ba_idx],
                    c=color, label=f"已知{label}", alpha=0.5, s=40, edgecolors="white")

# Unknown samples
for i, sid in enumerate(sample_ids):
    uk_ba = X_uk[i, list(df_unknown.columns).index([c for c in uk_oxide_cols if "BaO" in c][0]) - 2]
    uk_pb = X_uk[i, list(df_unknown.columns).index([c for c in uk_oxide_cols if "PbO" in c][0]) - 2]
    cons = df_results.loc[i, "共识"]
    color = "#FF5722" if cons == "铅钡" else ("#4CAF50" if cons == "高钾" else "#FF9800")
    axes[0].scatter(uk_pb, uk_ba, c=color, s=150, marker="*", edgecolors="black", linewidth=1,
                    zorder=5)
    axes[0].annotate(sid, (uk_pb, uk_ba), xytext=(5, 5), textcoords="offset points", fontsize=9, fontweight="bold")

axes[0].axhline(y=ba_threshold, color="gray", linestyle="--", alpha=0.5, label=f"BaO={ba_threshold:.1f}%")
axes[0].set_xlabel("PbO (%)")
axes[0].set_ylabel("BaO (%)")
axes[0].set_title("未知样品分类 (PbO-BaO 空间)")
axes[0].legend(fontsize=7)

# Plot 2: Sensitivity — threshold sweep
thresholds = np.linspace(ba_threshold * 0.5, ba_threshold * 1.5, 50)
n_pb = []
for t in thresholds:
    count = 0
    for i in range(len(X_uk)):
        ba_val = X_uk[i, list(df_unknown.columns).index([c for c in uk_oxide_cols if "BaO" in c][0]) - 2]
        if ba_val > t:
            count += 1
    n_pb.append(count)

axes[1].plot(thresholds, n_pb, "b-", linewidth=2)
axes[1].axvline(x=ba_threshold, color="red", linestyle="--", label=f"最优阈值={ba_threshold:.1f}%")
axes[1].fill_between([ba_threshold * 0.8, ba_threshold * 1.2], 0, 8, alpha=0.1, color="gray", label="+-20%")
axes[1].set_xlabel("BaO 阈值 (%)")
axes[1].set_ylabel("分类为铅钡的样品数")
axes[1].set_title("BaO 阈值敏感性")
axes[1].legend(fontsize=8)
axes[1].set_yticks(range(9))

plt.tight_layout()
plt.savefig("analysis/fig3_classification.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: analysis/fig3_classification.png")

# Save
df_results.to_csv("analysis/problem3_classification.csv", encoding="utf-8-sig", index=False)

summary = {
    "classification": df_results[["样品", "共识", "铅钡票数"]].to_dict("records"),
    "ba_threshold": float(ba_threshold),
    "pb_threshold": float(pb_threshold),
    "agreement_matrix": agreement.tolist(),
    "method_names": methods,
}
with open("analysis/problem3_results.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("Saved: analysis/problem3_classification.csv, analysis/problem3_results.json")
