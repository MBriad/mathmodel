"""Independent validation script — does NOT import any src/ modules.

Re-computes key assertions from raw data to verify modeling results.
Each check is self-contained and uses only pandas/scipy/numpy.
"""
import re

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

OXIDE_SHORT = ["SiO2","Na2O","K2O","CaO","MgO","Al2O3","Fe2O3","CuO",
               "PbO","BaO","P2O5","SrO","SnO2","SO2"]

failures = 0

def check(desc, condition):
    global failures
    status = "PASS" if condition else "FAIL"
    if not condition:
        failures += 1
    print(f"  [{status}] {desc}")

# ═══════════════════════════════════════════════════════════════════
# SECTION 0: Data Integrity
# ═══════════════════════════════════════════════════════════════════
print("── 0. Data Integrity ──")

df1 = pd.read_csv("data/表单1_文物信息.csv", skiprows=1)
df2 = pd.read_csv("data/表单2_化学成分.csv", skiprows=1)
df3 = pd.read_csv("data/表单3_未知类别.csv", skiprows=1)

check("表单1: 58 rows", len(df1) == 58)
check("表单1: 5 cols", df1.shape[1] == 5)

# Type counts
type_counts = df1.iloc[:, 2].value_counts()
check(f"高钾 count = 18 (got {type_counts.get('高钾', 0)})", type_counts.get("高钾", 0) == 18)
check(f"铅钡 count = 40 (got {type_counts.get('铅钡', 0)})", type_counts.get("铅钡", 0) == 40)

# Weathering counts
weath_counts = df1.iloc[:, 4].value_counts()
check(f"风化 count = 34 (got {weath_counts.get('风化', 0)})", weath_counts.get("风化", 0) == 34)
check(f"无风化 count = 24 (got {weath_counts.get('无风化', 0)})", weath_counts.get("无风化", 0) == 24)

check("表单2: 69 rows", len(df2) == 69)
check("表单3: 8 rows (A1-A8)", len(df3) == 8)
ids = df3.iloc[:, 0].tolist()
check(f"A1-A8 in order: {ids}", ids == ["A1","A2","A3","A4","A5","A6","A7","A8"])

# Validity: oxide sums 85%-105%
oxide_cols_2 = [df2.columns[i] for i in range(1, 15)]
sums_2 = df2[oxide_cols_2].fillna(0).sum(axis=1)
valid_2 = sums_2.between(85, 105)
check(f"表单2 valid (85-105%): {valid_2.sum()}/69 (expect 67)", valid_2.sum() == 67)

# Validity for 表单3
oxide_cols_3 = [df3.columns[i] for i in range(2, 16)]
sums_3 = df3[oxide_cols_3].fillna(0).sum(axis=1)
valid_3 = sums_3.between(85, 105)
check(f"表单3 valid (85-105%): {valid_3.sum()}/8", valid_3.sum() >= 7)

# ═══════════════════════════════════════════════════════════════════
# SECTION 1a: Weathering × Type association
# ═══════════════════════════════════════════════════════════════════
print("\n── 1a. Weathering Association ──")

# Build contingency table manually
ct = pd.crosstab(df1.iloc[:, 4], df1.iloc[:, 2])
check("2×2 table: 无风化-高钾 = 12", ct.loc["无风化", "高钾"] == 12)
check("2×2 table: 风化-铅钡 = 28", ct.loc["风化", "铅钡"] == 28)

# Weathering rates
hk_w_rate = ct.loc["风化", "高钾"] / ct["高钾"].sum()
pb_w_rate = ct.loc["风化", "铅钡"] / ct["铅钡"].sum()
check(f"高钾风化率 = 33.3% (got {hk_w_rate:.1%})", abs(hk_w_rate - 6/18) < 0.01)
check(f"铅钡风化率 = 70.0% (got {pb_w_rate:.1%})", abs(pb_w_rate - 28/40) < 0.01)
check("铅钡风化率 > 高钾风化率", pb_w_rate > hk_w_rate)

# Fisher exact test
_, fisher_p = fisher_exact(ct.values)
check(f"Fisher p < 0.05 (got {fisher_p:.4f})", fisher_p < 0.05)
check(f"Fisher p ≈ 0.011 (got {fisher_p:.4f})", abs(fisher_p - 0.0113) < 0.001)

# ═══════════════════════════════════════════════════════════════════
# SECTION 1b: Composition statistics
# ═══════════════════════════════════════════════════════════════════
print("\n── 1b. Composition Statistics ──")

# Merge for labeled data
def extract_id(name):
    m = re.match(r"(\d+)", str(name))
    return f"{int(m.group(1)):02d}" if m else None

df2_copy = df2.copy()
df2_copy["文物编号"] = df2_copy.iloc[:, 0].apply(extract_id)
df2_copy = df2_copy.set_index("文物编号")
df1_copy = df1.copy()
df1_copy["文物编号"] = df1_copy.iloc[:, 0].astype(str).str.zfill(2)
df1_copy = df1_copy.set_index("文物编号")

merged = df2_copy.merge(df1_copy.iloc[:, [1, 3]], left_index=True, right_index=True)
merged[oxide_cols_2] = merged[oxide_cols_2].fillna(0)
valid_m = merged[merged[oxide_cols_2].sum(axis=1).between(85, 105)]

# Spot-check two artifacts
a01 = valid_m[valid_m.index == "01"]
check("文物01 SiO2 = 69.33%", abs(a01[oxide_cols_2[0]].iloc[0] - 69.33) < 0.01)

a08 = valid_m[valid_m.index == "08"]
check("文物08 PbO = 28.68% (first 08 row)", abs(a08[oxide_cols_2[8]].iloc[0] - 28.68) < 0.01)

# High-K weathered SiO2 should be > 90%
hk_w = valid_m[(valid_m["类型"] == "高钾") & (valid_m["表面风化"] == "风化")]
hk_uw = valid_m[(valid_m["类型"] == "高钾") & (valid_m["表面风化"] == "无风化")]
check(f"高钾-风化 SiO2 mean = {hk_w[oxide_cols_2[0]].mean():.1f}% > 90%",
      hk_w[oxide_cols_2[0]].mean() > 90)
check("高钾-风化 K2O mean < 高钾-未风化 K2O mean",
      hk_w[oxide_cols_2[2]].mean() < hk_uw[oxide_cols_2[2]].mean())

# Pb-Ba weathered has higher PbO
pb_w = valid_m[(valid_m["类型"] == "铅钡") & (valid_m["表面风化"] == "风化")]
pb_uw = valid_m[(valid_m["类型"] == "铅钡") & (valid_m["表面风化"] == "无风化")]
check("铅钡-风化 PbO mean > 铅钡-未风化 PbO mean",
      pb_w[oxide_cols_2[8]].mean() > pb_uw[oxide_cols_2[8]].mean())

# ═══════════════════════════════════════════════════════════════════
# SECTION 1c: Prediction validation
# ═══════════════════════════════════════════════════════════════════
print("\n── 1c. Prediction Validation ──")

# Recompute paired correction for Pb-Ba (49, 50)
pb_deltas = []
for art_id in ["49", "50"]:
    sub = valid_m[valid_m.index == art_id]
    # Find 未风化点 and 普通 samples - use the sampling point name
    full_names = df2.iloc[:, 0].tolist()
    # Map: find rows for this artifact
    art_rows = [i for i, n in enumerate(full_names) if str(n).startswith(art_id)]
    uw_row = None
    reg_row = None
    for r in art_rows:
        name = str(full_names[r])
        if "未风化点" in name:
            uw_row = r
        elif "严重风化点" not in name and "未风化" not in name and "部位" not in name:
            reg_row = r
    if uw_row is not None and reg_row is not None:
        pre = df2.iloc[uw_row, 1:15].fillna(0).values.astype(float)
        post = df2.iloc[reg_row, 1:15].fillna(0).values.astype(float)
        pb_deltas.append(post - pre)

if len(pb_deltas) >= 2:
    mean_delta = np.mean(pb_deltas, axis=0)
    # Predict 49 pre-weathering and compare
    # Find 49's weathered sample
    idx_49_reg = None
    idx_49_uw = None
    for r in range(len(df2)):
        name = str(df2.iloc[r, 0])
        if name.startswith("49"):
            if "未风化点" in name:
                idx_49_uw = r
            elif "严重风化" not in name and "未风化" not in name and "部位" not in name:
                idx_49_reg = r

    if idx_49_reg is not None and idx_49_uw is not None:
        post_49 = df2.iloc[idx_49_reg, 1:15].fillna(0).values.astype(float)
        actual_pre_49 = df2.iloc[idx_49_uw, 1:15].fillna(0).values.astype(float)
        predicted_pre_49 = np.clip(post_49 - mean_delta, 0, 100)
        mae_49 = np.abs(predicted_pre_49 - actual_pre_49).mean()
        check(f"49 MAE = {mae_49:.2f}% < 2%", mae_49 < 2.0)

# ═══════════════════════════════════════════════════════════════════
# SECTION 2a: Classification rules
# ═══════════════════════════════════════════════════════════════════
print("\n── 2a. Classification Rules ──")

ba_idx = oxide_cols_2.index([c for c in oxide_cols_2 if "BaO" in c][0])
pb_idx_2a = oxide_cols_2.index([c for c in oxide_cols_2 if "PbO" in c][0])

# Only unweathered for training
train = valid_m[valid_m["表面风化"] == "无风化"]
ba_hk = train[train["类型"] == "高钾"][oxide_cols_2[ba_idx]]
ba_pb = train[train["类型"] == "铅钡"][oxide_cols_2[ba_idx]]

ba_threshold = (ba_hk.max() + ba_pb.min()) / 2
check(f"BaO threshold = {ba_threshold:.2f} (expect ~3.14)", abs(ba_threshold - 3.14) < 0.05)
check(f"高钾 max BaO ({ba_hk.max():.2f}) < 铅钡 min BaO ({ba_pb.min():.2f})",
      ba_hk.max() < ba_pb.min())

# PbO threshold
pb_hk = train[train["类型"] == "高钾"][oxide_cols_2[pb_idx_2a]]
pb_pb = train[train["类型"] == "铅钡"][oxide_cols_2[pb_idx_2a]]
pb_threshold = (pb_hk.max() + pb_pb.min()) / 2
check(f"PbO threshold = {pb_threshold:.2f} (expect ~5.46)", abs(pb_threshold - 5.46) < 0.1)

# On weathered samples: BaO rule accuracy
test = valid_m[valid_m["表面风化"] == "风化"]
ba_pred = test[oxide_cols_2[ba_idx]] > ba_threshold
ba_true = test["类型"] == "铅钡"
ba_acc = (ba_pred == ba_true).mean()
check(f"BaO rule on weathered: accuracy = {ba_acc:.1%} > 85%", ba_acc > 0.85)

# K2O AUC should be low (reverse discrimination)
k_idx_2a = oxide_cols_2.index([c for c in oxide_cols_2 if "K2O" in c][0])
k_hk = train[train["类型"] == "高钾"][oxide_cols_2[k_idx_2a]]
k_pb = train[train["类型"] == "铅钡"][oxide_cols_2[k_idx_2a]]
check(f"高钾 K2O mean ({k_hk.mean():.1f}) > 铅钡 K2O mean ({k_pb.mean():.1f})",
      k_hk.mean() > k_pb.mean())

# ═══════════════════════════════════════════════════════════════════
# SECTION 2b: Clustering sanity
# ═══════════════════════════════════════════════════════════════════
print("\n── 2b. Clustering Sanity ──")

for glass_type in ["高钾", "铅钡"]:
    data = valid_m[valid_m["类型"] == glass_type]
    X = data[oxide_cols_2].values
    # Remove near-zero-variance
    mask = X.var(axis=0) > 0.01
    X_a = X[:, mask]
    if len(X_a) >= 4:
        X_s = StandardScaler().fit_transform(X_a)
        pca = PCA()
        X_p = pca.fit_transform(X_s)
        nc = max(2, np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.85) + 1)
        km = KMeans(n_clusters=2, random_state=42, n_init=20)
        labels = km.fit_predict(X_p[:, :nc])
        sil = silhouette_score(X_p[:, :nc], labels)
        check(f"{glass_type}: silhouette(k=2) = {sil:.3f} in [-1, 1]", -1 <= sil <= 1)

        # Weathered samples should mostly cluster together
        weath_mask = (data["表面风化"] == "风化").values
        if weath_mask.sum() >= 3:
            w_labels = labels[weath_mask]
            dominant = np.bincount(w_labels).max()
            check(f"{glass_type}: {dominant}/{weath_mask.sum()} 风化样品在同一簇",
                  dominant >= weath_mask.sum() * 0.6)

# ═══════════════════════════════════════════════════════════════════
# SECTION 3: Unknown classification (A1-A8)
# ═══════════════════════════════════════════════════════════════════
print("\n── 3. Unknown Classification ──")

ba_idx3 = oxide_cols_3.index([c for c in oxide_cols_3 if "BaO" in c][0])
pb_idx3 = oxide_cols_3.index([c for c in oxide_cols_3 if "PbO" in c][0])

for i in range(len(df3)):
    sid = df3.iloc[i, 0]
    ba_val = df3.iloc[i, ba_idx3 + 2]  # +2 offset for cols A,B
    pb_val = df3.iloc[i, pb_idx3 + 2]
    ba_class = "铅钡" if ba_val > ba_threshold else "高钾"
    pb_class = "铅钡" if pb_val > pb_threshold else "高钾"

    # A1, A6, A7: BaO=0, PbO≈0 → should be 高钾
    if sid in ["A1", "A6", "A7"]:
        check(f"{sid}: BaO=0, PbO≈0 → BaO rule = 高钾", ba_class == "高钾")
    # A3, A4, A8: BaO > 3.14 → should be 铅钡
    if sid in ["A3", "A4", "A8"]:
        check(f"{sid}: BaO > 3.14 → BaO rule = 铅钡", ba_class == "铅钡")
    # A2: PbO high but BaO=0 → methods should disagree
    if sid == "A2":
        disagree = (ba_class != pb_class)
        check(f"A2: BaO rule={ba_class}, PbO rule={pb_class} → disagree", disagree)
    # A5: borderline
    if sid == "A5":
        check(f"A5: BaO={ba_val:.2f}, PbO={pb_val:.2f}", True)  # informational only

# ═══════════════════════════════════════════════════════════════════
# SECTION 4: Correlation analysis
# ═══════════════════════════════════════════════════════════════════
print("\n── 4. Correlation Analysis ──")

# CLR transform
X_raw = valid_m[oxide_cols_2].values
X_clr = np.zeros_like(X_raw)
for j in range(14):
    col = X_raw[:, j]
    col_pos = col[col > 0]
    min_pos = col_pos.min() if len(col_pos) > 0 else 0.01
    X_clr[:, j] = np.where(col <= 0, min_pos / 2, col)
gm = np.exp(np.mean(np.log(X_clr), axis=1, keepdims=True))
X_clr = np.log(X_clr / gm)

for glass_type in ["高钾", "铅钡"]:
    mask = (valid_m["类型"] == glass_type).values
    X_t = X_clr[mask]
    corr = np.corrcoef(X_t.T)

    # Basic checks
    check(f"{glass_type}: correlation matrix symmetric",
          np.allclose(corr, corr.T))
    check(f"{glass_type}: diagonal = 1",
          np.allclose(np.diag(corr), 1.0, atol=1e-10))
    check(f"{glass_type}: |r| <= 1 for all",
          np.all(np.abs(corr) <= 1.0))

    # Weathering-stratified avg |r|
    for weath in ["无风化", "风化"]:
        wm = (valid_m["类型"] == glass_type) & (valid_m["表面风化"] == weath)
        n_w = wm.sum()
        if n_w >= 4:
            c_w = np.corrcoef(X_clr[wm.values].T)
            upper = c_w[np.triu_indices(14, k=1)]
            avg_abs = np.abs(upper).mean()
            check(f"{glass_type}-{weath} (n={n_w}): avg |r| = {avg_abs:.3f}",
                  avg_abs > 0.1)  # should have some structure

# Specific: 高钾风化 avg |r| should be higher due to closure effect
hk_w_mask = (valid_m["类型"] == "高钾") & (valid_m["表面风化"] == "风化")
hk_uw_mask = (valid_m["类型"] == "高钾") & (valid_m["表面风化"] == "无风化")
if hk_w_mask.sum() >= 4 and hk_uw_mask.sum() >= 4:
    c_hk_w = np.corrcoef(X_clr[hk_w_mask.values].T)
    c_hk_uw = np.corrcoef(X_clr[hk_uw_mask.values].T)
    avg_w = np.abs(c_hk_w[np.triu_indices(14, k=1)]).mean()
    avg_uw = np.abs(c_hk_uw[np.triu_indices(14, k=1)]).mean()
    check(f"高钾风化 avg |r| ({avg_w:.3f}) > 高钾未风化 avg |r| ({avg_uw:.3f})",
          avg_w > avg_uw)

# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  TOTAL: {failures} FAILURES")
print(f"{'='*60}")
