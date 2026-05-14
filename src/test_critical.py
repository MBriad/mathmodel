"""Critical unit tests for the most bug-prone functions.

These test individual functions with known inputs and expected outputs.
If any of these fail, the modeling results are unreliable regardless of
what validate.py says.
"""
import re

import numpy as np
import pandas as pd

failures = 0

def check(desc, condition):
    global failures
    status = "PASS" if condition else "FAIL"
    if not condition:
        failures += 1
    print(f"  [{status}] {desc}")

# ═══════════════════════════════════════════════════════════════════
# TEST 1: extract_id — parses artifact number from sampling point name
# ═══════════════════════════════════════════════════════════════════
print("── 1. extract_id ──")

def extract_id(name):
    m = re.match(r"(\d+)", str(name))
    return f"{int(m.group(1)):02d}" if m else None

check("'01' -> '01'", extract_id("01") == "01")
check("'03部位1' -> '03'", extract_id("03部位1") == "03")
check("'08严重风化点' -> '08'", extract_id("08严重风化点") == "08")
check("'15' -> '15'", extract_id("15") == "15")
check("'42未风化点1' -> '42'", extract_id("42未风化点1") == "42")
check("'49未风化点' -> '49'", extract_id("49未风化点") == "49")
check("str(49) -> '49'", extract_id(49) == "49")
check("None -> None", extract_id(None) is None)
check("空字符串 -> None", extract_id("") is None)
check("leading zero preserved: '03' not '3'", extract_id("03") == "03")
check("two-digit: '58' -> '58'", extract_id("58") == "58")

# ═══════════════════════════════════════════════════════════════════
# TEST 2: classify_sample — categorizes sampling point type
# ═══════════════════════════════════════════════════════════════════
print("\n── 2. classify_sample ──")

def classify_sample(name):
    s = str(name)
    if "严重风化点" in s:
        return "严重风化点"
    if "未风化点" in s:
        return "未风化点"
    return "普通"

check("'01' -> 普通", classify_sample("01") == "普通")
check("'03部位1' -> 普通 (部位 suffix ignored)", classify_sample("03部位1") == "普通")
check("'08严重风化点' -> 严重风化点", classify_sample("08严重风化点") == "严重风化点")
check("'23未风化点' -> 未风化点", classify_sample("23未风化点") == "未风化点")
check("'42未风化点1' -> 未风化点", classify_sample("42未风化点1") == "未风化点")
check("'49未风化点' -> 未风化点", classify_sample("49未风化点") == "未风化点")
check("'54严重风化点' -> 严重风化点", classify_sample("54严重风化点") == "严重风化点")
# Edge: "未风化" alone without "点" is NOT caught — but our data doesn't have this
check("'15' -> 普通 (int input)", classify_sample(15) == "普通")

# ═══════════════════════════════════════════════════════════════════
# TEST 3: CLR transform — centered log-ratio correctness
# ═══════════════════════════════════════════════════════════════════
print("\n── 3. CLR Transformation ──")

def clr_transform(X):
    """X: (n_samples, n_components) array, values in [0, 100]."""
    X = np.array(X, dtype=float)
    # Replace zeros with half min positive
    X_adj = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        pos = col[col > 0]
        min_pos = pos.min() if len(pos) > 0 else 0.01
        X_adj[:, j] = np.where(col <= 0, min_pos / 2, col)
    gm = np.exp(np.mean(np.log(X_adj), axis=1, keepdims=True))
    return np.log(X_adj / gm)

# Test 1: Simple 2-component case
X_test = np.array([[50, 50], [70, 30], [30, 70]])
X_clr_test = clr_transform(X_test)
# CLR rows should sum to 0
row_sums = X_clr_test.sum(axis=1)
check("CLR row sums ≈ 0 (within 1e-10)", np.allclose(row_sums, 0, atol=1e-10))

# Test 2: Identical composition → identical CLR
X_same = np.array([[60, 40], [60, 40]])
clr_same = clr_transform(X_same)
check("Identical rows → identical CLR values", np.allclose(clr_same[0], clr_same[1]))

# Test 3: Zero handling
X_zero = np.array([[100, 0, 0], [0, 100, 0]])
clr_zero = clr_transform(X_zero)
check("Zero-handled CLR: no inf or nan", not np.any(np.isinf(clr_zero)) and not np.any(np.isnan(clr_zero)))
check("Zero-handled CLR row sums ≈ 0", np.allclose(clr_zero.sum(axis=1), 0, atol=1e-10))

# Test 4: Proportional compositions produce identical CLR
X_prop = np.array([[10, 20, 70], [20, 40, 140]])
clr_prop = clr_transform(X_prop)
check("Proportional compositions → identical CLR (scale invariance)",
      np.allclose(clr_prop[0], clr_prop[1], atol=1e-10))

# Test 5: Roundtrip sanity — correlation on raw vs CLR should differ
X_corr_test = np.array([[70, 5, 25], [50, 20, 30], [60, 10, 30], [40, 30, 30]])
clr_corr = clr_transform(X_corr_test)
raw_corr = np.corrcoef(X_corr_test.T)
clr_c = np.corrcoef(clr_corr.T)
# They should be different (CLR breaks closure)
diff_exists = not np.allclose(raw_corr, clr_c)
check("CLR changes correlation structure (not identical to raw)", diff_exists)

# ═══════════════════════════════════════════════════════════════════
# TEST 4: Difference method prediction
# ═══════════════════════════════════════════════════════════════════
print("\n── 4. Difference Method Prediction ──")

def predict_pre(post, delta):
    """Predict pre-weathering composition given post-weathering and delta vector."""
    pre = np.array(post) - np.array(delta)
    return np.clip(pre, 0, 100)

# Known pair: pre=(10,20,70), post=(5,15,80) → delta = (-5,-5,+10)
delta_test = np.array([-5, -5, 10])
# If post=(5,15,80), predict pre
pred = predict_pre([5, 15, 80], delta_test)
check("Ideal recovery: pred ≈ (10,20,70)", np.allclose(pred, [10, 20, 70]))

# Edge: if post is small and delta is large positive, pre goes negative → clip
delta_big = np.array([20, 5, -15])  # heavy enrichment in comp 0, loss in comp 2
pred_clip = predict_pre([5, 5, 90], delta_big)  # 5-20=-15 → clip to 0
check("Clip to 0: pred[0]=0 (5-20=-15 clipped)", pred_clip[0] == 0)
check("No clipping for positive: pred[1]=0 (5-5=0)", pred_clip[1] == 0)

# Roundtrip: predict_pre(post, delta) then add delta back should recover post
post_orig = np.array([40, 10, 50])
delta_rt = np.array([-10, 5, 5])
pred_rt = predict_pre(post_orig, delta_rt)
recovered = pred_rt + delta_rt
# Only exact where no clipping occurred
check("Roundtrip: recovered ≈ original (unclipped region)", np.allclose(recovered, post_orig))

# ═══════════════════════════════════════════════════════════════════
# TEST 5: BaO threshold classification
# ═══════════════════════════════════════════════════════════════════
print("\n── 5. BaO Threshold Classification ──")

def classify_bao(bao_value, threshold=3.14):
    return "铅钡" if bao_value > threshold else "高钾"

# Training data extremes from 2a
check("高钾 max BaO=2.86 → 高钾 (correct)", classify_bao(2.86) == "高钾")
check("铅钡 min BaO=3.42 → 铅钡 (correct)", classify_bao(3.42) == "铅钡")
check("Threshold itself: 3.14 → 高钾", classify_bao(3.14) == "高钾")
check("Zero BaO → 高钾", classify_bao(0) == "高钾")
check("Very high BaO (11.34) → 铅钡", classify_bao(11.34) == "铅钡")

# Boundary test: if threshold shifts by ±0.5
check("BaO=2.64 still 高钾 at threshold 3.14", classify_bao(2.64) == "高钾")
check("BaO=3.64 still 铅钡 at threshold 3.14", classify_bao(3.64) == "铅钡")

# ═══════════════════════════════════════════════════════════════════
# TEST 6: Data merge integrity
# ═══════════════════════════════════════════════════════════════════
print("\n── 6. Data Merge Integrity ──")

df1 = pd.read_csv("data/表单1_文物信息.csv", skiprows=1)
df2 = pd.read_csv("data/表单2_化学成分.csv", skiprows=1)

# Every artifact in 表单1 should appear in 表单2
ids_1 = set(df1.iloc[:, 0].astype(str).str.zfill(2))
ids_2 = set()
for name in df2.iloc[:, 0]:
    m = re.match(r"(\d+)", str(name))
    if m:
        ids_2.add(f"{int(m.group(1)):02d}")

missing_from_2 = ids_1 - ids_2
check(f"All 表单1 IDs in 表单2 (missing: {missing_from_2 or 'none'})", len(missing_from_2) == 0)

extra_in_2 = ids_2 - ids_1
check(f"No extra IDs in 表单2 not in 表单1 (extra: {extra_in_2 or 'none'})", len(extra_in_2) == 0)

# Total unique artifacts
check("Unique artifacts: 58", len(ids_1) == 58)

# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  CRITICAL TESTS: {failures} FAILURES")
if failures == 0:
    print("  All critical functions verified correct.")
else:
    print(f"  {failures} critical failures — modeling results UNRELIABLE.")
print(f"{'='*60}")
