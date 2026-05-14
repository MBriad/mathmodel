"""Problem 1c: Predict pre-weathering composition from weathered samples."""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loader import OXIDE_SHORT, load_and_merge

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

valid, oxide_cols = load_and_merge()

# ── Step 1: Pb-Ba paired correction vectors ────────────────────────
# 49, 50 have both 未风化点 (pre) and 普通 (post) samples on same artifact
pb_pairs = []
for art_id in ["49", "50"]:
    sub = valid[valid.index == art_id]
    pre_sample = sub[sub["采样类型"] == "未风化点"]
    post_sample = sub[sub["采样类型"] == "普通"]
    if len(pre_sample) == 1 and len(post_sample) == 1:
        pb_pairs.append({
            "artifact": art_id,
            "pre": pre_sample[oxide_cols].iloc[0],
            "post": post_sample[oxide_cols].iloc[0],
        })

print("Step 1 — Pb-Ba paired calibration (pre = 未风化点, post = 风化表面):")
corrections_pb = []
for pair in pb_pairs:
    delta = pair["post"] - pair["pre"]  # weathering effect: pre → post
    corrections_pb.append(delta)
    print(f"\n{pair['artifact']} 风化效应 (post - pre):")
    for ox, d in delta.items():
        short = OXIDE_SHORT[oxide_cols.index(ox)]
        if abs(d) > 1:
            print(f"  {short}: {d:+.2f}%")

# Mean correction vector for Pb-Ba
mean_delta_pb = pd.DataFrame(corrections_pb).mean()
print("\nPb-Ba 平均校正向量 (风化效应Δ = post - pre):")
print("  " + " ".join(f"{OXIDE_SHORT[i]}:{mean_delta_pb.iloc[i]:+.2f}" for i in range(14)))

# Validate direction with 08/26/54 severe pairs
print("\n方向验证 (08/26/54 严重风化点 vs 普通点):")
severe_pairs = []
for art_id in ["08", "26", "54"]:
    sub = valid[valid.index == art_id]
    severe = sub[sub["采样类型"] == "严重风化点"]
    reg = sub[sub["采样类型"] == "普通"]
    if len(severe) == 1 and len(reg) == 1:
        s_delta = severe[oxide_cols].iloc[0] - reg[oxide_cols].iloc[0]
        severe_pairs.append(s_delta)

severe_mean = pd.DataFrame(severe_pairs).mean()
# Check sign agreement (two-tier: all oxides, and meaningful deltas only)
sign_match = (np.sign(mean_delta_pb.values) == np.sign(severe_mean.values)).sum()
nonzero_mask = (np.abs(mean_delta_pb.values) > 0.1) | (np.abs(severe_mean.values) > 0.1)
sign_match_nz = (np.sign(mean_delta_pb.values[nonzero_mask]) == np.sign(severe_mean.values[nonzero_mask])).sum()
n_nz = nonzero_mask.sum()
print(f"  方向一致: {sign_match}/14 (含零值), {sign_match_nz}/{n_nz} (仅|Δ|>0.1)")
mismatched = []
for i, short in enumerate(OXIDE_SHORT):
    pb_sign = np.sign(mean_delta_pb.iloc[i])
    sv_sign = np.sign(severe_mean.iloc[i])
    match = "OK" if pb_sign == sv_sign else "!!"
    if nonzero_mask[i] and pb_sign != sv_sign:
        mismatched.append(short)
    if abs(mean_delta_pb.iloc[i]) > 0.1 or abs(severe_mean.iloc[i]) > 0.1:
        print(f"    {short}: 配对Δ={mean_delta_pb.iloc[i]:+.2f}, 严重点Δ={severe_mean.iloc[i]:+.2f} {match}")
if mismatched:
    print(f"  方向不一致氧化物: {', '.join(mismatched)}")
    print("  提示: 严重风化(08/26/54)与中度风化(49/50)可能遵循不同风化轨迹")

# ── Step 2: High-K group correction ────────────────────────────────
hk_weathered = valid[(valid["类型"] == "高钾") & (valid["表面风化"] == "风化")]
hk_unweathered = valid[(valid["类型"] == "高钾") & (valid["表面风化"] == "无风化")]
mean_delta_hk = hk_weathered[oxide_cols].mean() - hk_unweathered[oxide_cols].mean()

print("\nStep 2 — 高钾组间校正向量 (风化组均值 - 未风化组均值):")
print("  " + " ".join(f"{OXIDE_SHORT[i]}:{mean_delta_hk.iloc[i]:+.2f}" for i in range(14)))

# ── Step 3: Predict & validate ─────────────────────────────────────
# For Pb-Ba weathered samples: pre ≈ post - mean_delta_pb (reverse weathering)
# For high-K weathered samples: pre ≈ post - mean_delta_hk

# Apply prediction to weathered samples only
weathered_samples = valid[valid["表面风化"] == "风化"].copy()
predicted = pd.DataFrame(index=weathered_samples.index, columns=oxide_cols, dtype=float)

for i in range(len(weathered_samples)):
    row = weathered_samples.iloc[i]
    typ = row["类型"]
    delta = mean_delta_pb if typ == "铅钡" else mean_delta_hk
    pre_vals = row[oxide_cols].values - delta.values
    pre_vals = np.clip(pre_vals, 0, 100)
    predicted.iloc[i] = pre_vals

w_sum = weathered_samples[oxide_cols].sum(axis=1)
p_sum = predicted.sum(axis=1)

print("\nStep 3 — Predictions for weathered samples:")
print(f"Total weathered samples: {len(weathered_samples)}")
print(f"风化后总和范围: {w_sum.min():.1f}% - {w_sum.max():.1f}%")
print(f"预测前总和范围: {p_sum.min():.1f}% - {p_sum.max():.1f}%")

# Validate against actual unweathered spots on 49/50
print("\n── Validation: predicted vs actual 未风化点 ──")
for art_id in ["49", "50"]:
    sub = valid[valid.index == art_id]
    actual_pre = sub[sub["采样类型"] == "未风化点"][oxide_cols].iloc[0]
    predicted_pre = predicted.loc[art_id]
    if isinstance(predicted_pre, pd.DataFrame):
        predicted_pre = predicted_pre.iloc[0]

    mae = (predicted_pre - actual_pre).abs().mean()
    print(f"\n{art_id}: MAE = {mae:.2f}%")
    for ox in oxide_cols:
        short = OXIDE_SHORT[oxide_cols.index(ox)]
        a = actual_pre[ox]
        p = predicted_pre[ox]
        if max(a, p) > 1:
            print(f"  {short}: 实际={a:.2f}%, 预测={p:.2f}%, 误差={p-a:+.2f}%")

# ── Visualization ──────────────────────────────────────────────────
# Compare weathered vs predicted pre-weathering for top changes
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot: sample weathering comparison
for ax_idx, (art_id, typ_label) in enumerate([("49", "铅钡-49"), ("07", "高钾-07")]):
    ax = axes[ax_idx][0]
    sub = valid[valid.index == art_id]
    weathered = sub[sub["采样类型"] != "未风化点"][oxide_cols].iloc[0]
    if art_id in predicted.index:
        pre_row = predicted.loc[art_id]
        if isinstance(pre_row, pd.DataFrame):
            pre_row = pre_row.iloc[0]
        pre_vals = pre_row[oxide_cols].values
    else:
        ax.set_visible(False)
        continue

    x = range(len(OXIDE_SHORT))
    width = 0.35
    ax.bar([i - width/2 for i in x], weathered.values, width, label="风化后(实测)", color="#FF5722", alpha=0.8)
    ax.bar([i + width/2 for i in x], pre_vals, width, label="风化前(预测)", color="#4CAF50", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(OXIDE_SHORT, fontsize=8, rotation=45)
    ax.set_ylabel("含量 (%)")
    ax.set_title(f"{typ_label} 风化前后对比")
    ax.legend(fontsize=8)

# Subplot 3: Pb-Ba correction vector
ax = axes[0][1]
colors = ["#FF5722" if v > 0 else "#4CAF50" for v in mean_delta_pb.values]
ax.barh(OXIDE_SHORT, mean_delta_pb.values, color=colors)
ax.axvline(x=0, color="black", linewidth=0.5)
ax.set_title("铅钡校正向量 (Δ = 风化后 - 风化前)")
ax.set_xlabel("Δ 含量 (%)")

# Subplot 4: High-K correction vector
ax = axes[1][1]
colors = ["#FF5722" if v > 0 else "#4CAF50" for v in mean_delta_hk.values]
ax.barh(OXIDE_SHORT, mean_delta_hk.values, color=colors)
ax.axvline(x=0, color="black", linewidth=0.5)
ax.set_title("高钾校正向量 (Δ = 风化后 - 风化前)")
ax.set_xlabel("Δ 含量 (%)")

plt.tight_layout()
plt.savefig("analysis/fig1c_predictions.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: analysis/fig1c_predictions.png")

# ── Save prediction results ────────────────────────────────────────
output = pd.DataFrame(index=weathered_samples.index)
output["采样点"] = weathered_samples.iloc[:, 0]
output["类型"] = weathered_samples["类型"]
for ox, short in zip(oxide_cols, OXIDE_SHORT):
    output[f"{short}_风化后"] = weathered_samples[ox].values
    output[f"{short}_预测前"] = predicted[ox].values
output["总和_风化后"] = w_sum.values
output["总和_预测前"] = p_sum.values
output.to_csv("analysis/problem1c_predictions.csv", encoding="utf-8-sig")
print("Saved: analysis/problem1c_predictions.csv")

# Summary stats
print("\n── 校正摘要 ──")
print(f"Pb-Ba校正来源: 配对样本 ({len(pb_pairs)} 对: {[p['artifact'] for p in pb_pairs]})")
print(f"Pb-Ba预测样本数: {(weathered_samples['类型']=='铅钡').sum()}")
print(f"高钾校正来源: 组间均值差 (风化组 n={len(hk_weathered)}, 未风化组 n={len(hk_unweathered)})")
print(f"高钾预测样本数: {(weathered_samples['类型']=='高钾').sum()}")
