"""Problem 1b: Chemical composition statistics — weathered vs unweathered, by glass type."""
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from data_loader import OXIDE_SHORT, OXIDES, load_and_merge

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

valid, oxide_cols = load_and_merge()

# ── Step 3: Group statistics ───────────────────────────────────────
def group_stats(data, group_cols):
    """Mean ± std for each oxide per group."""
    return data.groupby(group_cols)[oxide_cols].agg(["mean", "std", "median"])

# 4-group: type × weathering
stats = group_stats(valid, ["类型", "表面风化"])
means = stats.xs("mean", axis=1, level=1)
stds = stats.xs("std", axis=1, level=1)
medians = stats.xs("median", axis=1, level=1)

print("Step 3 — Group means (4 groups):")
for (typ, weath), row in means.iterrows():
    print(f"\n{typ}-{weath}:")
    for ox, short in zip(OXIDES, OXIDE_SHORT):
        print(f"  {short}: {row[ox]:.2f}%")

# ── Step 4: Weathering difference analysis ──────────────────────────
# Delta = weathered mean - unweathered mean, per type
delta_data = {}
for typ in ["高钾", "铅钡"]:
    w_mean = means.loc[(typ, "风化")]
    uw_mean = means.loc[(typ, "无风化")]
    delta = w_mean - uw_mean
    rel_change = (delta / uw_mean.replace(0, np.nan) * 100).round(1)
    delta_data[typ] = {"delta": delta, "rel_change_pct": rel_change}

    print(f"\nStep 4 — {typ} 风化差异 (Top changes):")
    changes = pd.DataFrame({"Δ": delta, "Δ%": rel_change}).sort_values("Δ%", key=abs, ascending=False)
    print(changes.head(6).to_string())

# Paired validation: same artifact with severe + unweathered spots
paired_artifacts = ["08", "26", "54"]  # artifacts with 严重风化点
paired_data = valid[valid.index.isin(paired_artifacts)]
print(f"\nPaired samples available: {paired_data.index.tolist()}")
for art_id in paired_artifacts:
    sub = paired_data[paired_data.index == art_id]
    if len(sub) >= 2:
        severe = sub[sub["采样类型"] == "严重风化点"]
        reg = sub[sub["采样类型"] == "普通"]
        if len(severe) > 0 and len(reg) > 0:
            print(f"\n{art_id} 严重风化点 vs 普通采样点:")
            diff = severe[oxide_cols].iloc[0] - reg[oxide_cols].iloc[0]
            top = diff.abs().sort_values(ascending=False).head(5)
            for ox, d in top.items():
                print(f"  {ox}: Δ={d:.2f}%")

# ── Step 5: Visualization ──────────────────────────────────────────
# 5a. Boxplots for key oxides
key_oxides_short = ["SiO2", "K2O", "PbO", "BaO", "Na2O", "CaO", "Al2O3", "Fe2O3", "CuO", "P2O5"]
key_oxides_full = [OXIDES[OXIDE_SHORT.index(s)] for s in key_oxides_short]

fig, axes = plt.subplots(2, 5, figsize=(18, 8))
for i, (ox, ax) in enumerate(zip(key_oxides_full, axes.flat)):
    plot_data = [
        valid[(valid["类型"] == "高钾") & (valid["表面风化"] == "无风化")][ox].dropna(),
        valid[(valid["类型"] == "高钾") & (valid["表面风化"] == "风化")][ox].dropna(),
        valid[(valid["类型"] == "铅钡") & (valid["表面风化"] == "无风化")][ox].dropna(),
        valid[(valid["类型"] == "铅钡") & (valid["表面风化"] == "风化")][ox].dropna(),
    ]
    bp = ax.boxplot(plot_data, patch_artist=True, widths=0.6)
    colors = ["#4CAF50", "#FF5722", "#4CAF50", "#FF5722"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title(OXIDE_SHORT[OXIDES.index(ox)], fontsize=11)
    ax.set_xticklabels(["高钾\n无风化", "高钾\n风化", "铅钡\n无风化", "铅钡\n风化"], fontsize=7)
    ax.tick_params(axis="y", labelsize=8)

plt.suptitle("重点氧化物分布：风化 vs 未风化 × 玻璃类型", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("analysis/fig1b_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: analysis/fig1b_boxplots.png")

# 5b. Radar chart
angles = np.linspace(0, 2 * np.pi, len(OXIDES), endpoint=False).tolist()
angles += angles[:1]  # close

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for typ, weath, color, ls in [
    ("高钾", "无风化", "#4CAF50", "-"),
    ("高钾", "风化", "#FF5722", "-"),
    ("铅钡", "无风化", "#2196F3", "-"),
    ("铅钡", "风化", "#FF9800", "-"),
]:
    mean_vals = means.loc[(typ, weath)].values
    values = mean_vals.tolist() + [mean_vals[0]]
    ax.fill(angles, values, alpha=0.1, color=color)
    ax.plot(angles, values, color=color, linewidth=1.5, linestyle=ls, label=f"{typ}-{weath}")

ax.set_xticks(angles[:-1])
ax.set_xticklabels(OXIDE_SHORT, fontsize=8)
ax.set_yticklabels([])
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
ax.set_title("四组文物化学成分雷达图", fontsize=13, pad=25)
plt.savefig("analysis/fig1b_radar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: analysis/fig1b_radar.png")

# 5c. Difference bar chart
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, typ in zip(axes, ["高钾", "铅钡"]):
    deltas = delta_data[typ]["delta"]
    colors = ["#FF5722" if v > 0 else "#4CAF50" for v in deltas]
    bars = ax.bar(OXIDE_SHORT, deltas.values, color=colors, edgecolor="white")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_title(f"{typ}玻璃: 风化后 − 风化前 成分差值")
    ax.set_ylabel("Δ 含量 (%)")
    ax.tick_params(axis="x", rotation=45)
    for bar, val in zip(bars, deltas.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + (0.3 if val >= 0 else -0.8),
                f"{val:+.1f}", ha="center", fontsize=7)

plt.tight_layout()
plt.savefig("analysis/fig1b_diff_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: analysis/fig1b_diff_bar.png")

# ── Step 6: Statistical tests ──────────────────────────────────────
print("\nStep 6 — Mann-Whitney U tests (Bonferroni corrected):")
test_results = []
for typ in ["高钾", "铅钡"]:
    w_data = valid[(valid["类型"] == typ) & (valid["表面风化"] == "风化")][oxide_cols]
    uw_data = valid[(valid["类型"] == typ) & (valid["表面风化"] == "无风化")][oxide_cols]
    for ox, short in zip(oxide_cols, OXIDE_SHORT):
        x = w_data[ox].dropna()
        y = uw_data[ox].dropna()
        if len(x) >= 3 and len(y) >= 3:
            stat, p = mannwhitneyu(x, y, alternative="two-sided")
            test_results.append({"类型": typ, "氧化物": short, "U": stat, "p": p})

df_tests = pd.DataFrame(test_results)
df_tests["p_corrected"] = df_tests["p"] * len(df_tests)  # Bonferroni
df_tests["p_corrected"] = df_tests["p_corrected"].clip(upper=1.0)
df_tests["significant"] = df_tests["p_corrected"] < 0.05

print("Significant oxides (Bonferroni α=0.05):")
sig = df_tests[df_tests["significant"]]
print(sig[["类型", "氧化物", "p", "p_corrected"]].to_string(index=False))
print(f"\nTotal significant: {len(sig)}/{len(df_tests)}")

# Save results
results = {
    "valid_count": int(valid["valid"].sum()) if "valid" in valid.columns else len(valid),
    "delta": {typ: d["delta"].to_dict() for typ, d in delta_data.items()},
    "rel_change": {typ: d["rel_change_pct"].to_dict() for typ, d in delta_data.items()},
}

with open("analysis/problem1b_results.json", "w", encoding="utf-8") as f:
    safe = {}
    for k, v in results.items():
        if isinstance(v, dict):
            safe[k] = {str(k2): {str(k3): float(v3) if not isinstance(v3, str) else v3 for k3, v3 in v2.items()} for k2, v2 in v.items()}
        else:
            safe[k] = v
    json.dump(safe, f, ensure_ascii=False, indent=2)
print("\nResults saved: analysis/problem1b_results.json")
