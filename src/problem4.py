"""Problem 4: Chemical composition correlation analysis within and between glass types."""
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from data_loader import OXIDE_SHORT, load_and_merge

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

valid, oxide_cols = load_and_merge()

# ── CLR transformation (centered log-ratio) ────────────────────────
# Handle zeros: replace with half the minimum detected value per oxide
X_raw = valid[oxide_cols].values
X_clr = np.zeros_like(X_raw)
for j in range(14):
    col = X_raw[:, j]
    col_pos = col[col > 0]
    min_pos = col_pos.min() if len(col_pos) > 0 else 0.01
    col_adj = np.where(col <= 0, min_pos / 2, col)
    X_clr[:, j] = col_adj

# CLR: log(x / g) where g = geometric mean
gm = np.exp(np.mean(np.log(X_clr), axis=1, keepdims=True))
X_clr = np.log(X_clr / gm)

# ── Step 1: Correlation matrices ───────────────────────────────────
# Use CLR-transformed data for Pearson, raw for Spearman
def compute_corr(data, method="pearson"):
    if method == "pearson":
        corr = np.corrcoef(data.T)
    else:
        corr, _ = spearmanr(data)
    return pd.DataFrame(corr, index=OXIDE_SHORT, columns=OXIDE_SHORT)

results = {}
for glass_type in ["高钾", "铅钡"]:
    mask = valid["类型"] == glass_type
    X_type = X_clr[mask.values]

    pearson_corr = compute_corr(X_type, "pearson")
    spearman_corr = compute_corr(valid[mask][oxide_cols].values, "spearman")

    results[glass_type] = {
        "n_samples": int(mask.sum()),
        "pearson": pearson_corr,
        "spearman": spearman_corr,
    }

    print(f"\n{glass_type}: n={results[glass_type]['n_samples']}")

    # Top correlations
    print("  Top 5 positive correlations:")
    upper = pearson_corr.where(np.triu(np.ones(pearson_corr.shape), k=1).astype(bool))
    pairs = upper.stack().dropna().sort_values(ascending=False)
    for (a, b), v in pairs.head(5).items():
        print(f"    {a} — {b}: {v:.3f}")

    print("  Top 5 negative correlations:")
    neg_pairs = pairs.sort_values(ascending=True)
    for (a, b), v in neg_pairs.head(5).items():
        print(f"    {a} — {b}: {v:.3f}")

# ── Step 2: Heatmaps ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

for row, glass_type in enumerate(["高钾", "铅钡"]):
    for col, (method, cmap) in enumerate([("pearson", "RdBu_r"), ("spearman", "RdBu_r")]):
        ax = axes[row][col]
        corr_mat = results[glass_type][method]
        im = ax.imshow(corr_mat.values, cmap=cmap, vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks(range(14))
        ax.set_yticks(range(14))
        ax.set_xticklabels(OXIDE_SHORT, fontsize=8, rotation=45, ha="right")
        ax.set_yticklabels(OXIDE_SHORT, fontsize=8)
        ax.set_title(f"{glass_type} ({method}, n={results[glass_type]['n_samples']})", fontsize=12)
        # Annotate strong correlations
        for i in range(14):
            for j in range(14):
                if i != j and abs(corr_mat.iloc[i, j]) > 0.6:
                    ax.text(j, i, f"{corr_mat.iloc[i, j]:.1f}", ha="center", va="center", fontsize=6,
                            color="white" if abs(corr_mat.iloc[i, j]) > 0.8 else "black")

fig.subplots_adjust(right=0.91)
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
fig.colorbar(im, cax=cbar_ax, label="Correlation")
plt.suptitle("化学成分关联矩阵: 高钾 vs 铅钡", fontsize=14, y=1.01)
plt.savefig("analysis/fig4_correlation_heatmaps.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: analysis/fig4_correlation_heatmaps.png")

# ── Step 3: Difference analysis ────────────────────────────────────
print("\n── Correlation Differences (铅钡 - 高钾) ──")

# Fisher z-transformation for comparing Pearson correlations
def fisher_z(r):
    r_clipped = np.clip(r, -0.999, 0.999)
    return 0.5 * np.log((1 + r_clipped) / (1 - r_clipped))

pearson_hk = results["高钾"]["pearson"].values
pearson_pb = results["铅钡"]["pearson"].values
n_hk = results["高钾"]["n_samples"]
n_pb = results["铅钡"]["n_samples"]

z_hk = fisher_z(pearson_hk)
z_pb = fisher_z(pearson_pb)
se = np.sqrt(1/(n_hk - 3) + 1/(n_pb - 3))
z_diff = (z_pb - z_hk) / se

diff_matrix = pearson_pb - pearson_hk
diff_df = pd.DataFrame(diff_matrix, index=OXIDE_SHORT, columns=OXIDE_SHORT)
z_df = pd.DataFrame(z_diff, index=OXIDE_SHORT, columns=OXIDE_SHORT)

# Significant differences (|z| > 2)
sig_diffs = []
for i in range(14):
    for j in range(i+1, 14):
        if abs(z_diff[i, j]) > 2.0:
            sig_diffs.append({
                "pair": f"{OXIDE_SHORT[i]}-{OXIDE_SHORT[j]}",
                "高钾_r": round(float(pearson_hk[i, j]), 3),
                "铅钡_r": round(float(pearson_pb[i, j]), 3),
                "diff": round(float(diff_matrix[i, j]), 3),
                "z": round(float(z_diff[i, j]), 2),
            })

sig_diffs = sorted(sig_diffs, key=lambda x: abs(x["diff"]), reverse=True)
print(f"Significant differences (|z| > 2): {len(sig_diffs)}")
for d in sig_diffs[:10]:
    print(f"  {d['pair']}: 高钾={d['高钾_r']:.3f}, 铅钡={d['铅钡_r']:.3f}, Δ={d['diff']:+.3f}, z={d['z']:.2f}")

# ── Step 4: Difference heatmap + network ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Difference heatmap
im = axes[0].imshow(diff_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
axes[0].set_xticks(range(14))
axes[0].set_yticks(range(14))
axes[0].set_xticklabels(OXIDE_SHORT, fontsize=8, rotation=45, ha="right")
axes[0].set_yticklabels(OXIDE_SHORT, fontsize=8)
axes[0].set_title("相关系数差异 (铅钡 - 高钾)", fontsize=12)
# Annotate significant diffs
for i in range(14):
    for j in range(14):
        if i != j and abs(z_diff[i, j]) > 2.0:
            axes[0].text(j, i, f"{diff_matrix[i, j]:+.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(diff_matrix[i, j]) > 0.5 else "black", fontweight="bold")
plt.colorbar(im, ax=axes[0], shrink=0.8)

# Network of strong correlations
# Show edges where |r| > 0.5 in at least one type
ax = axes[1]
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.axis("off")
ax.set_title("强关联网络 (|r| > 0.5)", fontsize=12)

# Node positions on circle
angles = np.linspace(0, 2*np.pi, 14, endpoint=False)
pos = {s: (np.cos(a), np.sin(a)) for s, a in zip(OXIDE_SHORT, angles)}

# Draw nodes (dots at r=1) and labels (outside at r=1.18)
for short, (x, y) in pos.items():
    ax.scatter(x, y, s=200, c="lightgray", edgecolors="black", zorder=5)
    ax.annotate(short, (1.18 * x, 1.18 * y), ha="center", va="center",
                fontsize=9, fontweight="bold")

# Draw edges
for i in range(14):
    for j in range(i+1, 14):
        r_hk = pearson_hk[i, j]
        r_pb = pearson_pb[i, j]
        if abs(r_hk) > 0.5 or abs(r_pb) > 0.5:
            x1, y1 = pos[OXIDE_SHORT[i]]
            x2, y2 = pos[OXIDE_SHORT[j]]
            # Color: blue if both agree, red if only one type has it, purple if both
            both_pos = r_hk > 0.5 and r_pb > 0.5
            both_neg = r_hk < -0.5 and r_pb < -0.5
            only_hk = (abs(r_hk) > 0.5) and (abs(r_pb) <= 0.5)
            only_pb = (abs(r_pb) > 0.5) and (abs(r_hk) <= 0.5)

            if both_pos or both_neg:
                color, alpha, lw = "green" if both_pos else "red", 0.8, 2.0
                label = "both"
            elif only_hk:
                color, alpha, lw = "#4CAF50", 0.5, 1.0
                label = "高钾only"
            elif only_pb:
                color, alpha, lw = "#FF5722", 0.5, 1.0
                label = "铅钡only"
            else:
                continue
            ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=lw, zorder=1)

# Legend
legend_elements = [
    Line2D([0], [0], color="green", lw=2, label="两类共有强关联"),
    Line2D([0], [0], color="red", lw=2, label="两类共有强负关联"),
    Line2D([0], [0], color="#4CAF50", lw=1, label="仅高钾"),
    Line2D([0], [0], color="#FF5722", lw=1, label="仅铅钡"),
]
ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

plt.tight_layout()
plt.savefig("analysis/fig4_difference_network.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: analysis/fig4_difference_network.png")

# ── Summary ────────────────────────────────────────────────────────
print("\n── Summary ──")
print(f"高钾: {n_hk} samples, 铅钡: {n_pb} samples")
print(f"Significant correlation differences: {len(sig_diffs)} oxide pairs")

# Weathering-stratified analysis
print("\n── Weathering-stratified correlations ──")
for glass_type in ["高钾", "铅钡"]:
    for weath in ["无风化", "风化"]:
        mask = (valid["类型"] == glass_type) & (valid["表面风化"] == weath)
        n = mask.sum()
        if n >= 4:
            X_sub = X_clr[mask.values]
            c = np.corrcoef(X_sub.T)
            # Average absolute correlation
            upper_tri = c[np.triu_indices(14, k=1)]
            avg_abs_r = np.abs(upper_tri).mean()
            print(f"  {glass_type}-{weath} (n={n}): avg |r| = {avg_abs_r:.3f}")
        else:
            print(f"  {glass_type}-{weath} (n={n}): insufficient samples")

# Save
with open("analysis/problem4_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "n_high_K": n_hk,
        "n_Pb_Ba": n_pb,
        "significant_diffs": sig_diffs[:15],
        "total_sig_diffs": len(sig_diffs),
    }, f, ensure_ascii=False, indent=2)
print("\nSaved: analysis/problem4_results.json")
