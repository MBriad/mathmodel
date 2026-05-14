"""Problem 1a: Weathering vs glass type, decoration pattern, and color."""
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ── Step 1: Load & clean ──────────────────────────────────────────
df = pd.read_csv("data/表单1_文物信息.csv", skiprows=1)
assert df.shape == (58, 5), f"Expected 58 rows, got {df.shape[0]}"

results = {}  # store test results for summary

# ── Step 2: Weathering vs Type (2×2) ───────────────────────────────
ct_type = pd.crosstab(df["表面风化"], df["类型"])
print("Step 2 — 风化 × 类型 列联表:")
print(ct_type, "\n")

# 2×2 → Fisher exact preferred; chi2 with Yates correction also valid
_, p_chi2_type, _, _ = chi2_contingency(ct_type)
_, p_fisher_type = fisher_exact(ct_type)
print(f"Chi2 p = {p_chi2_type:.4f}, Fisher p = {p_fisher_type:.4f}")

results["风化×类型"] = {
    "table": ct_type.to_dict(),
    "chi2_p": round(float(p_chi2_type), 4),
    "fisher_p": round(float(p_fisher_type), 4),
    "significant": p_fisher_type < 0.05,
}
print(f"风化×类型 显著: {results['风化×类型']['significant']}\n")

# ── Step 3: Weathering vs Decoration (3×2) ─────────────────────────
ct_deco = pd.crosstab(df["表面风化"], df["纹饰"])
print("Step 3 — 风化 × 纹饰 列联表:")
print(ct_deco, "\n")

chi2_deco, p_deco, dof_deco, expected_deco = chi2_contingency(ct_deco)
n = ct_deco.sum().sum()
cramers_v = np.sqrt(chi2_deco / (n * (min(ct_deco.shape) - 1)))
print(f"Chi2 = {chi2_deco:.2f}, p = {p_deco:.4f}, Cramér's V = {cramers_v:.3f}")

results["风化×纹饰"] = {
    "table": ct_deco.to_dict(),
    "chi2": round(float(chi2_deco), 2),
    "p": round(float(p_deco), 4),
    "cramers_v": round(float(cramers_v), 3),
    "significant": p_deco < 0.05,
}
print(f"风化×纹饰 显著: {results['风化×纹饰']['significant']}\n")

# ── Step 4: Weathering vs Color ────────────────────────────────────
# Merge low-frequency colors: 黑(2) + 绿(1) → Other
df_color = df.copy()
rare_colors = df_color["颜色"].value_counts()
rare_colors = rare_colors[rare_colors < 5].index.tolist()
df_color["颜色_merged"] = df_color["颜色"].apply(
    lambda x: "其他" if x in rare_colors or pd.isna(x) else x
)

ct_color = pd.crosstab(df_color["表面风化"], df_color["颜色_merged"])
print("Step 4 — 风化 × 颜色 列联表 (低频合并):")
print(ct_color, "\n")

chi2_color, p_color, dof_color, expected_color = chi2_contingency(ct_color)
cramers_v_color = np.sqrt(chi2_color / (n * (min(ct_color.shape) - 1)))
print(f"Chi2 = {chi2_color:.2f}, p = {p_color:.4f}, Cramér's V = {cramers_v_color:.3f}")

results["风化×颜色"] = {
    "table": ct_color.to_dict(),
    "chi2": round(float(chi2_color), 2),
    "p": round(float(p_color), 4),
    "cramers_v": round(float(cramers_v_color), 3),
    "significant": p_color < 0.05,
    "merged_rare": rare_colors,
}
print(f"风化×颜色 显著: {results['风化×颜色']['significant']}")
print(f"低频颜色合并: {rare_colors}")

# Bonferroni correction for 3 simultaneous tests
alpha_corrected = 0.05 / 3
print(f"\n注: 同时进行了3次假设检验，Bonferroni校正后 α = 0.05/3 = {alpha_corrected:.4f}")
for factor, p_key in [("类型", "fisher_p"), ("纹饰", "p"), ("颜色", "p")]:
    p_val = results[f"风化×{factor}"][p_key]
    sig_str = "仍显著" if p_val < alpha_corrected else "不显著"
    print(f"  风化×{factor}: p={p_val:.4f} {'<' if p_val < alpha_corrected else '>'} {alpha_corrected:.4f} → {sig_str}")
print()

# ── Step 5: Combined visualization ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Subplot 1: 风化 × 类型
ct_type.plot(kind="bar", stacked=True, ax=axes[0], color=["#FF5722", "#4CAF50"], rot=0)
axes[0].set_title("风化状态 vs 玻璃类型")
axes[0].set_xlabel("表面风化")
axes[0].set_ylabel("文物数量")
axes[0].legend(title="类型")
for container in axes[0].containers:
    axes[0].bar_label(container, label_type="center", fontsize=9, color="white")

# Subplot 2: 风化 × 纹饰
ct_deco.plot(kind="bar", stacked=True, ax=axes[1], color=["#2196F3", "#FF9800", "#9C27B0"], rot=0)
axes[1].set_title("风化状态 vs 纹饰")
axes[1].set_xlabel("表面风化")
axes[1].set_ylabel("文物数量")
axes[1].legend(title="纹饰")
for container in axes[1].containers:
    axes[1].bar_label(container, label_type="center", fontsize=9, color="white")

# Subplot 3: 风化 × 颜色 (merged)
ct_color.plot(kind="bar", stacked=True, ax=axes[2], colormap="tab10", rot=0)
axes[2].set_title("风化状态 vs 颜色 (低频合并)")
axes[2].set_xlabel("表面风化")
axes[2].set_ylabel("文物数量")
axes[2].legend(title="颜色", fontsize=7)
for container in axes[2].containers:
    axes[2].bar_label(container, label_type="center", fontsize=8, color="white")

plt.tight_layout()
plt.savefig("analysis/fig1a_weathering_relations.png", dpi=150)
plt.close()
print("Figure saved: analysis/fig1a_weathering_relations.png")

# ── Summary table ──────────────────────────────────────────────────
summary = pd.DataFrame(
    {
        "Factor": ["玻璃类型", "纹饰", "颜色"],
        "Test": ["Fisher Exact", "Chi-square", "Chi-square"],
        "p-value": [
            results["风化×类型"]["fisher_p"],
            results["风化×纹饰"]["p"],
            results["风化×颜色"]["p"],
        ],
        "Effect Size": ["—", results["风化×纹饰"]["cramers_v"], results["风化×颜色"]["cramers_v"]],
        "Significant (α=0.05)": [
            "Yes" if results["风化×类型"]["significant"] else "No",
            "Yes" if results["风化×纹饰"]["significant"] else "No",
            "Yes" if results["风化×颜色"]["significant"] else "No",
        ],
        "Significant (Bonferroni)": [
            "Yes" if results["风化×类型"]["fisher_p"] < 0.05/3 else "No",
            "Yes" if results["风化×纹饰"]["p"] < 0.05/3 else "No",
            "Yes" if results["风化×颜色"]["p"] < 0.05/3 else "No",
        ],
    }
)
print("\n── Summary ──")
print(summary.to_string(index=False))

# ── Calc weathering rate per category ──────────────────────────────
print("\n── Weathering rates ──")
for col, label in [("类型", "玻璃类型"), ("纹饰", "纹饰"), ("颜色_merged", "颜色")]:
    rates = (
        pd.crosstab(df_color[col] if col == "颜色_merged" else df[col], df["表面风化"])
        .assign(Total=lambda x: x.sum(axis=1))
        .assign(风化率=lambda x: (x["风化"] / x["Total"] * 100).round(1))
    )
    print(f"\n{label}:")
    print(rates.to_string())

# Save results
with open("analysis/problem1a_results.json", "w", encoding="utf-8") as f:
    safe = {}
    for k, v in results.items():
        safe[k] = {}
        for sk, sv in v.items():
            if sk == "table":
                continue
            if isinstance(sv, (np.bool_,)):
                safe[k][sk] = bool(sv)
            elif isinstance(sv, (np.integer,)):
                safe[k][sk] = int(sv)
            elif isinstance(sv, (np.floating,)):
                safe[k][sk] = float(sv)
            elif isinstance(sv, list):
                safe[k][sk] = [str(x) for x in sv]
            else:
                safe[k][sk] = sv
    json.dump(safe, f, ensure_ascii=False, indent=2)
print("\nResults saved: analysis/problem1a_results.json")
