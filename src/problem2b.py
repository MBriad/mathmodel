"""Problem 2b: Subclassification within glass types via PCA + K-means."""
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from data_loader import OXIDE_SHORT, load_and_merge

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

valid, oxide_cols = load_and_merge()

# ── Subclassification per type ─────────────────────────────────────
results = {}

for glass_type in ["高钾", "铅钡"]:
    print(f"\n{'='*60}")
    print(f"  {glass_type} 亚类分析")
    print(f"{'='*60}")

    data = valid[valid["类型"] == glass_type].copy()
    X = data[oxide_cols].values

    # Exclude near-zero-variance features
    var = X.var(axis=0)
    active_mask = var > 0.01
    active_cols = [oxide_cols[i] for i in range(14) if active_mask[i]]
    active_short = [OXIDE_SHORT[i] for i in range(14) if active_mask[i]]
    X_active = X[:, active_mask]

    print(f"  Samples: {len(X)}, Active features: {len(active_cols)}/{14}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_active)

    # PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_components = max(2, np.searchsorted(cum_var, 0.85) + 1)
    print(f"  PCA components for 85% variance: {n_components}")
    print("  Top 3 PC loadings:")
    for pc in range(min(3, len(active_short))):
        loadings = pd.Series(pca.components_[pc], index=active_short).sort_values(key=abs, ascending=False)
        print(f"    PC{pc+1}: {', '.join(f'{k}({v:+.2f})' for k,v in loadings.head(4).items())}")

    # K-means: find optimal k
    X_use = X_pca[:, :n_components]
    inertias = []
    sil_scores = []
    max_k = max(2, min(4, len(X) // 6 + 2))
    k_range = range(2, max_k + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X_use)
        inertias.append(km.inertia_)
        if len(set(labels)) > 1:
            sil_scores.append(silhouette_score(X_use, labels))
        else:
            sil_scores.append(0)

    # Best k by silhouette
    best_k = k_range[np.argmax(sil_scores)]
    best_sil = max(sil_scores)

    print(f"  Best k: {best_k} (silhouette={best_sil:.3f})")
    print(f"  Silhouette scores: {dict(zip(k_range, [f'{s:.3f}' for s in sil_scores]))}")

    # Final clustering
    km = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    cluster_labels = km.fit_predict(X_use)
    data["亚类"] = [f"{glass_type}-{chr(65+c)}" for c in cluster_labels]  # A, B, C...

    # Subclass profiles
    print("\n  亚类均值轮廓:")
    for sub in sorted(data["亚类"].unique()):
        sub_data = data[data["亚类"] == sub]
        print(f"    {sub} (n={len(sub_data)}):")
        means = sub_data[oxide_cols].mean()
        for i, short in enumerate(OXIDE_SHORT):
            if means.iloc[i] > 1:
                print(f"      {short}: {means.iloc[i]:.1f}%")

    results[glass_type] = {
        "n_samples": len(X),
        "active_features": active_short,
        "best_k": int(best_k),
        "silhouette": float(best_sil),
        "cluster_labels": cluster_labels.tolist(),
        "pca_explained_var": cum_var[:n_components].tolist(),
    }

    # ── Visualization ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Subplot 1: PCA scatter with clusters
    colors = plt.cm.tab10(np.linspace(0, 1, best_k))
    for c in range(best_k):
        mask = cluster_labels == c
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=[colors[c]], label=f"{glass_type}-{chr(65+c)}", s=50, edgecolors="white")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axes[0].set_title(f"{glass_type} PCA (k={best_k})")
    axes[0].legend(fontsize=8)

    # Subplot 2: Elbow + Silhouette
    ax2 = axes[1]
    ax2.plot(list(k_range), inertias, "b-o", label="Inertia")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Inertia", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")
    ax3 = ax2.twinx()
    ax3.plot(list(k_range), sil_scores, "r-s", label="Silhouette")
    ax3.set_ylabel("Silhouette Score", color="red")
    ax3.tick_params(axis="y", labelcolor="red")
    ax2.set_title(f"{glass_type} 肘部法则 + 轮廓系数")

    # Subplot 3: Radar chart for subclasses
    ax = fig.add_subplot(1, 3, 3, polar=True)
    angles = np.linspace(0, 2 * np.pi, len(active_short), endpoint=False).tolist()
    angles += angles[:1]

    for c in range(best_k):
        sub_data = data[data["亚类"] == f"{glass_type}-{chr(65+c)}"]
        means = sub_data[[c for i, c in enumerate(oxide_cols) if active_mask[i]]].mean()
        values = means.values.tolist() + [means.values[0]]
        ax.fill(angles, values, alpha=0.1, color=colors[c])
        ax.plot(angles, values, color=colors[c], linewidth=1.5, label=f"{chr(65+c)} (n={len(sub_data)})")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(active_short, fontsize=8)
    ax.set_yticklabels([])
    ax.set_title(f"{glass_type} 亚类雷达图", pad=20)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(f"analysis/fig2b_{glass_type}_subclass.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: analysis/fig2b_{glass_type}_subclass.png")

    # ── Sensitivity: Bootstrap ─────────────────────────────────────
    ari_scores = []
    for seed in range(100):
        rng = np.random.RandomState(seed)
        idx_boot = rng.choice(len(X_use), size=len(X_use), replace=True)
        X_boot = X_use[idx_boot]
        try:
            km_boot = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            labels_boot = km_boot.fit_predict(X_boot)
            if len(np.unique(labels_boot)) > 1:
                ari = adjusted_rand_score(cluster_labels[idx_boot], labels_boot)
                ari_scores.append(ari)
        except Exception:
            pass

    # Perturbation sensitivity
    ari_perturb = []
    for seed in range(100):
        rng = np.random.RandomState(seed)
        X_noisy = X_use + rng.normal(0, 0.05 * X_use.std(axis=0), X_use.shape)
        try:
            km_noisy = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            labels_noisy = km_noisy.fit_predict(X_noisy)
            if len(np.unique(labels_noisy)) > 1:
                ari = adjusted_rand_score(cluster_labels, labels_noisy)
                ari_perturb.append(ari)
        except Exception:
            pass

    mean_ari_boot = np.mean(ari_scores) if ari_scores else 0
    mean_ari_perturb = np.mean(ari_perturb) if ari_perturb else 0
    print(f"  Bootstrap ARI: {mean_ari_boot:.3f} ({len(ari_scores)} valid)")
    print(f"  Perturbation ARI: {mean_ari_perturb:.3f} ({len(ari_perturb)} valid)")

    results[glass_type]["bootstrap_ari"] = float(mean_ari_boot)
    results[glass_type]["perturbation_ari"] = float(mean_ari_perturb)

# Save
with open("analysis/problem2b_results.json", "w", encoding="utf-8") as f:
    safe = {}
    for k, v in results.items():
        safe[k] = {}
        for sk, sv in v.items():
            if isinstance(sv, (np.integer,)):
                safe[k][sk] = int(sv)
            elif isinstance(sv, (np.floating,)):
                safe[k][sk] = float(sv)
            elif isinstance(sv, list):
                safe[k][sk] = [float(x) if isinstance(x, (np.floating, np.integer)) else str(x) for x in sv]
            else:
                safe[k][sk] = sv
    json.dump(safe, f, ensure_ascii=False, indent=2)
print("\nSaved: analysis/problem2b_results.json")
