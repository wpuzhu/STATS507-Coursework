"""Step 1: cohort description, harmonization diagnostics and exploratory analysis.

  * cohort: complete-case patients by center, acquisition batch and LVI;
  * harmonization diagnostics (label-free, whole cohort): for each setting and
    modality, the share of features whose location differs by batch
    (Kruskal-Wallis, BH q < 0.05), the PERMANOVA R^2 of batch and the
    silhouette of batch labels in the first 10 principal components;
  * cross-modality agreement on the training split, as in the original
    notebook: global structure (correlation between the pairwise patient
    distance matrices of two modalities) and local structure (t-SNE).
"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import multipletests

from common import (LABEL_COLORS, MODALITIES, PRIMARY, SEED, SETTINGS,
                    SPLIT_SEED, TAB, Harmonizer, batch_for, load_cohort, plt,
                    prepare_split, savefig)

warnings.filterwarnings("ignore")
BATCH_COLORS = {"C1": "#0072B2", "C2a": "#E69F00", "C2b": "#D55E00"}
PAIRS = [("T1", "T2"), ("T1", "ADC"), ("T2", "ADC")]


def kruskal_p(groups):
    try:
        return stats.kruskal(*groups).pvalue
    except ValueError:   # all values identical
        return 1.0


def permanova_r2(X, g):
    Xc = X - X.mean(0)
    between = sum((g == k).sum() * (Xc[g == k].mean(0) ** 2).sum()
                  for k in np.unique(g))
    return between / (Xc ** 2).sum()


def main():
    raw, ids = load_cohort()
    cohort = pd.crosstab([ids.center, ids.batch], ids.LVI, margins=True)
    cohort.columns = ["LVI-", "LVI+", "Total"]
    cohort.to_csv(TAB / "cohort.csv")
    print(cohort)

    rows, harmonized = [], {}
    for s in SETTINGS[1:]:
        harmonized[s] = {}
        for m in MODALITIES:
            X = Harmonizer(s).fit_transform(raw[m], batch_for(s, ids))
            harmonized[s][m] = X
            p = [kruskal_p([X.loc[ids.batch == b, f] for b in BATCH_COLORS])
                 for f in X.columns]
            pcs = PCA(10, random_state=SEED).fit_transform(X.values)
            rows.append({"setting": s, "modality": m,
                         "pct_batch_features": 100 * (multipletests(p, method="fdr_bh")[1] < 0.05).mean(),
                         "permanova_R2_batch": permanova_r2(X.values, ids.batch.values),
                         "silhouette_batch": silhouette_score(pcs, ids.batch)})
    diag = pd.DataFrame(rows)
    diag.to_csv(TAB / "harmonization_diagnostics.csv", index=False)
    print(diag.round(3).to_string())

    fig, axes = plt.subplots(3, 3, figsize=(9, 8))
    for i, s in enumerate(SETTINGS[1:]):
        for j, m in enumerate(MODALITIES):
            pcs = PCA(2, random_state=SEED).fit_transform(harmonized[s][m].values)
            for b, col in BATCH_COLORS.items():
                sel = (ids.batch == b).values
                axes[i, j].scatter(*pcs[sel].T, s=6, alpha=0.6, color=col,
                                   label=b, linewidths=0)
            r = diag[(diag.setting == s) & (diag.modality == m)].iloc[0]
            axes[i, j].set_title(f"{m}: {s}\nbatch R$^2$ = {r.permanova_R2_batch:.2f}",
                                 fontsize=8)
            axes[i, j].set_xlabel("PC1"); axes[i, j].set_ylabel("PC2")
    axes[0, 0].legend(title="Batch", markerscale=2, fontsize=7)
    fig.tight_layout()
    savefig(fig, "harmonization_pca")

    tr, te = train_test_split(ids.index, test_size=0.2, random_state=SPLIT_SEED,
                              stratify=ids["LVI"])
    ytr = ids.loc[tr, "LVI"].values
    agree = []
    for s in ["Original (raw z-score)", PRIMARY]:
        prepared = prepare_split(raw, ids, tr, te, s)
        D = {m: pairwise_distances(prepared[m][0].values) / np.sqrt(prepared[m][0].shape[1])
             for m in MODALITIES}
        iu = np.triu_indices(len(tr), 1)
        tag = "original" if s.startswith("Original") else "harmonized"
        fig, axes = plt.subplots(1, 3, figsize=(9.5, 3))
        for ax, (a, b) in zip(axes, PAIRS):
            r = np.corrcoef(D[a][iu], D[b][iu])[0, 1]
            agree.append({"setting": s, "pair": f"{a}-{b}", "distance_correlation": r})
            ax.scatter(D[a][iu], D[b][iu], s=1, alpha=0.03, color="#0072B2",
                       rasterized=True)
            lim = [0, max(D[a][iu].max(), D[b][iu].max())]
            ax.plot(lim, lim, "--", color="#D55E00", lw=1)
            ax.set_xlabel(f"{a} distance"); ax.set_ylabel(f"{b} distance")
            ax.set_title(f"{a} vs {b} (r = {r:.2f})")
        fig.tight_layout()
        savefig(fig, f"global_agreement_{tag}")

        fig, axes = plt.subplots(2, 3, figsize=(9.5, 6))
        btr = ids.loc[tr, "batch"].values
        for j, m in enumerate(MODALITIES):
            emb = TSNE(2, perplexity=30, init="pca", random_state=SPLIT_SEED,
                       max_iter=1000).fit_transform(prepared[m][0].values)
            for v, name in [(0, "LVI-"), (1, "LVI+")]:
                axes[0, j].scatter(*emb[ytr == v].T, s=8, color=LABEL_COLORS[v],
                                   label=name, alpha=0.8, linewidths=0)
            for bb, col in BATCH_COLORS.items():
                axes[1, j].scatter(*emb[btr == bb].T, s=8, color=col, label=bb,
                                   alpha=0.8, linewidths=0)
            for ax, t in zip(axes[:, j], ["LVI", "Acquisition batch"]):
                ax.set_title(f"{m}: {t}"); ax.set_xticks([]); ax.set_yticks([])
        axes[0, 0].legend(markerscale=2, fontsize=7)
        axes[1, 0].legend(markerscale=2, fontsize=7)
        fig.tight_layout()
        savefig(fig, f"local_neighborhood_{tag}")
    agree = pd.DataFrame(agree)
    agree.to_csv(TAB / "global_agreement.csv", index=False)
    print(agree.round(3).to_string())


if __name__ == "__main__":
    main()
