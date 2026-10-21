"""Step 4: single-modality spectral clustering (unsupervised analysis).

As in the original notebook, each modality is standardized and filtered
(|r| > 0.8), then clustered by spectral clustering (RBF affinity,
gamma = 1/p, k-means label assignment) for K = 2..5. Stability is measured
by the mean pairwise ARI over 100 subsamples of 80% and by the PAC of a
50-run consensus matrix. The analysis is repeated on
  * the original features (raw z-score), and
  * features harmonized label-free with ComBat by acquisition batch,
and each cluster is described by its LVI rate and its acquisition-batch
composition (Cramer's V). Finally, for the T1 partition with K = 3 we compare
logistic-regression CV AUC inside the mixed-label cluster with the AUC in
the whole cohort.
"""

import pickle
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from common import (CACHE, LABEL_COLORS, MODALITIES, PALETTE, PRIMARY, SEED,
                    TAB, Harmonizer, batch_for, correlation_filter,
                    load_cohort, plt, savefig, select_features)

warnings.filterwarnings("ignore")
K_RANGE = [2, 3, 4, 5]
SETTINGS_U = ["Original (raw z-score)", PRIMARY]
BATCH_COLORS = {"C1": "#0072B2", "C2a": "#E69F00", "C2b": "#D55E00"}


def spectral(X, k, seed):
    return SpectralClustering(n_clusters=k, affinity="rbf", gamma=1 / X.shape[1],
                              assign_labels="kmeans",
                              random_state=seed).fit_predict(X)


def stability(X, k, n_runs=100, frac=0.8, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(X)
    runs = []
    for _ in range(n_runs):
        idx = np.sort(rng.choice(n, int(frac * n), replace=False))
        runs.append((idx, spectral(X[idx], k, rng.randint(10000))))
    aris = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            common, a, b = np.intersect1d(runs[i][0], runs[j][0],
                                          return_indices=True)
            aris.append(adjusted_rand_score(runs[i][1][a], runs[j][1][b]))
    together = np.zeros((n, n)); co = np.zeros((n, n))
    for idx, lab in runs[:50]:
        onehot = np.eye(k)[lab]
        together[np.ix_(idx, idx)] += onehot @ onehot.T
        co[np.ix_(idx, idx)] += 1
    M = np.divide(together, co, out=np.zeros_like(together), where=co > 0)
    iu = np.triu_indices(n, 1)
    pac = np.mean((M[iu] > 0.1) & (M[iu] < 0.9))
    return np.mean(aris), pac, M


def cramers_v(a, b):
    tab = pd.crosstab(a, b)
    chi2 = stats.chi2_contingency(tab, correction=False)[0]
    return np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))


def features(raw, ids, setting):
    out = {}
    for m in MODALITIES:
        X = Harmonizer(setting).fit_transform(raw[m], batch_for(setting, ids))
        out[m] = X.loc[:, correlation_filter(X.values, 0.8)]
    return out


def cv_auc_lr(X, y, seed=SEED):
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
    aucs = []
    for tr, te in rcv.split(X, y):
        cols = select_features(X.iloc[tr], y[tr])
        m = LogisticRegression(C=0.1, solver="liblinear", max_iter=1000)
        m.fit(X.iloc[tr][cols], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X.iloc[te][cols])[:, 1]))
    return np.mean(aucs), np.std(aucs)


def main():
    raw, ids = load_cohort()
    y = ids["LVI"].values
    rows, comp, store = [], [], {}
    for setting in SETTINGS_U:
        feats = features(raw, ids, setting)
        store[setting] = {"features": feats, "labels": {}, "consensus": {}}
        for m in MODALITIES:
            X = feats[m].values
            for k in K_RANGE:
                lab = spectral(X, k, SEED)
                ari, pac, M = stability(X, k)
                store[setting]["labels"][(m, k)] = lab
                store[setting]["consensus"][(m, k)] = M
                rows.append({"setting": setting, "modality": m, "K": k,
                             "n_features": X.shape[1], "ARI": ari, "PAC": pac,
                             "sizes": "/".join(map(str, np.bincount(lab))),
                             "CramerV_LVI": cramers_v(lab, y),
                             "CramerV_batch": cramers_v(lab, ids["batch"])})
                print(rows[-1])
                for c in range(k):
                    sel = lab == c
                    comp.append({"setting": setting, "modality": m, "K": k,
                                 "cluster": c, "n": int(sel.sum()),
                                 "LVI_rate": y[sel].mean(),
                                 **{f"n_{b}": int((ids["batch"].values[sel] == b).sum())
                                    for b in BATCH_COLORS}})
    sel_df = pd.DataFrame(rows)
    sel_df.to_csv(TAB / "spectral_selection.csv", index=False)
    comp = pd.DataFrame(comp)
    comp.to_csv(TAB / "spectral_composition.csv", index=False)
    print(comp[comp.K == 3].round(3).to_string())

    # Mixed-cluster analysis (T1, K = 3), both settings
    mixed = []
    for setting in SETTINGS_U:
        lab = store[setting]["labels"][("T1", 3)]
        c = comp[(comp.setting == setting) & (comp.modality == "T1") & (comp.K == 3)]
        # mixed cluster: the largest cluster that is not (nearly) label-pure
        cand = c[(c.LVI_rate > 0.05) & (c.LVI_rate < 0.95)]
        mix = int(c.loc[(cand if len(cand) else c).n.idxmax(), "cluster"])
        inside = lab == mix
        for fs in ["T1+T2", "T1+T2+ADC"]:
            X = pd.concat([store[setting]["features"][m].add_prefix(m + ":")
                           for m in fs.split("+")], axis=1)
            full = cv_auc_lr(X, y)
            sub = cv_auc_lr(X[inside], y[inside]) if y[inside].sum() >= 10 else (np.nan, np.nan)
            mixed.append({"setting": setting, "feature_set": fs,
                          "mixed_cluster": mix, "n_mixed": int(inside.sum()),
                          "LVI_rate_mixed": y[inside].mean(),
                          "auc_all_mean": full[0], "auc_all_sd": full[1],
                          "auc_mixed_mean": sub[0], "auc_mixed_sd": sub[1]})
            print(mixed[-1])
    pd.DataFrame(mixed).to_csv(TAB / "spectral_mixed_cluster.csv", index=False)

    # ---- Figures --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
    for i, m in enumerate(MODALITIES):
        for setting, ls in zip(SETTINGS_U, ["--", "-"]):
            s = sel_df[(sel_df.setting == setting) & (sel_df.modality == m)]
            axes[0].plot(s.K, s.ARI, ls, marker="o", color=PALETTE[i],
                         label=f"{m} ({'orig.' if ls == '--' else 'ComBat'})")
            axes[1].plot(s.K, s.PAC, ls, marker="o", color=PALETTE[i])
    axes[0].set_ylabel("Mean pairwise ARI"); axes[1].set_ylabel("PAC (0.1, 0.9)")
    for ax in axes:
        ax.set_xlabel("K"); ax.set_xticks(K_RANGE)
    axes[0].legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    savefig(fig, "spectral_selection")

    for setting, tag in zip(SETTINGS_U, ["original", "harmonized"]):
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
        for ax, m in zip(axes, ["T1", "T2"]):
            lab = store[setting]["labels"][(m, 3)]
            M = store[setting]["consensus"][(m, 3)]
            order = np.argsort(lab, kind="stable")
            ax.imshow(M[np.ix_(order, order)], cmap="Blues", vmin=0, vmax=1,
                      interpolation="nearest")
            for cp in np.where(np.diff(lab[order]) != 0)[0] + 0.5:
                ax.axhline(cp, color="k", lw=0.6); ax.axvline(cp, color="k", lw=0.6)
            r = sel_df[(sel_df.setting == setting) & (sel_df.modality == m)
                       & (sel_df.K == 3)].iloc[0]
            ax.set_title(f"{m}, K = 3 (ARI {r.ARI:.2f}, PAC {r.PAC:.3f})")
            ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        savefig(fig, f"spectral_consensus_k3_{tag}")

        fig, axes = plt.subplots(2, 3, figsize=(9.5, 6))
        for row, m in enumerate(["T1", "T2"]):
            X = store[setting]["features"][m].values
            emb = TSNE(2, perplexity=30, init="pca", random_state=SEED).fit_transform(X)
            lab = store[setting]["labels"][(m, 3)]
            for c in range(3):
                axes[row, 0].scatter(*emb[lab == c].T, s=7, color=PALETTE[c],
                                     label=f"Cluster {c}", linewidths=0)
            for v, name in [(0, "LVI-"), (1, "LVI+")]:
                axes[row, 1].scatter(*emb[y == v].T, s=7, color=LABEL_COLORS[v],
                                     label=name, linewidths=0)
            for b, col in BATCH_COLORS.items():
                sel = (ids["batch"] == b).values
                axes[row, 2].scatter(*emb[sel].T, s=7, color=col, label=b,
                                     linewidths=0)
            for ax, t in zip(axes[row], ["Spectral clusters", "LVI", "Acquisition batch"]):
                ax.set_title(f"{m}: {t}"); ax.set_xticks([]); ax.set_yticks([])
                ax.legend(fontsize=6.5, markerscale=2)
        fig.tight_layout()
        savefig(fig, f"spectral_tsne_k3_{tag}")

    with open(CACHE / "spectral.pkl", "wb") as f:
        pickle.dump({"selection": sel_df, "composition": comp,
                     "labels": {s: store[s]["labels"] for s in SETTINGS_U}}, f)


if __name__ == "__main__":
    main()
