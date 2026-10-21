"""Step 3: clustering in the MOFA latent space and selection of k.

Patients are clustered with k-means (50 starts) on the standardized MOFA
factor scores for k = 2..8. Each k is scored by
  * silhouette width (Rousseeuw 1987), plus Calinski-Harabasz/Davies-Bouldin;
  * resampling stability: B = 200 subsamples of 80% of patients are
    re-clustered; we record the ARI between each subsample partition and the
    full-data partition restricted to the same patients, and the ARI between
    pairs of subsample partitions on their shared patients;
  * consensus clustering (Monti et al. 2003): the consensus matrix
    M_ij = (#times i, j co-clustered) / (#times i, j co-sampled), its CDF,
    and the proportion of ambiguous clustering PAC = P(0.1 < M_ij < 0.9)
    (Senbabaoglu et al. 2014);
  * minimum cluster proportion.
The optimal k maximizes the mean of the rank-normalized silhouette, mean ARI
and 1 - PAC among partitions whose smallest cluster has >= 10% of patients.

Each partition is also tested for association with acquisition batch and
center (Cramer's V): a partition that reproduces batch would indicate that
harmonization failed. The analysis is run on the primary representation and
on every sensitivity representation from step 2; ARI between the primary
clusters and each sensitivity clustering (same k, shared patients) measures
robustness to the harmonization choice.
"""

import pickle
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import leaves_list, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_samples,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from common import CACHE, PALETTE, SEED, TAB, plt, savefig

warnings.filterwarnings("ignore")
K_RANGE = range(2, 9)
B = 200
FRAC = 0.8
PRIMARY = "ComBat (batch)"
BATCH_COLORS = {"C1": "#0072B2", "C2a": "#E69F00", "C2b": "#D55E00"}


def kmeans(X, k, seed=SEED, n_init=50):
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(X)


def cramers_v(a, b):
    tab = pd.crosstab(a, b)
    chi2, p, _, _ = stats.chi2_contingency(tab, correction=False)
    v = np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))
    return v, p


def stability(X, k, full_labels, rng):
    n = len(X)
    m = int(round(FRAC * n))
    co = np.zeros((n, n))
    cs = np.zeros((n, n))
    ari_ref, ari_pair = [], []
    prev = None
    for b in range(B):
        idx = np.sort(rng.choice(n, m, replace=False))
        lab = kmeans(X[idx], k, seed=SEED + b, n_init=10)
        ari_ref.append(adjusted_rand_score(full_labels[idx], lab))
        if prev is not None:
            pidx, plab = prev
            shared, ia, ib = np.intersect1d(idx, pidx, return_indices=True)
            ari_pair.append(adjusted_rand_score(lab[ia], plab[ib]))
        prev = (idx, lab)
        same = (lab[:, None] == lab[None, :]).astype(float)
        co[np.ix_(idx, idx)] += same
        cs[np.ix_(idx, idx)] += 1
    M = np.divide(co, cs, out=np.zeros_like(co), where=cs > 0)
    np.fill_diagonal(M, 1)
    iu = np.triu_indices(n, 1)
    pac = np.mean((M[iu] > 0.1) & (M[iu] < 0.9))
    return {"M": M, "ari_ref": np.array(ari_ref),
            "ari_pair": np.array(ari_pair), "PAC": pac}


def evaluate(Z, ids, name, rng):
    X = StandardScaler().fit_transform(Z.values)
    rows, labels, cons = [], {}, {}
    for k in K_RANGE:
        lab = kmeans(X, k)
        st = stability(X, k, lab, rng)
        vb, pb = (cramers_v(lab, ids["batch"].values)
                  if ids["batch"].nunique() > 1 else (np.nan, np.nan))
        vl, pl = cramers_v(lab, ids["Label"].values)
        sizes = np.bincount(lab)
        rows.append({
            "representation": name, "k": k,
            "silhouette": silhouette_score(X, lab),
            "calinski_harabasz": calinski_harabasz_score(X, lab),
            "davies_bouldin": davies_bouldin_score(X, lab),
            "ARI_vs_full_mean": st["ari_ref"].mean(),
            "ARI_vs_full_sd": st["ari_ref"].std(),
            "ARI_pairwise_mean": st["ari_pair"].mean(),
            "PAC": st["PAC"],
            "min_cluster_prop": sizes.min() / sizes.sum(),
            "sizes": "/".join(map(str, sizes)),
            "CramerV_batch": vb, "p_batch": pb,
            "CramerV_LVI": vl, "p_LVI": pl,
        })
        labels[k] = lab
        cons[k] = st
        print({kk: (round(v, 3) if isinstance(v, float) else v)
               for kk, v in rows[-1].items()})
    df = pd.DataFrame(rows)
    ok = df.min_cluster_prop >= 0.10
    score = (df.silhouette.rank(pct=True) + df.ARI_vs_full_mean.rank(pct=True)
             + (1 - df.PAC).rank(pct=True)) / 3
    df["composite_score"] = np.where(ok, score, np.nan)
    best_k = int(df.loc[df.composite_score.idxmax(), "k"])
    return df, labels, cons, best_k, X


def relabel_by_size(lab):
    order = np.argsort(-np.bincount(lab))
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[l] for l in lab])


def plot_selection(df, name, best_k):
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.5))
    ks = df.k
    axes[0].plot(ks, df.silhouette, "o-", color=PALETTE[0])
    axes[0].set_ylabel("Mean silhouette")
    axes[1].errorbar(ks, df.ARI_vs_full_mean, yerr=df.ARI_vs_full_sd, fmt="o-",
                     color=PALETTE[0], capsize=2, label="vs full data")
    axes[1].plot(ks, df.ARI_pairwise_mean, "s--", color=PALETTE[1],
                 label="pairwise subsamples")
    axes[1].set_ylabel("ARI"); axes[1].legend(fontsize=6.5)
    axes[2].plot(ks, df.PAC, "o-", color=PALETTE[0])
    axes[2].set_ylabel("PAC (0.1, 0.9)")
    axes[3].plot(ks, df.CramerV_batch, "o-", color=PALETTE[5], label="batch")
    axes[3].plot(ks, df.CramerV_LVI, "s--", color=PALETTE[2], label="LVI")
    axes[3].set_ylabel("Cramer's V with clusters"); axes[3].legend(fontsize=6.5)
    axes[3].set_ylim(0, 1)
    for ax in axes:
        ax.axvline(best_k, color="#999999", lw=0.8, ls=":")
        ax.set_xlabel("k"); ax.set_xticks(list(ks))
    fig.suptitle(name, fontsize=9)
    fig.tight_layout()
    return fig


def main():
    with open(CACHE / "mofa.pkl", "rb") as f:
        mofa = pickle.load(f)
    ids_all = mofa["ids"]
    rng = np.random.default_rng(SEED)

    all_rows, out = [], {}
    for name, res in mofa["results"].items():
        Z = res["Z"]
        ids = ids_all.loc[Z.index]
        print("=" * 30, name)
        df, labels, cons, best_k, X = evaluate(Z, ids, name, rng)
        df["selected"] = df.k == best_k
        all_rows.append(df)
        out[name] = {"df": df, "labels": labels, "cons": cons,
                     "best_k": best_k, "X": X, "index": Z.index}
        print("selected k =", best_k)

    table = pd.concat(all_rows)
    table.to_csv(TAB / "cluster_selection_all.csv", index=False)

    prim = out[PRIMARY]
    k = prim["best_k"]
    lab = relabel_by_size(prim["labels"][k])
    X = prim["X"]
    idx = prim["index"]
    ids = ids_all.loc[idx]
    prim["df"].to_csv(TAB / "cluster_selection_primary.csv", index=False)

    # Robustness: algorithm and harmonization choice
    rob = []
    for alg, l2 in [
        ("Agglomerative (Ward)", AgglomerativeClustering(k).fit_predict(X)),
        ("Gaussian mixture", GaussianMixture(k, n_init=10, random_state=SEED)
         .fit(X).predict(X)),
    ]:
        rob.append({"comparison": f"k-means vs {alg}", "k": k,
                    "ARI": adjusted_rand_score(lab, l2), "n": len(lab)})
    for name, o in out.items():
        if name == PRIMARY:
            continue
        shared = idx.intersection(o["index"])
        l_other = pd.Series(o["labels"][k], index=o["index"]).loc[shared]
        l_prim = pd.Series(lab, index=idx).loc[shared]
        rob.append({"comparison": f"{PRIMARY} vs {name}", "k": k,
                    "ARI": adjusted_rand_score(l_prim, l_other),
                    "n": len(shared),
                    "other_selected_k": o["best_k"]})
    rob = pd.DataFrame(rob)
    rob.to_csv(TAB / "cluster_robustness.csv", index=False)
    print(rob)

    clusters = pd.DataFrame({"cluster": lab + 1}, index=idx)
    clusters["silhouette"] = silhouette_samples(X, lab)
    clusters = clusters.join(ids[["center", "batch", "Label"]])
    clusters.to_csv(TAB / "cluster_assignments.csv")
    ct = pd.crosstab(clusters.cluster, clusters.batch, margins=True)
    ct.to_csv(TAB / "cluster_by_batch.csv")
    print(ct)
    print(pd.crosstab(clusters.cluster, clusters.Label, normalize="index"))

    # ---- Figures --------------------------------------------------------
    savefig(plot_selection(prim["df"], f"MOFA + k-means ({PRIMARY})", k),
            "cluster_selection")
    for name, o in out.items():
        if name != PRIMARY:
            safe = (name.replace(" ", "_").replace("(", "").replace(")", "")
                    .replace(",", "").replace(".", ""))
            savefig(plot_selection(o["df"], name, o["best_k"]),
                    f"cluster_selection_{safe}")

    ks_show = [2, 3, 4, 5]
    fig, axes = plt.subplots(1, len(ks_show) + 1, figsize=(11, 2.5),
                             gridspec_kw={"width_ratios": [1] * len(ks_show)
                                          + [1.2]})
    for ax, kk in zip(axes, ks_show):
        M = prim["cons"][kk]["M"]
        order = leaves_list(linkage(1 - M[np.triu_indices(len(M), 1)],
                                    "average"))
        ax.imshow(M[np.ix_(order, order)], cmap="Blues", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_title(f"k = {kk}  (PAC {prim['cons'][kk]['PAC']:.2f})")
        ax.set_xticks([]); ax.set_yticks([])
    for i, kk in enumerate(K_RANGE):
        M = prim["cons"][kk]["M"]
        v = np.sort(M[np.triu_indices(len(M), 1)])
        axes[-1].plot(v, np.arange(1, len(v) + 1) / len(v),
                      color=plt.cm.viridis(i / (len(K_RANGE) - 1)),
                      label=f"k={kk}", lw=1)
    axes[-1].set_xlabel("Consensus index"); axes[-1].set_ylabel("CDF")
    axes[-1].legend(fontsize=6)
    fig.tight_layout()
    savefig(fig, "consensus_matrices")

    emb = TSNE(n_components=2, perplexity=30, random_state=SEED,
               init="pca").fit_transform(X)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.1))
    for c in range(k):
        axes[0].scatter(*emb[lab == c].T, s=7, color=PALETTE[c],
                        label=f"Cluster {c + 1}", linewidths=0)
    for b, col in BATCH_COLORS.items():
        sel = (ids.batch == b).values
        axes[1].scatter(*emb[sel].T, s=7, color=col, label=b, linewidths=0)
    for v, col, lname in [(0, "#BBBBBB", "LVI-"), (1, "#D55E00", "LVI+")]:
        sel = (ids.Label == v).values
        axes[2].scatter(*emb[sel].T, s=7, color=col, label=lname,
                        linewidths=0)
    for ax, t in zip(axes, ["Clusters", "Acquisition batch", "LVI"]):
        ax.set_title(t); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.legend(fontsize=6.5, markerscale=2)
    fig.tight_layout()
    savefig(fig, "cluster_tsne")

    fig, ax = plt.subplots(figsize=(4, 3))
    order = np.lexsort((-clusters.silhouette.values, lab))
    y0 = 0
    for c in range(k):
        s = np.sort(clusters.silhouette.values[lab == c])[::-1]
        ax.fill_betweenx(np.arange(y0, y0 + len(s)), 0, s, color=PALETTE[c],
                         label=f"Cluster {c + 1}")
        y0 += len(s) + 5
    ax.axvline(clusters.silhouette.mean(), ls="--", color="k", lw=0.8)
    ax.set_xlabel("Silhouette width"); ax.set_yticks([])
    ax.legend(fontsize=7)
    savefig(fig, "cluster_silhouette")

    with open(CACHE / "clusters.pkl", "wb") as f:
        pickle.dump({"clusters": clusters, "k": k, "out": {
            n: {kk: v for kk, v in o.items() if kk != "cons"}
            for n, o in out.items()}}, f)


if __name__ == "__main__":
    main()
