"""Step 2: MOFA+ latent representation of the harmonized T1/T2/ADC views.

Within each view a greedy |r| > 0.9 correlation filter removes near-duplicate
features (the radiomic panel contains many angle/offset replicates), then
MOFA+ is fitted with 10 initial factors, ARD + spike-and-slab weight priors,
drop-factor threshold 1% variance explained, and seed 507.

MOFA is fitted to every harmonization strategy from step 1 so that the batch
association of the latent factors can be compared: for each factor we report
the Kruskal-Wallis epsilon^2 for batch, and the CV AUC of a random forest
predicting batch from the retained factors.
"""

import pickle
import warnings

import numpy as np
import pandas as pd
from mofapy2.run.entry_point import entry_point
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from common import CACHE, MODALITIES, SEED, TAB, plt, savefig

warnings.filterwarnings("ignore")
STAGES = ["No correction", "ComBat (center)", "ComBat (batch)",
          "ComBat (batch, LVI protected)", "Rank-INT (batch)",
          "CovBat (batch)"]
PRIMARY = "ComBat (batch)"
STAGE_COLORS = ["#999999", "#E69F00", "#56B4E9", "#CC79A7", "#009E73",
                "#0072B2"]


def correlation_filter(X, threshold=0.9):
    corr = X.corr().abs().values
    keep = []
    for j in range(X.shape[1]):
        if all(corr[j, k] <= threshold for k in keep):
            keep.append(j)
    return X.iloc[:, keep]


def fit_mofa(views, n_factors=10):
    ent = entry_point()
    ent.set_data_options(scale_views=False)
    ent.set_data_matrix(
        [[views[m].values.astype(float)] for m in MODALITIES],
        views_names=MODALITIES, groups_names=["all"],
        samples_names=[[str(i) for i in views["T1"].index]],
        features_names=[[f"{m}:{c}" for c in views[m].columns]
                        for m in MODALITIES])
    ent.set_model_options(factors=n_factors, spikeslab_weights=True,
                          ard_weights=True)
    ent.set_train_options(iter=1000, convergence_mode="medium", dropR2=0.01,
                          seed=SEED, gpu_mode=False, verbose=False, quiet=True)
    ent.build()
    ent.run()
    model = ent.model
    Z = model.nodes["Z"].getExpectation()
    W = model.nodes["W"].getExpectation()
    r2 = np.array(model.calculate_variance_explained()[0]) * 100  # views x K
    if r2.shape[0] != len(MODALITIES):
        r2 = r2.T
    k = Z.shape[1]
    cols = [f"F{i + 1}" for i in range(k)]
    Z = pd.DataFrame(Z, index=views["T1"].index, columns=cols)
    W = {m: pd.DataFrame(W[i], index=views[m].columns, columns=cols)
         for i, m in enumerate(MODALITIES)}
    r2 = pd.DataFrame(r2, index=MODALITIES, columns=cols)
    return Z, W, r2


def batch_association(Z, batch):
    rows = []
    n, k = len(batch), len(np.unique(batch))
    for f in Z.columns:
        h = stats.kruskal(*[Z.loc[batch == b, f] for b in np.unique(batch)])
        groups = [Z.loc[batch == b, f] for b in np.unique(batch)]
        sds = [g.std() for g in groups]
        rows.append({"factor": f, "KW_p": h.pvalue,
                     "epsilon2": max(0, (h.statistic - k + 1) / (n - k)),
                     "sd_ratio": max(sds) / min(sds),
                     "BF_p": stats.levene(*groups, center="median").pvalue})
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    auc = cross_val_score(
        RandomForestClassifier(300, min_samples_leaf=3, random_state=SEED,
                               n_jobs=-1),
        Z.values, batch, cv=cv, scoring="roc_auc_ovr").mean()
    return pd.DataFrame(rows), auc


def main():
    with open(CACHE / "harmonized.pkl", "rb") as f:
        h = pickle.load(f)
    ids = h["ids"]
    batch = ids["batch"].values

    results, rows = {}, []
    for stage in STAGES:
        views = {}
        for m in MODALITIES:
            X = correlation_filter(h["stages"][stage][m])
            views[m] = pd.DataFrame(StandardScaler().fit_transform(X),
                                    index=X.index, columns=X.columns)
        Z, W, r2 = fit_mofa(views)
        assoc, auc = batch_association(Z, batch)
        results[stage] = {"views": views, "Z": Z, "W": W, "r2": r2,
                          "batch_assoc": assoc, "batch_auc": auc}
        rows.append({
            "stage": stage,
            "features_after_filter": "/".join(str(views[m].shape[1])
                                              for m in MODALITIES),
            "n_factors": Z.shape[1],
            "total_R2_T1": r2.loc["T1"].sum(), "total_R2_T2": r2.loc["T2"].sum(),
            "total_R2_ADC": r2.loc["ADC"].sum(),
            "max_factor_batch_epsilon2": assoc.epsilon2.max(),
            "n_factors_batch_eps2_gt_0.06": int((assoc.epsilon2 > 0.06).sum()),
            "max_factor_sd_ratio": assoc.sd_ratio.max(),
            "RF_batch_AUC_from_Z": auc})
        print(rows[-1])
        print(r2.round(1).to_string())
        print(assoc.round(3).to_string())

    # Sensitivity cohort: exclude batch C2b (fails cross-sequence coherence QC
    # and contains no LVI-positive patient); ComBat refitted on C1 vs C2a.
    sub = ids.index[ids["batch"] != "C2b"]
    from importlib import import_module
    combat = import_module("01_harmonization").combat
    views = {}
    for m in MODALITIES:
        X0 = h["stages"]["No correction"][m].loc[sub]
        X = correlation_filter(combat(X0, ids.loc[sub, "batch"]))
        views[m] = pd.DataFrame(StandardScaler().fit_transform(X),
                                index=X.index, columns=X.columns)
    Z, W, r2 = fit_mofa(views)
    assoc, auc = batch_association(Z, ids.loc[sub, "batch"].values)
    results["ComBat (batch), excl. C2b"] = {
        "views": views, "Z": Z, "W": W, "r2": r2, "batch_assoc": assoc,
        "batch_auc": auc}
    rows.append({
        "stage": "ComBat (batch), excl. C2b",
        "features_after_filter": "/".join(str(views[m].shape[1])
                                          for m in MODALITIES),
        "n_factors": Z.shape[1],
        "total_R2_T1": r2.loc["T1"].sum(), "total_R2_T2": r2.loc["T2"].sum(),
        "total_R2_ADC": r2.loc["ADC"].sum(),
        "max_factor_batch_epsilon2": assoc.epsilon2.max(),
        "n_factors_batch_eps2_gt_0.06": int((assoc.epsilon2 > 0.06).sum()),
        "max_factor_sd_ratio": assoc.sd_ratio.max(),
        "RF_batch_AUC_from_Z": auc})
    print(rows[-1])
    print(r2.round(1).to_string())

    # Confound-free check: center 1 only (single site, one acquisition batch,
    # so no harmonization is needed).
    sub = ids.index[ids["batch"] == "C1"]
    views = {}
    for m in MODALITIES:
        X = correlation_filter(h["stages"]["No correction"][m].loc[sub])
        views[m] = pd.DataFrame(StandardScaler().fit_transform(X),
                                index=X.index, columns=X.columns)
    Z, W, r2 = fit_mofa(views)
    results["Center 1 only"] = {"views": views, "Z": Z, "W": W, "r2": r2,
                                "batch_assoc": None, "batch_auc": np.nan}
    rows.append({
        "stage": "Center 1 only",
        "features_after_filter": "/".join(str(views[m].shape[1])
                                          for m in MODALITIES),
        "n_factors": Z.shape[1],
        "total_R2_T1": r2.loc["T1"].sum(), "total_R2_T2": r2.loc["T2"].sum(),
        "total_R2_ADC": r2.loc["ADC"].sum()})
    print(rows[-1])

    pd.DataFrame(rows).to_csv(TAB / "mofa_by_harmonization.csv", index=False)
    prim = results[PRIMARY]
    prim["r2"].T.round(2).to_csv(TAB / "mofa_variance_explained.csv")
    prim["Z"].to_csv(TAB / "mofa_factor_scores.csv")
    tw = []
    for m in MODALITIES:
        for f in prim["W"][m].columns:
            w = prim["W"][m][f]
            top = w.abs().sort_values(ascending=False).head(5).index
            for feat in top:
                tw.append({"factor": f, "view": m, "feature": feat,
                           "weight": w[feat]})
    pd.DataFrame(tw).to_csv(TAB / "mofa_top_weights.csv", index=False)

    # ---- Figures --------------------------------------------------------
    r2 = prim["r2"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8),
                             gridspec_kw={"width_ratios": [3, 1]})
    im = axes[0].imshow(r2.values, cmap="Blues", aspect="auto", vmin=0)
    axes[0].set_xticks(range(r2.shape[1]), r2.columns)
    axes[0].set_yticks(range(3), MODALITIES)
    for i in range(r2.shape[0]):
        for j in range(r2.shape[1]):
            v = r2.values[i, j]
            axes[0].text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                         color="white" if v > r2.values.max() * 0.6 else "black")
    axes[0].set_title("Variance explained per factor (%)")
    fig.colorbar(im, ax=axes[0], fraction=0.03)
    axes[1].barh(MODALITIES, r2.sum(1), color="#0072B2")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Total R$^2$ (%)")
    fig.tight_layout()
    savefig(fig, "mofa_variance_explained")

    fig, axes = plt.subplots(1, 2, figsize=(9, 2.9))
    for i, stage in enumerate(STAGES):
        a = results[stage]["batch_assoc"]
        x = np.arange(len(a)) + 1 + (i - 2.5) * 0.11
        axes[0].scatter(x, a.epsilon2, s=12, color=STAGE_COLORS[i],
                        label=stage)
        axes[1].scatter(x, a.sd_ratio, s=12, color=STAGE_COLORS[i])
    axes[0].axhline(0.06, ls="--", lw=0.8, color="k")
    axes[1].axhline(1, ls="--", lw=0.8, color="k")
    axes[1].set_yscale("log")
    axes[0].set_ylabel(r"Batch location effect $\epsilon^2$")
    axes[1].set_ylabel("Max/min within-batch SD")
    for ax in axes:
        ax.set_xlabel("MOFA factor")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    savefig(fig, "mofa_factor_batch_assoc")

    with open(CACHE / "mofa.pkl", "wb") as f:
        pickle.dump({"results": results, "ids": ids}, f)


if __name__ == "__main__":
    main()
