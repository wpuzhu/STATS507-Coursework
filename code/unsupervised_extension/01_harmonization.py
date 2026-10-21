"""Step 1: preprocessing, center/batch harmonization, and harmonization tests.

Exploration showed that center 2 contains two acquisition sub-batches: for
~100 texture features (mostly run-length / low-grey-level emphasis) 99 of the
181 center-2 patients have values orders of magnitude larger than everyone
else, consistently in T1, T2 and ADC (contiguous case IDs, i.e. a separate
extraction/acquisition run). These are detected from the raw data
(`detect_acquisition_batch`), giving three acquisition batches C1, C2a, C2b.
C2b contains no LVI-positive patient, so batch and outcome are confounded.

For each MRI sequence (T1, T2, ADC) separately:
  1. drop features with >10% missing values, and degenerate features
     (modal value shared by >20% of a batch, or zero within-batch IQR);
  2. impute remaining missing values with the within-batch median;
  3. log-transform strictly positive features (turns multiplicative scale
     differences into location shifts), Yeo-Johnson otherwise; standardize,
     winsorize at +/-5 SD;
  4. harmonize. Compared strategies:
       - no correction;
       - ComBat with hospital center as batch (the naive choice);
       - ComBat with acquisition batch           <- primary;
       - ComBat with acquisition batch, LVI protected (sensitivity);
       - CovBat with acquisition batch (sensitivity)
         (ComBat, then ComBat on the leading principal-component scores of
         the residuals, so that between-feature covariance is harmonized
         too; Chen et al. 2022). MOFA on ComBat output shows that C2b
         patients have almost no variance along the shared factors, i.e. part
         of the batch effect lives in the covariance. CovBat did not remove
         it (see 02_mofa.py), so plain ComBat is kept as primary;
       - within-batch rank inverse-normal transform (non-parametric
         sensitivity: matches the full marginal distribution per batch).

Harmonization is evaluated by
  * feature-wise Kruskal-Wallis (location) and Brown-Forsythe (scale) tests vs
    batch with BH-FDR, and epsilon^2 effect sizes;
  * multivariate batch separability: PERMANOVA-type R^2 (permutation p),
    silhouette of batch labels in PCA space, and repeated 5-fold CV AUC of a
    random forest predicting batch (sensitive to higher-order differences);
  * preservation of biological signal: features associated with LVI and CV
    AUC for predicting LVI.
"""

import pickle
import warnings

import numpy as np
import pandas as pd
from neuroCombat import neuroCombat
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

from common import CACHE, MODALITIES, SEED, TAB, bh, load_modalities, plt, savefig

warnings.filterwarnings("ignore")
rng = np.random.default_rng(SEED)

BATCHES = ["C1", "C2a", "C2b"]
BATCH_COLORS = {"C1": "#0072B2", "C2a": "#E69F00", "C2b": "#D55E00"}
STAGES = ["No correction", "ComBat (center)", "ComBat (batch)",
          "ComBat (batch, LVI protected)", "Rank-INT (batch)",
          "CovBat (batch)"]
PRIMARY = "ComBat (batch)"
STAGE_COLORS = ["#999999", "#E69F00", "#56B4E9", "#CC79A7", "#009E73",
                "#0072B2"]


def detect_acquisition_batch(raw, center, min_frac=0.2, cut=0.3):
    """Flag center-2 patients whose run-length features sit on a different scale.

    A feature value is "extreme" if it exceeds 10x the center-1 99th
    percentile. Features where >min_frac of center-2 patients are extreme are
    used; a patient's score is their mean extreme-rate over those features,
    averaged over the three sequences.
    """
    scores = []
    for m in MODALITIES:
        X = raw[m]
        q99 = X[center == 1].quantile(0.99)
        extreme = X.gt(10 * q99) & X.gt(0) & (q99 > 0).values
        feats = extreme[center == 2].mean()
        scores.append(extreme[feats[feats > min_frac].index].mean(1))
    score = pd.concat(scores, axis=1).mean(1)
    batch = pd.Series("C1", index=center.index)
    batch[(center == 2) & (score > cut)] = "C2b"
    batch[(center == 2) & (score <= cut)] = "C2a"
    return batch, score


def preprocess(X, batch, max_missing=0.10, max_mode=0.20):
    n0 = X.shape[1]
    X = X.loc[:, X.isna().mean() <= max_missing]
    mode = pd.concat([X[batch == b].round(10).apply(
        lambda s: s.value_counts(normalize=True).iloc[0]) for b in BATCHES],
        axis=1).max(1)
    iqr = pd.concat([X[batch == b].quantile(0.75) - X[batch == b].quantile(0.25)
                     for b in BATCHES], axis=1).min(1)
    X = X.loc[:, (mode <= max_mode) & (iqr > 0)].copy()
    for b in BATCHES:
        idx = batch == b
        X.loc[idx] = X.loc[idx].fillna(X.loc[idx].median())
    out = {}
    for f in X.columns:
        v = X[f].values.astype(float)
        if (v > 0).all():
            v = np.log(v)
        else:
            v = PowerTransformer(method="yeo-johnson").fit_transform(
                v.reshape(-1, 1)).ravel()
        out[f] = v
    Z = StandardScaler().fit_transform(pd.DataFrame(out, index=X.index))
    Z = np.clip(Z, -5, 5)
    return pd.DataFrame(Z, index=X.index, columns=X.columns), n0 - X.shape[1]


def combat(X, batch, protect=None):
    covars = pd.DataFrame({"batch": pd.factorize(batch)[0] + 1})
    cat = []
    if protect is not None:
        covars["protect"] = np.asarray(protect)
        cat = ["protect"]
    out = neuroCombat(dat=X.T.values, covars=covars, batch_col="batch",
                      categorical_cols=cat)["data"].T
    return pd.DataFrame(StandardScaler().fit_transform(out), index=X.index,
                        columns=X.columns)


def covbat(X, batch, pct_var=0.95):
    """CovBat (Chen et al. 2022): ComBat, then ComBat on residual PC scores."""
    Xc = combat(X, batch)
    mu = Xc.mean(0)
    R = (Xc - mu).values
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    scores = U * S
    k = int(np.searchsorted(np.cumsum(S ** 2) / (S ** 2).sum(), pct_var)) + 1
    covars = pd.DataFrame({"batch": pd.factorize(batch)[0] + 1})
    adj = neuroCombat(dat=scores[:, :k].T, covars=covars,
                      batch_col="batch")["data"].T
    scores = scores.copy()
    scores[:, :k] = adj
    out = scores @ Vt + mu.values
    return pd.DataFrame(StandardScaler().fit_transform(out), index=X.index,
                        columns=X.columns)


def rank_int(X, batch):
    """Within-batch rank-based inverse normal transform (Blom offsets)."""
    out = X.copy()
    for b in np.unique(batch):
        idx = np.asarray(batch == b)
        r = X[idx].rank(axis=0)
        out[idx] = stats.norm.ppf((r - 0.375) / (idx.sum() + 0.25))
    return pd.DataFrame(StandardScaler().fit_transform(out), index=X.index,
                        columns=X.columns)


def featurewise_batch_tests(X, batch):
    """Kruskal-Wallis (location), Brown-Forsythe (scale), epsilon^2 per feature."""
    levels = np.unique(batch)
    n, k = len(batch), len(levels)
    p_loc, p_scale, eps2 = [], [], []
    for f in X.columns:
        groups = [X.loc[batch == g, f].values for g in levels]
        h = stats.kruskal(*groups)
        p_loc.append(h.pvalue)
        eps2.append(max(0.0, (h.statistic - k + 1) / (n - k)))
        p_scale.append(stats.levene(*groups, center="median").pvalue)
    return pd.DataFrame({"feature": X.columns, "p_location": p_loc,
                         "q_location": bh(p_loc), "p_scale": p_scale,
                         "q_scale": bh(p_scale), "epsilon2": eps2})


def permanova_r2(X, groups, n_perm=499):
    """One-way PERMANOVA R^2 on Euclidean distances (= between-SS / total-SS)."""
    Xc = X - X.mean(0)
    tss = (Xc ** 2).sum()

    def r2(g):
        return sum((g == k).sum() * (Xc[g == k].mean(0) ** 2).sum()
                   for k in np.unique(g)) / tss

    obs = r2(groups)
    if n_perm == 0:
        return obs, np.nan
    null = np.array([r2(rng.permutation(groups)) for _ in range(n_perm)])
    return obs, (1 + (null >= obs).sum()) / (n_perm + 1)


def cv_auc(X, y, model, repeats=2):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats,
                                 random_state=SEED)
    scoring = "roc_auc" if len(np.unique(y)) == 2 else "roc_auc_ovr"
    return cross_val_score(model, X, y, cv=cv, scoring=scoring).mean()


def lr():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=5000))


def rf():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                  random_state=SEED, n_jobs=-1)


def main():
    raw, ids = load_modalities()
    center, lvsi = ids["center"], ids["Label"]
    batch, score = detect_acquisition_batch(raw, center)
    ids["batch"], ids["batch_score"] = batch, score
    ids.to_csv(TAB / "acquisition_batch.csv")
    comp = pd.crosstab(batch, lvsi, margins=True)
    comp.columns = ["LVI-", "LVI+", "Total"]
    comp.to_csv(TAB / "batch_composition.csv")
    print(comp)

    data = {s: {} for s in STAGES}
    dropped = {}
    for m in MODALITIES:
        X, dropped[m] = preprocess(raw[m], batch)
        data["No correction"][m] = X
        data["ComBat (center)"][m] = combat(X, center)
        data["ComBat (batch)"][m] = combat(X, batch)
        data["ComBat (batch, LVI protected)"][m] = combat(X, batch, lvsi)
        data["Rank-INT (batch)"][m] = rank_int(X, batch)
        data["CovBat (batch)"][m] = covbat(X, batch)
        print(m, "kept", X.shape[1], "dropped", dropped[m])

    rows, feat_tables = [], []
    views = MODALITIES + ["Concatenated"]
    for stage in STAGES:
        d = dict(data[stage])
        d["Concatenated"] = pd.concat([d[m].add_prefix(m + "_")
                                       for m in MODALITIES], axis=1)
        for m in views:
            X = d[m]
            Xv = X.values
            row = {"modality": m, "stage": stage, "n_features": X.shape[1]}
            if m != "Concatenated":
                ft = featurewise_batch_tests(X, batch.values)
                ft.insert(0, "stage", stage)
                ft.insert(0, "modality", m)
                feat_tables.append(ft)
                lvsi_q = bh([stats.mannwhitneyu(X.loc[lvsi == 1, f],
                                                X.loc[lvsi == 0, f]).pvalue
                             for f in X.columns])
                row.update({
                    "n_dropped_qc": dropped[m],
                    "pct_location_sig": 100 * (ft.q_location < 0.05).mean(),
                    "pct_scale_sig": 100 * (ft.q_scale < 0.05).mean(),
                    "median_epsilon2": ft.epsilon2.median(),
                    "pct_LVI_sig_features": 100 * (lvsi_q < 0.05).mean()})
            r2, r2_p = permanova_r2(Xv, batch.values)
            pcs = PCA(n_components=10, random_state=SEED).fit_transform(Xv)
            row.update({
                "permanova_R2_batch": r2, "permanova_p_batch": r2_p,
                "permanova_R2_center": permanova_r2(Xv, center.values, 0)[0],
                "silhouette_batch_PCA10": silhouette_score(pcs, batch),
                "batch_AUC_RF": cv_auc(Xv, batch, rf()),
                "LVI_AUC_LR": cv_auc(Xv, lvsi, lr()),
            })
            rows.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v)
                   for k, v in row.items()})

    summary = pd.DataFrame(rows)
    summary.to_csv(TAB / "harmonization_summary.csv", index=False)
    feat = pd.concat(feat_tables)
    feat.to_csv(TAB / "harmonization_featurewise.csv", index=False)

    # ---- Figures --------------------------------------------------------
    show = ["No correction", "ComBat (center)", "ComBat (batch)"]
    fig, axes = plt.subplots(3, 3, figsize=(9, 8.2))  # rows = stages in show
    for j, m in enumerate(MODALITIES):
        for i, stage in enumerate(show):
            ax = axes[i, j]
            pcs = PCA(n_components=2, random_state=SEED).fit_transform(
                data[stage][m])
            for b, col in BATCH_COLORS.items():
                ax.scatter(*pcs[batch == b].T, s=6, alpha=0.6, color=col,
                           label=b, linewidths=0)
            ax.set_title(f"{m}: {stage}", fontsize=8.5)
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    axes[0, 0].legend(loc="best", markerscale=2, title="Batch")
    fig.tight_layout()
    savefig(fig, "harm_pca_by_batch")

    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), sharey=True)
    bins = np.linspace(0, 0.6, 31)
    for ax, m in zip(axes, MODALITIES):
        for stage, col in zip(STAGES, STAGE_COLORS):
            if stage not in show:
                continue
            v = feat.query("modality == @m and stage == @stage").epsilon2
            ax.hist(v, bins=bins, alpha=0.6, color=col, label=stage)
        ax.set_title(m); ax.set_xlabel(r"Batch effect size $\epsilon^2$")
    axes[0].set_ylabel("Number of features"); axes[0].legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, "harm_effectsize_hist")

    fig, axes = plt.subplots(1, 2, figsize=(9, 2.9))
    x = np.arange(len(views)); w = 0.13
    for k, (stage, col) in enumerate(zip(STAGES, STAGE_COLORS)):
        s = summary[summary.stage == stage]
        axes[0].bar(x + (k - 2.5) * w, s.permanova_R2_batch, w, color=col,
                    label=stage)
        axes[1].bar(x + (k - 2.5) * w, s.batch_AUC_RF, w, color=col)
    axes[0].set_ylabel(r"PERMANOVA $R^2$ (batch)")
    axes[1].set_ylabel("RF CV AUC predicting batch")
    axes[1].axhline(0.5, ls="--", color="k", lw=0.8)
    axes[1].set_ylim(0.3, 1.02)
    for ax in axes:
        ax.set_xticks(x, views, fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7)
    savefig(fig, "harm_multivariate")

    with open(CACHE / "harmonized.pkl", "wb") as f:
        pickle.dump({"stages": data, "primary": PRIMARY, "ids": ids}, f)


if __name__ == "__main__":
    main()
