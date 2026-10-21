"""Shared data loading, acquisition-batch detection and harmonization.

Every learned transformation (Yeo-Johnson, ComBat, standardization, feature
selection) is fitted on training data only and then applied to held-out data,
so that harmonization never leaks test information into model fitting.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from neuroCombat import neuroCombat, neuroCombatFromTraining
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PowerTransformer, StandardScaler

SEED = 507
SPLIT_SEED = 42
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "multi_omics"
OUT = ROOT / "results" / "supervised_report"
FIG = OUT / "figures"
TAB = OUT / "tables"
CACHE = OUT / "cache"
for d in (FIG, TAB, CACHE):
    d.mkdir(parents=True, exist_ok=True)

MODALITIES = ["T1", "T2", "ADC"]
BATCHES = ["C1", "C2a", "C2b"]
FEATURE_SETS = ["T1", "T2", "ADC", "T1+T2", "T1+ADC", "T2+ADC", "T1+T2+ADC"]

# Harmonization settings compared throughout the supervised analysis.
SETTINGS = ["Original (raw z-score)", "No harmonization", "ComBat (center)",
            "ComBat (acquisition batch)"]
PRIMARY = "ComBat (acquisition batch)"
SETTING_COLORS = dict(zip(SETTINGS, ["#BBBBBB", "#999999", "#E69F00",
                                     "#0072B2"]))
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
LABEL_COLORS = {0: "#999999", 1: "#D55E00"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})


def savefig(fig, name):
    fig.savefig(FIG / f"{name}.png")
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def load_cohort():
    """Complete-case cohort (patients with no missing T1/T2/ADC feature).

    Returns {modality: raw feature DataFrame indexed by case} and an id table
    with center, LVI label and acquisition batch.
    """
    views = {m: pd.read_excel(DATA / f"{m}.xlsx").set_index("case").sort_index()
             for m in MODALITIES}
    ids = views["T1"][["center", "Label"]].rename(columns={"Label": "LVI"})
    for m in MODALITIES:
        assert (views[m]["Label"] == ids["LVI"]).all()
        views[m] = views[m].drop(columns=["center", "Label"])
    complete = np.logical_and.reduce([views[m].notna().all(1)
                                      for m in MODALITIES])
    ids["batch"] = detect_acquisition_batch(views, ids["center"])
    keep = ids.index[complete]
    return {m: views[m].loc[keep] for m in MODALITIES}, ids.loc[keep]


def detect_acquisition_batch(raw, center, min_frac=0.2, cut=0.3):
    """Label-free detection of the two acquisition batches inside center 2.

    Same rule as unsupervised_extension/01_harmonization.py: a center-2
    patient belongs to C2b when, on the run-length texture features whose
    center-2 values exceed 10x the center-1 99th percentile, more than 30% of
    their values are extreme (averaged over T1, T2 and ADC).
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
    return batch


class Harmonizer:
    """Per-modality preprocessing fitted on training data.

    setting:
      "Original (raw z-score)"     standardize raw features (original pipeline)
      "No harmonization"           log / Yeo-Johnson transform, standardize
      "ComBat (center)"            transform, ComBat with center as batch
      "ComBat (acquisition batch)" transform, ComBat with C1/C2a/C2b as batch
    """

    def __init__(self, setting):
        assert setting in SETTINGS
        self.setting = setting

    def _marginal(self, X):
        Z = np.empty_like(X, dtype=float)
        for j in range(X.shape[1]):
            v = X[:, j].astype(float)
            if self.log_[j]:
                Z[:, j] = np.log(np.maximum(v, self.floor_[j]))
            else:
                Z[:, j] = self.pt_[j].transform(v.reshape(-1, 1)).ravel()
        return Z

    def _transform(self, X):
        return np.clip((self._marginal(X) - self.mu_) / self.sd_, -5, 5)

    def fit_transform(self, X, batch):
        X = X.copy()
        self.cols_ = list(X.columns)
        if self.setting == "Original (raw z-score)":
            self.scaler_ = StandardScaler().fit(X.values)
            return pd.DataFrame(self.scaler_.transform(X.values),
                                index=X.index, columns=self.cols_)
        # QC: drop features whose modal value covers >20% of a training batch
        # (degenerate features become batch fingerprints after ComBat).
        b = pd.Series(np.asarray(batch), index=X.index)
        mode = pd.concat([X[b == g].round(10).apply(
            lambda s: s.value_counts(normalize=True).iloc[0])
            for g in b.unique()], axis=1).max(1)
        self.cols_ = list(X.columns[mode.values <= 0.2])
        V = X[self.cols_].values.astype(float)
        self.log_ = (V > 0).all(0)
        self.floor_ = np.where(self.log_, V.min(0), 0) / 2
        self.pt_ = [None if self.log_[j] else
                    PowerTransformer("yeo-johnson", standardize=False)
                    .fit(V[:, [j]]) for j in range(V.shape[1])]
        M = self._marginal(V)
        self.mu_, self.sd_ = M.mean(0), M.std(0) + 1e-12
        Z = self._transform(V)
        if self.setting != "No harmonization":
            covars = pd.DataFrame({"batch": np.asarray(batch)})
            res = neuroCombat(dat=Z.T, covars=covars, batch_col="batch")
            self.estimates_ = res["estimates"]
            Z = res["data"].T
        self.scaler_ = StandardScaler().fit(Z)
        return pd.DataFrame(self.scaler_.transform(Z), index=X.index,
                            columns=self.cols_)

    def transform(self, X, batch):
        V = X[self.cols_].values.astype(float)
        if self.setting == "Original (raw z-score)":
            Z = V
        else:
            Z = self._transform(V)
            if self.setting != "No harmonization":
                Z = neuroCombatFromTraining(dat=Z.T, batch=np.asarray(batch),
                                            estimates=self.estimates_)["data"].T
        return pd.DataFrame(self.scaler_.transform(Z), index=X.index,
                            columns=self.cols_)


def batch_for(setting, ids):
    return ids["center"].astype(str) if setting == "ComBat (center)" \
        else ids["batch"]


def correlation_filter(X, threshold=0.8):
    """Drop feature j if |r| > threshold with any earlier feature i.

    Matches the rule of the original notebooks (an earlier feature removes
    later ones even if it was itself removed).
    """
    corr = np.abs(np.corrcoef(X, rowvar=False))
    upper = np.triu(corr > threshold, k=1)
    return ~upper.any(axis=0)


def select_features(X_train, y_train, k=50, corr_threshold=0.8):
    """SelectKBest (ANOVA F) then greedy |r| > 0.8 filter, on training data."""
    sel = SelectKBest(f_classif, k=min(k, X_train.shape[1])).fit(X_train,
                                                                  y_train)
    cols = X_train.columns[sel.get_support()]
    keep = correlation_filter(X_train[cols].values, corr_threshold)
    return list(cols[keep])


def prepare_split(raw, ids, train_idx, test_idx, setting, k=50):
    """Harmonize + select features for one train/test split.

    Returns {modality: (X_train, X_test)} with selected columns.
    """
    b = batch_for(setting, ids)
    y_tr = ids.loc[train_idx, "LVI"].values
    out = {}
    for m in MODALITIES:
        h = Harmonizer(setting)
        Xtr = h.fit_transform(raw[m].loc[train_idx], b.loc[train_idx])
        Xte = h.transform(raw[m].loc[test_idx], b.loc[test_idx])
        cols = select_features(Xtr, y_tr, k=k)
        out[m] = (Xtr[cols].add_prefix(m + ":"), Xte[cols].add_prefix(m + ":"))
    return out


def feature_set(prepared, name, part):
    i = 0 if part == "train" else 1
    return pd.concat([prepared[m][i] for m in name.split("+")], axis=1)
