"""Step 2: supervised baselines for LVI prediction under each harmonization.

For each harmonization setting (common.SETTINGS) on the shared stratified
80/20 split (seed 42):
  * linear logistic regression and linear SVM (L2), C tuned by 5-fold CV AUC,
    for single modalities, pairs and all three (early fusion);
  * late fusion of the single-modality models, weights tuned on training AUC;
  * RBF kernel logistic regression and RBF SVM, tuned by 5-fold CV error;
  * threshold tuning (recall / precision / F1) for the best linear model.

Robustness of the logistic-regression results is assessed with
  * repeated stratified 5-fold CV (10 repeats) over the whole cohort, with
    harmonization, feature selection and C tuning refitted inside each fold;
  * center-1-only repeated CV (one acquisition batch: no batch effect, and no
    batch/LVI confounding);
  * leave-one-center-out validation (harmonization fitted label-free on all
    features, models trained on one center and tested on the other).
"""

import pickle
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     train_test_split)
from sklearn.svm import SVC, LinearSVC

from common import (CACHE, FEATURE_SETS, MODALITIES, PRIMARY, SETTING_COLORS,
                    SETTINGS, SPLIT_SEED, TAB, Harmonizer, batch_for,
                    feature_set, load_cohort, plt, prepare_split, savefig,
                    select_features)

warnings.filterwarnings("ignore")
C_GRID = [0.01, 0.1, 1, 10, 100]
GAMMA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
ALPHA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10]
SVC_C_GRID = [0.01, 0.1, 1, 10, 100, 1000]


# ---- linear models -----------------------------------------------------
def make_linear(method, C):
    if method == "Logistic regression":
        return LogisticRegression(C=C, solver="liblinear", max_iter=1000)
    return LinearSVC(C=C, max_iter=20000)


def score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def tune_linear(X, y, method, seed=SPLIT_SEED):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    best = (-np.inf, None)
    for C in C_GRID:
        aucs = [roc_auc_score(y[va], score(make_linear(method, C)
                                           .fit(X[tr], y[tr]), X[va]))
                for tr, va in skf.split(X, y)]
        if np.mean(aucs) > best[0]:
            best = (np.mean(aucs), C)
    return best[1], best[0]


# ---- kernel models -----------------------------------------------------
class KernelLogisticRegression:
    """RBF kernel logistic regression with an RKHS-norm penalty."""

    def __init__(self, gamma=1.0, alpha=1.0, maxiter=300):
        self.gamma, self.alpha, self.maxiter = gamma, alpha, maxiter

    def fit(self, X, y):
        self.X_ = np.asarray(X)
        y = np.asarray(y, dtype=float)
        K = rbf_kernel(self.X_, self.X_, gamma=self.gamma)

        def fun(w):
            f = K @ w
            loss = np.sum(np.logaddexp(0, f) - y * f) + 0.5 * self.alpha * w @ K @ w
            grad = K @ (expit(f) - y) + self.alpha * (K @ w)
            return loss, grad

        self.w_ = minimize(fun, np.zeros(len(y)), jac=True, method="L-BFGS-B",
                           options={"maxiter": self.maxiter}).x
        return self

    def decision_function(self, X):
        return rbf_kernel(np.asarray(X), self.X_, gamma=self.gamma) @ self.w_

    def predict_proba(self, X):
        p = expit(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)


def tune_kernel(X, y, kind):
    skf = StratifiedKFold(5, shuffle=True, random_state=SPLIT_SEED)
    grid = [(g, a) for g in GAMMA_GRID
            for a in (ALPHA_GRID if kind == "klr" else SVC_C_GRID)]
    best = (np.inf, None)
    for g, a in grid:
        errs = []
        for tr, va in skf.split(X, y):
            m = (KernelLogisticRegression(g, a) if kind == "klr"
                 else SVC(kernel="rbf", gamma=g, C=a)).fit(X[tr], y[tr])
            errs.append(1 - accuracy_score(y[va], m.predict(X[va])))
        if np.mean(errs) < best[0]:
            best = (np.mean(errs), (g, a))
    g, a = best[1]
    model = (KernelLogisticRegression(g, a) if kind == "klr"
             else SVC(kernel="rbf", gamma=g, C=a))
    return model.fit(X, y), best[0], {"gamma": g, ("alpha" if kind == "klr"
                                                   else "C"): a}


# ---- late fusion -------------------------------------------------------
def late_fusion_weights(scores_train, y, grid_size=21):
    best = (-np.inf, None)
    for w1 in np.linspace(0, 1, grid_size):
        for w2 in np.linspace(0, 1 - w1, int(round((1 - w1) * (grid_size - 1))) + 1):
            w = (w1, w2, 1 - w1 - w2)
            s = sum(wi * si for wi, si in zip(w, scores_train))
            auc = roc_auc_score(y, s)
            if auc > best[0]:
                best = (auc, w)
    return best[1], best[0]


def run_split(raw, ids, train_idx, test_idx, setting):
    prepared = prepare_split(raw, ids, train_idx, test_idx, setting)
    ytr = ids.loc[train_idx, "LVI"].values
    yte = ids.loc[test_idx, "LVI"].values
    rows, fitted = [], {}
    for fs in FEATURE_SETS:
        Xtr = feature_set(prepared, fs, "train").values
        Xte = feature_set(prepared, fs, "test").values
        for method in ["Logistic regression", "Linear SVM"]:
            C, cv_auc = tune_linear(Xtr, ytr, method)
            m = make_linear(method, C).fit(Xtr, ytr)
            fitted[(fs, method)] = m
            rows.append({
                "setting": setting, "feature_set": fs, "model": method,
                "n_features": Xtr.shape[1], "params": f"C={C}",
                "cv_auc": cv_auc,
                "train_auc": roc_auc_score(ytr, score(m, Xtr)),
                "test_auc": roc_auc_score(yte, score(m, Xte)),
                "train_error": 1 - accuracy_score(ytr, m.predict(Xtr)),
                "test_error": 1 - accuracy_score(yte, m.predict(Xte))})
        for kind, name in [("klr", "Kernel logistic regression"),
                           ("svm", "Kernel SVM")]:
            m, cv_err, params = tune_kernel(Xtr, ytr, kind)
            rows.append({
                "setting": setting, "feature_set": fs, "model": name,
                "n_features": Xtr.shape[1],
                "params": ", ".join(f"{k}={v:g}" for k, v in params.items()),
                "cv_error": cv_err,
                "train_auc": roc_auc_score(ytr, m.decision_function(Xtr)),
                "test_auc": roc_auc_score(yte, m.decision_function(Xte)),
                "train_error": 1 - accuracy_score(ytr, m.predict(Xtr)),
                "test_error": 1 - accuracy_score(yte, m.predict(Xte))})
    for method, cut in [("Logistic regression", 0.5), ("Linear SVM", 0.0)]:
        str_ = [score(fitted[(m, method)],
                      feature_set(prepared, m, "train").values) for m in MODALITIES]
        ste = [score(fitted[(m, method)],
                     feature_set(prepared, m, "test").values) for m in MODALITIES]
        w, tr_auc = late_fusion_weights(str_, ytr)
        s_tr = sum(wi * si for wi, si in zip(w, str_))
        s_te = sum(wi * si for wi, si in zip(w, ste))
        rows.append({
            "setting": setting, "feature_set": "Late fusion (T1, T2, ADC)",
            "model": method,
            "params": "w=(" + ", ".join(f"{x:.2f}" for x in w) + ")",
            "train_auc": tr_auc, "test_auc": roc_auc_score(yte, s_te),
            "train_error": 1 - accuracy_score(ytr, s_tr > cut),
            "test_error": 1 - accuracy_score(yte, s_te > cut)})
    return pd.DataFrame(rows), prepared, fitted


def threshold_curve(y, p):
    rows = []
    for t in np.round(np.arange(0.1, 0.9, 0.05), 2):
        pred = (p >= t).astype(int)
        rows.append({"threshold": t, "recall": recall_score(y, pred),
                     "precision": precision_score(y, pred, zero_division=0),
                     "f1": f1_score(y, pred, zero_division=0)})
    return pd.DataFrame(rows)


def lr_pipeline_auc(raw, ids, tr, te, setting, fs, harmonize_all=False):
    """Fit harmonization + selection + tuned LR on tr, return test AUC."""
    ytr = ids.loc[tr, "LVI"].values
    yte = ids.loc[te, "LVI"].values
    b = batch_for(setting, ids)
    Xtr_parts, Xte_parts = [], []
    for m in fs.split("+"):
        h = Harmonizer(setting)
        if harmonize_all:   # label-free, transductive (leave-one-center-out)
            H = h.fit_transform(raw[m], b)
            Xtr, Xte = H.loc[tr], H.loc[te]
        else:
            Xtr = h.fit_transform(raw[m].loc[tr], b.loc[tr])
            Xte = h.transform(raw[m].loc[te], b.loc[te])
        cols = select_features(Xtr, ytr)
        Xtr_parts.append(Xtr[cols]); Xte_parts.append(Xte[cols])
    Xtr = pd.concat(Xtr_parts, axis=1).values
    Xte = pd.concat(Xte_parts, axis=1).values
    C, _ = tune_linear(Xtr, ytr, "Logistic regression")
    m = make_linear("Logistic regression", C).fit(Xtr, ytr)
    return roc_auc_score(yte, score(m, Xte))


def robustness(raw, ids):
    rows = []
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10,
                                  random_state=SPLIT_SEED)
    strata = ids["LVI"].astype(str) + ids["batch"]
    for setting in SETTINGS:
        for fs in ["T1+T2", "T1+T2+ADC"]:
            aucs = [lr_pipeline_auc(raw, ids, ids.index[tr], ids.index[te],
                                    setting, fs)
                    for tr, te in rcv.split(ids, strata)]
            rows.append({"analysis": "Repeated 5-fold CV (all patients)",
                         "setting": setting, "feature_set": fs,
                         "auc_mean": np.mean(aucs), "auc_sd": np.std(aucs),
                         "n": len(ids)})
            print(rows[-1])
    c1 = ids[ids.batch == "C1"]
    for fs in ["T1", "T2", "ADC", "T1+T2", "T1+T2+ADC"]:
        aucs = [lr_pipeline_auc(raw, c1, c1.index[tr], c1.index[te],
                                "No harmonization", fs)
                for tr, te in rcv.split(c1, c1["LVI"])]
        rows.append({"analysis": "Repeated 5-fold CV (center 1 only)",
                     "setting": "Single batch", "feature_set": fs,
                     "auc_mean": np.mean(aucs), "auc_sd": np.std(aucs),
                     "n": len(c1)})
        print(rows[-1])
    for setting in ["No harmonization", "ComBat (center)", PRIMARY]:
        for train_c, test_c in [(1, 2), (2, 1)]:
            tr = ids.index[ids.center == train_c]
            te = ids.index[ids.center == test_c]
            rows.append({"analysis": f"Train center {train_c} -> test center {test_c}",
                         "setting": setting, "feature_set": "T1+T2",
                         "auc_mean": lr_pipeline_auc(raw, ids, tr, te, setting,
                                                     "T1+T2", harmonize_all=True),
                         "auc_sd": np.nan, "n": len(te)})
            print(rows[-1])
    return pd.DataFrame(rows)


def main():
    raw, ids = load_cohort()
    tr, te = train_test_split(ids.index, test_size=0.2, random_state=SPLIT_SEED,
                              stratify=ids["LVI"])
    pd.DataFrame({"case": ids.index, "split": np.where(ids.index.isin(tr),
                                                       "train", "test")}) \
        .merge(ids, left_on="case", right_index=True) \
        .to_csv(TAB / "cohort_split.csv", index=False)

    all_rows, store = [], {}
    for setting in SETTINGS:
        df, prepared, fitted = run_split(raw, ids, tr, te, setting)
        all_rows.append(df)
        store[setting] = {"prepared": prepared, "fitted": fitted}
        print(df[["setting", "feature_set", "model", "cv_auc", "train_auc",
                  "test_auc", "test_error"]].round(3).to_string())
    res = pd.concat(all_rows, ignore_index=True)
    res.to_csv(TAB / "baselines_all.csv", index=False)

    # Best linear model per setting (highest CV AUC), and threshold tuning
    lin = res[res.model.isin(["Logistic regression", "Linear SVM"])
              & res.cv_auc.notna()]
    best = lin.loc[lin.groupby("setting").cv_auc.idxmax()]
    best.to_csv(TAB / "baselines_best_linear.csv", index=False)
    print(best.round(3).to_string())

    p = store[PRIMARY]
    ytr = ids.loc[tr, "LVI"].values
    yte = ids.loc[te, "LVI"].values
    lr_best = lin[(lin.setting == PRIMARY) & (lin.model == "Logistic regression")]
    fs_best = lr_best.loc[lr_best.cv_auc.idxmax(), "feature_set"]
    m = p["fitted"][(fs_best, "Logistic regression")]
    ptr = m.predict_proba(feature_set(p["prepared"], fs_best, "train").values)[:, 1]
    pte = m.predict_proba(feature_set(p["prepared"], fs_best, "test").values)[:, 1]
    curve = threshold_curve(ytr, ptr)
    curve.to_csv(TAB / "threshold_curve_primary.csv", index=False)
    t_best = float(curve.loc[curve.f1.idxmax(), "threshold"])
    test_at = {"feature_set": fs_best, "threshold": t_best,
               "test_cm": confusion_matrix(yte, (pte >= t_best).astype(int)).tolist(),
               "test_recall": recall_score(yte, pte >= t_best),
               "test_precision": precision_score(yte, pte >= t_best, zero_division=0),
               "test_f1": f1_score(yte, pte >= t_best),
               "train_cm_05": confusion_matrix(ytr, (ptr >= 0.5).astype(int)).tolist(),
               "best_train_row": curve.loc[curve.f1.idxmax()].to_dict()}
    pd.Series(test_at).to_json(TAB / "threshold_primary.json", indent=1)
    print(test_at)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    cm = np.array(test_at["train_cm_05"])
    axes[0].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
    axes[0].set_xticks([0, 1], ["LVI-", "LVI+"]); axes[0].set_yticks([0, 1], ["LVI-", "LVI+"])
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    axes[0].set_title(f"Training confusion matrix ({fs_best}, t = 0.5)")
    for col, c in [("recall", "#0072B2"), ("precision", "#E69F00"),
                   ("f1", "#009E73")]:
        axes[1].plot(curve.threshold, curve[col], "o-", ms=3, color=c,
                     label=col.capitalize() if col != "f1" else "F1")
    axes[1].axvline(t_best, ls=":", color="#999999")
    axes[1].set_xlabel("Threshold"); axes[1].set_ylabel("Training score")
    axes[1].set_title("Threshold tuning"); axes[1].legend()
    fig.tight_layout()
    savefig(fig, "baseline_threshold")

    rob = robustness(raw, ids)
    rob.to_csv(TAB / "robustness.csv", index=False)

    # Harmonization comparison figure: test AUC (split) and repeated-CV AUC
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    lr = res[res.model == "Logistic regression"]
    x = np.arange(len(FEATURE_SETS)); w = 0.2
    for k, s in enumerate(SETTINGS):
        v = lr[lr.setting == s].set_index("feature_set").loc[FEATURE_SETS]
        axes[0].bar(x + (k - 1.5) * w, v.cv_auc, w, color=SETTING_COLORS[s],
                    label=s)
    axes[0].set_xticks(x, FEATURE_SETS, rotation=30, fontsize=7.5)
    axes[0].set_ylim(0.5, 1.08); axes[0].set_ylabel("5-fold CV AUC (training set)")
    axes[0].set_title("Logistic regression")
    axes[0].legend(fontsize=6.5, ncol=2, loc="upper center")
    r = rob[rob.analysis.str.startswith("Repeated 5-fold CV (all")]
    for k, s in enumerate(SETTINGS):
        v = r[r.setting == s]
        axes[1].errorbar(np.arange(len(v)) + (k - 1.5) * 0.15, v.auc_mean,
                         yerr=v.auc_sd, fmt="o", color=SETTING_COLORS[s],
                         capsize=2)
    c1 = rob[rob.analysis.str.contains("center 1 only")
             & rob.feature_set.isin(["T1+T2", "T1+T2+ADC"])]
    axes[1].scatter(np.arange(len(c1)) + 0.45, c1.auc_mean, marker="D",
                    color="#009E73", label="Center 1 only")
    axes[1].errorbar(np.arange(len(c1)) + 0.45, c1.auc_mean, yerr=c1.auc_sd,
                     fmt="none", color="#009E73", capsize=2)
    axes[1].set_xticks([0, 1], ["T1+T2", "T1+T2+ADC"])
    axes[1].set_ylim(0.5, 1); axes[1].set_ylabel("Repeated CV AUC (mean ± SD)")
    axes[1].set_title("Logistic regression, 10x5-fold CV"); axes[1].legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, "baseline_harmonization_auc")

    with open(CACHE / "baselines.pkl", "wb") as f:
        pickle.dump({"train_idx": tr, "test_idx": te, "store": store,
                     "results": res}, f)


if __name__ == "__main__":
    main()
