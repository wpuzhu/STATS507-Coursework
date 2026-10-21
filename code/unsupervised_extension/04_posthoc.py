"""Step 4: post-hoc characterization of the latent-space clusters.

Clinical, imaging-report and pathology variables from post_survey.xlsx are
compared across clusters (primary clustering, and the center-1-only
clustering as a confound-free check):
  * continuous: Kruskal-Wallis, effect size epsilon^2;
  * categorical: chi-squared, or a Monte-Carlo (permutation) chi-squared when
    any expected count is < 5; effect size Cramer's V;
  * BH-FDR across variables;
  * batch-adjusted test: likelihood-ratio test of the cluster term in
    rank(variable) ~ cluster + batch (OLS) for continuous variables and
    logistic / multinomial regression variable ~ cluster + batch for
    categorical variables, BH-FDR.
  * within-center-1 test (single acquisition batch, so free of the
    batch/outcome confounding present in center 2).
A variable is retained as "relevant and explanatory" when it is significant
unadjusted and batch-adjusted (q < 0.05) with at least a small effect size
(epsilon^2 >= 0.01 or V >= 0.10); a "candidate" tier lists variables with
nominal batch-adjusted p < 0.05 and at least a small effect, flagged by
whether the association replicates within center 1 (p < 0.05). Retained
(or, if none, candidate) variables then enter an L2
multinomial logistic model of cluster membership whose repeated 5-fold CV
macro-AUC is compared with (i) all post-hoc variables and (ii) batch alone.
Clusters are also described by their mean MOFA factor scores and the
highest-weighted radiomic features of the discriminating factors.
"""

import pickle
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from common import CACHE, DATA, PALETTE, SEED, TAB, bh, plt, savefig

warnings.filterwarnings("ignore")
rng = np.random.default_rng(SEED)

# source column -> (English name, type)
VARIABLES = {
    "Label": ("LVI", "binary"),
    "ADC值": ("ADC value (radiologist)", "continuous"),
    "最大径": ("Maximum diameter (cm)", "continuous"),
    "深肌层浸润": ("Deep myometrial invasion", "binary"),
    "年龄": ("Age (years)", "continuous"),
    "绝经情况": ("Postmenopausal", "binary"),
    "肿瘤边界": ("Tumour margin", "binary"),
    "DWI信号": ("DWI hyperintensity", "binary"),
    "腹水": ("Ascites", "binary"),
    "淋巴结阳性": ("Lymph-node positive", "binary"),
    "囊实性": ("Cystic/solid composition", "categorical"),
    "动脉早期1": ("Enhancement ratio, early arterial", "continuous"),
    "动脉晚期2": ("Enhancement ratio, late arterial", "continuous"),
    "实质期3": ("Enhancement ratio, parenchymal", "continuous"),
    "延迟期4": ("Enhancement ratio, delayed", "continuous"),
    "相对T2": ("Relative T2 signal", "continuous"),
    "T 2WI信号 ": ("T2WI signal grade", "binary"),
    "P53": ("p53 abnormal", "binary"),
    "ki67": ("Ki-67 (%)", "continuous"),
    "ER": ("ER (%)", "continuous"),
    "PR": ("PR (%)", "continuous"),
}


def load_posthoc():
    ps = pd.read_excel(DATA / "post_survey.xlsx").set_index("case").sort_index()
    cleaning = []
    bad = ~ps["P53"].isin([0, 1])
    for c in ps.index[bad]:
        cleaning.append({"case": c, "variable": "P53", "value": ps.at[c, "P53"],
                         "action": "set to missing (not 0/1)"})
    ps.loc[bad, "P53"] = np.nan
    bad = ps["PR"] > 100
    for c in ps.index[bad]:
        cleaning.append({"case": c, "variable": "PR", "value": ps.at[c, "PR"],
                         "action": "set to missing (>100%)"})
    ps.loc[bad, "PR"] = np.nan
    pd.DataFrame(cleaning).to_csv(TAB / "posthoc_data_cleaning.csv",
                                  index=False)
    df = ps[list(VARIABLES)].rename(columns={k: v[0] for k, v in
                                             VARIABLES.items()})
    types = {v[0]: v[1] for v in VARIABLES.values()}
    return df, types


def chi2_test(x, g, n_mc=4999):
    tab = pd.crosstab(x, g)
    chi2, p, _, exp = stats.chi2_contingency(tab, correction=False)
    v = np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))
    method = "chi-squared"
    if (exp < 5).any():
        null = np.empty(n_mc)
        xv, gv = np.asarray(x), np.asarray(g)
        for i in range(n_mc):
            null[i] = stats.chi2_contingency(
                pd.crosstab(xv, rng.permutation(gv)), correction=False)[0]
        p = (1 + (null >= chi2).sum()) / (n_mc + 1)
        method = "Monte-Carlo chi-squared"
    return p, v, method


def adjusted_lrt(y, cluster, batch, kind):
    """LRT for the cluster term with batch as nuisance covariate."""
    d = pd.DataFrame({"y": y, "cl": cluster, "b": batch}).dropna()
    Xb = pd.get_dummies(d["b"], drop_first=True, dtype=float)
    Xc = pd.get_dummies(d["cl"], prefix="cl", drop_first=True, dtype=float)
    X0 = sm.add_constant(Xb, has_constant="add")
    X1 = sm.add_constant(pd.concat([Xb, Xc], axis=1), has_constant="add")
    if d["b"].nunique() == 1:
        X0 = X0[["const"]]
        X1 = X1.drop(columns=Xb.columns)
    try:
        if kind == "continuous":
            yy = stats.rankdata(d["y"])
            l0 = sm.OLS(yy, X0).fit().llf
            l1 = sm.OLS(yy, X1).fit().llf
        elif kind == "binary":
            yy = (d["y"] == d["y"].max()).astype(float)
            l0 = sm.Logit(yy, X0).fit(disp=0, method="bfgs", maxiter=500).llf
            l1 = sm.Logit(yy, X1).fit(disp=0, method="bfgs", maxiter=500).llf
        else:
            yy = pd.factorize(d["y"], sort=True)[0]
            l0 = sm.MNLogit(yy, X0).fit(disp=0, method="bfgs", maxiter=500).llf
            l1 = sm.MNLogit(yy, X1).fit(disp=0, method="bfgs", maxiter=500).llf
    except Exception:
        return np.nan
    df_diff = (X1.shape[1] - X0.shape[1]) * (
        d["y"].nunique() - 1 if kind == "categorical" else 1)
    return stats.chi2.sf(2 * (l1 - l0), df_diff)


def univariate(df, types, cluster, batch):
    rows = []
    k = cluster.nunique()
    for var, kind in types.items():
        x = df[var]
        ok = x.notna()
        if kind == "continuous":
            groups = [x[ok & (cluster == c)] for c in sorted(cluster.unique())]
            h = stats.kruskal(*groups)
            p, eff, method = h.pvalue, max(0, (h.statistic - k + 1)
                                            / (ok.sum() - k)), "Kruskal-Wallis"
            summ = " | ".join(f"{np.median(g):.2f} [{np.percentile(g, 25):.2f}-"
                              f"{np.percentile(g, 75):.2f}]" for g in groups)
            eff_name = "epsilon2"
        else:
            p, eff, method = chi2_test(x[ok], cluster[ok])
            if kind == "binary":
                top = x.max()
                summ = " | ".join(
                    f"{(x[ok & (cluster == c)] == top).mean() * 100:.0f}%"
                    for c in sorted(cluster.unique()))
            else:
                summ = " | ".join(
                    "/".join(f"{v:.0f}" for v in x[ok & (cluster == c)]
                             .value_counts(normalize=True).sort_index() * 100)
                    for c in sorted(cluster.unique()))
            eff_name = "CramerV"
        rows.append({"variable": var, "type": kind, "test": method,
                     "summary_by_cluster": summ, "effect_size": eff,
                     "effect_measure": eff_name, "p": p,
                     "p_batch_adjusted": adjusted_lrt(x, cluster, batch, kind),
                     "n": int(ok.sum())})
    res = pd.DataFrame(rows)
    res["q"] = bh(res["p"])
    res["q_batch_adjusted"] = bh(res["p_batch_adjusted"])
    small = np.where(res.effect_measure == "epsilon2", 0.01, 0.10)
    res["retained"] = ((res.q < 0.05) & (res.q_batch_adjusted < 0.05)
                       & (res.effect_size >= small))
    res["candidate"] = (res.p_batch_adjusted < 0.05) & (res.effect_size >= small)
    return res


def cv_macro_auc(X, y, cat_cols=()):
    num_cols = [c for c in X.columns if c not in cat_cols]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), list(cat_cols))])
    model = make_pipeline(pre, LogisticRegression(C=1.0, max_iter=5000))
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    scoring = "roc_auc" if y.nunique() == 2 else "roc_auc_ovr"
    s = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return s.mean(), s.std()


def explanatory_models(df, types, res, cluster, batch):
    Xall = df.copy()
    for c in Xall.columns:
        Xall[c] = Xall[c].fillna(Xall[c].median())
    cat_all = [v for v, t in types.items() if t == "categorical"]
    keep = res.loc[res.retained, "variable"].tolist()
    label = "Retained variables"
    if not keep:
        keep = res.loc[res.candidate, "variable"].tolist()
        label = "Candidate variables"
    rows = []
    specs = [("Batch only", pd.get_dummies(batch, dtype=float), []),
             ("All post-hoc variables", Xall, cat_all)]
    if keep:
        specs.insert(1, (f"{label}: {', '.join(keep)}", Xall[keep],
                         [c for c in keep if c in cat_all]))
        specs.append((f"{label} + batch",
                      pd.concat([Xall[keep], pd.get_dummies(batch, dtype=float)],
                                axis=1), [c for c in keep if c in cat_all]))
    for name, X, cats in specs:
        if X.shape[1] == 1 and X.iloc[:, 0].nunique() == 1:
            continue
        m, s = cv_macro_auc(X, cluster, cats)
        rows.append({"model": name, "n_predictors": X.shape[1],
                     "cv_macro_AUC": m, "cv_sd": s})
    return pd.DataFrame(rows), keep


def plot_retained(df, types, res, cluster, name):
    keep = res.loc[res.retained | res.candidate].sort_values(
        "effect_size", ascending=False)
    if keep.empty:
        return
    cont = [v for v in keep.variable if types[v] == "continuous"]
    cat = [v for v in keep.variable if types[v] != "continuous"]
    n = len(cont) + len(cat)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.3 * nrow),
                             squeeze=False)
    cl = sorted(cluster.unique())
    for ax, v in zip(axes.flat, cont + cat):
        r = res.set_index("variable").loc[v]
        if v in cont:
            data = [df.loc[cluster == c, v].dropna() for c in cl]
            bp = ax.boxplot(data, widths=0.6, patch_artist=True,
                            showfliers=False)
            for patch, c in zip(bp["boxes"], cl):
                patch.set_facecolor(PALETTE[c - 1]); patch.set_alpha(0.7)
            for i, d in enumerate(data):
                ax.scatter(np.full(len(d), i + 1) + rng.normal(0, 0.06, len(d)),
                           d, s=2, color="k", alpha=0.3)
            ax.set_xticks(range(1, len(cl) + 1), [f"C{c}" for c in cl])
        else:
            tab = pd.crosstab(cluster, df[v], normalize="index") * 100
            bottom = np.zeros(len(tab))
            for j, col in enumerate(tab.columns):
                ax.bar([f"C{c}" for c in tab.index], tab[col], bottom=bottom,
                       color=plt.cm.Greys(0.3 + 0.5 * j / max(1, tab.shape[1] - 1)),
                       label=str(col))
                bottom += tab[col].values
            ax.set_ylabel("%")
            ax.legend(fontsize=6, title=None)
        ax.set_title(f"{v}\nq={r.q:.1e}, adj q={r.q_batch_adjusted:.1e}",
                     fontsize=7.5)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    savefig(fig, f"posthoc_retained_{name}")


def plot_effects(res, name):
    r = res.sort_values("effect_size")
    fig, ax = plt.subplots(figsize=(5, 0.22 * len(r) + 0.8))
    col = np.where(r.retained, PALETTE[0],
                   np.where(r.candidate, PALETTE[1], "#BBBBBB"))
    ax.barh(r.variable, r.effect_size, color=col)
    for i, (_, row) in enumerate(r.iterrows()):
        ax.text(row.effect_size, i, f"  q={row.q:.2g}", va="center", fontsize=6)
    ax.set_xlabel(r"Effect size ($\epsilon^2$ or Cramer's V)")
    ax.set_title("Blue: retained (unadjusted & batch-adjusted q<0.05)\n"
                 "orange: candidate (batch-adjusted p<0.05)", fontsize=8)
    savefig(fig, f"posthoc_effects_{name}")


def describe_factors(name, clusters, mofa_res):
    Z = mofa_res["Z"].loc[clusters.index]
    zs = (Z - Z.mean()) / Z.std()
    means = zs.groupby(clusters["cluster"]).mean()
    eps = {}
    for f in Z.columns:
        h = stats.kruskal(*[Z.loc[clusters.cluster == c, f]
                            for c in sorted(clusters.cluster.unique())])
        k = clusters.cluster.nunique()
        eps[f] = max(0, (h.statistic - k + 1) / (len(Z) - k))
    means.loc["epsilon2"] = pd.Series(eps)
    means.round(3).to_csv(TAB / f"cluster_factor_means_{name}.csv")
    r2 = mofa_res["r2"]
    top = sorted(eps, key=eps.get, reverse=True)[:3]
    rows = []
    for f in top:
        for m, W in mofa_res["W"].items():
            if r2.loc[m, f] < 1:
                continue
            w = W[f]
            for feat in w.abs().sort_values(ascending=False).index[:5]:
                rows.append({"factor": f, "factor_epsilon2": eps[f],
                             "view": m, "view_R2_pct": r2.loc[m, f],
                             "feature": feat, "weight": w[feat]})
    pd.DataFrame(rows).to_csv(TAB / f"cluster_driving_features_{name}.csv",
                              index=False)
    return means


def run(name, clusters, ids, df, types, mofa_res):
    cl = clusters["cluster"]
    d = df.loc[cl.index]
    batch = ids.loc[cl.index, "batch"]
    res = univariate(d, types, cl, batch)
    in_c1 = (batch == "C1").values
    if batch.nunique() > 1 and cl[in_c1].nunique() > 1:
        c1 = univariate(d[in_c1], types, cl[in_c1], batch[in_c1])
        res["p_within_C1"] = c1["p"].values
        res["candidate_replicates_C1"] = res.candidate & (res.p_within_C1 < 0.05)
    res.to_csv(TAB / f"posthoc_univariate_{name}.csv", index=False)
    models, keep = explanatory_models(d, types, res, cl, batch)
    models.to_csv(TAB / f"posthoc_models_{name}.csv", index=False)
    print("=" * 20, name)
    print(res[[c for c in ["variable", "effect_size", "p", "q",
                           "p_batch_adjusted", "q_batch_adjusted",
                           "p_within_C1", "retained", "candidate"]
               if c in res]].round(4).to_string())
    print(models.round(3).to_string())
    plot_retained(d, types, res, cl, name)
    plot_effects(res, name)
    means = describe_factors(name, clusters, mofa_res)
    print(means.round(2).to_string())
    return res, models


def main():
    with open(CACHE / "clusters.pkl", "rb") as f:
        cdata = pickle.load(f)
    with open(CACHE / "mofa.pkl", "rb") as f:
        mofa = pickle.load(f)
    ids = mofa["ids"]
    df, types = load_posthoc()

    run("primary", cdata["clusters"], ids, df, types,
        mofa["results"]["ComBat (batch)"])

    o = cdata["out"]["Center 1 only"]
    lab = o["labels"][o["best_k"]]
    order = np.argsort(-np.bincount(lab))
    lab = np.array([{old: new for new, old in enumerate(order)}[l]
                    for l in lab]) + 1
    c1 = pd.DataFrame({"cluster": lab}, index=o["index"])
    c1.to_csv(TAB / "cluster_assignments_center1_only.csv")
    run("center1_only", c1, ids, df, types, mofa["results"]["Center 1 only"])


if __name__ == "__main__":
    main()
