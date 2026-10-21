"""Step 6: write booktabs LaTeX tables and copy figures for the report."""

import shutil

import pandas as pd

from common import FIG, ROOT, TAB

REPORT = ROOT / "report" / "Unsupervised_Extension"
RT = REPORT / "tables"
RF = ROOT / "report" / "figures"  # shared figures folder, as on Overleaf
RT.mkdir(parents=True, exist_ok=True)
RF.mkdir(parents=True, exist_ok=True)


def esc(s):
    return (str(s).replace("%", r"\%").replace("&", r"\&")
            .replace("_", r"\_").replace("^2", r"$^2$"))


def write(name, df, caption, label, colfmt, note=None, size=r"\small",
          fit=False):
    body = df.to_latex(index=False, escape=False, column_format=colfmt).strip()
    if fit:
        body = r"\resizebox{\linewidth}{!}{%" + "\n" + body + "}"
    out = [r"\begin{table}[htbp]", r"\centering", size,
           rf"\caption{{{caption}}}", rf"\label{{{label}}}", body]
    if note:
        out.append(rf"\par\smallskip\parbox{{0.95\linewidth}}{{\footnotesize {note}}}")
    out.append(r"\end{table}")
    (RT / f"{name}.tex").write_text("\n".join(out) + "\n")


def fmt_p(p):
    if pd.isna(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def main():
    d = pd.read_csv(TAB / "data_integrity.csv")
    d = d.rename(columns={"batch": "Batch", "n": "$n$"})
    for c in d.columns[2:]:
        d[c] = d[c].map(lambda v: f"{v:.2f}")
    d.columns = ["Batch", "$n$", "Vol.\\ T1--T2", "Vol.\\ T1--ADC",
                 "Vol.\\ T2--ADC", "Vol.\\ vs diameter", "ADC vs radiologist"]
    write("integrity_table", d,
          "Data-integrity checks: Spearman correlations between quantities "
          "that should agree for the same patient.", "tab:ue-integrity",
          "lrrrrrr", "Vol.: radiomic VolumeCount; diameter: radiologist maximum "
          "diameter; ADC: whole-tumor radiomic mean ADC vs radiologist ADC.",
          fit=True)

    h = pd.read_csv(TAB / "harmonization_summary.csv")
    h = h[h.modality != "Concatenated"].copy()
    rows = []
    for _, r in h.iterrows():
        rows.append({
            "Strategy": r.stage, "Seq.": r.modality,
            "Loc.\\ sig.\\ (\\%)": f"{r.pct_location_sig:.0f}",
            "Scale sig.\\ (\\%)": f"{r.pct_scale_sig:.0f}",
            "$R^2_{\\text{batch}}$": f"{r.permanova_R2_batch:.3f}",
            "Silh.": f"{r.silhouette_batch_PCA10:.2f}",
            "RF AUC": f"{r.batch_AUC_RF:.2f}",
            "LVI feat.\\ (\\%)": f"{r.pct_LVI_sig_features:.0f}",
            "LVI AUC": f"{r.LVI_AUC_LR:.2f}"})
    t = pd.DataFrame(rows)
    t.loc[t.duplicated("Strategy"), "Strategy"] = ""
    write("harmonization_table", t,
          "Harmonization diagnostics against acquisition batch.",
          "tab:ue-harm", "llrrrrrrr",
          "Loc./Scale sig.: \\% of features with Kruskal--Wallis / "
          "Brown--Forsythe $q<0.05$ for batch; $R^2_{\\text{batch}}$: "
          "PERMANOVA; Silh.: silhouette of batch labels in 10 PCs; RF AUC: "
          "5-fold CV random-forest AUC predicting batch; LVI feat.: \\% of "
          "features with LVI $q<0.05$; LVI AUC: CV logistic AUC.",
          size=r"\scriptsize", fit=True)

    k = pd.read_csv(TAB / "cluster_selection_primary.csv")
    t = pd.DataFrame({
        "$k$": k.k.astype(str), "Silh.": k.silhouette.map("{:.3f}".format),
        "ARI (vs full)": [f"{m:.2f} $\\pm$ {s:.2f}" for m, s in
                          zip(k.ARI_vs_full_mean, k.ARI_vs_full_sd)],
        "ARI (pairwise)": k.ARI_pairwise_mean.map("{:.2f}".format),
        "PAC": k.PAC.map("{:.2f}".format),
        "Min.\\ prop.": k.min_cluster_prop.map("{:.2f}".format),
        "Sizes": k.sizes.str.replace("/", "/\\allowbreak "),
        "$V_{\\text{batch}}$": k.CramerV_batch.map("{:.2f}".format),
        "$V_{\\text{LVI}}$": k.CramerV_LVI.map("{:.2f}".format),
        "Score": k.composite_score.map(lambda v: "--" if pd.isna(v)
                                       else f"{v:.2f}")})
    t.loc[k.selected.values, "$k$"] = t.loc[k.selected.values, "$k$"].map(
        lambda v: f"\\textbf{{{v}}}")
    write("k_selection_table", t,
          "Internal validation of $k$-means partitions of the MOFA factor "
          "scores (primary harmonization).", "tab:ue-k", "rrrrrrlrrr",
          "ARI: mean $\\pm$ SD over 200 subsamples of 80\\%; PAC: proportion "
          "of consensus indices in (0.1, 0.9); Score: composite rank "
          "criterion (partitions with a cluster $<$10\\% excluded). Bold: "
          "selected $k$.", size=r"\scriptsize", fit=True)

    cb = pd.read_csv(TAB / "cluster_by_batch.csv")
    cb.columns = ["Cluster", "C1", "C2a", "C2b", "Total"]
    lv = pd.read_csv(TAB / "cluster_assignments.csv")
    pct = lv.groupby("cluster").Label.mean().mul(100).round(0)
    cb["LVI+ (\\%)"] = [f"{pct.get(int(c), lv.Label.mean() * 100):.0f}"
                         if str(c) != "All" else
                         f"{lv.Label.mean() * 100:.0f}" for c in cb.Cluster]
    write("cluster_batch_table", cb,
          "Primary clusters ($k=3$) by acquisition batch.",
          "tab:ue-clbatch", "lrrrrr")

    rb = pd.read_csv(TAB / "cluster_robustness.csv")
    t = pd.DataFrame({"Comparison": rb.comparison.map(esc),
                      "$n$": rb.n, "ARI": rb.ARI.map("{:.2f}".format),
                      "Selected $k$ (other)": rb.other_selected_k.map(
                          lambda v: "--" if pd.isna(v) else f"{int(v)}")})
    write("robustness_table", t,
          "Robustness of the primary partition ($k=3$) to the clustering "
          "algorithm and the harmonization choice.", "tab:ue-robust", "lrrr",
          fit=True)

    p = pd.read_csv(TAB / "posthoc_univariate_primary.csv")
    t = pd.DataFrame({
        "Variable": p.variable.map(esc),
        "Cluster 1 | 2 | 3": p.summary_by_cluster.str.replace(
            "|", "$|$", regex=False).map(lambda s: s.replace("%", r"\%")),
        "Effect": [f"{e:.3f} ({'$\\epsilon^2$' if m == 'epsilon2' else '$V$'})"
                   for e, m in zip(p.effect_size, p.effect_measure)],
        "$q$": p.q.map(fmt_p),
        "$p_{\\text{adj}}$": p.p_batch_adjusted.map(fmt_p),
        "$q_{\\text{adj}}$": p.q_batch_adjusted.map(fmt_p),
        "$p_{\\text{C1}}$": p.p_within_C1.map(fmt_p),
    })
    write("posthoc_table", t,
          "Post-hoc comparison of clinical, imaging-report and pathology "
          "variables across the primary clusters.", "tab:ue-posthoc",
          "llrrrrr",
          "Continuous: median [IQR], Kruskal--Wallis; binary: \\% positive; "
          "composition: \\% per category; $\\chi^2$ or Monte-Carlo $\\chi^2$. "
          "adj: batch-adjusted likelihood-ratio test; C1: within center 1. "
          "$q$: BH-FDR across variables. No variable met the retention rule "
          "($q<0.05$ unadjusted and adjusted with at least a small effect).",
          size=r"\scriptsize", fit=True)

    rows = []
    for name, f in [("Primary ($k=3$)", "posthoc_models_primary.csv"),
                    ("Center 1 only ($k=2$)", "posthoc_models_center1_only.csv")]:
        m = pd.read_csv(TAB / f)
        for _, r in m.iterrows():
            rows.append({"Clustering": name, "Predictors": esc(r.model),
                         "\\#": r.n_predictors,
                         "CV AUC": f"{r.cv_macro_AUC:.2f} $\\pm$ {r.cv_sd:.2f}"})
    t = pd.DataFrame(rows)
    t.loc[t.duplicated("Clustering"), "Clustering"] = ""
    write("posthoc_models_table", t,
          "Explanatory value of post-hoc variables for cluster membership "
          "(multinomial logistic regression, 10$\\times$5-fold CV macro-AUC).",
          "tab:ue-models", "llrr", size=r"\scriptsize", fit=True)

    for f in FIG.glob("*.pdf"):
        shutil.copy(f, RF / f.name)
    print("tables:", sorted(x.name for x in RT.iterdir()))


if __name__ == "__main__":
    main()
