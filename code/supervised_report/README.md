# Supervised Report

Code for the report *Multi-modal MRI for Identifying Lymphovascular Invasion in
Endometrial Cancer* (`report/Supervised_Report.tex`). It turns the original
notebooks in `code/notebooks/` into scripts and adds center harmonization.

Run all four steps (and the extension) with `../../run_all.sh`, or run the
scripts in order from this folder; step 3 reuses the split and features cached
by step 2.

| Script | What it does | Main outputs (`results/supervised_report/`) | Report section |
|---|---|---|---|
| `common.py` | Complete-case cohort (n = 329), label-free acquisition-batch detection, train-only harmonization (`Harmonizer`), feature selection | — | II-A, II-B |
| `01_eda.py` | Cohort table; harmonization diagnostics (batch-associated features, PERMANOVA R², batch silhouette); cross-modality global agreement and local t-SNE structure | `tables/cohort.csv`, `harmonization_diagnostics.csv`, `global_agreement.csv`; `figures/harmonization_pca`, `global_agreement_*`, `local_neighborhood_*` | III-A, III-B |
| `02_supervised_baselines.py` | Four harmonization settings × seven modality combinations: logistic regression, linear SVM, late fusion, RBF kernel models; threshold tuning; repeated CV, center-1-only CV, leave-one-center-out | `tables/baselines_*`, `robustness.csv`, `threshold_*`; `figures/baseline_*` | III-C |
| `03_fusion_networks.py` | Single-modality, early-fusion, global-weight and gated fusion networks, 10 seeds, original vs harmonized features | `tables/fusion_*`; `figures/fusion_gate_weights` | III-C |
| `04_spectral_clustering.py` | Single-modality spectral clustering, K = 2..5, ARI stability and consensus PAC, cluster composition by LVI and batch, mixed-cluster analysis | `tables/spectral_*`; `figures/spectral_*` | III-D |

Harmonization settings: `Original (raw z-score)` (reproduces the notebooks),
`No harmonization`, `ComBat (center)`, `ComBat (acquisition batch)` (primary).
