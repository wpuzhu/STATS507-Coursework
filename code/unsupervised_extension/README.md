# Unsupervised Extension

Code for the report *Multi-modal MRI for Identifying Lymphovascular Invasion in
Endometrial Cancer: Center Harmonization, Latent Representation and Clustering*
(`report/Unsupervised_Extension.tex`).

The analysis uses all 403 patients (missing values imputed within batch) and
asks whether the harmonized T1 / T2 / ADC radiomic features contain stable
patient subgroups, and whether those subgroups relate to clinical and
pathology variables.

## Run

```bash
./run_all.sh          # ~8 min; or run the numbered scripts in order
```

Inputs: `multi_omics/T1.xlsx`, `T2.xlsx`, `ADC.xlsx`, `post_survey.xlsx`.
Each script caches its result in `results/unsupervised_extension/cache/`
(git-ignored) for the next step, so the scripts must run in order.

## Steps

| Script | What it does | Main outputs (`results/unsupervised_extension/`) | Report section |
|---|---|---|---|
| `01_harmonization.py` | Preprocessing (QC, within-batch imputation, log / Yeo-Johnson); label-free detection of acquisition batches C1 / C2a / C2b; six harmonization strategies (none, ComBat by center, ComBat by batch, ComBat with LVI protected, rank-INT, CovBat); feature-wise and multivariate batch tests | `tables/harmonization_summary.csv`, `harmonization_featurewise.csv`, `acquisition_batch.csv`, `batch_composition.csv`; `figures/harm_*` | III-B, III-C, IV-B |
| `02_mofa.py` | Correlation filter (\|r\| > 0.9); MOFA+ (10 initial factors, ARD + spike-and-slab) for every harmonization strategy, plus excluding C2b and center 1 only; batch association of factors | `tables/mofa_*`; `figures/mofa_*` | III-D, IV-C |
| `03_clustering.py` | k-means on MOFA factor scores, k = 2..8; silhouette, subsampling ARI (B = 200), consensus matrix and PAC; composite choice of k; robustness to algorithm and harmonization | `tables/cluster_*`; `figures/cluster_*`, `consensus_matrices` | III-E, IV-D, IV-E |
| `04_posthoc.py` | 21 clinical / imaging-report / pathology variables vs clusters: Kruskal-Wallis / chi-squared, batch-adjusted LRT, within-center-1 tests, BH-FDR; retention rule; explanatory multinomial models | `tables/posthoc_*`; `figures/posthoc_*` | III-F, IV-F |
| `05_data_integrity.py` | Cross-file consistency (tumor volume across sequences, radiomic vs radiologist measurements) | `tables/data_integrity.csv` | IV-A |
| `06_latex_tables.py` | Writes the booktabs tables used by the report and copies figures to `report/figures/` | `report/Unsupervised_Extension/tables/*.tex` | all |

`common.py` holds shared paths, plotting style and helpers.

## Key settings

* Primary harmonization: ComBat by acquisition batch, no protected covariate.
* MOFA+: `mofapy2`, 10 factors, drop threshold 1% variance explained, seed 507.
* Clustering: k-means (50 starts); k chosen by the mean rank of silhouette,
  subsampling ARI and 1 - PAC among partitions whose smallest cluster has at
  least 10% of patients.
* Post-hoc retention: q < 0.05 unadjusted and batch-adjusted, with at least a
  small effect (epsilon^2 >= 0.01 or Cramer's V >= 0.10).

See the main [README](../../README.md) for the data caveats (acquisition batch
confounded with LVI; patient-linkage checks).
