#!/usr/bin/env bash
# Reproduce every result in both reports from the raw data (~30 min on a laptop CPU).
#   code/supervised_report       -> results/supervised_report        (Supervised Report)
#   code/unsupervised_extension  -> results/unsupervised_extension   (Unsupervised Extension)
#   report/tables, report/figures are refreshed for the LaTeX reports.
set -euo pipefail
cd "$(dirname "$0")"

LOG=results/supervised_report/cache
mkdir -p "$LOG"
cd code/supervised_report
echo "[Supervised Report 1/4] EDA and harmonization diagnostics"
python3 01_eda.py > "../../$LOG/eda_log.txt" 2>&1
echo "[Supervised Report 2/4] Supervised baselines"
python3 02_supervised_baselines.py > "../../$LOG/baselines_log.txt" 2>&1
echo "[Supervised Report 3/4] Fusion networks"
python3 03_fusion_networks.py > "../../$LOG/fusion_log.txt" 2>&1
echo "[Supervised Report 4/4] Spectral clustering"
python3 04_spectral_clustering.py > "../../$LOG/spectral_log.txt" 2>&1

echo "[Unsupervised Extension]"
../unsupervised_extension/run_all.sh

cd ../..
python3 code/sync_report_figures.py
echo "done"
