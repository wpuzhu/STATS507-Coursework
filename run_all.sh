#!/usr/bin/env bash
# Reproduce every result in the reports, end to end.
#   main pipeline            ~25 min (CPU)
#   unsupervised extension   ~8 min
# Outputs: results/main/{figures,tables}, results/unsupervised_extension/...
set -euo pipefail
cd "$(dirname "$0")"

LOG=results/main/cache
mkdir -p "$LOG"
cd code/pipeline
echo "[1/4] EDA and harmonization diagnostics"
python3 01_eda.py > "../../$LOG/eda_log.txt" 2>&1
echo "[2/4] Supervised baselines"
python3 02_supervised_baselines.py > "../../$LOG/baselines_log.txt" 2>&1
echo "[3/4] Fusion networks"
python3 03_fusion_networks.py > "../../$LOG/fusion_log.txt" 2>&1
echo "[4/4] Spectral clustering"
python3 04_spectral_clustering.py > "../../$LOG/spectral_log.txt" 2>&1
cd ../unsupervised_extension
echo "[extension] Harmonization, MOFA, clustering, post-hoc analysis"
./run_all.sh
python3 06_latex_tables.py > /dev/null
cd ../..
cp results/main/figures/*.pdf report/figures/
echo "done"
