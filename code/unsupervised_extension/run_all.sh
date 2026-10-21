#!/usr/bin/env bash
# Reproduce the Unsupervised Extension analysis end to end (~8 min).
# Outputs: results/unsupervised_extension/{figures,tables},
#          report/tables (LaTeX tables for the report).
set -euo pipefail
cd "$(dirname "$0")"
LOG=../../results/unsupervised_extension/cache
mkdir -p "$LOG"
echo "[1/6] Harmonization and tests";  python3 01_harmonization.py > "$LOG/harmonization_log.txt" 2>&1
echo "[2/6] MOFA latent factors";      python3 02_mofa.py > "$LOG/mofa_log.txt" 2>&1
echo "[3/6] Clustering and stability"; python3 03_clustering.py > "$LOG/cluster_log.txt" 2>&1
echo "[4/6] Post-hoc analysis";        python3 04_posthoc.py > "$LOG/posthoc_log.txt" 2>&1
echo "[5/6] Data-integrity checks";    python3 05_data_integrity.py > "$LOG/integrity_log.txt" 2>&1
echo "[6/6] LaTeX tables"; python3 06_latex_tables.py > "$LOG/latex_log.txt" 2>&1
echo done
