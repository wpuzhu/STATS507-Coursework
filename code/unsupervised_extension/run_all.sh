#!/usr/bin/env bash
# Reproduce the Unsupervised Extension analysis end to end (~8 min).
set -euo pipefail
cd "$(dirname "$0")"
LOG=../../results/unsupervised_extension/cache
python3 01_harmonization.py > "$LOG/harmonization_log.txt" 2>&1
python3 02_mofa.py > "$LOG/mofa_log.txt" 2>&1
python3 03_clustering.py > "$LOG/cluster_log.txt" 2>&1
python3 04_posthoc.py > "$LOG/posthoc_log.txt" 2>&1
python3 05_data_integrity.py > "$LOG/integrity_log.txt" 2>&1
echo done
