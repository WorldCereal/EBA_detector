#!/bin/bash
#SBATCH --job-name=outlier_proxy
#SBATCH --output=logs/proxy_%j.out
#SBATCH --error=logs/proxy_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
# CatBoost can use GPU if available; request one for large embedding matrices.
#SBATCH --gres=gpu:1

set -euo pipefail
mkdir -p logs results

# --- environment ----------------------------------------------------
module load Python 2>/dev/null || true
source "${OUTLIER_VENV:-$HOME/venvs/outliers}/bin/activate"

REPO="${OUTLIER_REPO:-$HOME/outlier_embeddings}"
cd "$REPO/experiments"

# --- run the full scenario x region x seed grid (Tier-1) ------------
python run_catboost_proxy.py \
    --scored-parquet "${SCORED_PARQUET:?set SCORED_PARQUET}" \
    --label-col      "${LABEL_COL:-downstream_class}" \
    --flag-col       "${FLAG_COL:-CTY24_anomaly_flag}" \
    --conf-col       "${CONF_COL:-CTY24_confidence_nonoutlier}" \
    --region-col     "${REGION_COL:-region}" \
    --quality-col    "${QUALITY_COL:-quality_score_ct}" \
    --group-col      "${GROUP_COL:-ref_id}" \
    --seeds          0 1 2 \
    --out-csv        "results/proxy_${SLURM_JOB_ID:-local}.csv"

echo "Tier-1 proxy grid complete."
