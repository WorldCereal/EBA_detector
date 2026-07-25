#!/bin/bash
#SBATCH --job-name=outlier_ft
#SBATCH --output=logs/ft_%A_%a.out
#SBATCH --error=logs/ft_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
# One array task per grid cell. Default grid below has
#   encoders(2) x regions(5) x scenarios(10) x seeds(3) = 300 cells.
# Launch with:  sbatch --array=0-299 finetune_array.slurm
#SBATCH --array=0-299%32        # cap to 32 concurrent jobs

set -euo pipefail
mkdir -p logs

module load Python 2>/dev/null || true
source "${OUTLIER_VENV:-$HOME/venvs/worldcereal}/bin/activate"
REPO="${OUTLIER_REPO:-$HOME/EBA_detector}"
cd "$REPO/experiments"

# --- grid axes (keep in sync with config.yaml) ----------------------
ENCODERS=(vanilla finetuned)
REGIONS=(ALL Europe Africa Americas Asia)
SCENARIOS=(baseline drop_candidate drop_suspect drop_flagged \
           downweight_linear downweight_power2 downweight_power4 \
           filter_conf_0.90 filter_conf_0.95 "drop_suspect+downweight")
SEEDS=(0 1 2)

nE=${#ENCODERS[@]}; nR=${#REGIONS[@]}; nC=${#SCENARIOS[@]}; nS=${#SEEDS[@]}

# --- decode SLURM_ARRAY_TASK_ID into (encoder, region, scenario, seed) ----
idx=${SLURM_ARRAY_TASK_ID:-0}
seed_i=$(( idx % nS )); idx=$(( idx / nS ))
scen_i=$(( idx % nC )); idx=$(( idx / nC ))
reg_i=$(( idx % nR ));  idx=$(( idx / nR ))
enc_i=$(( idx % nE ))

ENC=${ENCODERS[$enc_i]}
REG=${REGIONS[$reg_i]}
SCEN=${SCENARIOS[$scen_i]}
SEED=${SEEDS[$seed_i]}

echo "Grid cell: encoder=$ENC region=$REG scenario=$SCEN seed=$SEED"

python run_worldcereal_finetune.py \
    --data-parquet   "${SCORED_PARQUET:?set SCORED_PARQUET}" \
    --head-task      "${HEAD_TASK:-croptype}" \
    --season-id      "${SEASON_ID:-tc-s1}" \
    --encoder        "$ENC" \
    --presto-vanilla-path   "${PRESTO_VANILLA:?set PRESTO_VANILLA}" \
    --presto-finetuned-path "${PRESTO_FINETUNED:-$PRESTO_VANILLA}" \
    --region         "$REG" \
    --region-col     "${REGION_COL:-region}" \
    --scenario       "$SCEN" \
    --seed           "$SEED" \
    --split-column   "${SPLIT_COLUMN:-split}" \
    --label-col      "${LABEL_COL:-downstream_class}" \
    --flag-col       "${FLAG_COL:-CTY24_anomaly_flag}" \
    --conf-col       "${CONF_COL:-CTY24_confidence_nonoutlier}" \
    --quality-col    "${QUALITY_COL:-quality_score_ct}" \
    --output-dir     "${OUTPUT_ROOT:-runs}"

echo "Cell complete."
