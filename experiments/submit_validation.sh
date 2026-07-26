#!/bin/bash -l
# ==============================================================================
# Synthetic label-noise VALIDATION launcher (one SLURM job per REGION, parallel).
#
# Injects controlled label noise, re-runs the detector, and reports recovery.
# With the corrected defaults the key columns are flag_precision / flag_recall /
# flag_enrichment and lift_at_k (NOT global auroc) -- see run_validation.py.
# CPU-only; no GPU needed. One region = one job, fully parallel.
#
# To run: edit REGIONS (and optionally DOMAINS / CONFIG), then:
#     bash submit_validation.sh
# ==============================================================================
set -euo pipefail

# ------------------------------------------------------------------ EDIT THESE
REGIONS=(
    Eastern_Africa
    Middle_Africa
    South-Eastern_Asia
    South_America
    Southern_Asia
)
DOMAINS=(CTY24)                       # one or both of: CTY24 LC10
CONFIG=normal                         # normal | sharper (selects the merged-parquet dir)
MODES="within_context random parcel"  # parcel uses ref_id as the group
RATES="0.02 0.05 0.10 0.20"
SEEDS="0 1 2"
# ----------------------------------------------------------------------------

# ------------------------------------------------------------------ PATHS / ENV
PROXY_DIR=/path/to/TestFolder/wc_outliers/EBA_detector/experiments   # <-- set to your checkout
DUCKDB=/projects/worldcereal/data/cached_embeddings/embeddings_cache_LANDCOVER10_updated_new.duckdb
MERGED_DIR_normal=/projects/worldcereal/data/cached_wide_merged/Region_wise_files
MERGED_DIR_sharper=/projects/worldcereal/data/cached_wide_merged/Region_wise_files_sharper
OUTROOT=/path/to/TestFolder/wc_outliers/data_for_outlier
LOGDIR=/path/to/logs/WCValidation
CONDA_ENV=worldcereal-py311
PARTITION=cpu
CPUS=16
MEM=64gb
TIME=06:00:00
mkdir -p "$LOGDIR"

case "$CONFIG" in
    normal)  MERGED_DIR="$MERGED_DIR_normal" ;;
    sharper) MERGED_DIR="$MERGED_DIR_sharper" ;;
    *) echo "CONFIG must be 'normal' or 'sharper'"; exit 1 ;;
esac

# ------------------------------------------------------------------ SUBMIT LOOP
for DOMAIN in "${DOMAINS[@]}"; do
  OUTDIR="$OUTROOT/run_validation_outputs"
  mkdir -p "$OUTDIR"
  for REGION in "${REGIONS[@]}"; do
    MERGED="$MERGED_DIR/${REGION}.parquet"
    OUTCSV="$OUTDIR/validation_${DOMAIN}_${REGION}.csv"
    TAG="val_${DOMAIN}_${CONFIG}_${REGION}"
    echo "Submitting $TAG  ::  $MERGED -> $OUTCSV"

    CMD_STR=$(printf '%q ' python "$PROXY_DIR/run_validation.py" \
        --duckdb "$DUCKDB" \
        --merged-parquet "$MERGED" \
        --domain "$DOMAIN" \
        --modes $MODES \
        --rates $RATES \
        --seeds $SEEDS \
        --out-csv "$OUTCSV")

    sbatch <<SBATCH
#!/bin/bash -l
#SBATCH --partition=${PARTITION}
#SBATCH --job-name=${TAG}
#SBATCH --output=${LOGDIR}/%x.%j.out
#SBATCH --error=${LOGDIR}/%x.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --time=${TIME}
#SBATCH --mem=${MEM}
set -euo pipefail
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}
cd "${PROXY_DIR}"
export OMP_NUM_THREADS=${CPUS}

${CMD_STR}
SBATCH
  done
done
echo "All validation jobs submitted."
