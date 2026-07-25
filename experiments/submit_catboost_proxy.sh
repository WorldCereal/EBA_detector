#!/bin/bash -l
# ==============================================================================
# CatBoost outlier-PROXY launcher (one SLURM job per REGION x DOMAIN, in parallel).
#
# Trains a gradient-boosted classifier on the frozen Presto embeddings under each
# outlier-treatment scenario and evaluates on the three test views
# (full / candidates-removed / confidence-weighted). One region's merged parquet
# = one job, so regions run fully in parallel.
#
# To run: edit REGIONS (and optionally DOMAINS / CONFIG / SEEDS), then:
#     bash submit_catboost_proxy.sh
# Re-submitting more regions later = add them to REGIONS and run again.
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
DOMAINS=(CTY24)            # one or both of: CTY24 LC10
CONFIG=normal             # normal -> Region_wise_files ; sharper -> Region_wise_files_sharper
SEEDS="0 1 2"
TEST_FRAC=0.2
# ----------------------------------------------------------------------------

# ------------------------------------------------------------------ PATHS / ENV
# Directory that contains run_catboost_proxy.py + data_loader.py (this repo).
PROXY_DIR=/home/vito/shahs/TestFolder/All_repos/EBA_detector/experiments  # <-- set to your checkout
DUCKDB=/home/vito/shahs/TestFolder/wc_outliers/data_for_outlier/EMBEDDINGS_CACHE/embeddings_cache_LANDCOVER10_updated_new_model.duckdb
MERGED_DIR_normal=/projects/worldcereal/data/cached_wide_merged/Region_wise_files_newmodel
MERGED_DIR_sharper=/projects/worldcereal/data/cached_wide_merged/Region_wise_files_newmodel_sharper
OUTROOT=/home/vito/shahs/TestFolder/wc_outliers/data_for_outlier
LOGDIR=/home/vito/shahs/logs/WCProxyNewModel
CONDA_ENV=worldcereal-py311        # must have: catboost duckdb pandas pyarrow scikit-learn
PARTITION=batch                      # CatBoost on 128-d embeddings is fast on CPU; set to your CPU partition
CPUS=16
MEM=64gb
TIME=18:00:00
mkdir -p "$LOGDIR"

case "$CONFIG" in
    normal)  MERGED_DIR="$MERGED_DIR_normal" ;;
    sharper) MERGED_DIR="$MERGED_DIR_sharper" ;;
    *) echo "CONFIG must be 'normal' or 'sharper'"; exit 1 ;;
esac

# ------------------------------------------------------------------ SUBMIT LOOP
for DOMAIN in "${DOMAINS[@]}"; do
  OUTDIR="$OUTROOT/proxy_${DOMAIN}_${CONFIG}"
  mkdir -p "$OUTDIR"
  for REGION in "${REGIONS[@]}"; do
    MERGED="$MERGED_DIR/${REGION}.parquet"
    OUTCSV="$OUTDIR/proxy_${DOMAIN}_${REGION}.csv"
    TAG="proxy_${DOMAIN}_${CONFIG}_${REGION}"
    echo "Submitting $TAG  ::  $MERGED -> $OUTCSV"

    # Fully shell-quote so region names with spaces/dashes survive the heredoc.
    CMD_STR=$(printf '%q ' python "$PROXY_DIR/run_catboost_proxy.py" \
        --duckdb "$DUCKDB" \
        --merged-parquet "$MERGED" \
        --domain "$DOMAIN" --group-col ref_id \
        --seeds $SEEDS --test-frac "$TEST_FRAC" \
        --out-csv "$OUTCSV")

    sbatch <<SBATCH
#!/bin/bash -l
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
echo "All proxy jobs submitted."
