#!/bin/bash -l
# ==============================================================================
# Outlier-scoring (LC10 + CTY24) pipeline — high-memory CPU SLURM batch job.
# modified_algo branch: absolute cross-slice gate, regionally-shrunk null,
# per-ref_id slice grouping, quality gate, purity veto — see README.md
# ("How a sample gets flagged") for the full method writeup.
#
# Runs scripts/compute_anomaly_scores.py, the CLI version of
# notebooks/compute_outlier_scores.ipynb on this branch. Use this instead of
# the notebook when the LC10/CTY24 scoring step (which loads the full
# embeddings cache into RAM, plus an extra pass to build the cross-slice null)
# OOM-kills the notebook kernel — submit it to a high-memory CPU node instead.
#
# All parameters live in a config file (bash key=value) that this script
# sources, so nothing needs editing in this file itself.
#
# IMPORTANT: this script is a LAUNCHER, not the job itself — it runs briefly
# (as whatever job SLURM gives it) just to build the python command and
# `sbatch` a SECOND, separate job that does the actual work. Because SLURM
# copies this script into a spool directory before running it, it cannot
# locate itself via $BASH_SOURCE/$0 (that would resolve to the spool copy) —
# REPO_DIR below is hardcoded instead. If the outer launcher job fails, its
# log goes to LOGDIR too (see #SBATCH --output below); the real work runs as
# a second, separate job ID that appears after "Submitting ...".
#
# To run with the tracked default config:
#     sbatch compute_anomaly_scores.sh
#
# To run with your own config (recommended for anything other than a quick
# test — copy the default first so the tracked file stays a clean template).
# Use an ABSOLUTE path for the config argument (see note below):
#     cp scripts/configs/compute_anomaly_scores.conf scripts/configs/my_run.conf
#     # edit scripts/configs/my_run.conf
#     sbatch compute_anomaly_scores.sh /home/vito/shahs/TestFolder/All_repos/Reveiw_branches/EBA_detector/scripts/configs/my_run.conf
# ==============================================================================
#SBATCH --job-name=EBA_ANOMALY_NEWMETHOD_launcher
#SBATCH --output=/home/vito/shahs/logs/EBAOutlierScoring/%x.%j.out
#SBATCH --error=/home/vito/shahs/logs/EBAOutlierScoring/%x.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --mem=1gb
set -euo pipefail

REPO_DIR=/home/vito/shahs/TestFolder/All_repos/Reveiw_branches/EBA_detector
SCRIPT_DIR="${REPO_DIR}/scripts"
LOGDIR=/home/vito/shahs/logs/EBAOutlierScoring

# ------------------------------------------------------------------ LOAD CONFIG
# NOTE: a relative path here is resolved relative to whatever directory SLURM
# happened to launch this script from (its spool dir), NOT the directory you
# ran `sbatch` from — always pass an ABSOLUTE path as $1.
CONFIG_FILE="${1:-${SCRIPT_DIR}/configs/compute_anomaly_scores.conf}"
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Config file not found: ${CONFIG_FILE}" >&2
    echo "(if you passed a relative path, use an absolute path instead — see the note above)" >&2
    exit 1
fi
echo "Using config: ${CONFIG_FILE}"
# shellcheck disable=SC1090
source "${CONFIG_FILE}"

mkdir -p "${LOGDIR}"

# ------------------------------------------------------------------ BUILD CLI ARGS
ARGS=(
    --mode "${MODE:?MODE not set in config}"
    --input-format "${INPUT_FORMAT:?INPUT_FORMAT not set in config}"
    --input-long-dir "${INPUT_LONG_DIR:?INPUT_LONG_DIR not set in config}"
    --presto-url-or-path "${PRESTO_URL_OR_PATH:?PRESTO_URL_OR_PATH not set in config}"
    --batch-size "${BATCH_SIZE:-4096}"
    --num-workers "${NUM_WORKERS:-2}"
    --parquet-batch-rows "${PARQUET_BATCH_ROWS:-100000}"
    --lc10-class-mapping "${LC10_CLASS_MAPPING:-LANDCOVER10}"
    --lc10-h3-levels ${LC10_H3_LEVELS:-2 3}
    --lc10-min-slice-size "${LC10_MIN_SLICE_SIZE:-200}"
    --lc10-max-slice-size "${LC10_MAX_SLICE_SIZE:-10000}"
    --lc10-max-merge-iterations "${LC10_MAX_MERGE_ITERATIONS:-16}"
    --lc10-mad-k "${LC10_MAD_K:-3.3}"
    --cty24-class-mapping "${CTY24_CLASS_MAPPING:-CROPTYPE24}"
    --cty24-h3-levels ${CTY24_H3_LEVELS:-2 3 4}
    --cty24-min-slice-size "${CTY24_MIN_SLICE_SIZE:-100}"
    --cty24-max-slice-size "${CTY24_MAX_SLICE_SIZE:-5000}"
    --cty24-max-merge-iterations "${CTY24_MAX_MERGE_ITERATIONS:-8}"
    --cty24-mad-k "${CTY24_MAD_K:-3.3}"
    --threshold-mode "${THRESHOLD_MODE:-stable_mad}"
    --percentile-q "${PERCENTILE_Q:-0.96}"
    --norm-percentiles ${NORM_PERCENTILES:-2.0 98.0}
    --fdr-alpha "${FDR_ALPHA:-0.05}"
    --skip-classes ${SKIP_CLASSES:-ignore}
    --max-full-pairwise-n "${MAX_FULL_PAIRWISE_N:-0}"
    --abs-z-k "${ABS_Z_K:-3.3}"
    --abs-z-suspect "${ABS_Z_SUSPECT:-4.0}"
    --abs-z-candidate "${ABS_Z_CANDIDATE:-5.5}"
    --null-scale-estimator "${NULL_SCALE_ESTIMATOR:-left_tail}"
    --abs-combine "${ABS_COMBINE:-min}"
    --null-region-offset "${NULL_REGION_OFFSET:-2}"
    --null-region-min-level "${NULL_REGION_MIN_LEVEL:-1}"
    --null-region-level "${NULL_REGION_LEVEL:--1}"
    --null-shrink-k "${NULL_SHRINK_K:-5.0}"
    --purity-veto "${PURITY_VETO:-0.80}"
    --min-scoring-slice-size "${MIN_SCORING_SLICE_SIZE:-50}"
    --centroid-mode "${CENTROID_MODE:-trimmed}"
    --centroid-trim "${CENTROID_TRIM:-0.45}"
    --slice-trust-min "${SLICE_TRUST_MIN:-0.05}"
    --neighbour-rings "${NEIGHBOUR_RINGS:-1}"
    --sp-env-file "${SP_ENV_FILE:-$HOME/.sharepointenv}"
    --log-level "${LOG_LEVEL:-INFO}"
)

[ -n "${SUFFIX:-}" ]              && ARGS+=(--suffix "${SUFFIX}")
[ -n "${PARQUET_GLOB:-}" ]        && ARGS+=(--parquet-glob "${PARQUET_GLOB}")
[ -n "${WIDE_DIR:-}" ]            && ARGS+=(--wide-dir "${WIDE_DIR}")
[ -n "${MERGED_WIDE_PATH:-}" ]    && ARGS+=(--merged-wide-path "${MERGED_WIDE_PATH}")
[ -n "${EMBEDDINGS_DB_PATH:-}" ]  && ARGS+=(--embeddings-db-path "${EMBEDDINGS_DB_PATH}")
[ -n "${REVIEW_DIR:-}" ]          && ARGS+=(--review-dir "${REVIEW_DIR}")
[ -n "${OUTPUT_LONG_DIR:-}" ]     && ARGS+=(--output-long-dir "${OUTPUT_LONG_DIR}")
[ -n "${OUTPUT_WIDE_PATH:-}" ]    && ARGS+=(--output-wide-path "${OUTPUT_WIDE_PATH}")
[ -n "${CLASS_MAPPINGS_JSON:-}" ] && ARGS+=(--class-mappings-json "${CLASS_MAPPINGS_JSON}")
[ -n "${COMPUTE_H3_LEVELS:-}" ]   && ARGS+=(--compute_h3_levels "${COMPUTE_H3_LEVELS}")
[ -n "${TIME_COL:-}" ]            && ARGS+=(--time-col "${TIME_COL}")

# GROUP_COLS: unset/empty means "pool across datasets" (pass an explicit empty
# list to the CLI); a non-empty value means "join these extra slice-key columns".
# Bash can't distinguish "explicitly empty" from "unset" with ${VAR:-default},
# so this one always passes the flag — with zero following args when empty,
# which argparse's nargs="*" accepts as [].
ARGS+=(--group-cols ${GROUP_COLS:-})
# Same story for context columns, except omitting the flag entirely already
# gives the empty-list default, so just don't add it when unset.
[ -n "${CONTEXT_GROUP_COLS:-}" ]  && ARGS+=(--context-group-cols ${CONTEXT_GROUP_COLS})
[ -n "${NULL_EXTRA_KEYS:-}" ]     && ARGS+=(--null-extra-keys ${NULL_EXTRA_KEYS})
# Unset/empty -> omit the flag so the Python script's own default applies.
# Set to a non-empty value (including a literal "none") to override; the
# Python side has no built-in way to pass an explicit empty list here.
if [ -n "${POST_PROCESSING_SKIP_CLASSES:-}" ]; then
    ARGS+=(--post-processing-skip-classes ${POST_PROCESSING_SKIP_CLASSES})
fi

[ "${RESCAN_WIDE_DIR:-true}" = "false" ]           && ARGS+=(--no-rescan-wide-dir)
[ "${FORCE_RECOMPUTE:-false}" = "true" ]          && ARGS+=(--force-recompute)
[ "${PREMATCH:-true}" = "false" ]                 && ARGS+=(--no-prematch)
[ "${NO_ABSOLUTE_GATE:-false}" = "true" ]         && ARGS+=(--no-absolute-gate)
[ "${STRICT_QUALITY:-false}" = "true" ]           && ARGS+=(--strict-quality)
[ "${NO_QUALITY_GATE:-false}" = "true" ]          && ARGS+=(--no-quality-gate)
[ "${GATE_CONFIDENCE_BY_FLAG:-true}" = "false" ]  && ARGS+=(--no-gate-confidence-by-flag)
[ "${APPLY_SLICE_TRUST:-false}" = "true" ]        && ARGS+=(--apply-slice-trust)
[ "${OVERWRITE_WIDE:-false}" = "true" ]           && ARGS+=(--overwrite-wide)
[ "${OVERWRITE_MERGED:-false}" = "true" ]         && ARGS+=(--overwrite-merged)
[ "${OVERWRITE_SCORES:-false}" = "true" ]         && ARGS+=(--overwrite-scores)
[ "${OVERWRITE_MERGED_SCORES:-false}" = "true" ]  && ARGS+=(--overwrite-merged-scores)
[ "${OVERWRITE_WIDE_SCORES:-false}" = "true" ]    && ARGS+=(--overwrite-wide-scores)
[ "${SKIP_EMBEDDINGS:-false}" = "true" ]          && ARGS+=(--skip-embeddings)
[ "${SKIP_SCORING:-false}" = "true" ]             && ARGS+=(--skip-scoring)
[ "${SKIP_WRITE_BACK:-false}" = "true" ]          && ARGS+=(--skip-write-back)
[ "${SKIP_LONG_WRITE_BACK:-false}" = "true" ]     && ARGS+=(--skip-long-write-back)
[ "${SKIP_WIDE_WRITE_BACK:-false}" = "true" ]     && ARGS+=(--skip-wide-write-back)

# Fully shell-quote so paths/values with spaces survive the heredoc below.
CMD_STR=$(printf '%q ' python "${SCRIPT_DIR}/compute_anomaly_scores.py" "${ARGS[@]}")

TAG="EBA_ANOMALY_NEWMETHOD_${MODE}_$(basename "${CONFIG_FILE}" .conf)"
echo "Submitting ${TAG}"
echo "Command: ${CMD_STR}"

sbatch <<SBATCH
#!/bin/bash -l
#SBATCH --partition=${PARTITION:-normal}
#SBATCH --job-name=${TAG}
#SBATCH --output=${LOGDIR}/%x.%j.out
#SBATCH --error=${LOGDIR}/%x.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS:-16}
#SBATCH --time=${TIME:-24:00:00}
#SBATCH --mem=${MEM:-250gb}
set -euo pipefail
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV:-worldcereal-py311}
cd "${REPO_DIR}"

echo "[code] ${REPO_DIR} @ \$(git -C "${REPO_DIR}" rev-parse HEAD) (branch \$(git -C "${REPO_DIR}" rev-parse --abbrev-ref HEAD))"
echo "[config] ${CONFIG_FILE}"

${CMD_STR}
SBATCH
