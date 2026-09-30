#!/bin/bash
#SBATCH --job-name=VarDyn_GLO
#SBATCH --output=logs/output-%A/output-%a.out
#SBATCH --error=logs/error-%A/error-%a.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=v100_32g:1

#SBATCH --array=0-5          # Keep in sync with NUM_GPUS below: 0-$((NUM_GPUS-1))
#SBATCH --qos=gpu_max
#SBATCH --partition=gpu_std
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --mem=90G
#SBATCH --account=swot_duacs
#SBATCH --export=none

# -------------------- SLURM --------------------
NUM_GPUS=6                   # Number of GPU array tasks — also update #SBATCH --array above
NUM_MERGE_WORKERS=4
NUM_TILES_PER_GPU=4
ARRAY_ID=${SLURM_ARRAY_TASK_ID:-0}
NUM_ARRAY=${SLURM_ARRAY_TASK_COUNT:-$NUM_GPUS}
# Use SLURM_ARRAY_JOB_ID (common to all array tasks), fall back to SLURM_JOB_ID
JOB_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-$$}}

# -------------------- EXPERIMENT CONFIGURATION --------------------
# Keep experiment-specific settings in a separate, reproducible shell config.
# Submit with: sbatch slurm/run/VarDyn_GLO.sh --config path/to/config.sh
die() { echo "ERROR: $*" >&2; exit 1; }
# Parse once before sourcing the config; CLI overrides are applied afterwards.
declare -A CLI=()
while (( $# )); do
    option="${1%%=*}"
    case "$option" in
        --config|--tile-scope|--name_exp|--name_exp_background|--predecessor-job)
            if [[ "$1" == *=* ]]; then
                value="${1#*=}"
            else
                (( $# >= 2 )) && [[ "$2" != --* ]] || die "Missing value for $option"
                shift
                value="$1"
            fi
            [ -n "$value" ] || die "Missing value for $option"
            CLI[$option]="$value" ;;
        --skip-prepare|--restart|--force-merge|--merge-only)
            [[ "$1" != *=* ]] || die "$option takes no value"
            CLI[$option]=true ;;
        *) die "Unknown option: $1" ;;
    esac
    shift
done
CONFIG_FILE="${CLI[--config]:-${VAR_DYN_CONFIG:-}}"
if [ -z "$CONFIG_FILE" ] || [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: provide an experiment config with --config CONFIG_FILE" >&2
    exit 1
fi
CONFIG_FILE="$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")"
# shellcheck disable=SC1090
source "$CONFIG_FILE" || die "Cannot load $CONFIG_FILE"

for setting in MASH_DIR DIR_SAVE_PICKLE PATH_CONFIG PATH_CONFIG_EQ INIT_DATE FINAL_DATE; do
    [ -n "${!setting}" ] || die "Missing required setting: $setting"
done
for setting in PATH_CONFIG PATH_CONFIG_EQ; do
    [[ "${!setting}" = /* ]] || printf -v "$setting" '%s/%s' "$(dirname "$CONFIG_FILE")" "${!setting}"
done

# These defaults are orchestration settings and can be overridden by the
# external config without changing the reusable launcher.
NUM_MERGE_WORKERS="${NUM_MERGE_WORKERS:-4}"
NUM_TILES_PER_GPU="${NUM_TILES_PER_GPU:-4}"
ZARR_OUTPUT="${ZARR_OUTPUT:-false}"
OUTPUT_FLOAT64="${OUTPUT_FLOAT64:-false}"
CLEANUP_TILE_ZARR="${CLEANUP_TILE_ZARR:-true}"
ZARR_TIME_CHUNK="${ZARR_TIME_CHUNK:-4}"
ZARR_SPATIAL_CHUNK="${ZARR_SPATIAL_CHUNK:-256}"
ZARR_COMPRESSION_LEVEL="${ZARR_COMPRESSION_LEVEL:-3}"
# Maximum barrier wait / interval without tile completion progress.
BARRIER_TIMEOUT="${BARRIER_TIMEOUT:-7200}"
# Separate ceiling for one tile, which may legitimately take several hours.
TILE_TIMEOUT="${TILE_TIMEOUT:-172800}"
STAGE_TIMEOUT="${STAGE_TIMEOUT:-172800}"
FLAG_INIT_FROM_PREVIOUS="${FLAG_INIT_FROM_PREVIOUS:---flag_init_from_previous}"
FLAG_INIT="${FLAG_INIT:-false}"
FLAG_BACKGROUND="${FLAG_BACKGROUND:-false}"
NAME_EXP="${NAME_EXP:-}"

for setting in NUM_MERGE_WORKERS NUM_TILES_PER_GPU BARRIER_TIMEOUT TILE_TIMEOUT \
               STAGE_TIMEOUT ZARR_TIME_CHUNK ZARR_SPATIAL_CHUNK; do
    [[ "${!setting}" =~ ^[1-9][0-9]*$ ]] || die "$setting must be a positive integer"
done
for setting in ZARR_OUTPUT OUTPUT_FLOAT64 CLEANUP_TILE_ZARR FLAG_INIT FLAG_BACKGROUND; do
    [[ "${!setting}" = true || "${!setting}" = false ]] || die "$setting must be true or false"
done
[[ "$ZARR_COMPRESSION_LEVEL" =~ ^[0-9]$ ]] || die "ZARR_COMPRESSION_LEVEL must be 0..9"
# Apply CLI overrides after config loading.
SKIP_PREPARE="${CLI[--skip-prepare]:-false}"
FORCE_MERGE="${CLI[--force-merge]:-false}"
MERGE_ONLY="${CLI[--merge-only]:-false}"
$MERGE_ONLY && SKIP_PREPARE=true
RESTART=""
[ "${CLI[--restart]:-false}" = true ] && RESTART=--restart
TILE_SCOPE="${CLI[--tile-scope]:-all}"
NAME_EXP_OVERRIDE="${CLI[--name_exp]:-}"
NAME_EXP_BACKGROUND_OVERRIDE="${CLI[--name_exp_background]:-}"
PREDECESSOR_JOB="${CLI[--predecessor-job]:-}"

if [ "$TILE_SCOPE" != "all" ] && [ "$TILE_SCOPE" != "equatorial" ]; then
    echo "ERROR: --tile-scope must be 'all' or 'equatorial' (got '$TILE_SCOPE')" >&2
    exit 1
fi

# A dedicated background experiment necessarily requires background mode.
if [ -n "$NAME_EXP_BACKGROUND_OVERRIDE" ]; then
    FLAG_BACKGROUND=true
fi

FORCE_ARGS=()
$FORCE_MERGE && FORCE_ARGS+=(--force)

# EXP_NAME: --name_exp flag > name_experiment in PATH_CONFIG > filename fallback
if [ -n "$NAME_EXP_OVERRIDE" ]; then
    EXP_NAME="$NAME_EXP_OVERRIDE"
else
    EXP_NAME=$(python3 - "$PATH_CONFIG" <<'PY_NAME'
import ast
import sys
for node in ast.parse(open(sys.argv[1]).read()).body:
    if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == 'name_experiment'
        for target in node.targets
    ) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
        print(node.value.value)
        break
PY_NAME
    ) || die "Cannot read experiment name from $PATH_CONFIG"
    if [ -z "$EXP_NAME" ]; then
        EXP_NAME=$(basename "$PATH_CONFIG" .py | sed 's/^config_//')
    fi
fi
BASE_DIR="${DIR_SAVE_PICKLE}/${EXP_NAME}"
CONFIG_PATH="${BASE_DIR}/config.pkl"

_slurm_query() {
    timeout --kill-after=5 30 squeue "$@"
}

slurm_task_is_active() {
    local task_id="$1"
    local active_tasks
    if active_tasks=$(_slurm_query -r -h -j "$task_id" -o '%i'); then
        # A targeted query can return a canonical array ID for a raw job ID.
        [ -n "$active_tasks" ]
        return
    fi
    # Finished jobs can make a targeted query fail with "Invalid job id".
    # Only a successful full listing can establish that an array task is gone.
    if ! active_tasks=$(_slurm_query -r -h -o '%i'); then
        echo "WARNING: cannot determine Slurm owner ${task_id}; retaining tile lock" >&2
        return 0
    fi
    local active_task
    while read -r active_task; do
        [ "$active_task" = "$task_id" ] && return 0
        [[ "$task_id" != *_* && "$active_task" == "${task_id}_"* ]] && return 0
    done <<< "$active_tasks"
    if [[ ! "$task_id" =~ ^[0-9]+_[0-9]+$ && "$task_id" != "${PREDECESSOR_JOB:-}" ]]; then
        # A legacy raw ID may appear under a different canonical array ID.
        echo "WARNING: unresolved legacy Slurm owner ${task_id}; retaining tile lock" >&2
        return 0
    fi
    return 1
}

# -------------------- TIME-LIMIT CONTINUATION --------------------
# Slurm sends USR1 300 seconds before the wall-time limit. The first task
# obtaining the submission lock creates a continuation; completed work is
# skipped through their existing completion markers.
CONTINUATION_SCRIPT="${MASH_DIR}/slurm/run/VarDyn_GLO.sh"
FINAL_MARKER="${BASE_DIR}/experiment_complete.ok"
CONTINUATION_SUBMITTED=false

submit_continuation() {
    local submit_lock="${BASE_DIR}/.continuation_${JOB_ID}.lock"
    mkdir "$submit_lock" 2>/dev/null || return 0
    [ -f "${FINAL_MARKER}" ] && [ -z "$RESTART" ] \
        && ! $FORCE_MERGE && ! $MERGE_ONLY && return 0
    [ "${CONTINUATION_SUBMITTED}" = true ] && return 0

    local dependency="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
    local next_job
    local continuation_args=(--config "${CONFIG_FILE}" --skip-prepare)
    continuation_args+=(--predecessor-job "${dependency}")
    $MERGE_ONLY && continuation_args+=(--merge-only)
    [ "$TILE_SCOPE" != "all" ] && continuation_args+=(--tile-scope "$TILE_SCOPE")
    [ -n "${NAME_EXP_OVERRIDE}" ] && continuation_args+=(--name_exp "${NAME_EXP_OVERRIDE}")
    [ -n "${NAME_EXP_BACKGROUND_OVERRIDE}" ] && continuation_args+=(--name_exp_background "$NAME_EXP_BACKGROUND_OVERRIDE")
    if next_job=$(sbatch --parsable \
        --dependency="afterany:${dependency}" \
        "${CONTINUATION_SCRIPT}" "${continuation_args[@]}"); then
        CONTINUATION_SUBMITTED=true
        echo "$(date '+%F %T') | Submitted continuation array ${next_job}"
        if command -v scontrol >/dev/null 2>&1; then
            local next_job_id="${next_job%%;*}"
            local dependency_record
            dependency_record=$(scontrol show job "$next_job_id" -o 2>/dev/null || true)
            if [[ "$dependency_record" == *"Dependency=afterany:${dependency}"* ]]; then
                echo "$(date '+%F %T') | Verified continuation dependency afterany:${dependency}"
            else
                echo "$(date '+%F %T') | WARNING: could not verify continuation dependency afterany:${dependency}" >&2
            fi
        fi
    else
        rmdir "$submit_lock" 2>/dev/null || true
        echo "$(date '+%F %T') | ERROR: failed to submit continuation array" >&2
    fi
}

handle_timeout() {
    declare -F stop_tile_workers >/dev/null && stop_tile_workers
    declare -F stop_process_group >/dev/null && stop_process_group "${STAGE_PID:-}"
    echo "$(date '+%F %T') | Slurm wall-time signal received; requesting continuation"
    submit_continuation
    exit 124
}

handle_cancel() {
    declare -F stop_tile_workers >/dev/null && stop_tile_workers
    declare -F stop_process_group >/dev/null && stop_process_group "${STAGE_PID:-}"
    echo "$(date '+%F %T') | Cancellation signal received; no continuation will be submitted"
    exit 143
}

trap handle_timeout USR1
trap handle_cancel TERM

# Arrays preserve paths and experiment names as single arguments.
PREPARE_ARGS=(
    --init_date "$INIT_DATE" --final_date "$FINAL_DATE" --dir_save_pickle "$DIR_SAVE_PICKLE"
    --grid_type "$GRID_TYPE" --grid_type_eq "$GRID_TYPE_EQ"
    --nx_proc "$NX_PROC" --ny_proc "$NY_PROC" --nx_proc_eq "$NX_PROC_EQ" --ny_proc_eq "$NY_PROC_EQ"
    --dx "$DX" --dy "$DY"
    --space_window_size_proc_x "$SPACE_WIN_X" --space_window_size_proc_y "$SPACE_WIN_Y"
    --space_window_size_proc_x_eq "$SPACE_WIN_X_EQ" --space_window_size_proc_y_eq "$SPACE_WIN_Y_EQ"
    --space_overlap_x "$SPACE_OVERLAP_X" --space_overlap_y "$SPACE_OVERLAP_Y"
    --time_window_size_proc "$TIME_WIN" --time_overlap "$TIME_OVERLAP"
    --zarr_time_chunk "$ZARR_TIME_CHUNK" --zarr_spatial_chunk "$ZARR_SPATIAL_CHUNK"
    --zarr_compression_level "$ZARR_COMPRESSION_LEVEL"
)
[ -n "$FLAG_INIT_FROM_PREVIOUS" ] && PREPARE_ARGS+=("$FLAG_INIT_FROM_PREVIOUS")
$FLAG_INIT && PREPARE_ARGS+=(--flag_init)
$FLAG_BACKGROUND && PREPARE_ARGS+=(--flag_background)
[ -n "$NAME_EXP" ] && PREPARE_ARGS+=(--name_exp "$NAME_EXP")
NAME_EXP_BACKGROUND="${NAME_EXP_BACKGROUND_OVERRIDE:-${NAME_EXP_BACKGROUND:-}}"
[ -n "$NAME_EXP_BACKGROUND" ] && PREPARE_ARGS+=(--name_exp_background "$NAME_EXP_BACKGROUND")

# -------------------- ENVIRONMENT --------------------
source /home/il/${USER}/.bashrc
conda activate MASSHv2 || die "Cannot activate MASSHv2"

# Configure GPU allocation before any Python process imports JAX/XLA.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR=cuda_malloc_async
export HDF5_USE_FILE_LOCKING=FALSE

# Derive source and library paths from MASH_DIR (set in USER SETTINGS above).
# readlink -f "$0" is intentionally avoided: SLURM copies the script to
# /var/spool/slurmd/jobXXX/slurm_script before execution, making $0 useless.
SRC_DIR="${MASH_DIR}/slurm/src"
export MASSH_PATH="${MASH_DIR}/mapping"

# -------------------- LOG --------------------
LOGDIR="./logs/${EXP_NAME}_job-${JOB_ID}"
mkdir -p "$LOGDIR"
# GPU indices are local to each node (several nodes can each expose GPU 0).
# Include the Slurm array task, node, and allocated CUDA device in every log
# name so that all workers remain unambiguous in a multi-node allocation.
GPU_NODE="${SLURMD_NODENAME:-$(hostname -s)}"
GPU_DEVICE="${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-unassigned}}"
GPU_DEVICE_SAFE="${GPU_DEVICE//,/+}"
GPU_LOG_ID="task${ARRAY_ID}_${GPU_NODE}_cuda${GPU_DEVICE_SAFE}"
MAIN_LOGFILE="${LOGDIR}/${GPU_LOG_ID}.log"
exec > >(tee -a "$MAIN_LOGFILE") 2>&1

# -------------------- BARRIER DIR --------------------
BARRIER_DIR="${DIR_SAVE_PICKLE}/.barriers_${JOB_ID}"
# Retry mkdir to handle Lustre propagation delays and stale NFS handles
for _attempt in 1 2 3 4 5; do
    mkdir -p "$BARRIER_DIR" 2>/dev/null
    [ -d "$BARRIER_DIR" ] && break
    echo "$(date '+%F %T') | WARNING: mkdir barrier dir failed (attempt ${_attempt}), retrying..." >&2
    sleep $(( _attempt * 2 ))
done
if [ ! -d "$BARRIER_DIR" ]; then
    echo "$(date '+%F %T') | FATAL: Cannot create barrier directory: $BARRIER_DIR" >&2
    exit 1
fi

# -------------------- HEADER --------------------
echo "=========================================="
echo " Job ${JOB_ID} | GPU task ${ARRAY_ID}/${NUM_ARRAY}"
echo " Host: $(hostname)"
echo " Start time: $(date)"
echo " Python: $(which python)"
echo " CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo " SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-N/A}"
echo " SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-N/A}"
echo " GPU log identity=${GPU_LOG_ID}"
if command -v nvidia-smi >/dev/null 2>&1 && [ "$GPU_DEVICE" != "unassigned" ]; then
    echo " Allocated GPU status:"
    nvidia-smi -i "$GPU_DEVICE" \
        --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader || echo " WARNING: nvidia-smi failed" >&2
else
    echo " WARNING: no allocated GPU can be queried with nvidia-smi" >&2
fi
echo " Memory: $(ulimit -v 2>/dev/null || echo N/A)"
echo " SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-N/A}"
echo " SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-N/A}"
echo " NUM_MERGE_WORKERS=${NUM_MERGE_WORKERS}"
echo " NUM_TILES_PER_GPU=${NUM_TILES_PER_GPU}"
echo " BARRIER_TIMEOUT=${BARRIER_TIMEOUT}s TILE_TIMEOUT=${TILE_TIMEOUT}s STAGE_TIMEOUT=${STAGE_TIMEOUT}s"
echo " TILE_SCOPE=${TILE_SCOPE}"
echo " CLEANUP_TILE_ZARR=${CLEANUP_TILE_ZARR}"
echo " ZARR chunks=${ZARR_TIME_CHUNK}x${ZARR_SPATIAL_CHUNK}x${ZARR_SPATIAL_CHUNK}, zstd level=${ZARR_COMPRESSION_LEVEL}"
echo "=========================================="
if [ -n "$PREDECESSOR_JOB" ]; then
    if ! command -v squeue >/dev/null 2>&1; then
        echo "$(date '+%F %T') | ERROR: cannot guard continuation against predecessor ${PREDECESSOR_JOB}: squeue is unavailable" >&2
        exit 1
    fi
    predecessor_started=$SECONDS
    while slurm_task_is_active "$PREDECESSOR_JOB"; do
        if (( SECONDS - predecessor_started >= BARRIER_TIMEOUT )); then
            echo "ERROR: predecessor ${PREDECESSOR_JOB} still active or unknown after ${BARRIER_TIMEOUT}s" >&2
            exit 1
        fi
        sleep 5
    done
    echo "$(date '+%F %T') | Predecessor array ${PREDECESSOR_JOB} is no longer active"
fi
if [ -n "$RESTART" ] && [ -f "$FINAL_MARKER" ]; then
    rm -f "$FINAL_MARKER"
    echo "$(date '+%F %T') | Removed stale completion marker for explicit restart"
fi
if [ -f "$FINAL_MARKER" ] && ! $FORCE_MERGE && ! $MERGE_ONLY; then
    echo "$(date '+%F %T') | Experiment already complete; exiting late array task"
    exit 0
fi

# Waits are bounded in elapsed time, including scheduler query time.
wait_for_marker() {
    local marker="$1" failed="$2" lock="$3"
    local started=$SECONDS owner
    while [ ! -f "$marker" ]; do
        [ -f "$failed" ] || [ -f "${BARRIER_DIR}/run.failed" ] && return 1
        owner=$(cat "$lock/owner" 2>/dev/null) || owner=""
        if [ -n "$owner" ] && ! slurm_task_is_active "$owner"; then
            echo "ERROR: stage owner $owner disappeared while waiting for $marker" >&2
            return 1
        fi
        if (( SECONDS - started >= BARRIER_TIMEOUT )); then
            echo "ERROR: timed out waiting for $marker (owner=${owner:-unknown})" >&2
            return 1
        fi
        sleep 5
    done
    [ ! -f "$failed" ] && [ ! -f "${BARRIER_DIR}/run.failed" ]
}

stop_process_group() {
    local pid="$1"
    [ -n "$pid" ] || return 0
    kill -TERM -- "-$pid" 2>/dev/null || true
    local deadline=$((SECONDS + 10))
    while kill -0 "$pid" 2>/dev/null && (( SECONDS < deadline )); do sleep 1; done
    kill -KILL -- "-$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

STAGE_PID=""
run_stage() {
    [ ! -f "${BARRIER_DIR}/run.failed" ] || return 1
    setsid timeout --kill-after=10 "$STAGE_TIMEOUT" "$@" &
    STAGE_PID=$!
    while kill -0 "$STAGE_PID" 2>/dev/null; do
        if [ -f "${BARRIER_DIR}/run.failed" ]; then
            stop_process_group "$STAGE_PID"
            STAGE_PID=""
            return 1
        fi
        sleep 1
    done
    local status=0
    wait "$STAGE_PID" || status=$?
    stop_process_group "$STAGE_PID"
    [ -f "${BARRIER_DIR}/run.failed" ] && status=1
    STAGE_PID=""
    return "$status"
}

# Reap all workers, including a worker killed before publishing its failure.
wait_tile_workers() {
    local maximum="$1" pid
    local remaining=()
    while true; do
        [ ! -f "${BARRIER_DIR}/run.failed" ] || return 1
        remaining=()
        for pid in "${ACTIVE_TILE_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                remaining+=("$pid")
            else
                wait "$pid" || return 1
            fi
        done
        ACTIVE_TILE_PIDS=("${remaining[@]}")
        (( ${#ACTIVE_TILE_PIDS[@]} <= maximum )) && return 0
        sleep 1
    done
}

ACTIVE_TILE_PIDS=()
stop_tile_workers() {
    local pid
    for pid in "${ACTIVE_TILE_PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${ACTIVE_TILE_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
    ACTIVE_TILE_PIDS=()
}
launcher_exit() {
    local status=$?
    trap - EXIT
    if (( status != 0 )); then
        touch "${BARRIER_DIR}/run.failed"
    fi
    stop_tile_workers
    stop_process_group "$STAGE_PID"
    # Keep stage locks and diagnostics. A dependent new generation uses
    # its own barrier directory; peers must not restart an interrupted stage.
    exit "$status"
}
trap launcher_exit EXIT

# Completed stage locks remain in place, preventing late peers from repeating work.
claim_stage() {
    mkdir "$1" 2>/dev/null || return 1
    printf '%s\n' "${JOB_ID}_${ARRAY_ID}" > "$1/owner" || die "Cannot record stage owner: $1"
}

# -------------------- PREPARE SUBWINDOWS (one atomic owner) --------------------
# Pickles alone are not sufficient for a continuation: scratch directories
# may have been deleted between jobs. Validate every tile config before
# honoring --skip-prepare.
preparation_state_is_complete() {
    [ -f "$CONFIG_PATH" ] || return 1
    python3 - "$BASE_DIR" "$TILE_SCOPE" <<'PY_CHECK'
import os
import pickle
import sys
from pathlib import Path

base = Path(sys.argv[1])
tile_scope = sys.argv[2]
massh_path = os.environ.get('MASSH_PATH')
if massh_path:
    sys.path.insert(0, massh_path)
tile_configs = [p for p in base.glob('subwindow_*/subwindow_*/config.pkl')]
if not tile_configs:
    raise SystemExit(1)
for path in tile_configs:
    try:
        with path.open('rb') as stream:
            config = pickle.load(stream)
        if tile_scope == 'equatorial':
            lat_min = float(config.GRID.lat_min)
            lat_max = float(config.GRID.lat_max)
            if not lat_min < 0.0 < lat_max:
                continue
        scratch = Path(config.EXP.tmp_DA_path)
    except Exception as exc:
        print(f"cannot inspect tile config {path}: {exc}", file=sys.stderr)
        raise SystemExit(1)
    if not scratch.is_dir():
        print(f"missing tile scratch directory: {scratch}", file=sys.stderr)
        raise SystemExit(1)
raise SystemExit(0)
PY_CHECK
}

if claim_stage "${BARRIER_DIR}/prepare.lock"; then
    if $SKIP_PREPARE && preparation_state_is_complete; then
        echo "$(date '+%F %T') | Skipping preparation (--skip-prepare, pickles and tile scratch directories exist)"
    else
        if $SKIP_PREPARE; then
            echo "$(date '+%F %T') | --skip-prepare requested, but tile scratch state is incomplete; preparing again"
        fi
        echo "$(date '+%F %T') | Preparing subwindows and saving pickles"
        if ! MPLBACKEND=Agg run_stage python -u "${SRC_DIR}/prepare_VarDyn.py" "$PATH_CONFIG" "$PATH_CONFIG_EQ" "${PREPARE_ARGS[@]}"; then
            echo "$(date '+%F %T') | ERROR: Preparation failed!"
            touch "${BARRIER_DIR}/prepare_failed"
            exit 1
        fi
    fi
    echo "$(date '+%F %T') | Preparation complete"
    touch "${BARRIER_DIR}/prepared"

else
    echo "$(date '+%F %T') | Waiting for preparation to complete..."
    wait_for_marker "${BARRIER_DIR}/prepared" "${BARRIER_DIR}/prepare_failed" "${BARRIER_DIR}/prepare.lock" || exit 1
    echo "$(date '+%F %T') | Preparation detected, proceeding"
fi

# -------------------- TILE CLAIMING (atomic mkdir, works on Lustre/GPFS) --------------------

try_claim_tile() {
    local tile="$1"
    local lock_dir="${tile}/.tile_running.lock"
    local token="${JOB_ID}_${ARRAY_ID}_${BASHPID}"
    # Use the normalized array identity instead of Slurm's raw per-element
    # IDs, whose representation differs between squeue and sacct.
    local claimant_job="${JOB_ID}_${ARRAY_ID}"

    # The stable lock name prevents overlapping job generations from running
    # the same tile. mkdir is atomic on Lustre/GPFS.
    if mkdir "$lock_dir" 2>/dev/null; then
        printf '%s\n' "$claimant_job" > "${lock_dir}/job_id"
        printf '%s\n' "$token" > "${lock_dir}/token"
        printf '%s\n' "$token"
        return 0
    fi

    # A killed job can leave a directory behind. Reclaim it only when its
    # owning Slurm array task is no longer active; an unreadable/new lock
    # remains conservatively busy.
    local owner_job=""
    local owner_token=""
    [ -f "${lock_dir}/job_id" ] && owner_job=$(sed -n '1p' "${lock_dir}/job_id")
    [ -f "${lock_dir}/token" ] && owner_token=$(sed -n '1p' "${lock_dir}/token")
    [ -z "$owner_job" ] && return 1

    # Older locks stored only the array's base ID. Their token still identifies
    # the exact task, so do not keep a tile locked merely because a different
    # element of that array is pending or running.
    local owner_query="$owner_job"
    if [[ "$owner_job" != *_* && \
          "$owner_token" =~ ^${owner_job}_([0-9]+)_ ]]; then
        owner_query="${owner_job}_${BASH_REMATCH[1]}"
    fi
    # Never steal from another element of this array generation. A stale lock
    # from this generation is recovered by the dependent continuation job.
    local current_array_job="${SLURM_ARRAY_JOB_ID:-$JOB_ID}"
    if [[ "$owner_query" == "${current_array_job}_"* ]]; then
        return 1
    fi

    # Query the exact owner immediately before reclaiming an older lock.
    slurm_task_is_active "$owner_query" && return 1

    # Keep one nonempty tombstone per old lease. With a unique destination per
    # claimant, a second reclaimer could move the first claimant's NEW lock.
    # GNU mv -T cannot replace this nonempty directory: only one reclaimer wins.
    local stale_dir="${lock_dir}.stale-${owner_token:-$owner_query}"
    mv -T "$lock_dir" "$stale_dir" 2>/dev/null || return 1
    if mkdir "$lock_dir" 2>/dev/null; then
        printf '%s\n' "$claimant_job" > "${lock_dir}/job_id"
        printf '%s\n' "$token" > "${lock_dir}/token"
        printf '%s\n' "$token"
        return 0
    fi
    return 1
}

release_tile_claim() {
    local tile="$1"
    local token="$2"
    local lock_dir="${tile}/.tile_running.lock"
    local owner_token=""
    [ -f "${lock_dir}/token" ] && owner_token=$(sed -n '1p' "${lock_dir}/token")
    [ "$owner_token" = "$token" ] || return 0
    rm -f "${lock_dir}/job_id" "${lock_dir}/token"
    rmdir "$lock_dir" 2>/dev/null || true
}

tile_is_owned_by_task() {
    local tile_index="$1"
    (( tile_index % NUM_ARRAY == ARRAY_ID ))
}

# Only an explicit PENDING state authorizes borrowing another task's shard.
# Refresh once per dispatch pass, not per tile, to limit scheduler traffic.
# A task may start after this snapshot: tile locks still arbitrate ownership.
pending_array_tasks() {
    local listing task_id state
    listing=$(_slurm_query -r -h -j "$JOB_ID" -o '%i %T') || return 0
    while read -r task_id state; do
        if [[ "$task_id" == "${JOB_ID}_"* && "$state" == PENDING ]]; then
            local rank="${task_id#${JOB_ID}_}"
            [[ "$rank" =~ ^[0-9]+$ ]] && printf '%s\n' "$rank"
        fi
    done <<< "$listing"
    return 0
}

report_incomplete_tiles() {
    local tile_list="$1"
    local tile owner token
    while IFS= read -r tile; do
        [ -z "$tile" ] && continue
        [ -f "${tile}/.tile_complete.ok" ] && continue
        if [ -d "${tile}/.tile_running.lock" ]; then
            owner=$(sed -n '1p' "${tile}/.tile_running.lock/job_id" 2>/dev/null) || owner="unreadable"
            token=$(sed -n '1p' "${tile}/.tile_running.lock/token" 2>/dev/null) || token="unreadable"
            printf 'Incomplete tile: %s | lock owner=%s | token=%s\n' "$tile" "$owner" "$token" >&2
        else
            printf 'Incomplete tile: %s | no lock directory\n' "$tile" >&2
        fi
    done < "$tile_list"
}

# Inspect completed work across every deterministic shard rather than
# inferring completion from the number of running array elements.
window_tile_state() {
    local tile_list="$1"
    WINDOW_TILES_MISSING=0
    WINDOW_TILES_FAILED=0
    while IFS= read -r tile; do
        [ -z "$tile" ] && continue
        [ -f "${tile}/.tile_complete.ok" ] \
            || WINDOW_TILES_MISSING=$((WINDOW_TILES_MISSING + 1))
        [ -f "${tile}/.tile_failed" ] \
            && WINDOW_TILES_FAILED=$((WINDOW_TILES_FAILED + 1))
    done < "$tile_list"
}

incomplete_owners_are_active() {
    local tile index=0 owner checked=" "
    while IFS= read -r tile; do
        [ -z "$tile" ] && continue
        owner="${JOB_ID}_$((index % NUM_ARRAY))"
        index=$((index + 1))
        [ -f "$tile/.tile_complete.ok" ] && continue
        # A borrowed tile belongs to its actual claimant, not its pending rank.
        if [ -d "$tile/.tile_running.lock" ]; then
            owner=$(cat "$tile/.tile_running.lock/job_id" 2>/dev/null) || continue
            [ -n "$owner" ] || continue
        fi
        [[ "$checked" == *" $owner "* ]] && continue
        if ! slurm_task_is_active "$owner"; then
            [ -f "$tile/.tile_complete.ok" ] && continue
            echo "ERROR: owner $owner of incomplete tile $tile is no longer active" >&2
            return 1
        fi
        checked+="$owner "
    done < "$1"
    return 0
}

wait_for_spatial_merge_parts() {
    local iw="$1"
    local rank_count="$2"
    local merge_rank
    for ((merge_rank = 0; merge_rank < rank_count; merge_rank++)); do
        wait_for_marker "${BARRIER_DIR}/spatial_merge_iw${iw}_rank${merge_rank}.ok" \
            "${BARRIER_DIR}/spatial_merge_iw${iw}.failed" \
            "${BARRIER_DIR}/merge_iw${iw}_rank${merge_rank}.lock" || return 1
    done

}

# -------------------- TILE WORKER --------------------
run_single_tile() {
    local TILE="$1"
    local IW="$2"
    local CLAIM_TOKEN="$3"
    local child_pid=""
    trap - EXIT USR1
    trap 'stop_process_group "$child_pid"; release_tile_claim "$TILE" "$CLAIM_TOKEN"; exit 143' TERM INT
    local TILE_BASENAME=$(basename "$TILE")
    local TILE_PARENT=$(basename "$(dirname "$TILE")")
    local LOG_SUBDIR="${LOGDIR}/${TILE_PARENT}"
    mkdir -p "$LOG_SUBDIR"
    local TILE_LOG="${LOG_SUBDIR}/${TILE_BASENAME}_${GPU_LOG_ID}.log"

    echo "$(date '+%F %T') | GPU ${ARRAY_ID} | START tile ${TILE}" >> "$TILE_LOG"
    if [ -f "${BARRIER_DIR}/run.failed" ]; then
        release_tile_claim "$TILE" "$CLAIM_TOKEN"
        trap - TERM INT
        return 1
    fi
    rm -f "${TILE}/.tile_failed"
    OMP_NUM_THREADS=1 setsid timeout --kill-after=10 "$TILE_TIMEOUT" python "${SRC_DIR}/run_tile.py" "$TILE" $RESTART >> "$TILE_LOG" 2>&1 &
    child_pid=$!
    wait "$child_pid"
    local status=$?
    stop_process_group "$child_pid"
    child_pid=""
    if [ "$status" -eq 0 ] && [ ! -f "${TILE}/.tile_complete.ok" ]; then
        echo "ERROR: tile returned success without completion marker" >> "$TILE_LOG"
        status=2
    fi
    if [ "$status" -ne 0 ]; then
        touch "${BARRIER_DIR}/window_iw${IW}.failed" "${BARRIER_DIR}/run.failed"
    fi
    if [ $status -eq 0 ]; then
        rm -f "${TILE}/.tile_failed"
        echo "$(date '+%F %T') | GPU ${ARRAY_ID} | DONE  tile ${TILE}" >> "$TILE_LOG"
        if grep -Fq "Finished tile:" "$TILE_LOG"; then
            touch "${BARRIER_DIR}/computed_iw${IW}_${ARRAY_ID}"
        fi
    elif [ $status -eq 137 ]; then
        touch "${TILE}/.tile_failed"
        echo "$(date '+%F %T') | GPU ${ARRAY_ID} | KILLED (OOM?) tile ${TILE}" >> "$TILE_LOG"
    else
        touch "${TILE}/.tile_failed"
        echo "$(date '+%F %T') | GPU ${ARRAY_ID} | ERROR exit=${status} tile ${TILE}" >> "$TILE_LOG"
        # Echo the tail of the tile log to the main log so the error is visible
        # without having to dig into individual tile log files.
        echo "$(date '+%F %T') | GPU ${ARRAY_ID} | ERROR tile ${TILE} — last 40 lines of ${TILE_LOG}:" >&2
        tail -n 40 "$TILE_LOG" >&2
    fi
    release_tile_claim "$TILE" "$CLAIM_TOKEN"
    trap - EXIT TERM INT
    return "$status"
}

process_available_tiles() {
    local tile_list="$1"
    local iw="$2"
    local tile
    local claim_token
    local tile_index=0 owner_rank dispatch_mode
    local pending_ranks=" "
    ACTIVE_TILE_PIDS=()
    local pass_status=0
    TILES_PROCESSED_IN_PASS=0

    # Prefer our own shard, then help owners still pending in Slurm.
    for dispatch_mode in own pending; do
        tile_index=0
        if [ "$dispatch_mode" = pending ]; then
            pending_ranks=" $(pending_array_tasks | tr '\n' ' ') "
        fi
        while IFS= read -r tile; do
            [ -z "$tile" ] && continue

            if [ -f "${BARRIER_DIR}/run.failed" ]; then pass_status=1; break; fi
            owner_rank=$((tile_index % NUM_ARRAY))
            tile_index=$((tile_index + 1))
            if [ "$dispatch_mode" = own ]; then
                tile_is_owned_by_task "$((tile_index - 1))" || continue
            else
                (( owner_rank != ARRAY_ID )) || continue
                [[ "$pending_ranks" == *" $owner_rank "* ]] || continue
            fi

            # A previous pass or job generation may already have completed it.
            [ -f "${tile}/.tile_complete.ok" ] && continue

            claim_token=$(try_claim_tile "$tile") || continue
            # Another worker may have finished between our first check and mkdir.
            # Recheck under the claim before launching, including late owners.
            if [ -f "${tile}/.tile_complete.ok" ]; then
                release_tile_claim "$tile" "$claim_token"
                continue
            fi
            if [ "$dispatch_mode" = pending ]; then
                echo "$(date '+%F %T') | GPU ${ARRAY_ID} | Borrowing tile ${tile} from pending task ${JOB_ID}_${owner_rank}"
            fi
            run_single_tile "$tile" "$iw" "$claim_token" &
            ACTIVE_TILE_PIDS+=("$!")
            TILES_PROCESSED_IN_PASS=$((TILES_PROCESSED_IN_PASS + 1))
            echo "$(date '+%F %T') | GPU ${ARRAY_ID} | Active tiles: ${#ACTIVE_TILE_PIDS[@]}/${NUM_TILES_PER_GPU}"

            if ! wait_tile_workers "$((NUM_TILES_PER_GPU - 1))"; then
                pass_status=1
                break
            fi
        done < "$tile_list"
        (( pass_status == 0 )) || break
    done

    if (( pass_status == 0 )); then
        wait_tile_workers 0 || pass_status=1
    fi
    if (( pass_status != 0 )); then
        touch "${BARRIER_DIR}/run.failed"
        stop_tile_workers
        return 1
    fi
    ACTIVE_TILE_PIDS=()
    return 0
}

# Keep merge options available even when every durable window is skipped.
init_merge_args() {
    MERGE_ARGS=(--dir_save_pickle "$DIR_SAVE_PICKLE" --name_var_save "$NAME_VAR"
        --num_workers "$NUM_MERGE_WORKERS" --zarr_time_chunk "$ZARR_TIME_CHUNK"
        --zarr_spatial_chunk "$ZARR_SPATIAL_CHUNK" --zarr_compression_level "$ZARR_COMPRESSION_LEVEL")
    $ZARR_OUTPUT && MERGE_ARGS+=(--zarr_output)
    $OUTPUT_FLOAT64 && MERGE_ARGS+=(--output_float64)
    CLEANUP_ARGS=()
    $CLEANUP_TILE_ZARR && CLEANUP_ARGS+=(--cleanup_tile_zarr)
    return 0
}

merge_outputs() {
    run_stage python -u "${SRC_DIR}/merge_outputs.py" "$CONFIG_PATH" "${MERGE_ARGS[@]}" "$@"
}

publish_marker() {
    local marker="$1" temporary="${1}.tmp-${JOB_ID}_${ARRAY_ID}_${BASHPID}"
    printf 'Completed: %s\n' "$(date -Is)" > "$temporary" && mv -f "$temporary" "$marker"
}

finish_merge_stage() {
    local marker="$1" failed="$2" durable="$3"
    shift 3
    if merge_outputs "$@" && { [ -z "$durable" ] || publish_marker "$durable"; } \
        && touch "$marker"; then
        echo "$(date '+%F %T') | Merge completed: $marker"
    else
        touch "$failed"
        die "Merge or completion publication failed: $marker"
    fi
}

init_merge_args

# --------------- SEQUENTIAL TIME WINDOWS, TILE DISPATCH ---------------
mapfile -d '' -t TIME_WINDOWS < <(find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d -name 'subwindow_*' -print0 | LC_ALL=C sort -z)
(( ${#TIME_WINDOWS[@]} )) || die "No time windows found in $BASE_DIR"
IW=0

for TIME_DIR in "${TIME_WINDOWS[@]}"; do
    DURABLE_WINDOW_MARKER="${TIME_DIR}/.window_complete_${TILE_SCOPE}.ok"
    echo "$(date '+%F %T') | GPU ${ARRAY_ID} | Time window ${IW}: $TIME_DIR"

    if [ -z "$RESTART" ] && ! $FORCE_MERGE && [ -f "$DURABLE_WINDOW_MARKER" ]; then
        echo "$(date '+%F %T') | Time window ${IW} already complete; skipping assimilation and spatial merge"
        IW=$((IW + 1))
        continue
    fi

    # One actually-running task publishes the tile list atomically.
    TILE_LIST="${BARRIER_DIR}/tiles_iw${IW}"
    if claim_stage "${BARRIER_DIR}/queue_iw${IW}.lock"; then
        if [ "$TILE_SCOPE" = "equatorial" ]; then
            if ! python3 - "$TIME_DIR" > "${TILE_LIST}.tmp" <<'PY_TILE_SCOPE'
import os
import pickle
import sys
from pathlib import Path

time_dir = Path(sys.argv[1])
massh_path = os.environ.get('MASSH_PATH')
if massh_path:
    sys.path.insert(0, massh_path)
for tile in sorted(time_dir.glob("subwindow_*")):
    if not tile.is_dir():
        continue
    config_path = tile / "config.pkl"
    try:
        with config_path.open("rb") as stream:
            config = pickle.load(stream)
        lat_min = float(config.GRID.lat_min)
        lat_max = float(config.GRID.lat_max)
    except Exception as exc:
        print(f"ERROR: cannot inspect tile config {config_path}: {exc}",
              file=sys.stderr)
        raise SystemExit(1)
    if lat_min < 0.0 < lat_max:
        print(tile)
PY_TILE_SCOPE
            then
                echo "$(date '+%F %T') | ERROR: failed to select equatorial tiles in $TIME_DIR" >&2
                rm -f "${TILE_LIST}.tmp"
                touch "${BARRIER_DIR}/queue_failed_iw${IW}"
                exit 1
            fi
        else
            find "$TIME_DIR" -mindepth 1 -maxdepth 1 -type d -name "subwindow_*" | sort > "${TILE_LIST}.tmp"
        fi
        mv "${TILE_LIST}.tmp" "$TILE_LIST"
        TOTAL_TILES=$(wc -l < "$TILE_LIST")
        echo "$(date '+%F %T') | Found ${TOTAL_TILES} ${TILE_SCOPE} tiles for time window ${IW}"
        # Failures belong to an attempt. Clear them once, before releasing
        # this generation's workers; successful tiles remain resumable.
        while IFS= read -r tile; do
            [ -z "$tile" ] && continue
            [ -n "$RESTART" ] && rm -f "${tile}/.tile_complete.ok"
            rm -f "${tile}/.tile_failed"
        done < "$TILE_LIST"
        touch "${BARRIER_DIR}/queue_ready_iw${IW}"
    fi
    wait_for_marker "${BARRIER_DIR}/queue_ready_iw${IW}" "${BARRIER_DIR}/queue_failed_iw${IW}" "${BARRIER_DIR}/queue_iw${IW}.lock" || exit 1
    if [ ! -s "$TILE_LIST" ]; then
        echo "$(date '+%F %T') | ERROR: no ${TILE_SCOPE} tiles found in $TIME_DIR" >&2
        exit 1
    fi

    # Prefer each task's shard and borrow pending owners' tiles. Atomic claims
    # protect concurrent borrowers, late owners and overlapping generations.
    if ! $MERGE_ONLY; then
        tiles_done=0
        window_started=$SECONDS
        previous_missing=-1
        while true; do
            process_available_tiles "$TILE_LIST" "$IW" || exit 1
            tiles_done=$((tiles_done + TILES_PROCESSED_IN_PASS))

            window_tile_state "$TILE_LIST"
            if [ "$WINDOW_TILES_FAILED" -gt 0 ]; then
                echo "$(date '+%F %T') | Window failed: ${WINDOW_TILES_FAILED} tile(s) reported an error" >&2
                exit 1
            fi
            [ "$WINDOW_TILES_MISSING" -eq 0 ] && break

            if [ "$WINDOW_TILES_MISSING" -ne "$previous_missing" ]; then
                window_started=$SECONDS
                previous_missing=$WINDOW_TILES_MISSING
            fi
            if (( SECONDS - window_started >= BARRIER_TIMEOUT )); then
                echo "$(date '+%F %T') | Window timed out without completion progress: ${WINDOW_TILES_MISSING} tile(s) incomplete after scanning own and pending shards" >&2
                report_incomplete_tiles "$TILE_LIST"
                exit 1
            fi
            incomplete_owners_are_active "$TILE_LIST" || exit 1
            sleep 10

        done
        echo "$(date '+%F %T') | GPU ${ARRAY_ID} | Processed ${tiles_done} tiles in time window ${IW}"
    else
        echo "$(date '+%F %T') | GPU ${ARRAY_ID} | Skipping assimilation (--merge-only)"
    fi

    MERGE_MARKER="${BARRIER_DIR}/spatial_merge_iw${IW}.ok"
    MERGE_FAILED="${BARRIER_DIR}/spatial_merge_iw${IW}.failed"
    WINDOW_ARGS=(--iw_start "$IW" --iw_end "$((IW + 1))" "${FORCE_ARGS[@]}")
    MERGE_LOCK="${BARRIER_DIR}/merge_iw${IW}.lock"
    FINALIZE_ARGS=(--rank 0 --world 1)
    if $ZARR_OUTPUT; then
        # Every active task may claim merge ranks, regardless of pending peers.
        for ((MERGE_RANK = 0; MERGE_RANK < NUM_ARRAY; MERGE_RANK++)); do
            PART_MARKER="${BARRIER_DIR}/spatial_merge_iw${IW}_rank${MERGE_RANK}.ok"
            [ -f "$PART_MARKER" ] && continue
            [ -f "$MERGE_FAILED" ] && break
            if claim_stage "${BARRIER_DIR}/merge_iw${IW}_rank${MERGE_RANK}.lock"; then
                finish_merge_stage "$PART_MARKER" "$MERGE_FAILED" "" \
                    "${WINDOW_ARGS[@]}" --rank "$MERGE_RANK" --world "$NUM_ARRAY" --zarr_parts
            fi
        done
        wait_for_spatial_merge_parts "$IW" "$NUM_ARRAY" || exit 1
        MERGE_LOCK="${BARRIER_DIR}/merge_iw${IW}_finalize.lock"
        FINALIZE_ARGS=(--rank 0 --world "$NUM_ARRAY" --finalize_spatial_parts "${CLEANUP_ARGS[@]}")
    fi
    if claim_stage "$MERGE_LOCK"; then
        finish_merge_stage "$MERGE_MARKER" "$MERGE_FAILED" "$DURABLE_WINDOW_MARKER" \
            "${WINDOW_ARGS[@]}" "${FINALIZE_ARGS[@]}"
    else
        wait_for_marker "$MERGE_MARKER" "$MERGE_FAILED" "$MERGE_LOCK" || exit 1
    fi
    ((IW++))
done

# Publish durable completion before removing intermediate window products.
if claim_stage "${BARRIER_DIR}/final_merge.lock"; then
    finish_merge_stage "${BARRIER_DIR}/final_merge.ok" "${BARRIER_DIR}/run.failed" "$FINAL_MARKER" \
        --skip_spatial_merge --merge_time_windows "${FORCE_ARGS[@]}" "${CLEANUP_ARGS[@]}"
    if merge_outputs --skip_spatial_merge --cleanup_subwindow_outputs; then
        echo "$(date '+%F %T') | Subwindow merged outputs removed"
    else
        echo "$(date '+%F %T') | WARNING: post-completion subwindow cleanup failed" >&2
    fi
else
    wait_for_marker "${BARRIER_DIR}/final_merge.ok" "${BARRIER_DIR}/run.failed" "${BARRIER_DIR}/final_merge.lock" || exit 1
fi
