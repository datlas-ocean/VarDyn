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
#SBATCH --mem=120G
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
CONFIG_FILE="${VAR_DYN_CONFIG:-}"
args=("$@")
i=0
while [ $i -lt ${#args[@]} ]; do
    case "${args[$i]}" in
        --config)    i=$(( i + 1 )); CONFIG_FILE="${args[$i]}" ;;
        --config=*)  CONFIG_FILE="${args[$i]#--config=}" ;;
    esac
    i=$(( i + 1 ))
done

if [ -z "$CONFIG_FILE" ] || [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: provide an experiment config with --config CONFIG_FILE" >&2
    exit 1
fi
CONFIG_FILE="$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")"
# shellcheck disable=SC1090
source "$CONFIG_FILE"

# Paths in the experiment config are resolved relative to that config file,
# making submissions independent of the directory from which sbatch is run.
if [[ "$PATH_CONFIG" != /* ]]; then
    PATH_CONFIG="$(dirname "$CONFIG_FILE")/$PATH_CONFIG"
fi
if [[ "$PATH_CONFIG_EQ" != /* ]]; then
    PATH_CONFIG_EQ="$(dirname "$CONFIG_FILE")/$PATH_CONFIG_EQ"
fi

# These defaults are orchestration settings and can be overridden by the
# external config without changing the reusable launcher.
NUM_MERGE_WORKERS="${NUM_MERGE_WORKERS:-4}"
NUM_TILES_PER_GPU="${NUM_TILES_PER_GPU:-4}"
ZARR_OUTPUT="${ZARR_OUTPUT:-false}"
OUTPUT_FLOAT64="${OUTPUT_FLOAT64:-false}"
BARRIER_TIMEOUT="${BARRIER_TIMEOUT:-7200}"
FLAG_INIT_FROM_PREVIOUS="${FLAG_INIT_FROM_PREVIOUS:---flag_init_from_previous}"
FLAG_INIT="${FLAG_INIT:-false}"
FLAG_BACKGROUND="${FLAG_BACKGROUND:-false}"
NAME_EXP="${NAME_EXP:-}"

# -------------------- USER INPUT (optional CLI flags) --------------------
# Parse optional flags
SKIP_PREPARE=false
RESTART_ARGS=""
FORCE_MERGE=false
MERGE_ONLY=false
NAME_EXP_OVERRIDE=""
NAME_EXP_BACKGROUND_OVERRIDE=""
args=("$@")
i=0
while [ $i -lt ${#args[@]} ]; do
    case "${args[$i]}" in
        --config)       i=$(( i + 1 )) ;;
        --config=*)     ;;
        --skip-prepare)  SKIP_PREPARE=true ;;
        --restart)       RESTART_ARGS="--restart" ;;
        --force-merge)   FORCE_MERGE=true ;;
        --merge-only)    MERGE_ONLY=true; SKIP_PREPARE=true ;;
        --name_exp)      i=$(( i + 1 )); NAME_EXP_OVERRIDE="${args[$i]}" ;;
        --name_exp=*)    NAME_EXP_OVERRIDE="${args[$i]#--name_exp=}" ;;
        --name_exp_background) i=$(( i + 1 )); NAME_EXP_BACKGROUND_OVERRIDE="${args[$i]}" ;;
        --name_exp_background=*) NAME_EXP_BACKGROUND_OVERRIDE="${args[$i]#--name_exp_background=}" ;;
    esac
    i=$(( i + 1 ))
done

# A dedicated background experiment necessarily requires background mode.
if [ -n "$NAME_EXP_BACKGROUND_OVERRIDE" ]; then
    FLAG_BACKGROUND=true
fi

RESTART="$RESTART_ARGS"
FORCE_MERGE_ARG=""
$FORCE_MERGE && FORCE_MERGE_ARG="--force"

# Validate required settings
if [ -z "$MASH_DIR" ] || [ -z "$DIR_SAVE_PICKLE" ] || [ -z "$PATH_CONFIG" ] || \
   [ -z "$PATH_CONFIG_EQ" ] || [ -z "$INIT_DATE" ] || [ -z "$FINAL_DATE" ]; then
    echo "ERROR: One or more required USER SETTINGS are not set (MASH_DIR, DIR_SAVE_PICKLE, PATH_CONFIG, PATH_CONFIG_EQ, INIT_DATE, FINAL_DATE). Edit the USER SETTINGS block before submitting." >&2
    exit 1
fi

# EXP_NAME: --name_exp flag > name_experiment in PATH_CONFIG > filename fallback
if [ -n "$NAME_EXP_OVERRIDE" ]; then
    EXP_NAME="$NAME_EXP_OVERRIDE"
else
    EXP_NAME=$(python3 -c "
import re
txt = open('${PATH_CONFIG}').read()
m = re.search(r'^name_experiment\s*=\s*[\"\'](.*?)[\"\']', txt, re.MULTILINE)
print(m.group(1) if m else '')
" 2>/dev/null)
    if [ -z "$EXP_NAME" ]; then
        EXP_NAME=$(basename "$PATH_CONFIG" .py | sed 's/^config_//')
    fi
fi
BASE_DIR="${DIR_SAVE_PICKLE}/${EXP_NAME}"
CONFIG_PATH="${BASE_DIR}/config.pkl"

# -------------------- TIME-LIMIT CONTINUATION --------------------
# Slurm sends USR1 300 seconds before the wall-time limit. The first task
# obtaining the submission lock creates a continuation; completed work is
# skipped through their existing completion markers.
CONTINUATION_SCRIPT="${MASH_DIR}/slurm/run/VarDyn_GLO.sh"
FINAL_MARKER="${BASE_DIR}/experiment_complete.ok"
CONTINUATION_SUBMITTED=false
OWNED_STAGE_LOCK=""

submit_continuation() {
    local submit_lock="${BASE_DIR}/.continuation_${JOB_ID}.lock"
    mkdir "$submit_lock" 2>/dev/null || return 0
    [ -f "${FINAL_MARKER}" ] && [ -z "$RESTART" ] \
        && ! $FORCE_MERGE && ! $MERGE_ONLY && return 0
    [ "${CONTINUATION_SUBMITTED}" = true ] && return 0
    CONTINUATION_SUBMITTED=true

    local dependency="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
    local next_job
    local continuation_args=(--config "${CONFIG_FILE}" --skip-prepare)
    $MERGE_ONLY && continuation_args+=(--merge-only)
    [ -n "${NAME_EXP_OVERRIDE}" ] && continuation_args+=(--name_exp "${NAME_EXP_OVERRIDE}")
    if next_job=$(sbatch --parsable \
        --dependency="afterany:${dependency}" \
        "${CONTINUATION_SCRIPT}" "${continuation_args[@]}"); then
        echo "$(date '+%F %T') | Submitted continuation array ${next_job}"
    else
        rmdir "$submit_lock" 2>/dev/null || true
        echo "$(date '+%F %T') | ERROR: failed to submit continuation array" >&2
    fi
}

release_stage_lock() {
    if [ -n "$OWNED_STAGE_LOCK" ]; then
        rmdir "$OWNED_STAGE_LOCK" 2>/dev/null || true
    fi
}

handle_timeout() {
    release_stage_lock
    echo "$(date '+%F %T') | Slurm wall-time signal received; requesting continuation"
    submit_continuation
    exit 0
}

handle_cancel() {
    release_stage_lock
    echo "$(date '+%F %T') | Cancellation signal received; no continuation will be submitted"
    exit 0
}

trap handle_timeout USR1
trap handle_cancel TERM

INIT_BG_ARGS=""
$FLAG_INIT       && INIT_BG_ARGS+=" --flag_init"
$FLAG_BACKGROUND && INIT_BG_ARGS+=" --flag_background"
[ -n "$NAME_EXP" ] && INIT_BG_ARGS+=" --name_exp $NAME_EXP"
[ -n "$NAME_EXP_BACKGROUND_OVERRIDE" ] && NAME_EXP_BACKGROUND="$NAME_EXP_BACKGROUND_OVERRIDE"
[ -n "$NAME_EXP_BACKGROUND" ] && INIT_BG_ARGS+=" --name_exp_background $NAME_EXP_BACKGROUND"

PREPARE_ARGS="\
    --init_date $INIT_DATE \
    --final_date $FINAL_DATE \
    --dir_save_pickle $DIR_SAVE_PICKLE \
    --grid_type $GRID_TYPE \
    --grid_type_eq $GRID_TYPE_EQ \
    --nx_proc $NX_PROC --ny_proc $NY_PROC \
    --nx_proc_eq $NX_PROC_EQ --ny_proc_eq $NY_PROC_EQ \
    --dx $DX --dy $DY \
    --space_window_size_proc_x $SPACE_WIN_X \
    --space_window_size_proc_y $SPACE_WIN_Y \
    --space_window_size_proc_x_eq $SPACE_WIN_X_EQ \
    --space_window_size_proc_y_eq $SPACE_WIN_Y_EQ \
    --space_overlap_x $SPACE_OVERLAP_X --space_overlap_y $SPACE_OVERLAP_Y \
    --time_window_size_proc $TIME_WIN --time_overlap $TIME_OVERLAP \
    $FLAG_INIT_FROM_PREVIOUS \
    $INIT_BG_ARGS"

# -------------------- ENVIRONMENT --------------------
source /home/il/${USER}/.bashrc
conda activate MASSHv2
# Derive source and library paths from MASH_DIR (set in USER SETTINGS above).
# readlink -f "$0" is intentionally avoided: SLURM copies the script to
# /var/spool/slurmd/jobXXX/slurm_script before execution, making $0 useless.
SRC_DIR="${MASH_DIR}/slurm/src"
export MASSH_PATH="${MASH_DIR}/mapping"

# -------------------- LOG --------------------
LOGDIR="./logs/${EXP_NAME}_job-${JOB_ID}"
mkdir -p "$LOGDIR"
MAIN_LOGFILE="${LOGDIR}/gpu${ARRAY_ID}.log"
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
echo " Memory: $(ulimit -v 2>/dev/null || echo N/A)"
echo "=========================================="
if [ -n "$RESTART" ] && [ -f "$FINAL_MARKER" ]; then
    rm -f "$FINAL_MARKER"
    echo "$(date '+%F %T') | Removed stale completion marker for explicit restart"
fi
if [ -f "$FINAL_MARKER" ] && ! $FORCE_MERGE && ! $MERGE_ONLY; then
    echo "$(date '+%F %T') | Experiment already complete; exiting late array task"
    exit 0
fi


# -------------------- PREPARE SUBWINDOWS (one atomic owner) --------------------
# Pickles alone are not sufficient for a continuation: scratch directories
# may have been deleted between jobs. Validate every tile config before
# honoring --skip-prepare.
preparation_state_is_complete() {
    [ -f "$CONFIG_PATH" ] || return 1
    python3 - "$BASE_DIR" <<'PY_CHECK'
import pickle
import sys
from pathlib import Path

base = Path(sys.argv[1])
tile_configs = [p for p in base.glob('subwindow_*/subwindow_*/config.pkl')]
if not tile_configs:
    raise SystemExit(1)
for path in tile_configs:
    try:
        with path.open('rb') as stream:
            config = pickle.load(stream)
        scratch = Path(config.EXP.tmp_DA_path)
    except Exception:
        raise SystemExit(1)
    if not scratch.is_dir():
        print(f"missing tile scratch directory: {scratch}", file=sys.stderr)
        raise SystemExit(1)
raise SystemExit(0)
PY_CHECK
}

if mkdir "${BARRIER_DIR}/prepare.lock" 2>/dev/null; then
    OWNED_STAGE_LOCK="${BARRIER_DIR}/prepare.lock"
    if $SKIP_PREPARE && preparation_state_is_complete; then
        echo "$(date '+%F %T') | Skipping preparation (--skip-prepare, pickles and tile scratch directories exist)"
    else
        if $SKIP_PREPARE; then
            echo "$(date '+%F %T') | --skip-prepare requested, but tile scratch state is incomplete; preparing again"
        fi
        echo "$(date '+%F %T') | Preparing subwindows and saving pickles"
        MPLBACKEND=Agg python -u "${SRC_DIR}/prepare_VarDyn.py" "$PATH_CONFIG" "$PATH_CONFIG_EQ" $PREPARE_ARGS
        if [ $? -ne 0 ]; then
            echo "$(date '+%F %T') | ERROR: Preparation failed!"
            OWNED_STAGE_LOCK=""
            rmdir "${BARRIER_DIR}/prepare.lock" 2>/dev/null || true
            touch "${BARRIER_DIR}/prepare_failed"
            exit 1
        fi
    fi
    echo "$(date '+%F %T') | Preparation complete"
    touch "${BARRIER_DIR}/prepared"
    OWNED_STAGE_LOCK=""
else
    echo "$(date '+%F %T') | Waiting for preparation to complete..."
    while [ ! -f "${BARRIER_DIR}/prepared" ] && [ ! -f "${BARRIER_DIR}/prepare_failed" ]; do
        sleep 5
    done
    if [ -f "${BARRIER_DIR}/prepare_failed" ]; then
        echo "$(date '+%F %T') | ERROR: Preparation failed on the stage owner, aborting"
        exit 1
    fi
    echo "$(date '+%F %T') | Preparation detected, proceeding"
fi

# -------------------- TILE CLAIMING (atomic mkdir, works on Lustre/GPFS) --------------------
try_claim_tile() {
    # mkdir is atomic on all POSIX filesystems including Lustre/GPFS
    local tile="$1"
    local lock_dir="${tile}/.lock_${JOB_ID}"
    mkdir "$lock_dir" 2>/dev/null
}

# Wait on completed work rather than SLURM_ARRAY_TASK_COUNT: Slurm may start
# fewer array elements than requested, while every running element scans the
# same dynamically claimed queue.
wait_for_window_tiles() {
    local tile_list="$1"
    local waited=0
    while true; do
        local missing=0
        local failed=0
        while IFS= read -r tile; do
            [ -z "$tile" ] && continue
            [ -f "${tile}/.tile_complete.ok" ] || missing=$((missing + 1))
            [ -f "${tile}/.tile_failed" ] && failed=$((failed + 1))
        done < "$tile_list"
        if [ "$failed" -gt 0 ]; then
            echo "$(date '+%F %T') | Window failed: ${failed} tile(s) reported an error" >&2
            return 1
        fi
        [ "$missing" -eq 0 ] && return 0
        if [ "$waited" -ge "$BARRIER_TIMEOUT" ]; then
            echo "$(date '+%F %T') | Window incomplete: ${missing} tile(s) missing" >&2
            return 2
        fi
        sleep 10
        waited=$((waited + 10))
    done
}



wait_for_spatial_merge_parts() {
    local iw="$1"
    local rank_count="$2"
    while true; do
        if [ -f "${BARRIER_DIR}/spatial_merge_iw${iw}.failed" ]; then
            return 1
        fi
        local complete=0
        local merge_rank
        for ((merge_rank = 0; merge_rank < rank_count; merge_rank++)); do
            [ -f "${BARRIER_DIR}/spatial_merge_iw${iw}_rank${merge_rank}.ok" ] \
                && complete=$((complete + 1))
        done
        [ "$complete" -eq "$rank_count" ] && return 0
        sleep 10
    done
}

# -------------------- TILE WORKER --------------------
run_single_tile() {
    local TILE="$1"
    local IW="$2"
    local TILE_BASENAME=$(basename "$TILE")
    local TILE_PARENT=$(basename "$(dirname "$TILE")")
    local LOG_SUBDIR="${LOGDIR}/${TILE_PARENT}"
    mkdir -p "$LOG_SUBDIR"
    local TILE_LOG="${LOG_SUBDIR}/${TILE_BASENAME}_gpu${ARRAY_ID}.log"

    echo "$(date '+%F %T') | GPU ${ARRAY_ID} | START tile ${TILE}" >> "$TILE_LOG"
    rm -f "${TILE}/.tile_failed"
    OMP_NUM_THREADS=1 python "${SRC_DIR}/run_tile.py" "$TILE" $RESTART >> "$TILE_LOG" 2>&1
    local status=$?
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
}

# -------------------- SEQUENTIAL TIME WINDOWS, DYNAMIC TILE DISPATCH --------------------
TIME_WINDOWS=$(ls -d ${BASE_DIR}/subwindow_* 2>/dev/null | sort)
IW=0

for TIME_DIR in $TIME_WINDOWS; do
    echo "$(date '+%F %T') | GPU ${ARRAY_ID} | Time window ${IW}: $TIME_DIR"

    # One actually-running task publishes the tile list atomically.
    TILE_LIST="${BARRIER_DIR}/tiles_iw${IW}"
    if mkdir "${BARRIER_DIR}/queue_iw${IW}.lock" 2>/dev/null; then
        OWNED_STAGE_LOCK="${BARRIER_DIR}/queue_iw${IW}.lock"
        find "$TIME_DIR" -mindepth 1 -maxdepth 1 -type d -name "subwindow_*" | sort > "${TILE_LIST}.tmp"
        mv "${TILE_LIST}.tmp" "$TILE_LIST"
        TOTAL_TILES=$(wc -l < "$TILE_LIST")
        echo "$(date '+%F %T') | Found ${TOTAL_TILES} tiles for time window ${IW}"
        if [ -n "$RESTART" ]; then
            while IFS= read -r tile; do
                [ -z "$tile" ] && continue
                rm -f "${tile}/.tile_complete.ok"
                rm -f "${tile}/.tile_failed"
            done < "$TILE_LIST"
        fi
        touch "${BARRIER_DIR}/queue_ready_iw${IW}"
        OWNED_STAGE_LOCK=""
    fi
    while [ ! -f "${BARRIER_DIR}/queue_ready_iw${IW}" ]; do sleep 1; done

    # Each task dynamically claims tiles (first to mkdir wins)
    if ! $MERGE_ONLY; then
        running=0
        tiles_done=0
        while IFS= read -r TILE; do
            [ -z "$TILE" ] && continue

            # Try to claim this tile; skip if another GPU already got it
            try_claim_tile "$TILE" || continue

            run_single_tile "$TILE" "$IW" &
            ((running++))
            ((tiles_done++))

            if (( running >= NUM_TILES_PER_GPU )); then
                wait -n
                ((running--))
            fi
        done < "$TILE_LIST"
        wait
        echo "$(date '+%F %T') | GPU ${ARRAY_ID} | Processed ${tiles_done} tiles in time window ${IW}"
    else
        echo "$(date '+%F %T') | GPU ${ARRAY_ID} | Skipping assimilation (--merge-only)"
    fi

    if ! $MERGE_ONLY; then
        wait_for_window_tiles "$TILE_LIST"
        window_status=$?
        if [ "$window_status" -ne 0 ]; then
            # A timeout can benefit from a continuation; a deterministic tile
            # failure must be fixed instead of creating a relaunch loop.
            [ "$window_status" -eq 2 ] && submit_continuation
            exit 1
        fi
    fi


    ZARR_OUTPUT_ARG=""
    OUTPUT_FLOAT64_ARG=""
    $ZARR_OUTPUT && ZARR_OUTPUT_ARG="--zarr_output"
    $OUTPUT_FLOAT64 && OUTPUT_FLOAT64_ARG="--output_float64"
    MERGE_MARKER="${BARRIER_DIR}/spatial_merge_iw${IW}.ok"
    MERGE_FAILED="${BARRIER_DIR}/spatial_merge_iw${IW}.failed"

    if $ZARR_OUTPUT; then
        # Merge ranks are dynamically claimed, just like assimilation tiles.
        # If fewer array tasks start, each running task processes more ranks;
        # with all tasks running, every rank uses its own CPU allocation.
        for ((MERGE_RANK = 0; MERGE_RANK < NUM_ARRAY; MERGE_RANK++)); do
            PART_MARKER="${BARRIER_DIR}/spatial_merge_iw${IW}_rank${MERGE_RANK}.ok"
            [ -f "$PART_MARKER" ] && continue
            [ -f "$MERGE_FAILED" ] && break
            PART_LOCK="${BARRIER_DIR}/merge_iw${IW}_rank${MERGE_RANK}.lock"
            if mkdir "$PART_LOCK" 2>/dev/null; then
                OWNED_STAGE_LOCK="$PART_LOCK"
                echo "$(date '+%F %T') | GPU ${ARRAY_ID} | Spatial merge part ${MERGE_RANK}/${NUM_ARRAY} for window ${IW}"
                if python -u "${SRC_DIR}/merge_outputs.py" "$CONFIG_PATH" \
                    --dir_save_pickle "$DIR_SAVE_PICKLE" \
                    --name_var_save "$NAME_VAR" \
                    --num_workers "$NUM_MERGE_WORKERS" \
                    --iw_start "$IW" \
                    --iw_end "$((IW + 1))" \
                    --rank "$MERGE_RANK" \
                    --world "$NUM_ARRAY" \
                    --zarr_parts \
                    $FORCE_MERGE_ARG $ZARR_OUTPUT_ARG $OUTPUT_FLOAT64_ARG; then
                    touch "$PART_MARKER"
                    OWNED_STAGE_LOCK=""
                    echo "$(date '+%F %T') | Spatial merge part ${MERGE_RANK}/${NUM_ARRAY} done"
                else
                    echo "$(date '+%F %T') | Spatial merge part ${MERGE_RANK}/${NUM_ARRAY} failed" >&2
                    touch "$MERGE_FAILED"
                    OWNED_STAGE_LOCK=""
                    exit 1
                fi
            fi
        done

        wait_for_spatial_merge_parts "$IW" "$NUM_ARRAY"
        merge_parts_status=$?
        if [ "$merge_parts_status" -ne 0 ]; then
            exit 1
        fi

        FINALIZE_LOCK="${BARRIER_DIR}/merge_iw${IW}_finalize.lock"
        if mkdir "$FINALIZE_LOCK" 2>/dev/null; then
            OWNED_STAGE_LOCK="$FINALIZE_LOCK"
            echo "$(date '+%F %T') | Finalizing ${NUM_ARRAY} Zarr parts for time window ${IW}"
            if python -u "${SRC_DIR}/merge_outputs.py" "$CONFIG_PATH" \
                --dir_save_pickle "$DIR_SAVE_PICKLE" \
                --name_var_save "$NAME_VAR" \
                --num_workers "$NUM_MERGE_WORKERS" \
                --iw_start "$IW" \
                --iw_end "$((IW + 1))" \
                --rank 0 \
                --world "$NUM_ARRAY" \
                --finalize_spatial_parts \
                $FORCE_MERGE_ARG $ZARR_OUTPUT_ARG $OUTPUT_FLOAT64_ARG; then
                touch "$MERGE_MARKER"
                OWNED_STAGE_LOCK=""
                echo "$(date '+%F %T') | Spatial merge done for time window ${IW}"
            else
                echo "$(date '+%F %T') | Spatial merge finalization failed for time window ${IW}" >&2
                touch "$MERGE_FAILED"
                OWNED_STAGE_LOCK=""
                exit 1
            fi
        else
            echo "$(date '+%F %T') | Waiting for spatial merge finalization ${IW}"
            while [ ! -f "$MERGE_MARKER" ] && [ ! -f "$MERGE_FAILED" ]; do
                sleep 10
            done
            [ -f "$MERGE_FAILED" ] && exit 1
        fi
    else
        # NetCDF files are independent per date; retain the single-owner path.
        if mkdir "${BARRIER_DIR}/merge_iw${IW}.lock" 2>/dev/null; then
            OWNED_STAGE_LOCK="${BARRIER_DIR}/merge_iw${IW}.lock"
            echo "$(date '+%F %T') | Spatial NetCDF merge for time window ${IW}"
            if python -u "${SRC_DIR}/merge_outputs.py" "$CONFIG_PATH" \
                --dir_save_pickle "$DIR_SAVE_PICKLE" \
                --name_var_save "$NAME_VAR" \
                --num_workers "$NUM_MERGE_WORKERS" \
                --iw_start "$IW" \
                --iw_end "$((IW + 1))" \
                --rank 0 \
                --world 1 \
                $FORCE_MERGE_ARG $OUTPUT_FLOAT64_ARG; then
                touch "$MERGE_MARKER"
                OWNED_STAGE_LOCK=""
            else
                touch "$MERGE_FAILED"
                OWNED_STAGE_LOCK=""
                exit 1
            fi
        else
            while [ ! -f "$MERGE_MARKER" ] && [ ! -f "$MERGE_FAILED" ]; do
                sleep 10
            done
            [ -f "$MERGE_FAILED" ] && exit 1
        fi
    fi

    ((IW++))
done

# Final: exactly one running task merges all time windows.
if mkdir "${BARRIER_DIR}/final_merge.lock" 2>/dev/null; then
    OWNED_STAGE_LOCK="${BARRIER_DIR}/final_merge.lock"
    echo "$(date '+%Y-%m-%d %H:%M:%S') | Merging all time windows"
    if python -u "${SRC_DIR}/merge_outputs.py" "$CONFIG_PATH" \
        --dir_save_pickle "$DIR_SAVE_PICKLE" \
        --name_var_save "$NAME_VAR" \
        --num_workers "$NUM_MERGE_WORKERS" \
        --skip_spatial_merge \
        --merge_time_windows \
        $FORCE_MERGE_ARG $ZARR_OUTPUT_ARG $OUTPUT_FLOAT64_ARG; then
        tmp_marker="${FINAL_MARKER}.tmp-${SLURM_JOB_ID:-$$}"
        printf 'Experiment completed: %s\n' "$(date -Is)" > "$tmp_marker"
        mv -f "$tmp_marker" "$FINAL_MARKER"
        OWNED_STAGE_LOCK=""
        echo "$(date '+%Y-%m-%d %H:%M:%S') | All time windows processed"
        echo "$(date '+%Y-%m-%d %H:%M:%S') | Completion marker: $FINAL_MARKER"
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') | ERROR: final time-window merge failed" >&2
        OWNED_STAGE_LOCK=""
        rmdir "${BARRIER_DIR}/final_merge.lock" 2>/dev/null || true
        submit_continuation
        exit 1
    fi

    # Cleanup barrier files
    rm -rf "$BARRIER_DIR"
fi
