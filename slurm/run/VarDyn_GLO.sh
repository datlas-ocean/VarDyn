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
#SBATCH --signal=B:TERM@300
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
# Slurm sends TERM 300 seconds before the wall-time limit. Array task 0
# submits a dependent continuation array; completed tiles and merges are
# skipped through their existing completion markers.
CONTINUATION_SCRIPT="${MASH_DIR}/slurm/run/VarDyn_GLO.sh"
FINAL_MARKER="${BASE_DIR}/experiment_complete.ok"
CONTINUATION_SUBMITTED=false

submit_continuation() {
    [ "${ARRAY_ID}" -ne 0 ] && return 0
    [ -f "${FINAL_MARKER}" ] && return 0
    [ "${CONTINUATION_SUBMITTED}" = true ] && return 0
    CONTINUATION_SUBMITTED=true

    local dependency="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
    local next_job
    local continuation_args=(--config "${CONFIG_FILE}" --skip-prepare)
    [ -n "${NAME_EXP_OVERRIDE}" ] && continuation_args+=(--name_exp "${NAME_EXP_OVERRIDE}")
    if next_job=$(sbatch --parsable \
        --dependency="afterany:${dependency}" \
        "${CONTINUATION_SCRIPT}" "${continuation_args[@]}"); then
        echo "$(date '+%F %T') | Submitted continuation array ${next_job}"
    else
        echo "$(date '+%F %T') | ERROR: failed to submit continuation array" >&2
    fi
}

trap 'submit_continuation; exit 0' TERM

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
# Task 0 cleans up stale barriers before starting
if [ $ARRAY_ID -eq 0 ]; then
    rm -rf "$BARRIER_DIR" 2>/dev/null || true
    sleep 2  # Allow Lustre/GPFS to propagate the deletion to other nodes
fi
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

# -------------------- PREPARE SUBWINDOWS (task 0 only) --------------------
if [ $ARRAY_ID -eq 0 ]; then
    if [ -f "$CONFIG_PATH" ] && $SKIP_PREPARE; then
        echo "$(date '+%F %T') | Skipping preparation (--skip-prepare, pickles exist)"
    else
        echo "$(date '+%F %T') | Preparing subwindows and saving pickles"
        MPLBACKEND=Agg python -u "${SRC_DIR}/prepare_VarDyn.py" "$PATH_CONFIG" "$PATH_CONFIG_EQ" $PREPARE_ARGS
        if [ $? -ne 0 ]; then
            echo "$(date '+%F %T') | ERROR: Preparation failed!"
            touch "${BARRIER_DIR}/prepare_failed"
            exit 1
        fi
    fi
    echo "$(date '+%F %T') | Preparation complete"
    touch "${BARRIER_DIR}/prepared"
else
    echo "$(date '+%F %T') | Waiting for preparation to complete..."
    while [ ! -f "${BARRIER_DIR}/prepared" ] && [ ! -f "${BARRIER_DIR}/prepare_failed" ]; do
        sleep 5
    done
    if [ -f "${BARRIER_DIR}/prepare_failed" ]; then
        echo "$(date '+%F %T') | ERROR: Preparation failed on task 0, aborting"
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

# -------------------- BARRIER: wait for all array tasks --------------------
barrier_wait() {
    local tag="$1"
    # Retry touch in case BARRIER_DIR has a stale handle or needs to be recreated
    local _t
    for _t in 1 2 3 4 5; do
        mkdir -p "$BARRIER_DIR" 2>/dev/null
        touch "${BARRIER_DIR}/${tag}_${ARRAY_ID}" 2>/dev/null && break
        echo "$(date '+%F %T') | WARNING: barrier touch ${tag}_${ARRAY_ID} failed (attempt ${_t}), retrying..." >&2
        sleep $(( _t * 3 ))
    done
    if [ ! -f "${BARRIER_DIR}/${tag}_${ARRAY_ID}" ]; then
        echo "$(date '+%F %T') | FATAL: Could not write barrier file ${tag}_${ARRAY_ID}" >&2
        exit 1
    fi
    local waited=0
    while [ $(ls "${BARRIER_DIR}/${tag}_"* 2>/dev/null | wc -l) -lt $NUM_ARRAY ]; do
        sleep 5
        waited=$(( waited + 5 ))
        if [ $waited -ge $BARRIER_TIMEOUT ]; then
            echo "$(date '+%F %T') | WARNING: barrier_wait timeout (${BARRIER_TIMEOUT}s) for ${tag} — proceeding without all tasks" >&2
            break
        fi
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
    OMP_NUM_THREADS=1 python "${SRC_DIR}/run_tile.py" "$TILE" $RESTART >> "$TILE_LOG" 2>&1
    local status=$?
    if [ $status -eq 0 ]; then
        echo "$(date '+%F %T') | GPU ${ARRAY_ID} | DONE  tile ${TILE}" >> "$TILE_LOG"
        if grep -Fq "Finished tile:" "$TILE_LOG"; then
            touch "${BARRIER_DIR}/computed_iw${IW}_${ARRAY_ID}"
        fi
    elif [ $status -eq 137 ]; then
        echo "$(date '+%F %T') | GPU ${ARRAY_ID} | KILLED (OOM?) tile ${TILE}" >> "$TILE_LOG"
    else
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

    # Task 0 creates the tile list
    TILE_LIST="${BARRIER_DIR}/tiles_iw${IW}"
    if [ $ARRAY_ID -eq 0 ]; then
        find "$TIME_DIR" -mindepth 1 -maxdepth 1 -type d -name "subwindow_*" | sort > "$TILE_LIST"
        TOTAL_TILES=$(wc -l < "$TILE_LIST")
        echo "$(date '+%F %T') | Found ${TOTAL_TILES} tiles for time window ${IW}"
        touch "${BARRIER_DIR}/queue_ready_iw${IW}"
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

    # Barrier: wait for all GPUs to finish this time window
    barrier_wait "tw${IW}"

    if ! $FORCE_MERGE && ! compgen -G "${BARRIER_DIR}/computed_iw${IW}_"'*' > /dev/null; then
        echo "$(date '+%F %T') | GPU ${ARRAY_ID} | Skipping spatial merge for time window ${IW}: no tile computed and --force-merge not set"
        barrier_wait "merge${IW}"
        ((IW++))
        continue
    fi

    # Spatial merge: every array task processes its share of dates
    echo "$(date '+%F %T') | GPU ${ARRAY_ID} | Spatial merge for time window ${IW} (rank ${ARRAY_ID}/${NUM_ARRAY})"
    python -u "${SRC_DIR}/merge_outputs.py" "$CONFIG_PATH" \
        --dir_save_pickle "$DIR_SAVE_PICKLE" \
        --name_var_save "$NAME_VAR" \
        --num_workers "$NUM_MERGE_WORKERS" \
        --iw_start "$IW" \
        --iw_end "$((IW + 1))" \
        --rank "$ARRAY_ID" \
        --world "$NUM_ARRAY" \
        $FORCE_MERGE_ARG
    echo "$(date '+%F %T') | GPU ${ARRAY_ID} | Spatial merge done for time window ${IW}"

    # Wait for all ranks to finish their date shards before next time window
    barrier_wait "merge${IW}"

    ((IW++))
done

# Final: merge all time windows (task 0 only)
if [ $ARRAY_ID -eq 0 ]; then
    echo "$(date '+%F %T') | Merging all time windows"
    if python -u "${SRC_DIR}/merge_outputs.py" "$CONFIG_PATH" \
        --dir_save_pickle "$DIR_SAVE_PICKLE" \
        --name_var_save "$NAME_VAR" \
        --num_workers "$NUM_MERGE_WORKERS" \
        --skip_spatial_merge \
        --merge_time_windows \
        $FORCE_MERGE_ARG; then
        tmp_marker="${FINAL_MARKER}.tmp-${SLURM_JOB_ID:-$$}"
        printf 'Experiment completed: %s\n' "$(date -Is)" > "$tmp_marker"
        mv -f "$tmp_marker" "$FINAL_MARKER"
        echo "$(date '+%F %T') | All time windows processed"
        echo "$(date '+%F %T') | Completion marker: $FINAL_MARKER"
    else
        echo "$(date '+%F %T') | ERROR: final time-window merge failed" >&2
        exit 1
    fi

    # Cleanup barrier files
    rm -rf "$BARRIER_DIR"
fi
