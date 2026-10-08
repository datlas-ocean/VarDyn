import os
import re
from pathlib import Path
import subprocess
import shlex

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
LAUNCHER = REPOSITORY / "slurm" / "run" / "VarDyn_GLO.sh"


def _tile_lock_functions() -> str:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    names = ("_slurm_query", "slurm_task_is_active", "try_claim_tile",
             "release_tile_claim", "report_incomplete_tiles",
             "wait_for_marker", "stop_process_group", "run_stage", "wait_tile_workers",
             "stop_tile_workers", "run_single_tile",
             "process_available_tiles", "wait_for_spatial_merge_parts",
             "init_merge_args", "merge_outputs", "publish_marker", "claim_stage",
             "finish_merge_stage", "submit_continuation", "incomplete_owners_are_active",
             "initialize_run_state", "window_tile_state")
    return "\n".join(re.search(
        rf"^{name}\(\) \{{.*?^\}}", launcher, re.M | re.S).group()
        for name in names)



def _run_lock_scenario(tmp_path: Path, scenario: str) -> subprocess.CompletedProcess:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_squeue = bin_dir / "squeue"
    fake_squeue.write_text(
        "#!/bin/bash\n"
        "case \" $* \" in\n"
        "  *\" -j 101_0 \"*) printf '101_0\\n' ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_squeue.chmod(0o755)

    script = tmp_path / "scenario.sh"
    script.write_text(
        "#!/bin/bash\nset -u\n"
        + _tile_lock_functions()
        + "\n"
        + scenario,
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    return subprocess.run(
        ["bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


def test_tile_lock_blocks_an_active_predecessor(tmp_path):
    tile = tmp_path / "tile"
    tile.mkdir()
    result = _run_lock_scenario(
        tmp_path,
        f"""
tile={tile!s}
JOB_ID=101
ARRAY_ID=0
first_token=$(try_claim_tile "$tile") || exit 1
JOB_ID=202
if try_claim_tile "$tile" >/dev/null; then exit 2; fi
release_tile_claim "$tile" "$first_token"
second_token=$(try_claim_tile "$tile") || exit 3
release_tile_claim "$tile" "$second_token"
test ! -d "$tile/.tile_running.lock"
""",
    )
    assert result.returncode == 0, result.stderr


def test_tile_lock_reclaims_an_inactive_predecessor(tmp_path):
    tile = tmp_path / "tile"
    lock = tile / ".tile_running.lock"
    lock.mkdir(parents=True)
    (lock / "job_id").write_text("303\n", encoding="utf-8")
    (lock / "token").write_text("old-token\n", encoding="utf-8")
    result = _run_lock_scenario(
        tmp_path,
        f"""
tile={tile!s}
JOB_ID=404
ARRAY_ID=1
token=$(try_claim_tile "$tile") || exit 1
release_tile_claim "$tile" "$token"
test ! -d "$tile/.tile_running.lock"
""",
    )
    assert result.returncode == 0, result.stderr


def test_tile_lock_never_steals_from_current_array(tmp_path):
    tile = tmp_path / "tile"
    tile.mkdir()
    result = _run_lock_scenario(
        tmp_path,
        f"""
tile={tile!s}
JOB_ID=101
ARRAY_ID=0
first_token=$(try_claim_tile "$tile") || exit 1
ARRAY_ID=1
if try_claim_tile "$tile" >/dev/null; then exit 2; fi
test "$(cat "$tile/.tile_running.lock/token")" = "$first_token" || exit 3
release_tile_claim "$tile" "$first_token"
test ! -d "$tile/.tile_running.lock"
""",
    )
    assert result.returncode == 0, result.stderr


def test_launcher_has_resume_and_predecessor_guards():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert '--predecessor-job "${dependency}"' in launcher
    assert '[ -f "${tile}/.tile_complete.ok" ] && continue' in launcher
    assert '.window_complete_${TILE_SCOPE}.ok' in launcher


def test_old_array_lock_after_targeted_query_error(tmp_path):
    for index, (listing, status, reclaimed) in enumerate([
        ("", 0, True),
        ("12771183_4", 0, True),
        ("12771183_5", 0, False),
        ("", 1, False),
        ("12850304_0", 1, False),
    ]):
        case = tmp_path / str(index)
        tile = case / "tile"
        lock = tile / ".tile_running.lock"
        lock.mkdir(parents=True)
        (lock / "job_id").write_text("12771183\n")
        old_token = "12771183_5_3092055"
        (lock / "token").write_text(old_token + "\n")
        result = _run_lock_scenario(case, f"""
_slurm_query() {{
    if [[ " $* " == *" -j "* ]]; then
        echo "slurm_load_jobs error: Invalid job id specified" >&2
        return 1
    fi
    printf '%s\\n' '{listing}'
    return {status}
}}
JOB_ID=12850304
ARRAY_ID=0
if token=$(try_claim_tile "{tile}"); then
    echo RECLAIMED
else
    echo RETAINED
fi
""")
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == ("RECLAIMED" if reclaimed else "RETAINED")
        assert "Invalid job id specified" in result.stderr
        if reclaimed:
            assert (lock / "job_id").read_text().strip() == "12850304_0"
        else:
            assert (lock / "token").read_text().strip() == old_token
        if status:
            assert "retaining tile lock" in result.stderr


def test_incomplete_tile_diagnostics(tmp_path):
    tile = tmp_path / "tile"
    lock = tile / ".tile_running.lock"
    lock.mkdir(parents=True)
    (lock / "job_id").write_text("12771183\n")
    (lock / "token").write_text("12771183_5_3092055\n")
    done = tmp_path / "done"
    done.mkdir()
    (done / ".tile_complete.ok").touch()
    tile_list = tmp_path / "tiles"
    tile_list.write_text(f"{tile}\n{done}\n{tmp_path / 'unlocked'}\n")
    result = _run_lock_scenario(tmp_path, f'report_incomplete_tiles "{tile_list}"')
    assert result.returncode == 0, result.stderr
    assert "lock owner=12771183 | token=12771183_5_3092055" in result.stderr
    assert "no lock directory" in result.stderr
    assert str(done) not in result.stderr
    assert not result.stdout


def test_predecessor_invalid_id_falls_back_to_full_listing(tmp_path):
    result = _run_lock_scenario(tmp_path, """
PREDECESSOR_JOB=303
_slurm_query() {
    [[ " $* " == *" -j "* ]] && return 1
    printf '%s\n' "$listing"
}
listing=303_2
slurm_task_is_active 303 || exit 1
listing=404_0
if slurm_task_is_active 303; then exit 2; fi
""")
    assert result.returncode == 0, result.stderr


def test_barrier_dead_owner_and_unknown_owner_timeout(tmp_path):
    result = _run_lock_scenario(tmp_path, f"""
BARRIER_DIR={tmp_path}
BARRIER_TIMEOUT=1
mkdir "$BARRIER_DIR/stage.lock"
printf '303_0\n' > "$BARRIER_DIR/stage.lock/owner"
if wait_for_marker "$BARRIER_DIR/ready" "$BARRIER_DIR/failed" "$BARRIER_DIR/stage.lock"; then exit 1; fi
# Scheduler unavailable: never steal the lock, but still stop waiting.
_slurm_query() {{ return 1; }}
sleep() {{ SECONDS=$((SECONDS + 1)); }}
if wait_for_marker "$BARRIER_DIR/ready" "$BARRIER_DIR/failed" "$BARRIER_DIR/stage.lock"; then exit 2; fi
test -d "$BARRIER_DIR/stage.lock"
""")
    assert result.returncode == 0, result.stderr
    assert "disappeared" in result.stderr
    assert "timed out" in result.stderr


def test_barrier_success_and_shared_failure(tmp_path):
    result = _run_lock_scenario(tmp_path, f"""
BARRIER_DIR={tmp_path}
BARRIER_TIMEOUT=10
touch "$BARRIER_DIR/ready"
wait_for_marker "$BARRIER_DIR/ready" "$BARRIER_DIR/failed" "$BARRIER_DIR/lock" || exit 1
touch "$BARRIER_DIR/run.failed"
if wait_for_marker "$BARRIER_DIR/ready" "$BARRIER_DIR/failed" "$BARRIER_DIR/lock"; then exit 2; fi
""")
    assert result.returncode == 0, result.stderr


def _worker_setup(tmp_path):
    return f"""
BARRIER_DIR={tmp_path}
LOGDIR={tmp_path}/logs
SRC_DIR={tmp_path}
JOB_ID=404
ARRAY_ID=0
NUM_ARRAY=1
NUM_TILES_PER_GPU=2
TILE_TIMEOUT=15
GPU_LOG_ID=test
RESTART=""
ACTIVE_TILE_PIDS=()
mkdir -p "$LOGDIR" "$BARRIER_DIR/tile0" "$BARRIER_DIR/tile1" "$BARRIER_DIR/tile2"
printf '%s\n' "$BARRIER_DIR/tile0" "$BARRIER_DIR/tile1" "$BARRIER_DIR/tile2" > "$BARRIER_DIR/tiles"
"""


def test_tile_failure_stops_peers_and_prevents_later_launch(tmp_path):
    result = _run_lock_scenario(tmp_path, _worker_setup(tmp_path) + """
cat > "$BARRIER_DIR/bin/python" <<'SH'
#!/bin/bash
tile="$2"
printf '%s\n' "$$" > "$tile/child_pid"
if [[ "$tile" == *tile0 ]]; then sleep 1; exit 7; fi
sleep 20
touch "$tile/should_not_finish"
SH
chmod +x "$BARRIER_DIR/bin/python"
if process_available_tiles "$BARRIER_DIR/tiles" 0; then exit 1; fi
test -f "$BARRIER_DIR/run.failed" || exit 2
test -f "$BARRIER_DIR/window_iw0.failed" || exit 3
test ! -f "$BARRIER_DIR/tile2/child_pid" || exit 4
test ! -f "$BARRIER_DIR/tile1/should_not_finish" || exit 5
for tile in "$BARRIER_DIR/tile0" "$BARRIER_DIR/tile1"; do
    test ! -d "$tile/.tile_running.lock" || exit 6
    if [ -f "$tile/child_pid" ] && kill -0 "$(cat "$tile/child_pid")" 2>/dev/null; then exit 7; fi
done
""")
    assert result.returncode == 0, result.stderr
    log = next((tmp_path / "logs").glob("*/tile0_test.log"))
    assert "ERROR exit=7" in log.read_text()


def test_success_without_marker_is_failure_and_tile_timeout_is_bounded(tmp_path):
    result = _run_lock_scenario(tmp_path, _worker_setup(tmp_path) + """
cat > "$BARRIER_DIR/bin/python" <<'SH'
#!/bin/bash
if [[ "$2" == *tile1 ]]; then sleep 20; fi
exit 0
SH
chmod +x "$BARRIER_DIR/bin/python"
token=$(try_claim_tile "$BARRIER_DIR/tile0")
(run_single_tile "$BARRIER_DIR/tile0" 0 "$token")
test "$?" -eq 2 || exit 1
rm "$BARRIER_DIR/run.failed"
TILE_TIMEOUT=1
token=$(try_claim_tile "$BARRIER_DIR/tile1")
(run_single_tile "$BARRIER_DIR/tile1" 0 "$token")
test "$?" -eq 124 || exit 2
test ! -d "$BARRIER_DIR/tile1/.tile_running.lock"
""")
    assert result.returncode == 0, result.stderr


def test_reaps_abnormally_killed_worker_even_when_first_worker_is_running(tmp_path):
    result = _run_lock_scenario(tmp_path, f"""
BARRIER_DIR={tmp_path}
sleep 20 &
first=$!
(sleep 0.1; kill -KILL "$BASHPID") &
second=$!
ACTIVE_TILE_PIDS=("$first" "$second")
status=0
wait_tile_workers 0 || status=$?
kill "$first" 2>/dev/null || true
wait "$first" 2>/dev/null || true
test "$status" -ne 0
""")
    assert result.returncode == 0, result.stderr


def test_stage_stops_on_shared_failure(tmp_path):
    result = _run_lock_scenario(tmp_path, f"""
BARRIER_DIR={tmp_path}
STAGE_TIMEOUT=20
(sleep 1; touch "$BARRIER_DIR/run.failed") &
notifier=$!
if run_stage bash -c 'sleep 15; touch "$1"' bash "$BARRIER_DIR/should_not_finish"; then exit 1; fi
wait "$notifier"
test ! -f "$BARRIER_DIR/should_not_finish"
""")
    assert result.returncode == 0, result.stderr


def test_land_tile_publishes_explicit_state_without_inversion(tmp_path):
    import sys
    script = r'''
import pickle
import runpy
import sys
import types
from pathlib import Path
import numpy as np
root = Path(sys.argv[2])
config = types.SimpleNamespace(EXP=types.SimpleNamespace(path_save=str(root / "outputs")))
state = types.SimpleNamespace(mask=np.ones((2, 2), dtype=bool))
for name, obj in (("config", config), ("state", state)):
    with (root / (name + ".pkl")).open("wb") as stream:
        pickle.dump(obj, stream)
source = types.ModuleType("src")
def forbidden(**kwargs):
    raise AssertionError("land must not run inversion")
source.inv = types.SimpleNamespace(Inv_4Dvar=forbidden)
sys.modules["src"] = source
namespace = runpy.run_path(sys.argv[1])
namespace["run_tile"](root, restart=False)
assert (root / ".tile_complete.ok").read_text() == "SKIPPED_LAND\n"
assert not (root / "outputs").exists()
'''
    result = subprocess.run(
        [sys.executable, "-c", script,
         str(REPOSITORY / "slurm/src/run_tile.py"), str(tmp_path)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_shared_queue_finishes_work_without_other_tasks(tmp_path):
    result = _run_lock_scenario(tmp_path, _worker_setup(tmp_path) + r'''
NUM_ARRAY=6
NUM_TILES_PER_GPU=4
: > "$BARRIER_DIR/tiles"
for i in $(seq 0 135); do
    tile="$BARRIER_DIR/work$i"
    mkdir "$tile"
    printf '%s\n' "$tile" >> "$BARRIER_DIR/tiles"
    # Resume a partial window: the sole consumer finishes all remaining work.
    if (( i % 6 != 0 && i % 6 != 5 )); then touch "$tile/.tile_complete.ok"; fi
done
_slurm_query() { echo "dispatch must not query Slurm" >&2; exit 99; }
run_single_tile() {
    printf '%s\n' "$1" >> "$BARRIER_DIR/executed"
    touch "$1/.tile_complete.ok"
    release_tile_claim "$1" "$3"
}
process_available_tiles "$BARRIER_DIR/tiles" 0 || exit 1
test "$TILES_PROCESSED_IN_PASS" -eq 45 || exit 2
while read -r tile; do test -f "$tile/.tile_complete.ok" || exit 3; done < "$BARRIER_DIR/tiles"
# A late starter must skip all completed tiles.
ARRAY_ID=5
process_available_tiles "$BARRIER_DIR/tiles" 0 || exit 4
test "$TILES_PROCESSED_IN_PASS" -eq 0 || exit 5
test "$(wc -l < "$BARRIER_DIR/executed")" -eq 45
''')
    assert result.returncode == 0, result.stderr


def test_concurrent_consumers_and_late_starter_run_tile_once(tmp_path):
    result = _run_lock_scenario(tmp_path, _worker_setup(tmp_path) + r'''
NUM_ARRAY=3
touch "$BARRIER_DIR/tile0/.tile_complete.ok" "$BARRIER_DIR/tile1/.tile_complete.ok"
_slurm_query() { echo "dispatch must not query Slurm" >&2; exit 99; }
cat > "$BARRIER_DIR/bin/python" <<'SH'
#!/bin/bash
tile="$2"
printf '%s\n' started >> "$tile/executed"
touch "$tile/started"
sleep 1
touch "$tile/.tile_complete.ok"
SH
chmod +x "$BARRIER_DIR/bin/python"
(ARRAY_ID=0; process_available_tiles "$BARRIER_DIR/tiles" 0) &
first=$!
(ARRAY_ID=1; process_available_tiles "$BARRIER_DIR/tiles" 0) &
second=$!
for i in $(seq 1 100); do
    [ -f "$BARRIER_DIR/tile2/started" ] && break
    sleep 0.05
done
test -f "$BARRIER_DIR/tile2/started" || exit 1
ARRAY_ID=2
process_available_tiles "$BARRIER_DIR/tiles" 0 || exit 2
wait "$first" || exit 3
wait "$second" || exit 4
process_available_tiles "$BARRIER_DIR/tiles" 0 || exit 5
test "$(wc -l < "$BARRIER_DIR/tile2/executed")" -eq 1 || exit 6
test ! -d "$BARRIER_DIR/tile2/.tile_running.lock"
''')
    assert result.returncode == 0, result.stderr


def test_free_slot_takes_other_running_tasks_work_before_slow_worker_finishes(tmp_path):
    result = _run_lock_scenario(tmp_path, _worker_setup(tmp_path) + r'''
NUM_ARRAY=3
# All tasks are RUNNING; tile1 and tile2 formerly belonged to other ranks.
_slurm_query() { printf '%s\n' '404_0 RUNNING' '404_1 RUNNING' '404_2 RUNNING'; }
cat > "$BARRIER_DIR/bin/python" <<'SH'
#!/bin/bash
tile="$2"
root="$(dirname "$tile")"
if [[ "$tile" == *tile0 ]]; then
    touch "$tile/started"
    for i in $(seq 1 200); do
        [ -f "$root/tile2/started" ] && break
        sleep 0.02
    done
    test -f "$root/tile2/started" || exit 9
elif [[ "$tile" == *tile2 ]]; then
    test ! -f "$root/tile0/.tile_complete.ok" || exit 8
    touch "$tile/started"
fi
touch "$tile/.tile_complete.ok"
SH
chmod +x "$BARRIER_DIR/bin/python"
process_available_tiles "$BARRIER_DIR/tiles" 0 || exit 1
test "$TILES_PROCESSED_IN_PASS" -eq 3 || exit 2
for i in 0 1 2; do
    test -f "$BARRIER_DIR/tile$i/.tile_complete.ok" || exit 3
done
''')
    assert result.returncode == 0, result.stderr


def test_full_gpu_does_not_reserve_waiting_work_from_free_peer(tmp_path):
    result = _run_lock_scenario(tmp_path, _worker_setup(tmp_path) + r'''
NUM_ARRAY=2
NUM_TILES_PER_GPU=1
touch "$BARRIER_DIR/tile2/.tile_complete.ok"
cat > "$BARRIER_DIR/bin/python" <<'SH'
#!/bin/bash
tile="$2"
root="$(dirname "$tile")"
printf '%s\n' "$ARRAY_ID" >> "$tile/executed"
if [[ "$tile" == *tile0 ]]; then
    touch "$tile/started"
    for i in $(seq 1 200); do
        [ -f "$root/tile1/.tile_complete.ok" ] && break
        sleep 0.02
    done
    test -f "$root/tile1/.tile_complete.ok" || exit 9
fi
touch "$tile/.tile_complete.ok"
SH
chmod +x "$BARRIER_DIR/bin/python"
export ARRAY_ID
(ARRAY_ID=0; process_available_tiles "$BARRIER_DIR/tiles" 0) &
first=$!
for i in $(seq 1 100); do
    [ -f "$BARRIER_DIR/tile0/started" ] && break
    sleep 0.02
done
test -f "$BARRIER_DIR/tile0/started" || exit 1
ARRAY_ID=1
process_available_tiles "$BARRIER_DIR/tiles" 0 || exit 2
wait "$first" || exit 3
test "$(cat "$BARRIER_DIR/tile0/executed")" = 0 || exit 4
test "$(cat "$BARRIER_DIR/tile1/executed")" = 1 || exit 5
''')
    assert result.returncode == 0, result.stderr


def test_six_gpu_consumers_complete_shared_queue_once_with_capacity_limits(tmp_path):
    result = _run_lock_scenario(tmp_path, _worker_setup(tmp_path) + r'''
NUM_ARRAY=6
NUM_TILES_PER_GPU=4
: > "$BARRIER_DIR/tiles"
for i in $(seq 0 135); do
    mkdir "$BARRIER_DIR/work$i"
    printf '%s\n' "$BARRIER_DIR/work$i" >> "$BARRIER_DIR/tiles"
done
run_single_tile() {
    printf 'START %s\n' "$1" >> "$BARRIER_DIR/events_$ARRAY_ID"
    printf '%s\n' "$ARRAY_ID" >> "$1/executed"
    sleep 0.03
    touch "$1/.tile_complete.ok"
    printf 'END %s\n' "$1" >> "$BARRIER_DIR/events_$ARRAY_ID"
    release_tile_claim "$1" "$3"
}
pids=()
for rank in $(seq 0 5); do
    (ARRAY_ID=$rank; process_available_tiles "$BARRIER_DIR/tiles" 0) &
    pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid" || exit 1; done
while read -r tile; do
    test -f "$tile/.tile_complete.ok" || exit 2
    test "$(wc -l < "$tile/executed")" -eq 1 || exit 3
    test ! -d "$tile/.tile_running.lock" || exit 4
done < "$BARRIER_DIR/tiles"
''')
    assert result.returncode == 0, result.stderr
    for events in tmp_path.glob('events_*'):
        active = 0
        for line in events.read_text().splitlines():
            active += 1 if line.startswith('START ') else -1
            assert 0 <= active <= 4
        assert active == 0


def test_unclaimed_shared_work_has_no_nominal_owner_to_check(tmp_path):
    result = _run_lock_scenario(tmp_path, _worker_setup(tmp_path) + r'''
NUM_ARRAY=6
slurm_task_is_active() { echo "unexpected owner check" >&2; return 1; }
incomplete_owners_are_active "$BARRIER_DIR/tiles" || exit 1
''')
    assert result.returncode == 0, result.stderr
    assert not result.stderr


def test_tile_completed_between_check_and_claim_is_not_relaunched(tmp_path):
    result = _run_lock_scenario(tmp_path, _worker_setup(tmp_path) + r'''
# Simulate another worker finishing and releasing the lock just before mkdir.
eval "$(declare -f try_claim_tile | sed '1s/try_claim_tile/original_claim/')"
try_claim_tile() {
    touch "$1/.tile_complete.ok"
    original_claim "$1"
}
run_single_tile() { touch "$BARRIER_DIR/should_not_run"; }
process_available_tiles "$BARRIER_DIR/tiles" 0 || exit 1
test "$TILES_PROCESSED_IN_PASS" -eq 0 || exit 2
test ! -f "$BARRIER_DIR/should_not_run" || exit 3
for i in 0 1 2; do test ! -d "$BARRIER_DIR/tile$i/.tile_running.lock" || exit 4; done
''')
    assert result.returncode == 0, result.stderr


def test_concurrent_stale_reclaim_cannot_move_new_lease(tmp_path):
    tile = tmp_path / 'tile'
    lock = tile / '.tile_running.lock'
    lock.mkdir(parents=True)
    (lock / 'job_id').write_text('303_5\n')
    (lock / 'token').write_text('303_5_old\n')
    result = _run_lock_scenario(tmp_path, f'''
JOB_ID=404
tile={shlex.quote(str(tile))}
slurm_task_is_active() {{
    touch "$tile/seen_$ARRAY_ID"
    for i in $(seq 1 100); do
        [ -f "$tile/seen_0" ] && [ -f "$tile/seen_1" ] && break
        sleep 0.02
    done
    if [ "$ARRAY_ID" = 1 ]; then
        for i in $(seq 1 100); do
            [ -f "$tile/winner" ] && break
            sleep 0.02
        done
    fi
    return 1
}}
(ARRAY_ID=0; try_claim_tile "$tile" > "$tile/winner_token" && touch "$tile/winner") &
first=$!
(ARRAY_ID=1; if try_claim_tile "$tile"; then exit 9; fi) &
second=$!
wait "$first" || exit 1
wait "$second" || exit 2
test "$(cat "$tile/.tile_running.lock/token")" = "$(cat "$tile/winner_token")"
''')
    assert result.returncode == 0, result.stderr
    assert (tile / '.tile_running.lock.stale-303_5_old' / 'job_id').read_text().strip() == '303_5'


def test_shared_queue_health_checks_actual_claimant(tmp_path):
    result = _run_lock_scenario(tmp_path, _worker_setup(tmp_path) + r'''
NUM_ARRAY=3
touch "$BARRIER_DIR/tile0/.tile_complete.ok" "$BARRIER_DIR/tile1/.tile_complete.ok"
mkdir "$BARRIER_DIR/tile2/.tile_running.lock"
printf '404_0\n' > "$BARRIER_DIR/tile2/.tile_running.lock/job_id"
slurm_task_is_active() { [ "$1" = 404_0 ]; }
incomplete_owners_are_active "$BARRIER_DIR/tiles" || exit 1
slurm_task_is_active() { return 1; }
if incomplete_owners_are_active "$BARRIER_DIR/tiles"; then exit 2; fi
''')
    assert result.returncode == 0, result.stderr
    assert 'owner 404_0' in result.stderr


@pytest.mark.parametrize('restart', [False, True])
def test_initialization_invalidates_products_once(tmp_path, restart):
    window = tmp_path / 'experiment' / 'subwindow_2024-04-10'
    window.mkdir(parents=True)
    (window / '.window_complete_all.ok').touch()
    result = _run_lock_scenario(tmp_path, f'''
die() {{ exit 99; }}
BASE_DIR={shlex.quote(str(window.parent))}
BARRIER_DIR={shlex.quote(str(tmp_path / 'barrier'))}
mkdir "$BARRIER_DIR"
FINAL_MARKER="$BASE_DIR/experiment_complete.ok"
JOB_ID=404
ARRAY_ID=0
TILE_SCOPE=all
RESTART={'--restart' if restart else '""'}
FORCE_MERGE={'false' if restart else 'true'}
BARRIER_TIMEOUT=1
touch "$FINAL_MARKER"
initialize_run_state || exit 1
test ! -f "$FINAL_MARKER" || exit 2
test ! -f "{window}/.window_complete_all.ok" || exit 3
test -f "{window}/.window_rebuild_all.required" || exit 4
# A late task must not invalidate fresh completion.
touch "$FINAL_MARKER" "{window}/.window_complete_all.ok"
ARRAY_ID=1
initialize_run_state || exit 5
test -f "$FINAL_MARKER" || exit 6
test -f "{window}/.window_complete_all.ok" || exit 7
''')
    assert result.returncode == 0, result.stderr
    assert (window / '.window_restart_all.required').exists() == restart


@pytest.mark.parametrize('claimed', [False, True])
def test_idle_gpu_waits_for_healthy_remote_tile_but_bounds_unclaimed_work(tmp_path, claimed):
    launcher = LAUNCHER.read_text()
    loop = re.split(r'^        echo .*Processed', launcher.split(
        '        tiles_done=0\n', 1)[1], maxsplit=1, flags=re.M)[0]
    scenario = _worker_setup(tmp_path) + r'''
BARRIER_TIMEOUT=1
TILE_LIST="$BARRIER_DIR/tiles"
IW=0
touch "$BARRIER_DIR/tile0/.tile_complete.ok" "$BARRIER_DIR/tile1/.tile_complete.ok"
process_available_tiles() { TILES_PROCESSED_IN_PASS=0; }
slurm_task_is_active() { [ "$1" = 404_1 ]; }
ticks=0
sleep() {
    SECONDS=$((SECONDS + 5))
    ticks=$((ticks + 1))
    if (( ticks == 3 )); then touch "$BARRIER_DIR/tile2/.tile_complete.ok"; fi
}
tiles_done=0
'''
    if claimed:
        scenario += '''
mkdir "$BARRIER_DIR/tile2/.tile_running.lock"
printf '404_1\n' > "$BARRIER_DIR/tile2/.tile_running.lock/job_id"
'''
    result = _run_lock_scenario(tmp_path, scenario + loop)
    assert result.returncode == (0 if claimed else 1), result.stderr
    if claimed:
        assert (tmp_path / 'tile2' / '.tile_complete.ok').exists()
    else:
        assert 'timed out without completion progress' in result.stderr


def test_restart_continuation_keeps_completed_tiles_and_rebuilds_every_window(tmp_path):
    main = LAUNCHER.read_text().split('\ninit_merge_args\n', 1)[1]
    setup = _worker_setup(tmp_path) + r'''
die() { echo "$*" >&2; exit 99; }
BASE_DIR="$BARRIER_DIR/experiment"
mkdir -p "$BASE_DIR/subwindow_2024-04-10/subwindow_0_0" \
    "$BASE_DIR/subwindow_2024-04-10/subwindow_1_0" "$BASE_DIR/subwindow_2024-05-20/subwindow_0_0"
FINAL_MARKER="$BASE_DIR/experiment_complete.ok"
for window in "$BASE_DIR"/subwindow_*; do
    touch "$window/.window_complete_all.ok"
    for tile in "$window"/subwindow_*; do touch "$tile/.tile_complete.ok"; done
done
touch "$FINAL_MARKER"
DIR_SAVE_PICKLE="$BASE_DIR"
CONFIG_PATH="$BASE_DIR/config.pkl"
NAME_VAR=sla
NUM_ARRAY=1
NUM_TILES_PER_GPU=1
NUM_MERGE_WORKERS=1
ZARR_TIME_CHUNK=4
ZARR_SPATIAL_CHUNK=128
ZARR_COMPRESSION_LEVEL=3
ZARR_OUTPUT=true
OUTPUT_FLOAT64=false
CLEANUP_TILE_ZARR=false
FORCE_ARGS=()
RESTART=--restart
FORCE_MERGE=false
MERGE_ONLY=false
TILE_SCOPE=all
BARRIER_TIMEOUT=1
BARRIER_DIR="$BASE_DIR/barrier-first"
mkdir "$BARRIER_DIR"
run_single_tile() {
    if [ "$PHASE" = first ] && [[ "$1" == *subwindow_1_0 ]]; then
        release_tile_claim "$1" "$3"
        return 7
    fi
    printf '%s %s\n' "$1" "$WINDOW_RESTART" >> "$BASE_DIR/executed"
    touch "$1/.tile_complete.ok"
    release_tile_claim "$1" "$3"
}
run_stage() {
    printf '%s\n' "$*" >> "$BASE_DIR/merges"
}
PHASE=first
initialize_run_state || exit 1
init_merge_args
'''
    resume = r'''
test "$?" -ne 0 || exit 2
test -f "$BASE_DIR/subwindow_2024-04-10/subwindow_0_0/.tile_complete.ok" || exit 3
test ! -f "$BASE_DIR/subwindow_2024-05-20/.window_complete_all.ok" || exit 4
PHASE=second
JOB_ID=505
RESTART=""
BARRIER_DIR="$BASE_DIR/barrier-second"
mkdir "$BARRIER_DIR"
initialize_run_state || exit 5
init_merge_args
'''
    result = _run_lock_scenario(tmp_path, setup + '\n(\n' + main + '\n)\n' + resume + main)
    assert result.returncode == 0, result.stderr
    executed = (tmp_path / 'experiment' / 'executed').read_text().splitlines()
    assert len(executed) == 3
    assert len(set(executed)) == 3
    assert all(line.endswith(' --restart') for line in executed)
    calls = (tmp_path / 'experiment' / 'merges').read_text().splitlines()
    spatial = [line for line in calls if '--iw_start' in line]
    assert len(spatial) == 4  # One rank and one finalization per window.
    assert all('--force' in line for line in spatial)
    for window in (tmp_path / 'experiment').glob('subwindow_*'):
        assert (window / '.window_complete_all.ok').exists()
        assert not list(window.glob('.window_*.required'))
    assert (tmp_path / 'experiment' / 'experiment_complete.ok').exists()


@pytest.mark.parametrize('zarr', [False, True])
@pytest.mark.parametrize('already_done', [False, True])
def test_merge_flow_preserves_options_and_publication_order(tmp_path, zarr, already_done):
    base = tmp_path / 'experiment with spaces'
    window = base / 'subwindow_2023-01-26'
    (window / 'subwindow_0_0').mkdir(parents=True)
    if already_done:
        (window / '.window_complete_all.ok').touch()
    main = LAUNCHER.read_text().split('\ninit_merge_args\n', 1)[1]
    setup = f'''
die() {{ echo "$*" >&2; exit 1; }}
BASE_DIR={shlex.quote(str(base))}
BARRIER_DIR={shlex.quote(str(tmp_path))}
DIR_SAVE_PICKLE="$BASE_DIR"
CONFIG_PATH="$BASE_DIR/config.pkl"
SRC_DIR="$BASE_DIR/source code"
FINAL_MARKER="$BASE_DIR/experiment_complete.ok"
NAME_VAR=sla,ug
JOB_ID=404
ARRAY_ID=0
NUM_ARRAY=3
NUM_MERGE_WORKERS=4
ZARR_TIME_CHUNK=10
ZARR_SPATIAL_CHUNK=128
ZARR_COMPRESSION_LEVEL=3
ZARR_OUTPUT={str(zarr).lower()}
OUTPUT_FLOAT64=true
CLEANUP_TILE_ZARR=true
FORCE_ARGS=()
RESTART=""
FORCE_MERGE=false
MERGE_ONLY=true
TILE_SCOPE=all
BARRIER_TIMEOUT=1
run_stage() {{
    printf '%s\\n' CALL "$@" >> "$BARRIER_DIR/calls"
    if [[ " $* " == *" --cleanup_subwindow_outputs "* ]]; then
        test -f "$FINAL_MARKER" || exit 99
    fi
}}
init_merge_args
'''
    result = _run_lock_scenario(tmp_path, setup + main)
    assert result.returncode == 0, result.stderr
    calls = (tmp_path / 'calls').read_text().split('CALL\n')[1:]
    args = [call.splitlines() for call in calls]
    assert len(args) == (2 if already_done else (6 if zarr else 3))
    for call in args:
        assert str(base / 'source code' / 'merge_outputs.py') in call
        assert call[call.index('--dir_save_pickle') + 1] == str(base)
        assert '--output_float64' in call
        assert ('--zarr_output' in call) == zarr
        assert call[call.index('--zarr_time_chunk') + 1] == '10'
    assert '--merge_time_windows' in args[-2]
    assert '--cleanup_tile_zarr' in args[-2]
    assert '--cleanup_subwindow_outputs' in args[-1]
    assert '--cleanup_tile_zarr' not in args[-1]
    assert (base / 'experiment_complete.ok').is_file()
    assert (tmp_path / 'final_merge.ok').is_file()
    assert (window / '.window_complete_all.ok').is_file()


@pytest.mark.parametrize('failure', ['merge', 'publication'])
def test_failed_merge_stage_never_publishes_success(tmp_path, failure):
    result = _run_lock_scenario(tmp_path, f'''
die() {{ exit 1; }}
JOB_ID=404
ARRAY_ID=0
merge_outputs() {{ return {1 if failure == 'merge' else 0}; }}
{'publish_marker() { return 1; }' if failure == 'publication' else ''}
finish_merge_stage "{tmp_path}/ok" "{tmp_path}/failed" "{tmp_path}/durable"
''')
    assert result.returncode == 1
    assert (tmp_path / 'failed').exists()
    assert not (tmp_path / 'ok').exists()
    assert not (tmp_path / 'durable').exists()


def test_continuation_retries_failed_submission_and_preserves_background(tmp_path):
    result = _run_lock_scenario(tmp_path, f'''
BASE_DIR={shlex.quote(str(tmp_path))}
JOB_ID=404
SLURM_JOB_ID=404
FINAL_MARKER="$BASE_DIR/final"
CONTINUATION_SUBMITTED=false
CONFIG_FILE="config with spaces.sh"
CONTINUATION_SCRIPT=launcher.sh
RESTART=""
FORCE_MERGE=false
MERGE_ONLY=false
TILE_SCOPE=all
NAME_EXP_OVERRIDE="experiment name"
NAME_EXP_BACKGROUND_OVERRIDE="background name"
sbatch() {{
    if [ ! -f "$BASE_DIR/retry" ]; then touch "$BASE_DIR/retry"; return 1; fi
    printf '%s\\n' "$@" > "$BASE_DIR/submission"
    echo 505
}}
scontrol() {{ echo 'Dependency=afterany:404'; }}
submit_continuation
[ "$CONTINUATION_SUBMITTED" = false ] || exit 1
submit_continuation
[ "$CONTINUATION_SUBMITTED" = true ] || exit 2
''')
    assert result.returncode == 0, result.stderr
    args = (tmp_path / 'submission').read_text().splitlines()
    assert args[args.index('--name_exp_background') + 1] == 'background name'
    assert args[args.index('--name_exp') + 1] == 'experiment name'
    assert args[args.index('--config') + 1] == 'config with spaces.sh'
    assert '--dependency=afterany:404' in args


@pytest.mark.parametrize('args', [
    ['--config'], ['--config='], ['--typo'], ['--restart=yes'],
    ['--config', '--skip-prepare'], ['--name_exp'],
])
def test_cli_rejects_bad_options_before_environment_setup(args):
    result = subprocess.run(['bash', str(LAUNCHER), *args], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'ERROR:' in result.stderr
    assert 'conda' not in result.stderr


def test_cli_and_prepare_arguments_preserve_spaces_and_quotes(tmp_path):
    config = tmp_path / 'experiment settings.sh'
    pyconfig = tmp_path / "user's config.py"
    pyconfig.write_text("name_experiment = 'sample experiment'\n")
    config.write_text('\n'.join([
        f'MASH_DIR={shlex.quote(str(tmp_path))}',
        f'DIR_SAVE_PICKLE={shlex.quote(str(tmp_path / "pickle files"))}',
        f'PATH_CONFIG={shlex.quote(pyconfig.name)}',
        f'PATH_CONFIG_EQ={shlex.quote(pyconfig.name)}',
        'INIT_DATE=2023-01-01', 'FINAL_DATE=2023-02-01',
    ]))
    prefix = LAUNCHER.read_text().split('# -------------------- ENVIRONMENT')[0]
    script = tmp_path / 'prefix.sh'
    script.write_text(prefix + '\nprintf "%s\\n" "$EXP_NAME" "${PREPARE_ARGS[@]}"\n')
    result = subprocess.run(
        ['bash', str(script), '--config', str(config), '--name_exp_background=background with spaces'],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    assert args[0] == 'sample experiment'
    assert args[args.index('--dir_save_pickle') + 1] == str(tmp_path / 'pickle files')
    assert args[args.index('--name_exp_background') + 1] == 'background with spaces'
    assert '--flag_background' in args
