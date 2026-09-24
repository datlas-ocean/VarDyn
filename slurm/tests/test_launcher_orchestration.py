import os
from pathlib import Path
import subprocess


REPOSITORY = Path(__file__).resolve().parents[2]
LAUNCHER = REPOSITORY / "slurm" / "run" / "VarDyn_GLO.sh"


def _tile_lock_functions() -> str:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    start = launcher.index("slurm_task_is_active() {")
    end = launcher.index("window_tile_state() {", start)
    return launcher[start:end]


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


def test_deterministic_tile_shards_are_disjoint_and_complete(tmp_path):
    result = _run_lock_scenario(
        tmp_path,
        """
NUM_ARRAY=6
for tile_index in $(seq 0 135); do
    owners=0
    for ARRAY_ID in $(seq 0 $((NUM_ARRAY - 1))); do
        if tile_is_owned_by_task "$tile_index"; then
            owners=$((owners + 1))
        fi
    done
    test "$owners" -eq 1 || exit 1
done
""",
    )
    assert result.returncode == 0, result.stderr


def test_launcher_has_resume_and_predecessor_guards():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert '--predecessor-job "${dependency}"' in launcher
    assert '[ -f "${tile}/.tile_complete.ok" ] && continue' in launcher
    assert 'tile_is_owned_by_task "$tile_index"' in launcher
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
squeue() {{
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
