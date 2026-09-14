import os
from pathlib import Path
import subprocess


REPOSITORY = Path(__file__).resolve().parents[2]
LAUNCHER = REPOSITORY / "slurm" / "run" / "VarDyn_GLO.sh"


def _tile_lock_functions() -> str:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    start = launcher.index("try_claim_tile() {")
    end = launcher.index(
        "# Wait on completed work rather than SLURM_ARRAY_TASK_COUNT", start
    )
    return launcher[start:end]


def _run_lock_scenario(tmp_path: Path, scenario: str) -> subprocess.CompletedProcess:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_squeue = bin_dir / "squeue"
    fake_squeue.write_text(
        "#!/bin/bash\n"
        "case \" $* \" in\n"
        "  *\" -j 101 \"*) printf '101_0\\n' ;;\n"
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


def test_launcher_waits_instead_of_submitting_on_barrier_diagnostic():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    wait_function = launcher[
        launcher.index("wait_for_window_tiles() {") : launcher.index(
            "wait_for_spatial_merge_parts()", launcher.index("wait_for_window_tiles() {")
        )
    ]
    assert "continuing to wait" in wait_function
    assert "submit_continuation" not in wait_function
    assert 'return 2' not in wait_function


def test_launcher_has_resume_and_predecessor_guards():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert '--predecessor-job "${dependency}"' in launcher
    assert '[ -f "${TILE}/.tile_complete.ok" ] && continue' in launcher
    assert '.window_complete_${TILE_SCOPE}.ok' in launcher
