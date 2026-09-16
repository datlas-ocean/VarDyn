#!/usr/bin/env python3
import argparse
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import os
from pathlib import Path
from datetime import datetime
import pickle

import traceback
import warnings
from pathlib import Path

# MASSH imports — override path with the MASSH_PATH environment variable
# Default: resolve relative to this script's location (slurm/ → mapping/)
_MASSH_PATH = os.environ.get('MASSH_PATH', str(Path(__file__).parent.parent / 'mapping'))
sys.path.append(_MASSH_PATH)
from src import inv

TILE_SUCCESS_MARKER = "outputs_saved.ok"
ORCHESTRATION_SUCCESS_MARKER = ".tile_complete.ok"


def marker_path(config) -> Path:
    """File written when a tile completed through output saving."""
    return Path(config.EXP.path_save) / TILE_SUCCESS_MARKER

def write_orchestration_marker(path: Path, message: str):
    """Atomically publish completion to the Slurm coordinator."""
    temporary = path.with_name(f'{path.name}.tmp-{os.getpid()}')
    temporary.write_text(message, encoding='utf-8')
    temporary.replace(path)



def run_tile(tile_dir: Path, restart:bool):
    """
    Run one data assimilation tile located in:
    subwindow_<time>/subwindow_<space>
    """

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "CPU-only")

    print(f"[{datetime.now()}] Starting tile")
    print(f"Using GPU(s): {gpu}")
    print(f"Tile directory: {tile_dir}")

    # --------------------------------------------------
    # Expected input files
    # --------------------------------------------------
    config_path = tile_dir / "config.pkl"
    state_path  = tile_dir / "state.pkl"

    if not config_path.exists():
        raise RuntimeError(f"Missing config file: {config_path}")
    if not state_path.exists():
        raise RuntimeError(f"Missing state file: {state_path}")

    print(f"Using config: {config_path.name}")
    print(f"Using state : {state_path.name}")

    # --------------------------------------------------
    # Load inputs
    # --------------------------------------------------
    with open(config_path, "rb") as f:
        config = pickle.load(f)

    with open(state_path, "rb") as f:
        State = pickle.load(f)

    orchestration_marker = tile_dir / ORCHESTRATION_SUCCESS_MARKER
    if restart and orchestration_marker.exists():
        orchestration_marker.unlink()

    # Skip tiles with no ocean points: nothing to assimilate and the SW model
    # core will hard-fail on an all-zero mask.
    if getattr(State, 'mask', None) is not None and State.mask.all():
        print(f"[SKIP] Tile is all land (no ocean points), skipping: {tile_dir}")
        write_orchestration_marker(orchestration_marker, "all-land tile\n")
        return

    print(f"Running inversion, output path: {config.EXP.path_save}")

    success_marker = marker_path(config)

    # --------------------------------------------------
    # Run algorithm
    # --------------------------------------------------
    if restart and success_marker.exists():
        success_marker.unlink()

    if restart or not success_marker.exists():
        inv.Inv_4Dvar(config=config, State=State, verbose=0)

        success_marker.parent.mkdir(parents=True, exist_ok=True)
        tmp_marker = success_marker.with_suffix(success_marker.suffix + ".tmp")
        tmp_marker.write_text(
            f"Tile completed successfully through output saving.\n"
            f"timestamp: {datetime.now().isoformat()}\n"
            f"tile_dir: {tile_dir}\n",
            encoding="utf-8",
        )
        tmp_marker.replace(success_marker)
        write_orchestration_marker(
            orchestration_marker, f"completed: {datetime.now().isoformat()}\n")

        print(f"[{datetime.now()}] Finished tile: {tile_dir}")
    else:
        write_orchestration_marker(
            orchestration_marker,
            f"already completed: {datetime.now().isoformat()}\n")
        print(f"[{datetime.now()}] Non-processed tile: {tile_dir}")
        print(f"Because you did not ask for restart and {success_marker} exists.")


def main():

    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser(
        description="Run one data assimilation spatial tile"
    )
    parser.add_argument(
        "tile_path",
        type=Path,
        help="Path to subwindow_<time>/subwindow_<space>"
    )

    parser.add_argument(
        "--restart",
        action="store_true",
        help="Restart assimilation if set"
    )

    args = parser.parse_args()
    tile_dir = args.tile_path.resolve()
    RESTART = args.restart

    if not tile_dir.exists():
        print(f"ERROR: tile directory does not exist: {tile_dir}", file=sys.stderr)
        sys.exit(1)

    if not tile_dir.is_dir():
        print(f"ERROR: tile_path is not a directory: {tile_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        run_tile(tile_dir, restart=RESTART)
    except Exception as e:
        print(f"[ERROR] Tile failed: {tile_dir}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
