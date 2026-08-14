#!/bin/bash
# Reproducible experiment configuration for slurm/run/VarDyn_GLO.sh.
# Paths below may be relative to this file.

MASH_DIR="/home/il/leguilf/VarDyn"
DIR_SAVE_PICKLE="/work/scratch/data/leguilf/Studies/VarDyn_SWOT-v3.0/pickles"

PATH_CONFIG="../configs/config_VarDyn-QG_GLO_nadirs-minus-al.py"
PATH_CONFIG_EQ="../configs/config_VarDyn-QG_GLO_nadirs-minus-al_eq.py"
INIT_DATE="2023-12-15"
FINAL_DATE="2025-01-15"

NAME_VAR="sla,SSH_tot,ug,vg"
ZARR_OUTPUT=false       # Store merged outputs as Zarr and remove subwindow NetCDF files
OUTPUT_FLOAT64=false     # Save floating-point outputs as float64 (default: float32)

# Spatial subwindow grid
GRID_TYPE="GRID_CAR"
GRID_TYPE_EQ="GRID_GEO"
NX_PROC=512
NY_PROC=256
NX_PROC_EQ=512
NY_PROC_EQ=256
DX=10
DY=10

# Spatial subwindow size and overlap (degrees)
SPACE_WIN_X=50
SPACE_WIN_Y=25
SPACE_WIN_X_EQ=50
SPACE_WIN_Y_EQ=25
SPACE_OVERLAP_X=2.5
SPACE_OVERLAP_Y=2.5

# Temporal subwindow size and overlap (days)
TIME_WIN=50
TIME_OVERLAP=10

# Orchestration settings
# Output storage
# ZARR_OUTPUT=true converts subwindow NetCDF files to Zarr before final merge and deletes .nc files.
# OUTPUT_FLOAT64=true stores merged floating-point variables as float64 (default: float32).

NUM_MERGE_WORKERS=4
NUM_TILES_PER_GPU=4
BARRIER_TIMEOUT=7200
FLAG_INIT_FROM_PREVIOUS="--flag_init_from_previous"
FLAG_INIT=false
FLAG_BACKGROUND=false
NAME_EXP=""
