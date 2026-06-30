#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration file for running the CSW internal-tide example on the JET BM-IT
OSSE data.

"""

name_experiment = 'VarDyn-CSWM-JET' # Experiment label reused in output filenames and directories.

myPath = '.' # Root directory for generated outputs, scratch data, cached observations, and operators.

# JET reference and vertical-mode files are available from:
# https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/catalog/meomopendap/extract/MEOM/JET_BM-IT/catalog.html
path_modes = '/data1/data/models/JET_BM-IT/full_resolution/vmodes_jet_cfg1_wp6_2d_coarse2.nc' # Vertical-mode, grid, MDT, and phase-speed file.
path_ref = '/data1/data/models/JET_BM-IT/full_resolution/wp6_surf_data_ref_coarse2.nc' # Reference surface fields used for BM coupling, observations, and diagnostics.

compute_obs = True # Recompute cached observations and observation operators when True.

#################################################################################################################################
# Global libraries
#################################################################################################################################

from datetime import datetime, timedelta
from math import pi

#################################################################################################################################
# EXPERIMENTAL PARAMETERS
#################################################################################################################################

EXP = dict(

    name_experiment = name_experiment, # Experiment label stored in the merged configuration.

    name_exp_save = name_experiment, # Prefix used when writing output NetCDF files.

    path_save = f'{myPath}/outputs/{name_experiment}', # Directory where model/analysis outputs are written.

    tmp_DA_path = f'{myPath}/scratch/{name_experiment}', # Scratch directory for temporary assimilation files and copied config.

    init_date = datetime(2010, 5, 1, 0), # First analysis time.

    final_date = datetime(2010, 7, 1, 0), # Last analysis time.

    assimilation_time_step = timedelta(hours=1), # Time spacing between analysis/control states.

    saveoutput_time_step = timedelta(hours=3), # Time spacing between saved model states.

    flag_plot = 1, # Plotting verbosity used during setup and execution.

    write_obs = True, # Save preprocessed observations for reuse.

    path_obs = f'{myPath}/obs', # Directory for cached observation dictionaries.

    compute_obs = compute_obs, # Forward the global observation/operator recomputation flag.

)

#################################################################################################################################
# GRID parameters
#################################################################################################################################

NAME_GRID = 'myGRID' # Name of the grid block selected for this experiment.

myGRID = dict(

    super = 'GRID_FROM_FILE', # Build the computational grid from an existing NetCDF file.

    path_init_grid = path_modes, # NetCDF file containing the longitude/latitude grid.

    name_init_lon = 'lon', # Longitude variable name in path_init_grid.

    name_init_lat = 'lat', # Latitude variable name in path_init_grid.

)

#################################################################################################################################
# Model parameters
#################################################################################################################################

NAME_MOD = 'CSWM' # Name of the dynamical-model block selected for this experiment.

CSWM = dict(

    super = 'MOD_CSW1L', # Use the current one-layer coupled shallow-water internal-tide model.

    name_var = {'U': 'u_it', 'V': 'v_it', 'SSH': 'ssh_it'}, # Map model roles to state/output variable names.

    name_params = ['He_mean', 'hbc', 'alpha_He', 'alpha_Uu', 'alpha_Uz', 'alpha_Up'], # Controlled IT parameters: equivalent depth, boundary waves, and BM coupling terms.

    g = 9.81, # Gravity acceleration in m s-2.

    dtmodel = 300, # Nominal model timestep in seconds before CFL adjustment.

    cfl = .3, # CFL factor used to compute a stable timestep from grid spacing and wave speed.

    time_scheme = 'rk4', # Time integration scheme used by the CSW core.

    path_mdt = path_modes, # File containing mean dynamic topography used in coupling terms.

    name_var_mdt = {'lon': 'lon', 'lat': 'lat', 'var': 'mdt'}, # Coordinate and variable names for MDT.

    filec_aux = path_modes, # File containing first baroclinic phase speed.

    name_var_c = {'lon': 'lon', 'lat': 'lat', 'var': 'c1'}, # Coordinate and variable names for phase speed.

    H = 4000, # Constant bathymetry depth in meters when file_H_aux is not provided.

    w_waves = [2 * pi / 12. / 3600], # Internal-tide angular frequencies in rad s-1.

    Ntheta = 1, # Number of positive incoming-wave angle samples for boundary waves.

    flag_coupling_from_bm = True, # Enable Balanced Motion corrections to IT equivalent depth/advective terms.

    path_vertical_modes = path_modes, # File used to compute BM/IT interaction terms when path_interaction_terms is absent.

    path_interaction_terms = None, # Optional precomputed interaction-term file; None computes from vertical modes.

    path_bm = path_ref, # Balanced Motion reference file used by the coupling.

    name_var_bm = {'time': 'time', 'lon': 'lon', 'lat': 'lat', 'ssh_bm': 'ssh_bm'}, # Coordinate and SSH names for BM input.

    periodic_x = True, # Apply periodic boundary conditions in the zonal direction.

    periodic_y = False, # Do not apply periodic boundary conditions in the meridional direction.

    flag_bc_sponge = True, # Enable sponge relaxation near boundaries when dist_sponge_bc is set.

    dist_sponge_bc = 400, # Sponge width in kilometers.

    sponge_coef = 0.05, # Sponge damping coefficient.

    mask_sponge_bc = True, # Mask observations inside sponge layers.

    bc_it_method = 'plane_wave_bdy', # Boundary wave phase method evaluated from boundary-edge medium.

    extend_it_open_boundary_sponge = True, # Extend entering-wave medium from the sponge interior edge across open-boundary sponge bands.

)

#################################################################################################################################
# Observation parameters
#################################################################################################################################

NAME_OBS = ['NR'] # Observation blocks used by the experiment.

NR = dict(

    super = 'OBS_L4', # Use gridded L4-style observations.

    path = path_ref, # Observation NetCDF file.

    name_time = 'time', # Time coordinate name in the observation file.

    name_lon = 'lon', # Longitude coordinate name in the observation file.

    name_lat = 'lat', # Latitude coordinate name in the observation file.

    name_var = {'SSH': 'ssh_it'}, # Map observed SSH role to source variable name.

    sigma_noise = .02, # Constant observation-error standard deviation in meters.

    subsampling = 81, # Observation temporal subsampling factor in model/assimilation steps. --> 3h x 81 = 10 days between observations

)

#################################################################################################################################
# OBSERVATIONAL OPERATORS
#################################################################################################################################

NAME_OBSOP = 'OBSOP_NR' # Observation-operator block selected for this experiment.

OBSOP_NR = dict(

    super = 'OBSOP_INTERP_L4', # Interpolate gridded observations onto model state/time.

    name_var = 'SSH', # Model/observation role handled by this operator.

    name_obs = ['NR'], # Restrict this operator to the NR observation block.

    write_op = True, # Cache the interpolation operator on disk.

    path_save = f'{myPath}/H', # Directory where cached operators are written/read.

    compute_op = compute_obs, # Recompute the operator when compute_obs is True.

    interp_method = 'linear', # Spatial interpolation method for gridded observations.

)

#################################################################################################################################
# Reduced basis parameters
#################################################################################################################################

NAME_BASIS = ['Basis_He', 'Basis_alpha_He', 'Basis_alpha_adv', 'Basis_alpha_Uz', 'Basis_hbc'] # Reduced-basis blocks composing the control vector.

Basis_He = dict(

    super = 'BASIS_GAUSS3D', # Use Gaussian basis functions in space and time.

    name_mod_var = 'He_mean', # Controlled equivalent-depth anomaly parameter.

    flux = False, # Treat this as a parameter field rather than a transient source term.

    facns = 2., # Spacing factor between neighboring spatial basis centers.

    facnlt = 2., # Spacing factor between neighboring temporal basis centers.

    sigma_D = 1000, # Horizontal Gaussian scale in kilometers.

    sigma_T = 50, # Temporal Gaussian scale in days.

    sigma_Q = 1e-2, # Prior standard deviation for equivalent-depth coefficients.

    normalize_fact = False, # Keep the legacy normalization behavior for this basis.

)

Basis_alpha_He = dict(

    super = 'BASIS_OFFSET', # Use one domain-wide offset basis vector.

    name_mod_var = 'alpha_He', # Controlled BM-to-IT equivalent-depth coupling coefficient.

    sigma_B = .1, # Prior standard deviation for the alpha_He offset.

)

Basis_alpha_adv = dict(

    super = 'BASIS_OFFSET', # Use one shared domain-wide offset basis vector.

    name_mod_var = ['alpha_Uu', 'alpha_Up'], # Link advective coupling coefficients to the same control.

    sigma_B = .1, # Prior standard deviation for the shared advective-alpha offset.

)

Basis_alpha_Uz = dict(

    super = 'BASIS_OFFSET', # Use one domain-wide offset basis vector.

    name_mod_var = 'alpha_Uz', # Controlled vertical-shear coupling coefficient.

    sigma_B = .1, # Prior standard deviation for the alpha_Uz offset.

)

Basis_hbc = dict(

    super = 'BASIS_HBC', # Use the current internal-tide boundary-condition basis.

    name_params = ['hbcx', 'hbcy'], # Control south/north and west/east boundary-wave amplitudes.

    sigma_B_bc = 1e-2, # Prior standard deviation for boundary-condition coefficients.

    time_dependant = False, # Use constant-in-time boundary controls, matching the old CST basis.

    D_bc = 1000, # Along-boundary Gaussian scale in kilometers.

    T_bc = 50, # Temporal scale in days; ignored when time_dependant is False.

    Nwaves = 1, # Number of tidal frequencies represented in the boundary basis.

    Ntheta = 1, # Number of positive incoming-wave angle samples in the boundary basis.

)

#################################################################################################################################
# Analysis parameters
#################################################################################################################################

NAME_INV = 'myINV' # Inversion block selected for this experiment.

myINV = dict(

    super = 'INV_4DVAR', # Use the 4DVar minimization driver.

    save_minimization = True, # Save cost-function and gradient history during minimization.

    compute_test = False, # Disable tangent-linear/adjoint/gradient tests by default.

    ftol = 5e-5, # Relative cost-function decrease threshold.

    maxiter = 50, # Maximum number of minimizer iterations.

    freq_it_plot = 10, # Iteration interval for minimization diagnostic plots.

    opt_method = 'L-BFGS-B', # SciPy optimizer used by the 4DVar driver.

    path_save_control_vectors = f'{myPath}/controls/{name_experiment}', # Directory for saved control vectors/restarts.

    timestep_checkpoint = timedelta(hours=24), # Time spacing between checkpointed analysis states.

    prec = True, # Enable control-vector preconditioning.

    path_init_4Dvar = None, # Optional path to an initial control vector.

    restart_4Dvar = False, # Do not restart from the latest saved control vector by default.

)

#################################################################################################################################
# Diagnostics
#################################################################################################################################

NAME_DIAG = 'DIAG_IT' # Diagnostic block selected for this experiment.

DIAG_IT = dict(

    super = 'DIAG_OSSE', # Compare outputs against a known reference simulation.

    dir_output = f'{myPath}/diags/{name_experiment}', # Directory for diagnostic figures and products.

    name_ref = path_ref, # Reference NetCDF file or glob used for OSSE comparison.

    name_ref_time = 'time', # Time coordinate name in the reference file.

    name_ref_lon = 'lon', # Longitude coordinate name in the reference file.

    name_ref_lat = 'lat', # Latitude coordinate name in the reference file.

    name_ref_var = 'ssh_it', # Reference variable compared against the experiment.

    lenght_scale = 1000, # Diagnostic spatial decorrelation/filtering scale in kilometers.

    name_exp_var = 'ssh_it', # Experiment output variable compared with the reference.

    path_images2mp4 = '/data1/packages/climporn/ffmpeg/images2mp4.sh', # Optional helper script for movie creation.

    compare_to_baseline = False, # Disable comparison against a separate baseline experiment.

    name_bas = None, # Baseline file or glob when compare_to_baseline is True.

    name_bas_time = 'time', # Time coordinate name in the baseline file.

    name_bas_lon = 'lon', # Longitude coordinate name in the baseline file.

    name_bas_lat = 'lat', # Latitude coordinate name in the baseline file.

    name_bas_var = 'ssh_it', # Baseline variable name when baseline comparison is enabled.

)
