#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Florian Le Guillou on June 2026.

Configuration file for running VarDyn-QG on the Agulhas region for the WHIRLS project. 
This configuration file is used to set up the parameters for the experiment, including paths, model settings, observational operators, reduced basis parameters, analysis parameters, observation parameters, and diagnostics.
This configuration file is only a template and should be modified according to the specific needs of the experiment. In particular, the paths to the data and output directories should be updated to reflect the user's environment
"""

name_experiment = 'VarDyn-QG' # Short experiment name reused in output paths and files.

myPath = '.' # Root directory for relative outputs, scratch files, and cached operators.

compute_obs = True # Set True to rebuild cached observations and operators.

path_mdt = '/data1/data/obs/level4/MDT/mdt_hybrid_cnes_cls22_cmems2020_global.nc' # Mean dynamic topography file used by the grid mask and model.

name_var_mdt = {'lon':'longitude','lat':'latitude','var':'mdt'} # Coordinate and variable names used to read MDT.

#################################################################################################################################
# Global libraries     
#################################################################################################################################

from datetime import datetime,timedelta
 
#################################################################################################################################
# EXPERIMENTAL PARAMETERS
#################################################################################################################################
EXP = dict(

    name_experiment = name_experiment, # Experiment label used in logs and saved metadata.

    name_exp_save = name_experiment, # Prefix used for output NetCDF files.

    path_save = f'{myPath}/outputs/{name_experiment}', # Directory where analysis and diagnostic outputs are written.

    tmp_DA_path = f"{myPath}/scratch/{name_experiment}", # Scratch directory for cached observations, operators, and restart files.

    init_date = datetime(2025,6,1,0), # First analysis time, as datetime(year, month, day, hour).

    final_date = datetime(2025,7,1,0), # Last analysis time, as datetime(year, month, day, hour).

    assimilation_time_step = timedelta(hours=6), # Time spacing between two assimilated model states.

    saveoutput_time_step = timedelta(hours=6), # Time spacing between saved output states.

    flag_plot = 1, # Plotting level for quick-look/debug figures.

    write_obs = True, # Save preprocessed observations for reuse.

    path_obs = f'{myPath}/obs', # Directory used to save/read cached preprocessed observations.

)
    
#################################################################################################################################
# GRID parameters
#################################################################################################################################
NAME_GRID = 'myGRID'

myGRID = dict(

    super = 'GRID_GEO', # Inherit defaults for a regular longitude/latitude grid.

    lon_min = 0, # Western boundary of the model domain.

    lon_max = 50., # Eastern boundary of the model domain.

    lat_min = -45., # Southern boundary of the model domain.

    lat_max = -20., # Northern boundary of the model domain.

    dlon = 0.1, # Longitude grid spacing, in degrees.

    dlat = 0.1, # Latitude grid spacing, in degrees.

    name_init_mask = path_mdt, # Here the mask is set using the MDT file. The mask is set to 1 where the MDT is NaN (i.e. Land) and 0 where it is not NaN (i.e. Ocean).

    name_var_mask = name_var_mdt # Coordinate and variable names used to read the land/sea mask.

)

#################################################################################################################################
# Model parameters
#################################################################################################################################
NAME_MOD = 'QG'

QG = dict(

    super = 'MOD_QG1L', # Inherit defaults for the 1.5-layer quasi-geostrophic model.

    formulation = 'sf', # Use streamfunction as the prognostic formulation.

    advect_tracer = True, # Also transport passive tracers listed in name_var.

    name_var = {'SSH':'sla', 'SST':'sst'}, # Map VarDyn variable roles to model variable names.

    save_diagnosed_variables = True, # Save diagnosed variables in the model output

    cfl = .3, # Choose the model time step from a CFL constraint instead of fixed dtmodel.

    time_scheme = 'rk2', # Time-integration scheme used by the QG model.

    init_from_bc = True, # Initialize the model state from the external boundary-condition fields.

    filec_aux = '../../aux_files/aux_first_baroclinic_speed.nc', # First baroclinic phase-speed field.

    name_var_c = {'lon':'lon','lat':'lat','var':'c1'}, # Coordinate and variable names inside filec_aux.

    cmin = 2., # Lower bound applied to the phase-speed field.

    Kdiffus = 150, # Diffusion coefficient applied by the model.

    path_mdt = path_mdt, # Mean dynamic topography file used by the QG model.

    name_var_mdt = name_var_mdt, # Coordinate and variable names for MDT.
    
    bc_files = {
                "SSH": {
                    "file": "/data1/data/obs/level4/DUACS_NRT/nrt_global_allsat_phy_l4_202506*.nc", # Boundary-condition file path or glob pattern.
                    "name_var": "sla", # Variable name in the boundary-condition files.
                    "name_lon": "longitude", # Longitude coordinate name in the boundary-condition files.
                    "name_lat": "latitude", # Latitude coordinate name in the boundary-condition files.
                    "name_time": "time", # Time coordinate name in the boundary-condition files.
                    "c_grid": False, # Whether these boundary fields are on a C-grid.
                    },
                "SST": {
                    "file": "/data1/flo/ODYSSEA/SST_GLO_PHY_L4_NRT_010_043/cmems_obs-sst_glo_phy_nrt_l4_P1D-m_202303/2025/06/*.nc", # Boundary-condition file path or glob pattern.
                    "name_var": "analysed_sst", # Variable name in the boundary-condition files.
                    "name_lon": "lon", # Longitude coordinate name in the boundary-condition files.
                    "name_lat": "lat", # Latitude coordinate name in the boundary-condition files.
                    "name_time": "time", # Time coordinate name in the boundary-condition files.
                    "c_grid": False, # Whether these boundary fields are on a C-grid.
                },
}
    
)

#################################################################################################################################
# Observation parameters
#################################################################################################################################
NAME_OBS = ['SST_MW','AL','C2N','H2B','S3A','S3B','S6A','SWON']

sigma_nadir = 0.05 # Optional fallback nadir SSH error standard deviation, in meters.

path_err = '' # Directory containing per-satellite SSH error files.

name_var_err = {'lon':'lon', 'lat':'lat', 'var':'noise'} # Coordinate and variable names for the measurement-error field.

SST_MW = dict(

    super = 'OBS_L4', # Inherit defaults for gridded L4 observations.

    path = '/data1/flo/ODYSSEA/SST_GLO_SST_L3S_NRT_OBSERVATIONS_010_010/cmems_obs-sst_glo_phy_l3s_pmw_P1D-m_202311/2025/06/*.nc', # Observation file path or glob pattern.

    name_time = 'time', # Time coordinate name in the input files.
    
    name_lon = 'lon', # Longitude coordinate name in the input files.

    name_lat = 'lat', # Latitude coordinate name in the input files.
    
    name_var = {'SST':'adjusted_sea_surface_temperature'}, # Mapping from VarDyn variable role to source variable name.
    
    sigma_noise = .3 # Constant observation-error standard deviation used when no error variable is read.

)

AL = dict(

    super = 'OBS_SSH_NADIR', # Inherit defaults for along-track nadir SSH observations.

    path = '/data1/data/obs/level3/Alti_l3_cmems_NRT/al/*202506*.nc', # Observation file path or glob pattern.

    name_time = 'time', # Time coordinate name in the input files.
    
    name_lon = 'longitude', # Longitude coordinate name in the input files.

    name_lat = 'latitude', # Latitude coordinate name in the input files.
    
    name_var = {'SSH':'sla_unfiltered'}, # Mapping from VarDyn variable role to source variable name.

    path_err = f'{path_err}/noise_alg.nc', # Measurement-error file for this satellite product. 

    name_var_err = name_var_err, # Coordinate and variable names for the measurement-error field.
    
    sigma_noise = sigma_nadir

)

C2N = dict(

    super = 'OBS_SSH_NADIR', # Inherit defaults for along-track nadir SSH observations.

    path = '/data1/data/obs/level3/Alti_l3_cmems_NRT/c2/*202506*.nc', # Observation file path or glob pattern.

    name_time = 'time', # Time coordinate name in the input files.
    
    name_lon = 'longitude', # Longitude coordinate name in the input files.

    name_lat = 'latitude', # Latitude coordinate name in the input files.
    
    name_var = {'SSH':'sla_unfiltered'}, # Mapping from VarDyn variable role to source variable name.

    path_err = f'{path_err}/noise_c2.nc', # Measurement-error file for this satellite product. 

    name_var_err = name_var_err, # Coordinate and variable names for the measurement-error field.
    
    sigma_noise = sigma_nadir

)

H2B = dict(

    super = 'OBS_SSH_NADIR', # Inherit defaults for along-track nadir SSH observations.

    path = '/data1/data/obs/level3/Alti_l3_cmems_NRT/h2b/*202506*.nc', # Observation file path or glob pattern.

    name_time = 'time', # Time coordinate name in the input files.
    
    name_lon = 'longitude', # Longitude coordinate name in the input files.

    name_lat = 'latitude', # Latitude coordinate name in the input files.
    
    name_var = {'SSH':'sla_unfiltered'}, # Mapping from VarDyn variable role to source variable name.
    
    path_err = f'{path_err}/noise_h2g.nc', # Measurement-error file for this satellite product. 

    name_var_err = name_var_err, # Coordinate and variable names for the measurement-error field.
    
    sigma_noise = sigma_nadir,

)

S3A = dict(

    super = 'OBS_SSH_NADIR', # Inherit defaults for along-track nadir SSH observations.

    path = '/data1/data/obs/level3/Alti_l3_cmems_NRT/s3a/*202506*.nc', # Observation file path or glob pattern.

    name_time = 'time', # Time coordinate name in the input files.
    
    name_lon = 'longitude', # Longitude coordinate name in the input files.

    name_lat = 'latitude', # Latitude coordinate name in the input files.
    
    name_var = {'SSH':'sla_unfiltered'}, # Mapping from VarDyn variable role to source variable name.
    
    path_err = f'{path_err}/noise_s3a.nc', # Measurement-error file for this satellite product. 

    name_var_err = name_var_err, # Coordinate and variable names for the measurement-error field.
    
    sigma_noise = sigma_nadir,

)

S3B = dict(

    super = 'OBS_SSH_NADIR', # Inherit defaults for along-track nadir SSH observations.

    path = '/data1/data/obs/level3/Alti_l3_cmems_NRT/s3b/*202506*.nc', # Observation file path or glob pattern.

    name_time = 'time', # Time coordinate name in the input files.
    
    name_lon = 'longitude', # Longitude coordinate name in the input files.

    name_lat = 'latitude', # Latitude coordinate name in the input files.
    
    name_var = {'SSH':'sla_unfiltered'}, # Mapping from VarDyn variable role to source variable name.
    
    path_err = f'{path_err}/noise_s3b.nc', # Measurement-error file for this satellite product. 

    name_var_err = name_var_err, # Coordinate and variable names for the measurement-error field.

    sigma_noise = sigma_nadir,

)

S6A = dict(

    super = 'OBS_SSH_NADIR', # Inherit defaults for along-track nadir SSH observations.

    path = '/data1/data/obs/level3/Alti_l3_cmems_NRT/s6a/*202506*.nc', # Observation file path or glob pattern.

    name_time = 'time', # Time coordinate name in the input files.
    
    name_lon = 'longitude', # Longitude coordinate name in the input files.

    name_lat = 'latitude', # Latitude coordinate name in the input files.
    
    name_var = {'SSH':'sla_unfiltered'}, # Mapping from VarDyn variable role to source variable name.
    
    path_err = f'{path_err}/noise_s6a_hr.nc', # Measurement-error file for this satellite product. 

    name_var_err = name_var_err, # Coordinate and variable names for the measurement-error field.

    sigma_noise = sigma_nadir,

)

SWON = dict(

    super = 'OBS_SSH_NADIR', # Inherit defaults for along-track nadir SSH observations.

    path = '/data1/data/obs/level3/Alti_l3_cmems_NRT/SWOTnadir/*202506*.nc', # Observation file path or glob pattern.

    name_time = 'time', # Time coordinate name in the input files.
    
    name_lon = 'longitude', # Longitude coordinate name in the input files.

    name_lat = 'latitude', # Latitude coordinate name in the input files.
    
    name_var = {'SSH':'sla_unfiltered'}, # Mapping from VarDyn variable role to source variable name.
    
    path_err = f'{path_err}/noise_swonc.nc', # Measurement-error file for this satellite product. 

    name_var_err = name_var_err, # Coordinate and variable names for the measurement-error field.

    sigma_noise = sigma_nadir,

)


#################################################################################################################################
# OBSERVATIONAL OPERATORS
#################################################################################################################################
NAME_OBSOP = ['OBSOP_SSH', 'OBSOP_SST']

OBSOP_SSH = dict(

    super = 'OBSOP_INTERP_L3', # Interpolate model fields to along-track SSH observations.

    write_op = True, # Cache the interpolation operator on disk.

    path_save = f'{myPath}/H', # Directory where the cached operator is written.

    compute_op = compute_obs, # Recompute the operator when True; otherwise reuse cached files when available.

)

OBSOP_SST = dict(

    super = 'OBSOP_INTERP_L4', # Interpolate gridded SST observations onto the model grid/time.

    name_var = 'SST', # Model variable role observed by this operator.

    write_op = True, # Cache the interpolation operator on disk.

    path_save = f'{myPath}/H', # Directory where the cached operator is written.

    compute_op = compute_obs, # Recompute the operator when True; otherwise reuse cached files when available.


)

#################################################################################################################################
# Reduced basis parameters
#################################################################################################################################
NAME_BASIS =  ['LargeScales_SSH', 'SmallScales_SSH', 'LargeFastScales_SST', 'LargeSlowScales_SST', 'SmallScales_SST']


LargeScales_SSH = dict(

    super = 'BASIS_GAUSS3D', # Use Gaussian basis functions in longitude, latitude, and time.

    name_mod_var = 'sla', # Model variable controlled by this basis block.
    
    flux = True, # Let basis amplitudes appear/disappear in time, acting like a time-localized forcing.

    facns = 3., # Spacing factor between neighboring basis centers in space.

    facnlt = 3., # Spacing factor between neighboring basis centers in time.

    sigma_D = 970, # Horizontal decorrelation scale, in km.

    sigma_T = 25, # Temporal decorrelation scale, in days.

    sigma_Q = 0.03, # Prior standard deviation for SSH basis coefficients.

)


SmallScales_SSH = dict(

    super = 'BASIS_BMaux', # Use balanced-motion basis scales read from an auxiliary file.

    name_mod_var = 'sla', # Model variable controlled by this basis block.
    
    wavelet_init = False, # Do not use the wavelet basis to initialize the model state. # Do not use the wavelet basis to initialize the model state.
    
    file_aux = '../../aux_files/aux_reduced_basis_BM.nc', # Auxiliary file containing wavelength-dependent std and decorrelation times.

    lmax = 1000., # Largest wavelength represented by this small-scale basis, in km.

    factdec = 7.5, # Multiplier applied to the auxiliary decorrelation time.

    tdecmin = 2., # Minimum allowed decorrelation time, in days.

    tdecmax = 40., # Maximum allowed decorrelation time, in days.

    lc = None, # Optional cutoff length scale; None keeps the basis default behavior.
)

LargeFastScales_SST = dict(

    super = 'BASIS_GAUSS3D', # Use Gaussian basis functions in longitude, latitude, and time.

    name_mod_var = 'sst', # Model variable controlled by this basis block.
    
    flux = True, # Let basis amplitudes appear/disappear in time, acting like a time-localized forcing.

    facns = 3., # Spacing factor between neighboring basis centers in space.

    facnlt = 3., # Spacing factor between neighboring basis centers in time.

    sigma_D = 1000, # Horizontal decorrelation scale, in km.

    sigma_T = 3, # Temporal decorrelation scale, in days.

    sigma_Q = 0.5, # Prior standard deviation for fast SST basis coefficients.


)

LargeSlowScales_SST = dict(

    super = 'BASIS_GAUSS3D', # Use Gaussian basis functions in longitude, latitude, and time.

    name_mod_var = 'sst', # Model variable controlled by this basis block.
    
    flux = True, # Let basis amplitudes appear/disappear in time, acting like a time-localized forcing.

    facns = 3., # Spacing factor between neighboring basis centers in space.

    facnlt = 3., # Spacing factor between neighboring basis centers in time.

    sigma_D = 970, # Horizontal decorrelation scale, in km.

    sigma_T = 25, # Temporal decorrelation scale, in days.

    sigma_Q = 0.5, # Prior standard deviation for slow SST basis coefficients.

)



SmallScales_SST = dict(

    super = 'BASIS_BMaux', # Use balanced-motion basis scales read from an auxiliary file.

    name_mod_var = 'sst', # Model variable controlled by this basis block.
    
    wavelet_init = False, # Do not use the wavelet basis to initialize the model state. # Do not use the wavelet basis to initialize the model state.
    
    file_aux = '../../aux_files/aux_reduced_basis_BM.nc', # Auxiliary file containing wavelength-dependent std and decorrelation times.

    lmax = 1000., # Largest wavelength represented by this small-scale basis, in km.

    factdec = 7.5, # Multiplier applied to the auxiliary decorrelation time.

    tdecmin = 2., # Minimum allowed decorrelation time, in days.

    tdecmax = 20., # Maximum allowed decorrelation time, in days.

    facQ = 10, # Multiplier applied to the estimated prior coefficient standard deviation.

    lc = None, # Optional cutoff length scale; None keeps the basis default behavior.
)



#################################################################################################################################
# Analysis parameters
#################################################################################################################################
NAME_INV = 'myINV'

myINV = dict(

    super = 'INV_4DVAR', # Inherit defaults for the 4DVar minimization driver.

    save_minimization = True, # Save cost-function and gradient history during minimization.

    ftol = 5e-5, # Relative cost-function decrease threshold used for convergence.

    convergence_nit = 10, # Required number of consecutive converged iterations before stopping.

    maxiter = 50, # Maximum number of minimizer iterations. Here is we set 100 for testing purposes, but in practice, it can be set to a higher value (e.g. 1000).

    gradient_max_norm = 1e12, # Restart from the best state if the gradient norm exceeds this value.
    
    max_retries = 10, # Maximum restart attempts after unstable minimizer steps.

    path_save_control_vectors = f'{myPath}/controls/{name_experiment}', # Directory where control vectors are saved.

    timestep_checkpoint = timedelta(hours=6), # Time spacing between two checkpointed analysis states.

    prec = True, # Enable control-vector preconditioning.

    restart_4Dvar = False, # Restart from the latest saved control vector when available.

 
)


