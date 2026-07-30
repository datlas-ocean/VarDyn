#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Florian Le Guillou on June 2026.

Default experiment, grid, model, and diagnostics configuration blocks.
"""

#################################################################################################################################
# Global libraries     
#################################################################################################################################

from datetime import datetime,timedelta

#################################################################################################################################
# EXPERIMENTAL PARAMETERS
#################################################################################################################################
EXP = dict(

    name_experiment = 'my_exp', # name of the experiment

    saveoutputs = True, # save outputs flag (True or False)

    name_exp_save = 'my_output_name', # name of output files

    path_save = 'outputs', # path of output files

    tmp_DA_path = "scratch/", # temporary data assimilation directory path

    flag_plot = 0, # between 0 and 4. 0 for none plot, 4 for full plot

    name_lon = 'lon',

    name_lat = 'lat',

    name_time = 'time',

    init_date = datetime(2012,10,1,0), # initial date (yyyy,mm,dd,hh) 

    final_date = datetime(2012,12,2,0),  # final date (yyyy,mm,dd,hh) 

    assimilation_time_step = timedelta(hours=1),  # assimilation time step (corresponding to observation update timestep)

    saveoutput_time_step = timedelta(hours=1),  # time step at which the states are saved 

    plot_time_step = timedelta(days=1),  #  time step at which the states are plotted (for debugging),

    time_obs_min = None, 

    time_obs_max = None,

    lon_obs_min = None,

    lon_obs_max = None,

    lat_obs_min = None,

    lat_obs_max = None,

    write_obs = False, # save observation dictionary in *path_obs*

    compute_obs = False, # force computing observations 

    path_obs = None # if set to None, observations are saved in *tmp_DA_path*

)


#################################################################################################################################
# GRID 
#################################################################################################################################
NAME_GRID = 'GRID_GEO'

# Read grid from file
GRID_FROM_FILE = dict(

    path_init_grid = '', 

    name_init_lon = 'lon',

    name_init_lat = 'lat',

    subsampling = None,

    name_init_mask = None,

    name_var_mask = {'lon':'','lat':'','var':''},

)

# Regular geodetic grid
GRID_GEO = dict(

    lon_min = 294.,                                        # domain min longitude

    lon_max = 306.,                                        # domain max longitude

    lat_min = 32.,                                         # domain min latitude

    lat_max = 44.,                                         # domain max latitude

    dlon = 1/10.,                                            # zonal grid spatial step (in degree)

    dlat = 1/10.,                                            # meridional grid spatial step (in degree)

    name_init_mask = None,

    name_var_mask = {'lon':'','lat':'','var':''}

)

# Regular cartesian grid 
GRID_CAR = dict(

    super = 'GRID_CAR',

    lon_min = 295.,                                        # domain min longitude

    lon_max = 305.,                                        # domain max longitude

    lat_min = 33.,                                         # domain min latitude

    lat_max = 43.,                                         # domain max latitude

    dx = 25.,                                              # grid spacing in km

    nx = None,                                             # If not None, use nx to compute dx 

    ny = None,                                             #

    name_init_mask = None,

    name_var_mask = {'lon':'','lat':'','var':''}

)

# Restart from previous run 
GRID_RESTART = dict(

    name_grid = 'restart',

)

#################################################################################################################################
# MODELS
#################################################################################################################################
NAME_MOD = None # Either DIFF, QG1L, QG1LM, SW1L, SW1LM    

# Diffusion model
MOD_DIFF = dict(

    name_var = {'SSH':"ssh"},

    var_to_save = None,

    name_init_var = {},

    dtmodel = 300, # model timestep

    Kdiffus = 0, # coefficient of diffusion. Set to 0 for Identity model

    init_from_bc = False,

    dist_sponge_bc = None,  # distance (in km) for which boundary fields are spatially spread close to the borders

    bc_file = None,  # Path to NetCDF file containing boundary conditions

    bc_files = None,  # Optional dict mapping model variable names to BC files or per-variable BC source dicts

    bc_name_lon = 'lon',  # Name of longitude coordinate in BC file

    bc_name_lat = 'lat',  # Name of latitude coordinate in BC file

    bc_name_time = None,  # Name of time coordinate in BC file

    bc_name_var = {},  # Dict mapping variable names to names in BC file

    bc_c_grid = False,  # Whether BC grid is C-grid (True) or A-grid (False)
)

# 1.5-layer Quasi-Geostrophic models
MOD_QG1L = dict(

    name_class = 'Qgm', # Name of the model class in jqgm.py

    formulation = 'ssh', # Dynamical formulation: 'ssh' (work in SSH space) or 'sf' (work in streamfunction space)

    name_var = {'SSH':"ssh"}, # Dictionnary of variable name (need to be at least SSH, and optionaly tracer variables SST, SSS etc. and/or ageostrophic velocities U, V)

    name_params = None, # List of parameters to jointly estimate.  Recognised values: 'c' (effective phase-speed field c_eff(x,y)).

    name_init_var = {}, # Only if grid is a GRID_FROM_FILE type. Dictionnary of variable names to initialize from the file 

    dir_model = None, # directory of the model (if other than mapping/models/model_qg1l)

    var_to_save = None, # List of variable names (among of the values of name_var dictionary) to save

    save_diagnosed_variables = False, # Whether to save diagnosed variables (e.g. SSH, geostrophic velocies and cyclogeostrophic velocities) in the output netcdf files

    save_params = False, # Whether to save control parameters (e.g. corrective fluxes) in the output netcdf files

    upwind = 3, # Order of the upwind scheme for PV advection (either 1,2 or 3) 

    advect_pv = True, # Whether or not to advect PV. 

    advect_tracer = False, # Whether or not to advect tracers. If True, need to add tracer variables (e.g. SST) in *name_var*

    dtmodel = 1200, # model timestep

    cfl = None, # If not None, dtmodel is set such as dtmodel=cfl*dx/c

    time_scheme = 'Euler', # Time scheme of the model (e.g. Euler,rk2,rk3)

    c0 = 2.7, # If not None, fixed value for phase velocity 

    filec_aux = None, # if c0==None, auxilliary file to be used as phase velocity field (the spatial interpolation is handled inline)

    name_var_c = {'lon':'','lat':'','var':''}, # Variable names for the phase velocity auxilliary file 

    cmin = None, # Minimum value of phase velocity to consider

    cmax = None, # Maximum value of phase velocity to consider

    file_bathy_aux = None, # Name of netcdf file for ocean bathymetry field. If prescribed, bathymetry will be taken into account in the model

    name_var_bathy = {'lon':'', 'lat':'', 'var':''}, # Name of longitude,latitude and variable of bathymetry netcdf file

    bathy_ratio_max = None, # Maximum value of bathymetry-related PV term

    solver = 'spectral', # Solver for Elliptical Equation inversion (either spectral or cg - for Conjugate Gradient)

    init_from_bc = False, # Whether or not to initialize the model with boundary fields.

    dist_sponge_bc = None, # Width (in km) of the band where boundary conditions are applied to edges of the domain and to coastal aeras

    Kdiffus = None,

    Kdiffus_trac = None,

    bc_trac = 'OBC', # Either OBC or fixed

    forcing_tracer_from_bc = False, # Whether to use BC fields to force tracer advection,

    sponge_coef = 0., # Rayleigh damping coefficient applied to tracers in the sponge zone (dimensionless, per model step). Typical values: 0.01–0.2

    constant_c = True,

    constant_f = True,

    f0 = None,

    tile_size = 32, # Only for name_class=='QgmWithTiles'
            
    tile_overlap = 16,  # Only for name_class=='QgmWithTiles'

    path_mdt = None, # path of MDT 

    name_var_mdt = None, # dictionary of MDT coordinates and variable {'lon':<name_lon>, 'lat':<name_lat>, 'var':<name_var>}

    bc_file = None,  # Path to NetCDF file containing boundary conditions

    bc_files = None,  # Optional dict mapping model variable names to BC files or per-variable BC source dicts

    bc_name_lon = 'lon',  # Name of longitude coordinate in BC file

    bc_name_lat = 'lat',  # Name of latitude coordinate in BC file

    bc_name_time = None,  # Name of time coordinate in BC file

    bc_name_var = {},  # Dict mapping variable names to names in BC file

    bc_c_grid = False,  # Whether BC grid is C-grid (True) or A-grid (False)

)

# 1.5-layer Shallow-Water model
MOD_CSW1L = dict(

    # Model parameters

    name_var = {'U':'u','V':'v','SSH':'ssh'}, # Dictionnary of variable name 

    name_params = ['He_mean', 'hbc', 'alpha'], # List of parameters to control (among 'He_mean', 'hbc', 'alpha', 'alpha_He', 'alpha_Uu', , 'alpha_Up', 'alpha_Uz')

    name_init_var = {}, # Only if grid is a GRID_FROM_FILE type. Dictionnary of variable names to initialize from the file

    dir_model = None, # directory of the model (if other than mapping/models/model_sw1l)

    var_to_save = None, # List of variable names (among of the values of name_var dictionary) to save. If None, all variables in name_var will be saved

    g = 9.81, # Gravity acceleration (in m/s^2)

    # Time stepping parameters

    dtmodel = 300, # model timestep

    cfl = None, # If not None, dtmodel is set such as dtmodel=cfl*dx/sqrt(gHe)

    time_scheme = 'rk4', # Time scheme of the model (e.g. Euler,rk3,rk4)

    # MDT data

    path_mdt = None, # path of MDT

    name_var_mdt = {'lon':'','lat':'','mdt':'','mdu':'','mdv':''},

    # First baroclinic mode phase velocity field 

    c0 = 2.7, # If filec_aux is None, fixed value for phase velocity (m/s)

    filec_aux = None, # auxilliary file to be used as phase velocity field (the spatial interpolation is handled inline)

    name_var_c = {'lon':'','lat':'','var':''}, # Variable names for the phase velocity auxilliary file 

    constant_He = False, # if True, use the spatial mean of c to derive a
                         # spatially constant equivalent depth He = mean(c)^2 / g

    # Bathymetry parameters

    H = 4e3, # Mean depth (in m)

    file_H_aux = None, # if H is None, netcdf file for spatially varying depth field. The spatial interpolation is handled inline.

    name_var_H = {'lon':'','lat':'','var':''}, # Variable names for the depth netcdf file

    # IT parameters

    w_waves = [2*3.14/12/3600], # igw frequencies (in seconds)

    Ntheta = 1, # Number of angles (computed from the normal of the border) of incoming waves.
               # Set to -1 to auto-compute the minimum Ntheta from the boundary tangential Nyquist:
               #   Ntheta >= L_bdy / lambda_min, where L_bdy = max boundary length and
               #   lambda_min = 2*pi*c_min/omega_max  (c_min on the boundary).
               # Set to 0 for normal incidence only (theta=0).

    # BM coupling parameters

    flag_coupling_from_bm = False, # Whether to compute He corrections from the balanced motion field

    path_vertical_modes = None, # Path of the vertical modes netcdf file

    path_interaction_terms = None, # Path of the interaction terms netcdf file. If None, interaction terms will be computed from the vertical modes (if path_vertical_modes is not None) or from the analytical formula of the modes (if path_vertical_modes is None)

    name_var_interaction_terms = {'lon':'','lat':'','U11_u':'','U11_p':'','U11_z':'', 'dc2':''}, # Variable names for the interaction terms netcdf file

    path_bm = None, # Path of the balanced motion netcdf file

    name_var_bm = {'time':'','lon':'','lat':'','ssh_bm':''}, # Variable names for the balanced motion netcdf file

    # Sponge-layer boundary relaxation parameters

    periodic_x = False, # Whether to apply periodic boundary conditions in the zonal direction

    periodic_y = False, # Whether to apply periodic boundary conditions in the meridional direction

    flag_bc_sponge = True, # Whether to apply a sponge boundary condition (i.e. a damping term nudging the solution towards the boundary conditions) close to the borders and coastal areas (if *dist_sponge_bc* is not None)

    dist_sponge_bc = None, # Width (in km) of the band where boundary conditions are applied to edges of the domain and to coastal aeras. If None, no sponge boundary condition is applied

    sponge_coef = 0.05, # Damping coefficient of the sponge boundary condition (in s^-1). Typical values are between 0.01 and 0.1 (e.g. 0.05 means a damping timescale of 20 seconds

    use_sponge_on_coast = True, # Whether to apply sponge near coastal areas (land mask)

    tangential_sponge_factor = 1., # factor [0,1] reducing sponge on tangential velocity at open boundaries (1=isotropic, 0=no tangential damping)

    mask_sponge_bc = True, # Whether to set the mask to True in the sponge boundary areas (i.e. to avoid assimilating observations in these areas)

    bc_it_method = 'plane_wave_bdy', # Wave phase method for the sponge IT boundary conditions.
                                     # 'plane_wave'     : original method, kept for backward compatibility
                                     # 'plane_wave_bdy' : k evaluated at the boundary edge (recommended default)
                                     # 'wkb'            : WKB cumulative-phase integral + He^{-1/4} amplitude correction

    bc_it_corner_weight_power = 1.0, # Power applied to smooth S/N/W/E corner partition weights

    extend_it_open_boundary_sponge = False, # If True, extend IT-side Heb, He_mean, alpha* controls, and Bathymetry H
                                            # from the Sponge Interior Edge across open-boundary S/N/W/E sponge bands
                                            # before constructing entering-wave/generation media. Coast/island/land
                                            # sponge extension is intentionally left for a future implementation.

    bc_file = None,  # Path to NetCDF file containing boundary conditions

    bc_files = None,  # Optional dict mapping model variable names to BC files or per-variable BC source dicts

    bc_name_lon = 'lon',  # Name of longitude coordinate in BC file

    bc_name_lat = 'lat',  # Name of latitude coordinate in BC file

    bc_name_time = 'time',  # Name of time coordinate in BC file

    bc_name_var = {},  # Dict mapping variable names to names in BC file

    bc_c_grid = False,  # Whether BC grid is C-grid (True) or A-grid (False)

)

# QG-SW Models
MOD_QGSW = dict(

    name_class = 'qg', # Name of the model class (either qg or sw)

    name_var = {'U':'u', 'V':'v', 'SSH':'ssh'},

    name_init_var = {}, 

    var_to_save = None,

    name_params = None, #['H', 'hbcx', 'hbcy', 'itg'], # list of parameters to control. H denotes dimensionless equivalent-depth log-control.

    nl = 1, # number of layers in the model (for nl>1, set H and g_prime as lists/arrays)

    dtmodel = 1200, # model timestep

    f0 = None, # Coriolis parameter (in s^-1). If None, f0 will be computed from the grid

    constant_f = False,

    c0 = 2.7,

    filec_aux = None, # if c0==None, auxilliary file to be used as phase velocity field (the spatial interpolation is handled inline)

    name_var_c = {'lon':'','lat':'','var':''}, # Variable names for the phase velocity auxilliary file 

    cmin = None, # Minimum value of phase velocity to consider

    cmax = None, # Maximum value of phase velocity to consider

    H = None, # mean layer depth(s) in meters.  Scalar or list, e.g. H=[500., 2500.] for nl=2

    constant_H = False, # if True and H is None (nl=1), use the spatial mean of c to derive a
                        # spatially constant H = mean(c)^2 / g_prime instead of the full 2-D field

    g_prime = None, # reduced gravity(ies).  Scalar or list, e.g. g_prime=[9.81, 0.02] for nl=2

    # Meaning of the 2-D state field named by name_var['SSH']:
    # 'ssh' keeps the historical direct-height formulation.
    # 'interface_displacement' (nl=1 SW only) stores reduced-gravity interface
    # displacement eta. External SSH is diagnosed as (g_prime / physical_gravity) * eta.
    # With H set and g_prime=None, g_prime is diagnosed from c1**2 / H.
    # 'modal_two_layer' (nl=2 SW only) is a physical mixed layer over a first
    # baroclinic layer. SSH/surface U/V are public diagnosed fields; the
    # kernel carries h_ml/u_ml/v_ml and h_bc1/u_bc1/v_bc1. Set H_ml/H_bc1 and
    # include their names in name_params to control the positive log-depths.
    height_representation = 'ssh',

    # Restart coordinate for the field mapped to name_init_var['SSH'] in
    # interface_displacement mode. The default is the physical `ssh` written
    # by save_output; use `interface_displacement` only when explicitly
    # selecting that saved internal field.
    restart_height_coordinate = 'physical_ssh',

    H_ml = None,       # modal_two_layer mixed-layer reference depth [m]
    H_ml_floor = None, # strict lower bound for H_ml [m]
    H_ml_max = None,   # optional upper bound for H_ml [m]
    H_bc1 = None,       # modal_two_layer first-baroclinic-layer reference depth [m]
    H_bc1_floor = None, # strict lower bound for H_bc1 [m]
    H_bc1_max = None,   # optional upper bound for H_bc1 [m]
    c_mode2_ratio = None, # c_mode2/c1 used with g_prime=None; must be below the two-layer admissibility limit
    interface_amplification = None, # diagnostic only in modal_two_layer; never a control
    forcing_vertical_projection = 'surface_baroclinic', # lift surface forcing/BCs into the modal two-layer stack
    wind_forcing_layer = 'mixed_layer', # modal_two_layer: wind acts only on layer 0 using current H_ml

    physical_gravity = 9.81, # m s^-2, used only for SSH <-> interface-displacement conversion

    init_from_bc = True,

    cfl = .25,

    bottom_drag_coef = 0.,

    slip_coef = 1., # Lateral wall slip coefficient (dimensionless, in [0,1]): 1 = free-slip, 0 = no-slip, in-between = partial slip. Use 1 when use_sponge_on_coast=True so the sponge is the sole near-coast damping mechanism (no double damping).

    taux = 0., # wind stress in N/m^2

    tauy = 0., # wind stress in N/m^2

    path_mdt = None, # path of MDT

    name_var_mdt = {'lon':'','lat':'','var':''}, # dictionary of MDT coordinates and variable {'lon':<name_lon>, 'lat':<name_lat>, 'var':<name_var>}

    name_var_mdu = {'lon':'','lat':'','var':''}, # dictionary of MDT coordinates and variable {'lon':<name_lon>, 'lat':<name_lat>, 'var':<name_var>}

    name_var_mdv = {'lon':'','lat':'','var':''}, # dictionary of MDT coordinates and variable {'lon':<name_lon>, 'lat':<name_lat>, 'var':<name_var>}

    dist_sponge_bc = None,

    use_sponge_on_coast = True,

    sponge_coef = 0.,

    tangential_sponge_factor = 1., # factor [0,1] reducing sponge on tangential velocity at open boundaries (1=isotropic, 0=no tangential damping)

    mask_sponge_bc = False, # Whether to set the mask to True in the sponge boundary areas (i.e. to avoid assimilating observations in these areas). Defaults to False for backward compatibility with existing Model_qgsw experiments.

    qg_balanced_sponge_bc = False, # (SW mode only) Whether to use a QG-projected, dynamically-balanced sponge target instead of the raw external boundary field. Requires an active sponge (sponge_coef > 0, dist_sponge_bc set). Incompatible with the 'bc' control parameter.

    visc_coef = 0., # viscosity coefficient (in m^2/s). Typical values 10–30 m²/s, 50–100 m²/s if unstable

    H_floor = None, # minimum Total Equivalent Depth for controlled H. If None, use cmin**2/g_prime when cmin is set; otherwise 0.

    H_min = None, # legacy hard minimum equivalent-depth clamp used only when H is not controlled. None means no clamping
    
    H_max = None, # optional hard maximum Total Equivalent Depth safety rail. None means no clamping

    diff_coef = 0., # diffusivity coefficient for h (in m^2/s). Typical values 20–50 m²/s, 100–200 m²/s if unstable

    diff_coef_trac = 0., # diffusivity coefficient for passive tracers (in m^2/s). Typical values 50–200 m²/s

    time_scheme = 'rk3',      # temporal scheme: 'rk3' (SSP-RK3, default) | 'rk2' (explicit midpoint, matches Qgm) | 'rk2_ssp' (Heun)
    h_adv_scheme = 'weno',       # h-continuity scheme: 'weno' (conservative WENO-6, default) | 'linear_upwind3'/'linear_upwind5' (fixed linear conservative fluxes for adjoint tests) | 'rusanov1'/'upwind1' (diffusive conservative option)
    mom_adv_scheme = 'weno',     # momentum-advection scheme: 'weno' (WENO-6, default) | 'upwind3' | 'upwind5' (fixed linear vorticity face reconstruction)
    tracer_adv_scheme = 'weno',  # tracer-advection scheme: 'weno' (WENO-6, default) | 'linear_upwind3'/'linear_upwind5' | 'rusanov1'/'upwind1'
    solver = 'dst_cmm',       # elliptic solver: 'dst_cmm' (DST + capacitance-matrix for irregular boundaries, default) | 'dst' (plain DST, matches inverse_elliptic_dst in Qgm)

    advect_tracer = None, # If True/False, override automatic tracer detection from name_var. None = auto.

    path_wind = None, # path to NetCDF wind file containing u10/v10 (if None, no wind forcing)

    name_var_wind = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time',
                     'u10': 'u10', 'v10': 'v10'}, # variable names in the wind NetCDF file

    rho_air = 1.225, # air density (kg/m³) used in the bulk wind-stress formula

    Cd_wind = 1.3e-3, # drag coefficient used in the bulk wind-stress formula tau = rho_air * Cd * |U10| * U10

    Cd_wind_formula = None, # Use the Large & Pond formula for drag coefficient. Set to None to use a constant drag coefficient (Cd_wind)

    rho_water = 1025.0, # ocean water density (kg/m³) used to convert wind stress [Pa] to acceleration [m²/s²]: tau/(rho_water*H)*dx

    # Physical layer depth (m) for the wind-stress denominator:  tau / (rho_water * h_wind) * dx
    # IMPORTANT for 1-layer QG/SW models: the model equivalent depth H = c²/g ≈ 0.4–1 m is
    # NOT the physical mixed-layer depth (~50–200 m).  Without setting h_wind, wind forcing
    # is 100–500× too large.  Set h_wind to the actual mixed-layer depth, e.g.:
    #   h_wind = 100.     # 100 m mixed layer
    # Leave None to use the model's reference layer thickness (correct only for multi-layer
    # models where H represent the true physical layer depths).
    h_wind = None,

    # Bounds for the dimensionless logarithmic h_wind control. When h_wind is
    # controlled, VarDyn uses h_wind_total = floor + (h_wind-floor)*exp(alpha).
    h_wind_floor = None, # None means 0
    h_wind_max = None, # None means no upper bound

    wind_timestep = 3600, # wind update interval in seconds (default: 1 hour). Wind stress is
                          # precomputed at this cadence and held constant between updates.
                          # Reduces memory when the model timestep is very small.

    max_nstep = 240, # maximum number of model steps per JIT call. Large nstep values are
                     # split into chunks of max_nstep to limit GPU memory usage.
                     # Decrease if running out of GPU memory.

    # Momentum forcing mode for external forcing (Fu, Fv, Fh).
    # 'direct'          : use Fu, Fv as provided (default).
    # 'mass_consistent' : derive Fu, Fv from Fh so that velocity is conserved
    #                     when mass is added:  Fu = -u/h * Fh,  Fv = -v/h * Fh.
    forcing_momentum = 'direct',

    bc_file = None,  # Path to NetCDF file containing boundary conditions

    bc_files = None,  # Optional dict mapping model variable names to BC files or per-variable BC source dicts

    bc_name_lon = 'lon',  # Name of longitude coordinate in BC file

    bc_name_lat = 'lat',  # Name of latitude coordinate in BC file

    bc_name_time = 'time',  # Name of time coordinate in BC file

    bc_name_var = {},  # Dict mapping variable names to names in BC file

    bc_c_grid = True,  # Whether BC grid is C-grid (True) or A-grid (False)

)

# Balanced Motion + Internal Tide model
MOD_BMIT = dict(

    # Coupled BM/IT state variables. BM velocity variables are prognostic when
    # bm_model uses a shallow-water core and diagnosed when bm_model is SSH-only.
    name_var = {
        'U_IT':'u_it', 'V_IT':'v_it', 'SSH_IT':'ssh_it',
        'U_BM':'u_bm', 'V_BM':'v_bm', 'SSH_BM':'ssh_bm',
        'SSH':'ssh'
    },

    name_init_var = {},

    var_to_save = None,

    dtmodel = 300, # coupled model timestep; component models are run on this timestep

    # Balanced Motion component. Component-specific BM options live here.
    bm_model = dict(
        super = 'MOD_QG1L',
        name_var = {'SSH':'ssh_bm'},
        name_params = None,
        name_init_var = {'SSH':'ssh_bm'},
        dtmodel = 300,
        init_from_bc = False,
        time_scheme = 'Euler',
        Kdiffus = 0,
        c0 = 2.7,
        filec_aux = None,
        name_var_c = {'lon':'','lat':'','var':''},
        cmin = None,
        cmax = None,
        path_mdt = None,
        name_var_mdt = None,
        bc_file = None,
        bc_files = None,
        bc_name_lon = 'lon',
        bc_name_lat = 'lat',
        bc_name_time = 'time',
        bc_name_var = {},
        bc_c_grid = False,
    ),

    # Height used by BM->IT equivalent-depth coupling: 'anomaly' uses SSH_BM,
    # 'full' uses SSH_BM plus MDT when available. Advective coupling always uses
    # full BM velocities.
    bm_height_for_He = 'anomaly',

    mask_sponge_bc = True, # Whether to set the mask to True in sponge boundary areas for the coupled BM/IT model.

    # Internal Tide component. Component-specific IT options live here.
    it_model = dict(
        super = 'MOD_CSW1L',
        name_var = {'U':'u_it','V':'v_it','SSH':'ssh_it'},
        name_params = ['He_mean', 'hbc'],
        name_init_var = {'U':'u_it', 'V':'v_it', 'SSH':'ssh_it'},
        dtmodel = 300,
        time_scheme = 'rk4',
        c0 = 2.7,
        filec_aux = None,
        name_var_c = {'lon':'','lat':'','var':''},
        cmin = None,
        cmax = None,
        constant_He = False,
        H = 4e3,
        file_H_aux = None,
        name_var_H = {'lon':'','lat':'','var':''},
        w_waves = [2*3.14/12/3600],
        Ntheta = 1,
        g = 9.81,
        periodic_x = False,
        periodic_y = False,
        flag_bc_sponge = False,
        dist_sponge_bc = None,
        sponge_coef = 0.05,
        use_sponge_on_coast = True,
        tangential_sponge_factor = 1.,
        bc_it_method = 'plane_wave_bdy',
        bc_it_corner_weight_power = 1.0,
        extend_it_open_boundary_sponge = False,
    ),

    alpha_eps = 1e-6, # safety margin for alpha logit references loaded from background files

    alpha_background_physical = None, # If None, auto-detect new physical alpha backgrounds via *_control companions; False reads legacy centered alpha anomalies.

    flag_coupling_from_bm = False, # Whether to compute He and advective corrections from Balanced Motion

    path_vertical_modes = None, # Path of the vertical modes netcdf file

    path_interaction_terms = None, # Path of the interaction terms netcdf file

    name_var_interaction_terms = {'lon':'lon','lat':'lat','U11_u':None,'U11_p':None,'dc2':None}, # Variable names for the interaction terms

    max_nstep = 240, # maximum number of coupled model steps per JIT call.

)

#################################################################################################################################
# OBSERVATIONS 
#################################################################################################################################
NAME_OBS = None

# L4 products (has to be on 2D latitude x longitude grids)
OBS_L4 = dict(

    path = '', # path of observation netcdf file(s)

    name_time = '', # name of time coordinate
    
    name_lon = '', # name of longitude coordinate

    name_lat = '', # name of latitude coordinate
    
    name_var = {}, # dictionnary of observed variables (keys: variable types [SSH,SST etc...]; values: name of observed variables)

    name_err = {}, # dictionnary of measurement error variables (keys: variable types [SSH,SST etc...]; values: name of error variables)

    subsampling = None, # Subsampling in time (in number of model time step). Set to None for no subsampling

    sigma_noise = None,  # Value of (constant) measurement error (will be used if *name_err* is not provided)

    offset = None, # Value to add to observations

)

# Nadir altimetry
OBS_SSH_NADIR = dict(

    path = '', # path of observation netcdf file(s)

    name_time = '', # name of time coordinate
    
    name_lon = '', # name of longitude coordinate

    name_lat = '', # name of latitude coordinate
    
    name_var = {'SSH':''}, # dictionnary of observed variables (keys: only SSH because altimetry; values: name of observed SSH, can be a lost of variables to combine, see *combine_var* parameter below)

    combine_var = None, # If not None, dictionnary of variable to combine to get the observed variable (keys: same as name_var; values: list of -1 or +1 to indicate how to combine variables in name_var, e.g. {'SSH':[-1,1]} to compute SSH as the difference between the second and the first variable in name_var['SSH'] list)
    
    synthetic_noise = None, # Std of synthetic noise (std in meters) to artificially add to the data

    varmax = 1e2, # Maximal value of observations considered 

    sigma_noise = None, # Value of (constant) measurement error 

    add_mdt = None, # Whether to add MDT or not (if observations are SLA and dynamical model works with SSH)

    substract_mdt = None, # Whether to remove MDT or not (if observations are SSH and dynamical model works with SLA)

    path_mdt = None, # path of MDT 

    name_var_mdt = None, # dictionary of MDT coordinates and variable {'lon':<name_lon>, 'lat':<name_lat>, 'var':<name_var>}
    
    path_err = None, # path of error file 

    name_var_err = None, # dictionary of error coordinates and variable {'lon':<name_lon>, 'lat':<name_lat>, 'var':<name_var>}
    
    nudging_params_ssh = None, # dictionary of nudging parameters on SSH {'sigma':<float>,'K':<float>,'Tau':<datetime.timedelta>}. Note that 'sigma' parameter is useless now, and will be removed soon,

    delta_t = None, # Sampling period of the satellite (in s), used for computing geostrophic current 

    velocity = None # Velocity of the satellite (in m/s), used for computing geostrophic current 

)

# Swath altimetry
OBS_SSH_SWATH = dict(

    path = '', # path of observation netcdf file(s)

    name_time = '', # name of time coordinate
    
    name_lon = '', # name of longitude coordinate

    name_lat = '', # name of latitude coordinate

    name_xac = None, # name of across track coordinate (like in SWOTsimulator output files)
    
    name_var = {'SSH':''}, # dictionnary of observed variables (keys: only SSH because altimetry; values: name of observed SSH)
    
    subsampling = None,
    
    synthetic_noise = None, # Std of synthetic noise (std in meters) to artificially add to the data

    sigma_noise = None, # Value of (constant) measurement error 

    add_mdt = None, # Whether to add MDT or not (if observations are SLA and dynamical model works with SSH)

    substract_mdt = None, # Whether to remove MDT or not (if observations are SSH and dynamical model works with SLA)

    path_mdt = None, # path of MDT 

    name_var_mdt = None, # dictionary of MDT coordinates and variable {'lon':<name_lon>, 'lat':<name_lat>, 'var':<name_var>}
    
    path_err = None, # path of error file 

    name_var_err = None, # dictionary of error coordinates and variable {'lon':<name_lon>, 'lat':<name_lat>, 'var':<name_var>}
    
    nudging_params_ssh = None, # dictionary of nudging parameters on SSH {'sigma':<float>,'K':<float>,'Tau':<datetime.timedelta>}. Note that *sigma* parameter is useless now, and will be removed soon

    nudging_params_relvort = None, # dictionary of nudging parameters on Relative Vorticity {'sigma':<float>,'K':<float>,'Tau':<datetime.timedelta>}. Note that *sigma* parameter is useless now, and will be removed soon
    
)

#################################################################################################################################
# OBSERVATIONAL OPERATORS
#################################################################################################################################
NAME_OBSOP = None

OBSOP_INTERP_L3 = dict(

    name_obs = None, # List of observation class names. If None, all observation will be considered. 

    name_var = 'SSH',

    write_op = False, # Write operator data to *path_save*

    path_save = None, # Directory where to save observational operator

    compute_op = True, # Force computing H 

    Npix = 4, # Number of pixels to perform projection y=Hx

    mask_borders = False,

)

OBSOP_INTERP_L4 = dict(

    name_obs = None, # List of observation class names. If None, all observation will be considered. 

    name_var = 'SSH',

    write_op = False, # Write operator data to *path_save*

    path_save = None, # Directory where to save observational operator

    compute_op = True, # Force computing H 

    mask_borders = False,

    interp_method = 'linear', # either 'nearest', 'linear', 'cubic' (use only 'cubic' when data is full of non-NaN)

    gradients = False

)

#################################################################################################################################
# REDUCED BASIS
#################################################################################################################################

NAME_BASIS = None

# Offset basis (i.e. a single basis vector with value 1 everywhere)
BASIS_OFFSET = dict(

    name_mod_var = None, # String or list of model parameter names sharing this basis

    sigma_B = None, 

)

# Gaussian basis: both 2D (space) and 3D (space+time) versions are available. 
BASIS_GAUSS2D = dict(

    super = 'BASIS_GAUSS2D',

    name_mod_var = '', # String or list of related model parameter names

    c_grid_var = None, # C-grid variable type: None (default h-grid), 'U' (shape ny,nx+1), or 'V' (shape ny+1,nx)

    compute_velocities = False, # Whether to compute geostrophic velocities associated to the SSH basis vectors

    name_mod_u = 'u', # Name of the zonal-velocity model variable (if *compute_velocities* is True)

    name_mod_v = 'v', # Name of the meridional-velocity model variable (if *compute_velocities* is True)

    facns = 2., # Factor for gaussian spacing in space (controls centre density relative to sigma_D)

    sigma_D = 300, # Spatial scale (km): Gaussian half-width / truncation radius

    sigma_Q = 0.01, # Prior standard deviation for each control coefficient

    facQ = 1., # Factor multiplied to the estimated Q

    flag_variable_Q = False, # If True, read spatially varying std from *path_sad*

    path_sad = None, # Path to a netcdf file with a spatially varying std field (used when flag_variable_Q=True)

    name_var_sad = {'lon':'', 'lat':'', 'var':''}, # Variable names inside *path_sad*

    path_background = None, # Path to a netcdf file with background control-vector values

    var_background = None # Variable name inside *path_background*

)

BASIS_GAUSS3D = dict(

    name_mod_var = '', # String or list of related model parameter names 

    c_grid_var = None, # C-grid variable type: None (default h-grid), 'U' (shape ny,nx+1), or 'V' (shape ny+1,nx)

    compute_velocities = False, # Whether to compute geostrophic velocities associated to the SSH basis vectors

    name_mod_u = 'u', # Name of the zonal-velocity model variable (if *compute_velocities* is True)

    name_mod_v = 'v', # Name of the meridional-velocity model variable (if *compute_velocities* is True)

    flux = False,

    facns = 2., # Factor for gaussian spacing in space

    facnlt = 1., # Factor for gaussian spacing in time

    sigma_D = 300, # Spatial scale (km)

    sigma_T = 20, # Time scale (days)

    sigma_Q = 0.01, # Standard deviation for matrix Q 

    facQ = 1., # Factor multiplied to the estimated Q

    normalize_fact = True,

    time_spinup = None, # days

    flag_variable_Q = False,

    path_sad = None,

    name_var_sad = {'lon':'', 'lat':'', 'var':''}, # Name of longitude,latitude and variable of depth netcdf file

    path_background = None, # path netcdf file of a basis vector (e.g. coming from a previous run) to use as background

    var_background = None # name of the variable of the basis vector

) 

# Balanced Motions with auxilliary data 
BASIS_BMaux = dict(

    name_mod_var = None, # String or list of related model parameter names 

    c_grid_var = None, # C-grid variable type: None (default h-grid), 'U' (shape ny,nx+1), or 'V' (shape ny+1,nx)

    compute_velocities = False, # Whether to compute geostrophic velocities associated to the SSH basis vectors

    name_mod_u = 'u', # Name of the zonal-velocity model variable (if *compute_velocities* is True)

    name_mod_v = 'v', # Name of the meridional-velocity model variable (if *compute_velocities* is True)
    
    flux = False, # Whether making a component signature in space appear/disappear in time. For dynamical mapping, use flux=False

    facns = 1., #factor for wavelet spacing in space 

    facnlt = 2., #factor for wavelet spacing in time

    npsp = 3.5, # Defines the wavelet shape

    facpsp = 1.5, # factor to fix df between wavelets

    file_aux = '', # Name of auxilliary file in which are stored the std and tdec for each locations at different wavelengths.

    lmin = 80, # minimal wavelength (in km)

    lmax = 970., # maximal wavelength (in km)

    factdec = 0.5, # factor to be multiplied to the computed time of decorrelation 

    tdecmin = 2.5, # minimum time of decorrelation 

    tdecmax = 40., # maximum time of decorrelation 

    facQ = 1, # factor to be multiplied to the estimated Q

    facQ_aux_path = None,

    l_largescale = 500, # factor to be multiplied to the estimated Q

    facQ_largescale = 1, # factor to be multiplied to the estimated Q

    file_depth = None, # Name of netcdf file for ocean depth field. If prescribed, wavelet components will be attenuated for small depth considering arguments depth1 & depth2

    name_var_depth = {'lon':'', 'lat':'', 'var':''}, # Name of longitude,latitude and variable of depth netcdf file

    depth1 = 0.,

    depth2 = 30.,

    path_background = None, # path netcdf file of a basis vector (e.g. coming from a previous run) to use as background

    var_background = None, # name of the variable of the basis vector

    norm_time = True,

    file_facQaux = None,

    name_var_facQaux = {'wavenumber':'', 'lon':'', 'lat':'', 'var':''}

)

# Internal Tides boundary conditions basis (i.e. a set of basis vectors with values at the open boundaries)
BASIS_HBC = dict(

    name_params = ['hbcx', 'hbcy'], # list of parameters to control (among 'He', 'hbcx', 'hbcy', 'itg')

    ### COMMON PARAMETER ### 

    # facgauss = 3.5,  # factor for gaussian spacing= both space/time

    facns = 3.5, # factor for gaussian spacing in space

    facnlt = 2.5, # factor for gaussian spacing in time 

    time_dependant = True, # True if gaussian basis is time dependant

    ### - HBC PARAMETER ### 

    sigma_B_bc = 1e-2, # Background variance for bc

    D_bc = 200, # Space scale of gaussian decomposition for boundary conditions (in km)

    T_bc = 20, # Time scale of gaussian decomposition for boundary conditions (in days)

    Nwaves = 1, # igw frequencies (in seconds)

    Ntheta = 1, # Number of angles (computed from the normal of the border) of incoming waves,

)


#################################################################################################################################
# INVERSION METHODS
#################################################################################################################################
NAME_INV = None

# 4-Dimensional Variational 
INV_4DVAR = dict(

    minimizer = 'scipy', # 'scipy': historical host L-BFGS-B; 'optax-decoupled': device L-BFGS with a scalar Python line search

    compute_test = False, # TLM, ADJ & GRAD tests

    freq_it_plot = 10, # Frequency of iteration to plot the cost function and its gradient  

    plot_state_during_minimization = False, # Opt in to costly device-to-host state plots from cost evaluations

    print_time = False, # Whether to print the time taken for each iteration, split by model, obs operator and gradient computation

    JAX_mem_fraction = None,

    cost_float64 = True, # Accumulate cost/control terms in float64 while model kernels may stay float32

    jit_cost_and_grad = False, # Compile the complete forward/adjoint checkpoint schedule as one device executable

    cost_and_grad_schedule = 'python', # 'python': historical unrolled checkpoint loops; 'scan': rolled lax.scan loops

    path_init_4Dvar = None, # To restart the minimization process from a specified control vector

    restart_4Dvar = False, # To restart the minimization process from the last control vector

    ftol = None, # Relative accepted-cost change criterion; supported by both minimizers

    gtol = None, # Gradient tolerance relative to the initial norm; projected max norm in SciPy, L2 norm in Optax

    convergence_nit = None, # Consecutive accepted iterations satisfying a configured criterion; None means one

    minimum_iterations = 1, # Do not apply sustained convergence before this accepted iteration

    maxiter = 10, # Maximal number of iterations for the minimization process

    gradient_max_norm = 1e6, # Reject unstable Optax trials; SciPy restarts from its best state

    max_retries = 5, # SciPy-only retries after unstable gradients

    save_minimization = False, # save cost function and its gradient at each iteration 

    path_save_control_vectors = None, # Path where to save the control vector at each 4Dvar iteration 

    timestep_checkpoint = timedelta(hours=12), # timestep separating two consecutive analysis 

    sigma_R = None, # Observational standard deviation

    sigma_B = None,

    prec = False, # preconditoning

    path_background = None, # Path of a control vector from another experiment to use as the background 

    anomaly_from_bc = False # Whether to perform the minimization with anomalies from boundary condition field(s)
 
)


#################################################################################################################################
# DIAGNOSTICS
#################################################################################################################################
NAME_DIAG = None

# Observatory System Simulation Experiment 
DIAG_OSSE = dict(

    dir_output = None,

    time_min = None,

    time_max = None,

    time_step = None,

    lon_min = None,

    lon_max = None,

    lat_min = None,

    lat_max = None,
    
    path_images2mp4 = None,

    name_ref = '',

    name_ref_time = '',

    name_ref_lon = '',

    name_ref_lat = '',

    name_ref_var = '',

    options_ref =  {},

    name_exp_time = None,

    name_exp_lon = None,

    name_exp_lat = None,

    name_exp_var = '',

    exp_grid_type = None,  # None for h-grid, 'u' for u-grid, 'v' for v-grid

    compare_to_baseline = False,

    name_bas = None,

    name_bas_time = None,

    name_bas_lon = None,

    name_bas_lat = None,

    name_bas_var = None,

    name_mask = None,

    name_var_mask = {'lon':'','lat':'','var':''}

)

# Observatory System Experiment (e.g. validation with real data)
DIAG_OSE = dict(

    dir_output = None,

    time_min = None,

    time_max = None,

    lon_min = None,

    lon_max = None,

    lat_min = None,

    lat_max = None,

    bin_lon_step = 1,

    bin_lat_step = 1,

    bin_time_step = '1D',

    name_ref = '',

    name_ref_time = '',

    name_ref_lon = '',

    name_ref_lat = '',

    name_ref_var = '',

    options_ref =  {},

    add_mdt_to_ref = False,

    path_mdt = None,

    name_var_mdt = None,
    
    delta_t_ref = None, # s

    velocity_ref = None, # km/s

    lenght_scale = 1000, # km

    nb_min_obs = 10,

    name_exp_var = '',

    compare_to_baseline = False,

    name_bas = None,

    name_bas_time = None,

    name_bas_lon = None,

    name_bas_lat = None,

    name_bas_var = None

)
