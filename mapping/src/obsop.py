#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Florian Le Guillou on June 2026.

Builds observation operators and interpolation mappings.
"""
from .config import USE_FLOAT64
import os,sys
import math
from numbers import Real, Integral
from copy import deepcopy
# Disable HDF5 file locking BEFORE importing xarray/netCDF4 — spawn worker
# subprocesses re-import this module and would otherwise hit
# "NetCDF: Not a valid ID" on shared filesystems when several tiles read
# the same obs files concurrently.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import xarray as xr 
import numpy as np 
from src import tools as grid
import pickle
import matplotlib.pylab as plt
from scipy.interpolate import CloughTocher2DInterpolator, griddata
from scipy.sparse import csc_matrix, csr_matrix
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree, Delaunay, KDTree
import pyinterp
import jax.numpy as jnp 
from jax import jit
import jax

jax.config.update("jax_enable_x64", USE_FLOAT64)


def _linear_barycentric_weights(source_coords, target_coords):
    """Return a sparse linear-interpolation operator for fixed 2-D grids."""
    triangulation = Delaunay(source_coords)
    simplex = triangulation.find_simplex(target_coords)
    inside = simplex >= 0

    target_inside = target_coords[inside]
    simplex_inside = simplex[inside]
    transforms = triangulation.transform[simplex_inside]
    first_weights = np.einsum(
        'nij,nj->ni',
        transforms[:, :2, :],
        target_inside - transforms[:, 2, :],
    )
    barycentric = np.column_stack(
        (first_weights, 1.0 - first_weights.sum(axis=1))
    )

    rows = np.repeat(np.flatnonzero(inside), 3)
    columns = triangulation.simplices[simplex_inside].ravel()
    weights = csr_matrix(
        (barycentric.ravel(), (rows, columns)),
        shape=(target_coords.shape[0], source_coords.shape[0]),
    )
    return weights, ~inside


def _open_obs_file(path, _retries=4, _sleep=0.5):
    """xr.open_dataset with retry, to absorb transient HDF5/netCDF read
    failures ('NetCDF: Not a valid ID') that can occur when many sibling
    subprocesses read the same files from a shared filesystem.
    """
    import time as _time
    last_exc = None
    for _i in range(_retries):
        try:
            return xr.open_dataset(path)
        except Exception as e:
            last_exc = e
            _time.sleep(_sleep * (2 ** _i))
    raise last_exc


def _stacked_obs_cache_path(operator, kind):
    """Return a cache path for already padded observation arrays."""
    if not operator.write_op or len(operator.date_obs) == 0:
        return None
    first = operator.date_obs[0].strftime('%Y%m%d_%H%M')
    last = operator.date_obs[-1].strftime('%Y%m%d_%H%M')
    observations = '_'.join(operator.name_obs)
    return os.path.join(
        operator.path_save,
        f'{operator.name_H}_{observations}_{first}_{last}_'
        f'{len(operator.date_obs)}_{kind}_stacked_v1.pic',
    )


def Obsop(config, State, dict_obs, Model, verbose=1, *args, **kwargs):
    """
    NAME
        basis

    DESCRIPTION
        Main function calling obsopt for specific observational operators
    """
    
    if config.OBSOP is None:
        return 
    
    if verbose:
        print(config.OBSOP)
    
    if config.OBSOP.super is None:
        return Obsop_multi(config, State, dict_obs, Model)
    
    elif config.OBSOP.super=='OBSOP_INTERP_LWE_L3':
        return Obsop_interp_lwe_l3(config, State, dict_obs, Model)

    elif config.OBSOP.super=='OBSOP_INTERP_L3':
        if any(source.get('lwe_enabled', False) for source in
               _lwe_selected_sources(config).values()):
            raise ValueError('lwe_enabled requires OBSOP_INTERP_LWE_L3')
        return Obsop_interp_l3(config, State, dict_obs, Model)
    
    elif config.OBSOP.super=='OBSOP_INTERP_L4':
        return Obsop_interp_l4(config, State, dict_obs, Model)
    
    else:
        sys.exit(config.OBSOP.super + ' not implemented yet')



class Obsop_interp:

    def __init__(self,config,State,dict_obs,Model):

        self.compute_H = config.OBSOP.compute_op

        # Date obs list
        self.date_obs = []
        
        # Pattern for saving files
        if config.EXP.lon_obs_max is not None:
            lon_obs_max = config.EXP.lon_obs_max
        else:
            lon_obs_max = State.lon_max

        if config.EXP.lon_obs_min is not None:
            lon_obs_min = config.EXP.lon_obs_min
        else:
            lon_obs_min = State.lon_min
        
        if config.EXP.lat_obs_max is not None:
            lat_obs_max = config.EXP.lat_obs_max
        else:
            lat_obs_max = State.lat_max

        if config.EXP.lat_obs_min is not None:
            lat_obs_min = config.EXP.lat_obs_min
        else:
            lat_obs_min = State.lat_min

        box = f'{int(lon_obs_min)}_{int(lon_obs_max)}_{int(lat_obs_min)}_{int(lat_obs_max)}'
        self.name_H = f'H_{box}_{int(config.EXP.assimilation_time_step.total_seconds())}_{int(State.dx)}_{int(State.dy)}'

        # Path to save operators
        self.compute_op = config.OBSOP.compute_op
        self.write_op = config.OBSOP.write_op
        if self.write_op:
            # We'll save or read operator data to *path_save*
            self.path_save = config.OBSOP.path_save
            if not os.path.exists(self.path_save):
                os.makedirs(self.path_save)
        else:
            self.path_save = config.EXP.tmp_DA_path

        # Temporary path where to save misfit
        self.tmp_DA_path = config.EXP.tmp_DA_path
        
        # Model variable and optional semantic observation role.  The
        # observation operator remains model-agnostic; the model resolves the
        # role to its own prognostic field/layer representation.
        self.name_mod_var = Model.name_var
        self.model = Model
        self.observation_role = getattr(config.OBSOP, 'observation_role', None)

        # For grid interpolation:
        lon = +State.lon
        lat = +State.lat
        self.shape_grid = [State.ny, State.nx]
        self.coords_geo = np.column_stack((lon.ravel(), lat.ravel()))
        self.coords_car = grid.geo2cart(self.coords_geo)

        # Mask land and sponge zones.
        # State.mask  : land / coast (always excluded from obs).
        # State.sponge_mask : sponge footprint set by Model_qgsw when
        #                     mask_sponge_bc=True.  Kept separate so that
        #                     other models sharing State are unaffected.
        _obs_mask = np.zeros((State.ny, State.nx), dtype=bool)
        if State.mask is not None:
            _obs_mask |= State.mask
        if getattr(State, 'sponge_mask', None) is not None:
            _obs_mask |= State.sponge_mask
        self.obs_mask = _obs_mask
        self.ind_mask = set(np.flatnonzero(_obs_mask.ravel()).tolist())
        
        # Mask boundary pixels
        self.ind_borders = []
        if config.OBSOP.mask_borders:
            coords_geo_borders = np.column_stack((
                np.concatenate((State.lon[0,:],State.lon[1:-1,-1],State.lon[-1,:],State.lon[:,0])),
                np.concatenate((State.lat[0,:],State.lat[1:-1,-1],State.lat[-1,:],State.lat[:,0]))
                ))
            if len(coords_geo_borders)>0:
                for i in range(self.coords_geo.shape[0]):
                    if np.any(np.all(np.isclose(coords_geo_borders,self.coords_geo[i]), axis=1)):
                        self.ind_borders.append(i)
        
        # Process obs
        self.dict_obs = dict_obs

    def _observation_model_field(self, State):
        """Return the field selected by the model for this observation."""
        if self.observation_role is not None and hasattr(self.model, 'observation_field'):
            return self.model.observation_field(State, self.name_var, self.observation_role)
        fields = State.var if hasattr(State, 'var') else State
        return fields[self.name_mod_var[self.name_var]]

    def _add_observation_adjoint(self, adState, increment):
        """Pass an observation adjoint back through the model interface."""
        if self.observation_role is not None and hasattr(self.model, 'add_observation_adjoint'):
            self.model.add_observation_adjoint(
                adState, self.name_var, self.observation_role, increment)
        else:
            name = self.name_mod_var[self.name_var]
            adState.setvar(adState.var[name] + increment, name)



    def process_obs(self, var_bc=None):

        return

    def is_obs(self,t):

        return t in self.date_obs

    @staticmethod
    def _model_to_observation_scale(Model, name_var):
        """Return the pointwise model-height -> observed-variable scale."""
        if name_var != 'SSH':
            return jnp.asarray(1.)
        return jnp.asarray(getattr(Model, 'ssh_observation_scale', 1.))
    
    def is_obs_time(self,t):

        return self.is_obs(t)
                
    def misfit(self,t,State):

        return

    def adj(self, t, adState, R):

        return

class Obsop_interp_l3(Obsop_interp):

    def __init__(self,config,State,dict_obs,Model):

        super().__init__(config,State,dict_obs,Model)

        # Date obs
        self.name_var = config.OBSOP.name_var
        self.model_to_observation_scale = self._model_to_observation_scale(Model, self.name_var).reshape(-1)
        self.date_obs = []
        self.t_obs = []
        self.name_obs = []
        date_obs = list(dict_obs.keys())
        for t,timestamp in zip(Model.T,Model.timestamps):
            delta_t = [(timestamp - date).total_seconds() for date in date_obs]
            if len(delta_t)>0:
                
                if np.min(np.abs(delta_t))<=Model.dt/2:
                    
                    ind_obs = np.argmin(np.abs(delta_t))

                    for obs_name, sat_info in zip(dict_obs[date_obs[ind_obs]]['obs_name'], 
                                                  dict_obs[date_obs[ind_obs]]['attributes']):
                        
                        if (self.name_var in sat_info['name_var']) and ((config.OBSOP.name_obs is None) or (config.OBSOP.name_obs is not None and obs_name in config.OBSOP.name_obs)):
                            if obs_name not in self.name_obs:
                                self.name_obs.append(obs_name)
                            if t not in self.t_obs:
                                self.date_obs.append(date_obs[ind_obs])
                                self.t_obs.append(t)
                    
        # For grid interpolation:
        self.Npix = config.OBSOP.Npix
        self.dmax = self.Npix*np.mean(np.sqrt(State.DX**2 + State.DY**2))*1e-3*np.sqrt(2)/2 # maximal distance for space interpolation

        self.t_obs = np.array(self.t_obs)
        self.t_obs_jax = jnp.array(self.t_obs)

        self.name_H += f'_L3-JAX_{self.name_var}_{config.OBSOP.Npix}'

        self._misfit_reduced_jit = jit(self._misfit_reduced)
        self._misfit_jit = jit(self._misfit)
    
    def _sparse_op(self, lon_obs, lat_obs):
        """Optimized sparse observation operator using KDTree."""
        
        coords_geo_obs = np.column_stack((lon_obs, lat_obs))
        coords_car_obs = grid.geo2cart(coords_geo_obs)

        # KDTree for nearest neighbor search
        tree = KDTree(self.coords_car)
        D, ind_closest = tree.query(coords_car_obs, k=self.Npix)
        if self.Npix == 1:
            D = D[:, None]
            ind_closest = ind_closest[:, None]

        # Apply distance threshold
        valid_mask = (D <= self.dmax)
        if len(self.ind_mask) > 0:
            masked_candidates = np.isin(ind_closest, list(self.ind_mask))
            nearest_is_masked = masked_candidates[:, 0]
            valid_mask &= ~masked_candidates
            valid_mask[nearest_is_masked, :] = False

        if np.any(valid_mask):

            if self.Npix>1:
                # Compute weights
                weights = np.exp(-D**2 / (2 * (0.5 * self.dmax)**2)) * valid_mask

                # Normalize
                sum_weights = np.sum(weights, axis=1, keepdims=True)
                weights = np.divide(
                                weights,
                                sum_weights,
                                out=np.zeros_like(weights),
                                where=sum_weights > 0
                            )
            else:
                weights = np.ones_like(D) 

        else:
            data = D * 0.

        # Extract valid indices
        try:
            row, col = np.where(valid_mask)
            data = weights[row, col]
            indices = jnp.array([row, ind_closest[row, col]], dtype=jnp.int32)
            data = jnp.array(data)
        except:
            data = jnp.array([])
            indices = jnp.array([])
            
        return data, indices
    
    def explicit_proj_operation(self, data, indices, X, n_obs):
        """
        Perform the projection operation explicitly without using csc_matrix.

        Parameters:
        - data: list or numpy array of interpolation coefficients (non-zero values of the sparse matrix).
        - row: list or numpy array of observation grid indices corresponding to each non-zero entry.
        - col: list or numpy array of state grid indices corresponding to each non-zero entry.
        - X: numpy array representing the state grid field to be projected.
        - n_obs: number of observation points (rows of the sparse matrix).

        Returns:
        - proj_X: projected observation values as a numpy array.
        """
        
        row = indices[0]
        col = indices[1]

        # Compute contributions of all sparse matrix entries to the corresponding observation indices
        proj_X = jnp.zeros(n_obs)
        proj_X = jnp.add.at(proj_X, row, data * X[col], inplace=False)

        return proj_X
    
    def process_obs(self, var_bc=None):

        stacked_cache = _stacked_obs_cache_path(self, 'l3')
        if (
            var_bc is None
            and not self.compute_op
            and stacked_cache is not None
            and os.path.exists(stacked_cache)
        ):
            with open(stacked_cache, 'rb') as stream:
                cached = pickle.load(stream)
            self.n_data = jnp.asarray(cached['n_data'])
            self.n_obs = jnp.asarray(cached['n_obs'])
            self.data_arr = jnp.asarray(cached['data_arr'])
            self.indices_arr = jnp.asarray(
                cached['indices_arr'], dtype=jnp.int32
            )
            self.varobs_arr = jnp.asarray(cached['varobs_arr'])
            self.errobs_arr = jnp.asarray(cached['errobs_arr'])
            return

        self.varobs = {}
        self.errobs = {}
        self.data = {}
        self.indices = {}

        for i,(date,t) in enumerate(zip(self.date_obs, self.t_obs)):

            self.varobs[t] = {}
            self.errobs[t] = {}
            self.data[t] = {}
            self.indices[t] = {}

            sat_info_list = self.dict_obs[date]['attributes']
            obs_file_list = self.dict_obs[date]['obs_path']
            obs_name_list = self.dict_obs[date]['obs_name']

            file_L3 = (
                f"{self.path_save}/{self.name_H}_{'_'.join(self.name_obs)}_"
                f"{date.strftime('%Y%m%d_%H%M')}.pic"
            )
            cached_payload = None
            if (
                not self.compute_op
                and self.write_op
                and os.path.exists(file_L3)
            ):
                with open(file_L3, "rb") as f:
                    cached_payload = pickle.load(f)
                if isinstance(cached_payload, dict):
                    data = cached_payload['data']
                    indices = cached_payload['indices']
                    var_obs = np.asarray(
                        cached_payload['var_obs']
                    ).copy()
                    err_obs = np.asarray(cached_payload['err_obs'])
                    lon_obs = np.asarray(cached_payload['lon_obs'])
                    lat_obs = np.asarray(cached_payload['lat_obs'])
                    if var_bc is not None and self.name_var in var_bc:
                        coords_obs = np.column_stack((lon_obs, lat_obs))
                        mask = np.any(np.isnan(self.coords_geo), axis=1)
                        var_bc_interp = griddata(
                            self.coords_geo[~mask],
                            var_bc[self.name_var][i].ravel()[~mask],
                            coords_obs,
                            method='cubic',
                        )
                        var_obs -= var_bc_interp
                    self.data[t] = data
                    self.indices[t] = indices
                    self.varobs[t] = var_obs
                    self.errobs[t] = err_obs
                    continue

            # Init lists
            var_obs = []
            err_obs = []
            lon_obs = []
            lat_obs = []

            ####################
            # Merge observations
            ####################
            for obs_name, sat_info, obs_file in zip(obs_name_list, sat_info_list, obs_file_list):

                if obs_name not in self.name_obs:
                    continue

                with _open_obs_file(obs_file) as ncin:

                    lon = ncin[sat_info['name_lon']].values
                    lat = ncin[sat_info['name_lat']].values

                    if self.name_var not in ncin:
                        continue

                    # Observed variable
                    var = ncin[self.name_var].values

                    # Reshape to 1D arrays
                    lon = lon.ravel()
                    lat = lat.ravel()
                    var = var.ravel()

                    # Observed error
                    name_err = self.name_var + '_err'
                    if name_err in ncin:
                        err = ncin[name_err].values.ravel() 
                    elif sat_info['sigma_noise'] is not None:
                        err = sat_info['sigma_noise'] * np.ones_like(var)
                    else:
                        err = np.ones_like(var)    

                    # Append to lists
                    var_obs.append(var)
                    err_obs.append(err)
                    lon_obs.append(lon)
                    lat_obs.append(lat)
            
            # Concatenations of lists
            var_obs = np.concatenate(var_obs)
            err_obs = np.concatenate(err_obs)
            lon_obs = np.concatenate(lon_obs)
            lat_obs = np.concatenate(lat_obs)

            coords_obs = np.column_stack((lon_obs, lat_obs))
            raw_var_obs = var_obs.copy()
            if var_bc is not None and self.name_var in var_bc:
                mask = np.any(np.isnan(self.coords_geo),axis=1)
                var_bc_interp = griddata(self.coords_geo[~mask], var_bc[self.name_var][i].flatten()[~mask], coords_obs, method='cubic')
                var_obs -= var_bc_interp

            # Fill dictionnaries
            self.varobs[t] = var_obs
            self.errobs[t] = err_obs

            # Compute Sparse operator
            if cached_payload is not None:
                data, indices = cached_payload
                self.data[t] = data
                self.indices[t] = indices
            else:
                # Compute operator
                result = self._sparse_op(lon_obs,lat_obs)
                data, indices = result
                self.data[t] = data
                self.indices[t] = indices
            # Versioned cache includes observation vectors as well as H,
            # so repeated inversions no longer reopen every altimetry NetCDF.
            if self.write_op:
                with open(file_L3, "wb") as f:
                    pickle.dump({
                        'version': 2,
                        'data': data,
                        'indices': indices,
                        'var_obs': raw_var_obs,
                        'err_obs': err_obs,
                        'lon_obs': lon_obs,
                        'lat_obs': lat_obs,
                    }, f)
        
        self.n_data = np.array([self.data[t].size for t in self.t_obs])
        self.n_obs = np.array([self.varobs[t].size for t in self.t_obs])
        
        if self.n_data.size>0:
            self.data_arr = np.zeros((self.t_obs.size, self.n_data.max()))
            self.indices_arr = np.zeros((self.t_obs.size, 2, self.n_data.max()), dtype=np.int32)
            self.varobs_arr = np.zeros((self.t_obs.size, self.n_obs.max()))
            self.errobs_arr = np.ones((self.t_obs.size, self.n_obs.max())) * 1e7
        
            for i,t in enumerate(self.t_obs):

                self.data_arr[i,:self.n_data[i]] = self.data[t]
                self.indices_arr[i,:,:self.n_data[i]] = self.indices[t]
                self.varobs_arr[i,:self.n_obs[i]] = self.varobs[t]
                self.errobs_arr[i,:self.n_obs[i]] = self.errobs[t] 
            
            if stacked_cache is not None and var_bc is None:
                with open(stacked_cache, 'wb') as stream:
                    pickle.dump({
                        'version': 1,
                        'n_data': self.n_data,
                        'n_obs': self.n_obs,
                        'data_arr': self.data_arr,
                        'indices_arr': self.indices_arr,
                        'varobs_arr': self.varobs_arr,
                        'errobs_arr': self.errobs_arr,
                    }, stream)

            self.n_data = jnp.array(self.n_data)
            self.n_obs = jnp.array(self.n_obs)
            self.data_arr = jnp.array(self.data_arr)
            self.indices_arr = jnp.asarray(self.indices_arr, dtype=jnp.int32)
            self.varobs_arr = jnp.array(self.varobs_arr)
            self.errobs_arr = jnp.array(self.errobs_arr)
        else:
            # Observation-free tiles are valid in tiled experiments (for
            # example over polar regions). Keep the scan interface defined
            # with zero-length, fixed-rank arrays; the misfit is never called
            # for these tiles because t_obs_jax is empty.
            self.n_data = jnp.zeros((0,), dtype=int)
            self.n_obs = jnp.zeros((0,), dtype=int)
            self.data_arr = jnp.zeros((0, 0))
            self.indices_arr = jnp.zeros((0, 2, 0), dtype=int)
            self.varobs_arr = jnp.zeros((0, 0))
            self.errobs_arr = jnp.zeros((0, 0))
        
    def is_obs_time(self,t):
        """Check if t is in observation times."""
        # This result controls a Python branch in the current orchestration.
        return bool(np.any(np.isclose(t, self.t_obs)))

    def is_obs_time_jax(self, t):
        """Device-side observation mask for a rolled checkpoint scan."""
        return jnp.any(jnp.isclose(t, self.t_obs_jax))

    def _misfit(self, t, X):
        
        # Get data at time t
        idt = jnp.where(self.t_obs_jax==t, size=1)[0]
        data_t = self.data_arr[idt][0]
        indices_t = self.indices_arr[idt][0]
        varobs_t = self.varobs_arr[idt][0]
        errobs_t = self.errobs_arr[idt][0]

        # Project the model coordinate after converting eta -> physical SSH
        # when the QGSW model uses the nl=1 interface formulation.
        HX = self.explicit_proj_operation(
            data_t, indices_t, X * self.model_to_observation_scale, varobs_t.size
        )

        # Compute misfit & errors
        misfit = HX - varobs_t
        inverr = 1/errobs_t
        misfit = jnp.where(jnp.isnan(misfit), 0, misfit) 
        inverr = jnp.where(jnp.isnan(inverr), 0, inverr) 

        return inverr * misfit
    
    def _misfit_reduced(self, t, grad_obs, X):

        """
        Projects a gradient in observation space back to the model state space.
        
        Parameters:
            t: Current time
            grad_obs: Gradient in observation space (adjoint variable associated with misfit).
            X: Current model state (used to ensure consistent adjoint mapping).
            
        Returns:
            Gradient in the model state space (reduced space).
        """

        # Define a wrapper for _misfit that computes the forward misfit
        def misfit_func(X):
            return self._misfit(t, X)

        # Compute the vector-Jacobian product (vjp) for the forward operation
        _, vjp_func = jax.vjp(misfit_func, X)

        # Use the vjp function to project the gradient from observation space to model state space
        grad_model_state, = vjp_func(grad_obs)

        return grad_model_state
    
    def misfit(self, t, State):

        X = self._observation_model_field(State)

        return self._misfit_jit(t, X.ravel() )
    
    def misfit_jax(self, t, State_var):

        X = self._observation_model_field(State_var).ravel()

        return self._misfit_jit(t, X)

    def scan_misfit_size(self):
        return int(self.varobs_arr.shape[1])

    def scan_misfit(self, t, State_var):
        """Return a fixed-size, zero-masked L3 misfit for every checkpoint."""
        if self.varobs_arr.shape[0] == 0:
            # Observation-free tiles still execute the forecast scan, but
            # have no observation cost contribution.
            return jnp.zeros((0,), dtype=jnp.asarray(t).dtype)
        raw = self.misfit_jax(t, State_var)
        return jnp.where(self.is_obs_time_jax(t), raw, jnp.zeros_like(raw))

    def scan_adj(self, t, adState_var, State_var, misfit):
        """Pure-pytree L3 adjoint used inside ``lax.scan``."""
        if self.varobs_arr.shape[0] == 0:
            return adState_var
        var = self._observation_model_field(State_var)
        adX = self._misfit_reduced_jit(t, misfit, var.ravel())
        result = dict(adState_var)
        if self.observation_role is not None and hasattr(self.model, 'add_observation_adjoint'):
            self.model.add_observation_adjoint(
                result, self.name_var, self.observation_role, adX.reshape(var.shape))
        else:
            name = self.name_mod_var[self.name_var]
            result[name] = result[name] + adX.reshape(result[name].shape)
        return result
    
    def adj(self, t, adState, State, misfit):

        # Compute adjoint operation of y = Hx
        var = self._observation_model_field(State)
        adX = self._misfit_reduced_jit(t, misfit, var.ravel())
        self._add_observation_adjoint(adState, adX.reshape(var.shape))
    
def _lwe_selected_sources(config):
    """Select only sources used by this operator (including multi-operators)."""
    sources = getattr(config, 'OBS', None) or {}
    selected = config.OBSOP.get('name_obs')
    if sources.get('super') is not None:
        if selected is not None and len(selected) != 1:
            raise ValueError('A single OBS source requires one name_obs')
        return {(selected[0] if selected else 'OBS'): sources}
    if selected is None:
        return sources
    missing = set(selected) - set(sources)
    if missing:
        raise ValueError(f'Unknown LWE observation sources: {sorted(missing)}')
    return {name: sources[name] for name in selected}


class Obsop_interp_lwe_l3(Obsop_interp_l3):
    """Native geometry and sparse whitened LWE projection; inversion is separate."""

    @classmethod
    def prepare_joint(cls, config, State, dict_obs, Model):
        """Prepare joint observations; LWE never enters the physical State."""
        operator = cls.prepare_projection(config, State, dict_obs, Model)
        operator._geometry_only = False
        operator._misfit_jit = jit(operator._misfit)
        return operator

    @classmethod
    def prepare_projection(cls, config, State, dict_obs, Model):
        """Prepare L and sqrt(Q), without enabling the observation cost.

        project_lwe returns metres of SSH, aligned with native_geometry_arr.
        A checkpoint argument instead selects its unpadded native rows.
        """
        operator = cls.prepare_geometry(config, State, dict_obs, Model)
        operator._prepare_lwe_projection()
        return operator

    @staticmethod
    def _sample_lwe_map(path, names, lon, lat, label, _maps=None):
        """Bilinear sampling of a rectilinear NetCDF or zipped Zarr map."""
        from scipy.interpolate import RegularGridInterpolator
        cache_key = (os.fspath(path), tuple(sorted(names.items())))
        if _maps is not None and cache_key in _maps:
            x, y, field = _maps[cache_key]
        else:
            if str(path).endswith('.zip'):
                # MIOST SAD is Zarr v2 with `dimensions` rather than xarray's
                # `_ARRAY_DIMENSIONS`. Read its chunks without a runtime MIOST
                # dependency; preserve fill_value=0 as a physical zero.
                import json
                import zipfile
                from itertools import product
                from numcodecs import get_codec

                def read_array(archive, name):
                    meta = json.loads(archive.read(name + '/.zarray'))
                    attrs = json.loads(archive.read(name + '/.zattrs'))
                    if meta.get('zarr_format') != 2:
                        raise ValueError(f'{label}: only Zarr v2 SAD maps are supported')
                    shape, chunks = tuple(meta['shape']), tuple(meta['chunks'])
                    dtype = np.dtype(meta['dtype'])
                    fill = meta.get('fill_value')
                    if isinstance(fill, str) and fill in ('NaN', 'Infinity', '-Infinity'):
                        fill = float(fill)
                    array = np.full(shape, 0 if fill is None else fill, dtype=dtype)
                    compressor = get_codec(meta['compressor']) if meta.get('compressor') else None
                    filters = [get_codec(codec) for codec in meta.get('filters') or []]
                    for index in product(*(range((n + c - 1) // c) for n, c in zip(shape, chunks))):
                        key = name + '/' + meta.get('dimension_separator', '.').join(map(str, index))
                        try:
                            data = archive.read(key)
                        except KeyError:
                            continue
                        if compressor is not None:
                            data = compressor.decode(data)
                        for codec in reversed(filters):
                            data = codec.decode(data)
                        block = np.frombuffer(data, dtype=dtype).reshape(chunks, order=meta.get('order', 'C'))
                        slices = tuple(slice(i*c, min((i+1)*c, n)) for i, c, n in zip(index, chunks, shape))
                        local = tuple(slice(0, s.stop-s.start) for s in slices)
                        array[slices] = block[local]
                    dims = attrs.get('_ARRAY_DIMENSIONS', attrs.get('dimensions'))
                    if not dims or len(dims) != len(shape):
                        raise ValueError(f'{label}: missing Zarr array dimensions for {name}')
                    result = xr.DataArray(array, dims=dims)
                    missing = attrs.get('_FillValue')
                    if missing is not None:
                        result = result.where(result != missing)
                    return result

                with zipfile.ZipFile(path) as archive:
                    variables = {name: read_array(archive, name) for name in names.values()}
                    mask_name = names['var'] + '_mask'
                    if mask_name + '/.zarray' in archive.namelist():
                        variables[mask_name] = read_array(archive, mask_name)
                    ds = xr.Dataset(variables)
            else:
                ds = xr.open_dataset(path)
            with ds:
                xvar, yvar = ds[names['lon']], ds[names['lat']]
                if xvar.ndim != 1 or yvar.ndim != 1 or xvar.dims == yvar.dims:
                    raise ValueError(f'{label}: map coordinates must be separate 1-D axes')
                value = ds[names['var']].transpose(yvar.dims[0], xvar.dims[0])
                x, y = np.asarray(xvar, dtype=float), np.asarray(yvar, dtype=float)
                field = np.asarray(value, dtype=float)
                mask_name = names['var'] + '_mask'
                if mask_name in ds:
                    mask = ds[mask_name].transpose(yvar.dims[0], xvar.dims[0])
                    field = np.where(np.asarray(mask, dtype=bool), np.nan, field)
            if (min(x.size, y.size) < 2 or not np.all(np.isfinite(x))
                    or not np.all(np.isfinite(y))):
                raise ValueError(f'{label}: map axes must contain at least two finite coordinates')
            ix, iy = np.argsort(x), np.argsort(y)
            x, y, field = x[ix], y[iy], field[np.ix_(iy, ix)]
            if np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0):
                raise ValueError(f'{label}: map axes must be strictly monotone without duplicates')
            # Accept global grids with either a duplicated seam or one missing endpoint.
            span = x[-1] - x[0]
            dx = np.diff(x)
            global_map = (np.isclose(span, 360., atol=1e-7, rtol=0.) or
                          (np.allclose(dx, dx[0]) and
                           np.isclose(span + dx[0], 360., atol=1e-7, rtol=0.)))
            if global_map and np.isclose(span, 360., atol=1e-7, rtol=0.):
                if not np.allclose(field[:, 0], field[:, -1], equal_nan=True):
                    raise ValueError(f'{label}: conflicting values at the duplicated longitude seam')
                x, field = x[:-1], field[:, :-1]
            if global_map:
                x = np.r_[x, x[0] + 360.]
                field = np.column_stack((field, field[:, 0]))
            if _maps is not None:
                _maps[cache_key] = (x, y, field)
        # Equivalent longitude conventions also work for regional maps.
        query_x = (np.asarray(lon) - x[0]) % 360. + x[0]
        query_y = np.asarray(lat)
        outside = ((query_x < x[0]) | (query_x > x[-1]) |
                   (query_y < y[0]) | (query_y > y[-1]) |
                   ~np.isfinite(query_x) | ~np.isfinite(query_y))
        if np.any(outside):
            i = int(np.flatnonzero(outside)[0])
            raise ValueError(f'{label}: center {i} outside map domain at '
                             f'lon={lon[i]}, lat={lat[i]}')
        sampled = RegularGridInterpolator((y, x), field, bounds_error=True)(
            np.column_stack((query_y, query_x)))
        invalid = ~np.isfinite(sampled) | (sampled < 0)
        if np.any(invalid):
            i = int(np.flatnonzero(invalid)[0])
            raise ValueError(f'{label}: invalid value {sampled[i]} at center {i} '
                             f'(lon={lon[i]}, lat={lat[i]})')
        return sampled

    def _prepare_lwe_projection(self):
        from .obs import _lwe_signature
        import hashlib
        from scipy.interpolate import interp1d

        def file_hash(path):
            digest = hashlib.sha256()
            with open(path, 'rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            return digest.hexdigest()

        paths = {source.path_lwe for source in self.lwe_sources.values()}
        if paths:
            paths.add(self.lwe_config.facQ_file)
        signature = _lwe_signature(dict(schema=1, geometry=self.geometry_signature,
            maps={path: file_hash(path) for path in sorted(paths)},
            common=self.lwe_config, sources=self.lwe_sources))
        self.projection_signature = signature
        self.projection_cache_path = os.path.join(self.path_save,
            'lwe_projection_v1_' + signature + '.pic')
        payload = None
        if self.write_op and not self.compute_op:
            try:
                with open(self.projection_cache_path, 'rb') as stream:
                    cached = pickle.load(stream)
                if (cached['version'] == 1 and cached['signature'] == signature
                        and cached['digest'] == self._payload_digest(cached)):
                    self._validate_lwe_payload(cached)
                    payload = cached
            except (OSError, ValueError, KeyError, TypeError, AttributeError,
                    IndexError, EOFError, pickle.UnpicklingError):
                pass
        if payload is None:
            geom = self.native_geometry_arr
            valid = geom['valid']
            flat = {key: value[valid] for key, value in geom.items() if key != 'valid'}
            keys = ('center_time_ns', 'center_lon', 'center_lat', 'source_id',
                    'segment_id', 'variance')
            pieces = {key: [] for key in keys}
            rows, columns, weights = [], [], []
            ncoeff = 0
            map_grids = {}  # Decode each auxiliary only once across all segments.
            halfwidth = float(self.lwe_config.tdec_lw)
            step = halfwidth / self.lwe_config.facnlt
            # MIOST's fixed phase is day 11930 since 1950, independent of window.
            reference_ns = int((np.datetime64('1950-01-01', 'ns') +
                               np.timedelta64(11930, 'D')).astype(np.int64))
            for name, source in self.lwe_sources.items():
                source_id = self.source_names.index(name)
                selected = (flat['source_id'] == source_id) & flat['lwe_eligible']
                for segment_id in np.unique(flat['segment_id'][selected]):
                    segment_rows = np.flatnonzero(selected & (flat['segment_id'] == segment_id))
                    native_ns = flat['time_ns'][segment_rows]
                    unique_ns, first = np.unique(native_ns, return_index=True)
                    origin = int(unique_ns[0])
                    times = (unique_ns - origin) / 86400e9
                    tref = (reference_ns - origin) / 86400e9
                    start = times[0] - (times[0] - tref) % step
                    centers = np.arange(start - 3 * step, times[-1] + 3 * step, step)
                    # Match MIOST's (center-4*step, center+4*step] selection.
                    keep = (np.searchsorted(times, centers + 4 * step, side='right') !=
                            np.searchsorted(times, centers - 4 * step, side='right'))
                    centers = centers[keep]
                    if not centers.size:
                        continue
                    unique_rows = segment_rows[first]
                    lon = np.rad2deg(np.unwrap(np.deg2rad(flat['lon'][unique_rows])))
                    lat = flat['lat'][unique_rows]
                    if not np.all(np.isfinite(lon)) or not np.all(np.isfinite(lat)):
                        raise ValueError(f'{name}, segment {segment_id}: nonfinite track coordinates')
                    if times.size == 1:
                        cx, cy = np.full(centers.size, lon[0] % 360.), np.full(centers.size, lat[0])
                    else:
                        cx = interp1d(times, lon, fill_value='extrapolate')(centers) % 360.
                        cy = interp1d(times, lat, fill_value='extrapolate')(centers)
                    cy = np.clip(cy, -90., 90.)
                    label = f'{name}, segment {segment_id}'
                    std = self._sample_lwe_map(source.path_lwe, source.name_var_lwe,
                                               cx, cy, label + ': ' + source.path_lwe, _maps=map_grids)
                    facq = self._sample_lwe_map(self.lwe_config.facQ_file,
                        self.lwe_config.name_var_facQ, cx, cy,
                        label + ': ' + self.lwe_config.facQ_file, _maps=map_grids)
                    with np.errstate(over='ignore', invalid='ignore'):
                        variance = facq * std**2
                    if not np.all(np.isfinite(variance)):
                        raise ValueError(f'{label}: nonfinite coefficient variance')
                    positive = variance > 0
                    centers, cx, cy, variance = (a[positive] for a in (centers, cx, cy, variance))
                    count = centers.size
                    values = (origin + np.rint(centers * 86400e9).astype(np.int64),
                              cx, cy, np.full(count, source_id, dtype=np.int64),
                              np.full(count, segment_id, dtype=np.int64), variance)
                    for key, value in zip(keys, values):
                        pieces[key].append(value)
                    # Sorted rows enable compact support searches without a dense matrix.
                    order = np.argsort(native_ns, kind='stable')
                    obs_times = (native_ns[order] - origin) / 86400e9
                    for j, center in enumerate(centers):
                        lo, hi = np.searchsorted(obs_times,
                            [center - halfwidth, center + halfwidth], side='right')
                        local = np.arange(lo, hi)
                        u = (obs_times[local] - center) / halfwidth
                        w = np.cos(.5 * np.pi * u)**2
                        rows.append(segment_rows[order[local]])
                        columns.append(np.full(local.size, ncoeff + j, dtype=np.int64))
                        weights.append(w)
                    ncoeff += count
            payload = {key: np.concatenate(value) if value else np.empty(0,
                       dtype=np.int64 if key.endswith('_id') or key == 'center_time_ns' else float)
                       for key, value in pieces.items()}
            for key, value, dtype in (('rows', rows, np.int64),
                                       ('columns', columns, np.int64), ('weights', weights, float)):
                payload[key] = np.concatenate(value).astype(dtype) if value else np.empty(0, dtype=dtype)
            payload.update(version=1, signature=signature)
            self._validate_lwe_payload(payload)
            payload['digest'] = self._payload_digest(payload)
            if self.write_op:
                temporary = self.projection_cache_path + f'.tmp-{os.getpid()}'
                try:
                    with open(temporary, 'wb') as stream:
                        pickle.dump(payload, stream)
                    os.replace(temporary, self.projection_cache_path)
                finally:
                    if os.path.exists(temporary):
                        os.remove(temporary)
        self._install_lwe_payload(payload)

    def _validate_lwe_payload(self, payload):
        n = payload['variance'].size
        for key in ('center_time_ns', 'center_lon', 'center_lat', 'source_id', 'segment_id', 'variance'):
            if payload[key].shape != (n,):
                raise ValueError('Invalid LWE coefficient metadata')
        for key in ('center_time_ns', 'source_id', 'segment_id'):
            if payload[key].dtype != np.dtype('int64'):
                raise ValueError('Invalid LWE coefficient identity dtype')
        if (not np.all(np.isfinite(payload['center_lon']))
                or not np.all(np.isfinite(payload['center_lat']))
                or np.any(np.abs(payload['center_lat']) > 90)
                or np.any(payload['source_id'] < 0)
                or np.any(payload['source_id'] >= len(self.source_names))
                or np.any(payload['segment_id'] < 0)):
            raise ValueError('Invalid LWE coefficient geometry')
        rows, cols, weights = (payload[key] for key in ('rows', 'columns', 'weights'))
        if (rows.ndim != 1 or cols.shape != rows.shape or weights.shape != rows.shape
                or rows.dtype != np.dtype('int64') or cols.dtype != np.dtype('int64')
                or np.any(rows < 0) or np.any(rows >= np.count_nonzero(self.native_geometry_arr['valid']))
                or np.any(cols < 0) or np.any(cols >= n)
                or not np.all(np.isfinite(weights)) or np.any(weights < 0)
                or not np.all(np.isfinite(payload['variance'])) or np.any(payload['variance'] <= 0)):
            raise ValueError('Invalid sparse LWE projection')
        valid = self.native_geometry_arr['valid']
        for key in ('source_id', 'segment_id'):
            if not np.array_equal(self.native_geometry_arr[key][valid][rows], payload[key][cols]):
                raise ValueError('LWE projection crosses source or segment identities')

    def _install_lwe_payload(self, payload):
        self.lwe_coefficients = {key: payload[key] for key in
            ('center_time_ns', 'center_lon', 'center_lat', 'source_id', 'segment_id', 'variance')}
        self.n_lwe = payload['variance'].size
        valid = self.native_geometry_arr['valid']
        self.lwe_matrix = csr_matrix((payload['weights'], (payload['rows'], payload['columns'])),
                                    shape=(int(valid.sum()), self.n_lwe))
        self.lwe_sqrtq = jnp.asarray(np.sqrt(payload['variance']))
        self._lwe_columns = jnp.asarray(payload['columns'])
        self._lwe_weights = jnp.asarray(payload['weights'])
        # Map compact true rows to the existing padded observation array layout.
        self._lwe_padded_rows = jnp.asarray(np.flatnonzero(valid)[payload['rows']])
        self._lwe_shape = valid.shape
        self._lwe_views = {}
        offsets = np.r_[0, np.cumsum(np.asarray(self.n_obs, dtype=np.int64))]
        for i, t in enumerate(self.t_obs):
            select = (payload['rows'] >= offsets[i]) & (payload['rows'] < offsets[i + 1])
            self._lwe_views[t] = (jnp.asarray(payload['rows'][select] - offsets[i]),
                                  jnp.asarray(payload['columns'][select]),
                                  jnp.asarray(payload['weights'][select]), int(offsets[i + 1] - offsets[i]))

    def project_lwe(self, z, t=None):
        """L sqrt(Q) z in metres SSH; all padded rows or one unpadded checkpoint."""
        z = jnp.asarray(z)
        if z.shape != (self.n_lwe,):
            raise ValueError(f'LWE control must have shape {(self.n_lwe,)}')
        if t is None:
            rows, cols, weights = self._lwe_padded_rows, self._lwe_columns, self._lwe_weights
            size = math.prod(self._lwe_shape)
        else:
            rows, cols, weights, size = self._lwe_views[t]
        result = jnp.zeros(size, dtype=jnp.result_type(z, weights)).at[rows].add(
            weights * self.lwe_sqrtq[cols] * z[cols])
        return result.reshape(self._lwe_shape) if t is None else result

    def adjoint_lwe(self, residual, t=None):
        """sqrt(Q) L.T residual, without observation-error weighting."""
        residual = jnp.asarray(residual)
        if t is None:
            if residual.shape != self._lwe_shape:
                raise ValueError(f'LWE residual must have shape {self._lwe_shape}')
            rows, cols, weights = self._lwe_padded_rows, self._lwe_columns, self._lwe_weights
        else:
            rows, cols, weights, size = self._lwe_views[t]
            if residual.shape != (size,):
                raise ValueError(f'LWE checkpoint residual must have shape {(size,)}')
        return jnp.zeros(self.n_lwe, dtype=jnp.result_type(residual, weights)).at[cols].add(
            weights * self.lwe_sqrtq[cols] * residual.ravel()[rows])

    def scan_adjoint_lwe(self, t, residual):
        """Transpose of the padded SSH projection at a possibly traced time."""
        if not self._lwe_shape[0] or not self._lwe_shape[1]:
            return jnp.zeros(self.n_lwe, dtype=jnp.asarray(residual).dtype)
        width = self._lwe_shape[1]
        batch = self._lwe_padded_rows // width
        row = self._lwe_padded_rows % width
        weight = jnp.where(self.t_obs_jax[batch] == t, self._lwe_weights, 0.)
        return jnp.zeros(self.n_lwe, dtype=jnp.result_type(residual, weight)).at[
            self._lwe_columns].add(weight * self.lwe_sqrtq[self._lwe_columns] * residual[row])

    @staticmethod
    def resolve_config(config):
        from .config_default import OBSOP_INTERP_LWE_L3
        from .exp import Config

        common = Config(deepcopy(OBSOP_INTERP_LWE_L3))
        common.update(deepcopy(config.OBSOP))
        if common.name_var != 'SSH':
            raise ValueError('OBSOP_INTERP_LWE_L3 requires name_var=SSH')
        for key in ('tdec_lw', 'facnlt', 'gap_seconds'):
            value = common[key]
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f'{key} must be finite and positive')
        if (isinstance(common.min_segment_size, bool)
                or not isinstance(common.min_segment_size, Integral)
                or common.min_segment_size < 1):
            raise ValueError('min_segment_size must be a positive integer')
        if type(common.p_param) is not int or common.p_param != 0:
            raise ValueError('LWE requires p_param=0 (nadir)')

        def descriptor(value, label):
            if not isinstance(value, dict) or any(
                    not isinstance(value.get(key), str) or not value[key].strip()
                    for key in ('lon', 'lat', 'var')):
                raise ValueError(f'{label} requires nonempty lon, lat, var names')

        def map_path(path, label, directory=None):
            if path is None or not os.fspath(path).strip():
                raise ValueError(f'{label} requires an explicit filename')
            path = os.fspath(path)
            if directory is not None and not os.path.isabs(path):
                path = os.path.join(directory, path)
            if not os.path.isfile(path):
                raise ValueError(f'{label} does not exist: {path}')
            return path

        missions = {}
        for name, source in _lwe_selected_sources(config).items():
            enabled = source.get('lwe_enabled', False)
            if type(enabled) is not bool:
                raise ValueError(f'{name}.lwe_enabled must be a boolean')
            if not enabled:
                continue
            if source.get('super') != 'OBS_SSH_NADIR':
                raise ValueError(f'{name}: LWE supports OBS_SSH_NADIR only')
            from .obs import _validate_lwe_sampling
            _validate_lwe_sampling(source)
            # The existing loader evaluates this expression; never subtract again.
            names = (source.get('name_var') or {}).get('SSH')
            factors = (source.get('combine_var') or {}).get('SSH')
            if (not isinstance(names, (list, tuple))
                    or list(names) != ['sla_unfiltered', 'lwe']
                    or not isinstance(factors, (list, tuple))
                    or list(factors) != [1, -1]):
                raise ValueError(f'{name}: before DUACS requires '
                                 'sla_unfiltered - lwe via name_var/combine_var')
            from .config_default import OBS_SSH_NADIR
            names = deepcopy(source.get('name_var_lwe', OBS_SSH_NADIR['name_var_lwe']))
            descriptor(names, f'{name}.name_var_lwe')
            missions[name] = Config(
                path_lwe=map_path(source.get('path_lwe'), f'{name}.path_lwe',
                                  source.get('sad_path')),
                name_var_lwe=names,
            )
        if missions:
            descriptor(common.name_var_facQ, 'name_var_facQ')
            common.facQ_file = map_path(common.facQ_file, 'facQ_file')
        return common, missions

    @classmethod
    def prepare_geometry(cls, config, State, dict_obs, Model):
        """Prepare native rows and spatial H only; this is not an inversion.

        Native time is int64 nanoseconds since Unix epoch, on the host. Mission
        is the configured OBS key. Row IDs index that source's signed input
        dataset before spatial/checkpoint selection. No duplicate times are lost.
        """
        from .obs import Obs, _LweGeometryMissing

        def prepare(observations):
            operator = cls.__new__(cls)
            operator.lwe_config, operator.lwe_sources = cls.resolve_config(config)
            operator.source_configs = deepcopy(_lwe_selected_sources(config))
            operator.source_names = tuple(operator.source_configs)
            operator._geometry_only = True
            Obsop_interp_l3.__init__(operator, config, State, observations, Model)
            operator.process_obs()
            return operator

        try:
            return prepare(dict_obs)
        except (_LweGeometryMissing, FileNotFoundError):
            # Legacy prepared files cannot recover native times from checkpoints.
            fresh_config = deepcopy(config)
            fresh_config.EXP.compute_obs = True
            return prepare(Obs(fresh_config, State))

    _geometry_fields = ('time_ns', 'source_id', 'row_id', 'segment_id',
                        'lwe_eligible', 'lon', 'lat')

    @staticmethod
    def _payload_digest(payload):
        import hashlib
        digest = hashlib.sha256()
        for key, value in sorted(payload.items()):
            if isinstance(value, np.ndarray):
                digest.update(key.encode())
                digest.update(str((value.dtype.str, value.shape)).encode())
                digest.update(np.ascontiguousarray(value).tobytes())
        return digest.hexdigest()

    def _read_geometry_cache(self, path, stacked=False):
        try:
            with open(path, 'rb') as stream:
                payload = pickle.load(stream)
            if (not isinstance(payload, dict) or payload.get('version') != 1
                    or payload.get('signature') != self.geometry_signature
                    or payload.get('digest') != self._payload_digest(payload)):
                return None
            rank = 2 if stacked else 1
            shape = payload['var_obs'].shape
            if (len(shape) != rank or payload['err_obs'].shape != shape
                    or any(payload[key].shape != shape for key in self._geometry_fields)
                    or any(payload[key].dtype != np.dtype('int64') for key in
                           ('time_ns', 'source_id', 'row_id', 'segment_id'))
                    or payload['lwe_eligible'].dtype != np.dtype('bool')):
                return None
            n_obs = payload['n_obs'] if stacked else np.array([shape[0]])
            n_data = payload['n_data'] if stacked else np.array([payload['data'].size])
            if stacked and n_obs.shape != (len(self.t_obs),):
                return None
            if stacked:
                if (n_data.shape != n_obs.shape or shape[0] != len(self.t_obs)
                        or np.any(n_obs < 0) or np.any(n_obs > shape[1])
                        or np.any(n_data < 0) or np.any(n_data > payload['data'].shape[1])
                        or payload['indices'].shape != (shape[0], 2, payload['data'].shape[1])
                        or payload['valid'].shape != shape
                        or not np.array_equal(payload['valid'], np.arange(shape[1])[None, :] < n_obs[:, None])):
                    return None
            elif payload['indices'].shape != (2, payload['data'].size):
                return None
            for i, (nr, nd) in enumerate(zip(n_obs, n_data)):
                ix = payload['indices'][i, :, :nd] if stacked else payload['indices']
                sources = payload['source_id'][i, :nr] if stacked else payload['source_id']
                if (np.any(sources < 0) or np.any(sources >= len(self.source_names))
                        or (nd and (np.any(ix < 0) or np.any(ix[0] >= nr)
                                    or np.any(ix[1] >= len(self.coords_geo))))):
                    return None
            return payload
        except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError,
                EOFError, pickle.UnpicklingError):
            return None

    def _write_geometry_cache(self, path, payload):
        payload = dict(payload, version=1, signature=self.geometry_signature)
        payload['digest'] = self._payload_digest(payload)
        temporary = f'{path}.tmp-{os.getpid()}'
        try:
            with open(temporary, 'wb') as stream:
                pickle.dump(payload, stream)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)

    def _read_native_batch(self, date):
        from .obs import _lwe_native_rows, _lwe_signature, _LweGeometryMissing
        lists = {key: [] for key in self._geometry_fields + ('var_obs', 'err_obs')}
        batch = self.dict_obs[date]
        for name, info, path in zip(batch['obs_name'], batch['attributes'], batch['obs_path']):
            if name not in self.source_names or self.name_var not in info['name_var']:
                continue
            with _open_obs_file(path) as ds:
                if self.name_var not in ds:
                    continue
                var = ds[self.name_var].values.ravel()
                lon = ds[info['name_lon']].values.ravel()
                lat = ds[info['name_lat']].values.ravel()
                if name in self.lwe_sources:
                    if _lwe_signature(info) != _lwe_signature(self.source_configs[name]):
                        raise _LweGeometryMissing('Prepared observation configuration has changed')
                    native = _lwe_native_rows(ds, name, input_signature=
                        self.input_signatures[name], source_signature=
                        _lwe_signature(self.source_configs[name]))
                    time_ns, row_id = native['lwe_time_ns'], native['lwe_row_id']
                    actual = ds[info['name_time']].values.astype('datetime64[ns]').astype(np.int64).ravel()
                    if not np.array_equal(actual, time_ns):
                        raise _LweGeometryMissing('Prepared native time differs from observation coordinate')
                else:
                    # Disabled sources still occupy spatial H rows. No LWE segment.
                    time_ns = ds[info['name_time']].values.astype('datetime64[ns]').astype(np.int64).ravel()
                    row_id = np.arange(var.size, dtype=np.int64)
                err_name = self.name_var + '_err'
                noise = info.get('sigma_noise')
                err = (ds[err_name].values.ravel() if err_name in ds else
                       np.full(var.size, 1. if noise is None else noise))
                values = dict(time_ns=time_ns, row_id=row_id, lon=lon, lat=lat,
                              source_id=np.full(var.size, self.source_names.index(name), dtype=np.int64),
                              segment_id=np.full(var.size, -1, dtype=np.int64),
                              lwe_eligible=np.zeros(var.size, dtype=bool), var_obs=var, err_obs=err)
                if any(value.shape != var.shape for value in values.values()):
                    raise ValueError(f'{name}: native metadata and SSH rows must align (nadir only)')
                for key, value in values.items():
                    lists[key].append(value)
        return {key: (np.concatenate(values) if values else np.empty(
            0, dtype=bool if key == 'lwe_eligible' else
            np.int64 if key.endswith('_id') or key == 'time_ns' else float))
                for key, values in lists.items()}

    def _segment_native_batches(self, batches):
        """Segment each mission globally, never independently at checkpoints."""
        if not batches:
            return
        sizes = [batch['time_ns'].size for batch in batches]
        times = np.concatenate([batch['time_ns'] for batch in batches])
        sources = np.concatenate([batch['source_id'] for batch in batches])
        # Disabled missions have no native prepared IDs; give their rows a
        # deterministic source-local index across the signed batch sequence.
        for name in set(self.source_names) - set(self.lwe_sources):
            offset = 0
            for batch in batches:
                selected = batch['source_id'] == self.source_names.index(name)
                count = np.count_nonzero(selected)
                batch['row_id'][selected] = np.arange(offset, offset + count, dtype=np.int64)
                offset += count
        segments = np.full(times.size, -1, dtype=np.int64)
        eligible = np.zeros(times.size, dtype=bool)
        for name in self.lwe_sources:
            selected = np.flatnonzero(sources == self.source_names.index(name))
            # NaT is retained as an ocean row, but cannot define LWE geometry.
            selected = selected[times[selected] != np.iinfo(np.int64).min]
            unique, inverse = np.unique(times[selected], return_inverse=True)
            segment = np.cumsum(np.r_[False, np.diff(unique) > self.lwe_config.gap_seconds * 1e9]).astype(np.int64)
            if not unique.size:
                continue
            counts = np.bincount(segment)
            segments[selected] = segment[inverse]
            eligible[selected] = counts[segment[inverse]] >= self.lwe_config.min_segment_size
        start = 0
        for batch, size in zip(batches, sizes):
            batch['segment_id'] = segments[start:start + size]
            batch['lwe_eligible'] = eligible[start:start + size]
            start += size

    def process_obs(self, var_bc=None):
        if not getattr(self, '_geometry_only', False):
            if not hasattr(self, 'native_geometry_arr'):
                return super().process_obs(var_bc)  # LWE disabled.
            # Prepared native rows stay in their signed order; ordinary L3
            # caches must never replace these arrays during driver setup.
            if not hasattr(self, '_lwe_raw_varobs'):
                self._lwe_raw_varobs = np.asarray(self.varobs_arr).copy()
            values = self._lwe_raw_varobs.copy()
            if var_bc is not None and self.name_var in var_bc:
                mask = np.any(np.isnan(self.coords_geo), axis=1)
                for i, t in enumerate(self.t_obs):
                    valid = self.native_geometry_arr['valid'][i]
                    points = np.column_stack((self.native_geometry_arr['lon'][i, valid],
                                              self.native_geometry_arr['lat'][i, valid]))
                    values[i, valid] -= griddata(self.coords_geo[~mask],
                        var_bc[self.name_var][i].ravel()[~mask], points, method='cubic')
            self.varobs_arr = jnp.asarray(values)
            self._misfit_jit = jit(self._misfit)
            self._misfit_reduced_jit = jit(self._misfit_reduced)
            return
        from .obs import _lwe_signature, _lwe_file_inventory, _lwe_source_inventory
        paths = [path for date in self.date_obs
                 for name, path in zip(self.dict_obs[date]['obs_name'], self.dict_obs[date]['obs_path'])
                 if name in self.source_names]
        inputs = {name: _lwe_source_inventory(source) for name, source in self.source_configs.items()}
        self.input_signatures = {name: _lwe_signature(files) for name, files in inputs.items()}
        # H also depends on the exact grid, masking and interpolation settings.
        self.geometry_signature = _lwe_signature(dict(
            schema=1, sources=self.source_configs, source_names=self.source_names,
            enabled=tuple(self.lwe_sources),
            inputs=inputs,
            files=_lwe_file_inventory(paths), dates=self.date_obs, model_times=self.t_obs.tolist(),
            common=self.lwe_config, coords=self._payload_digest({'grid': self.coords_geo,
                'mask': self.obs_mask}), npix=self.Npix, dmax=self.dmax))
        prefix = os.path.join(self.path_save, 'lwe_geometry_v1_' + self.geometry_signature)
        self.geometry_cache_path = prefix + '_stacked.pic'
        stacked = (self._read_geometry_cache(self.geometry_cache_path, stacked=True)
                   if self.write_op and not self.compute_op and var_bc is None else None)
        if stacked is not None:
            self._install_geometry_arrays(stacked)
            return
        batches = []
        for date in self.date_obs:
            path = prefix + '_' + date.strftime('%Y%m%d_%H%M%S%f') + '.pic'
            cached = self._read_geometry_cache(path) if self.write_op and not self.compute_op else None
            batch = cached if cached is not None else self._read_native_batch(date)
            if cached is None:
                data, indices = self._sparse_op(batch['lon'], batch['lat']) if batch['lon'].size else (np.empty(0), np.empty((2, 0), dtype=np.int32))
                batch['data'] = np.asarray(data)
                batch['indices'] = np.asarray(indices, dtype=np.int32).reshape(2, -1)
            batches.append(batch)
        self._segment_native_batches(batches)
        if self.write_op:
            for date, batch in zip(self.date_obs, batches):
                self._write_geometry_cache(prefix + '_' + date.strftime('%Y%m%d_%H%M%S%f') + '.pic', batch)
        # Boundary subtraction never enters the persistent observation caches.
        if var_bc is not None and self.name_var in var_bc:
            for i, batch in enumerate(batches):
                batch['var_obs'] = batch['var_obs'].copy()
                mask = np.any(np.isnan(self.coords_geo), axis=1)
                batch['var_obs'] -= griddata(self.coords_geo[~mask],
                    var_bc[self.name_var][i].ravel()[~mask], np.column_stack((batch['lon'], batch['lat'])), method='cubic')
        n_obs = np.array([batch['var_obs'].size for batch in batches], dtype=np.int64)
        n_data = np.array([batch['data'].size for batch in batches], dtype=np.int64)
        nr, nd = int(n_obs.max(initial=0)), int(n_data.max(initial=0))
        stacked = dict(n_obs=n_obs, n_data=n_data,
                       data=np.zeros((len(batches), nd)),
                       indices=np.zeros((len(batches), 2, nd), dtype=np.int32),
                       valid=np.arange(nr)[None, :] < n_obs[:, None])
        for key in self._geometry_fields + ('var_obs', 'err_obs'):
            dtype = (bool if key == 'lwe_eligible' else np.int64
                     if key.endswith('_id') or key == 'time_ns' else float)
            fill = -1 if key.endswith('_id') else 1e7 if key == 'err_obs' else 0
            if key == 'time_ns':
                fill = np.iinfo(np.int64).min
            stacked[key] = np.full((len(batches), nr), fill, dtype=dtype)
        for i, batch in enumerate(batches):
            stacked['data'][i, :n_data[i]] = batch['data']
            stacked['indices'][i, :, :n_data[i]] = batch['indices']
            for key in self._geometry_fields + ('var_obs', 'err_obs'):
                stacked[key][i, :n_obs[i]] = batch[key]
        if self.write_op and var_bc is None:
            self._write_geometry_cache(self.geometry_cache_path, stacked)
        self._install_geometry_arrays(stacked)

    def _install_geometry_arrays(self, payload):
        # Keep exact geometry on host regardless of jax_enable_x64.
        self.native_geometry_arr = {key: payload[key] for key in self._geometry_fields}
        self.native_geometry_arr['valid'] = payload['valid']
        self.native_geometry = {t: {key: value[i, :payload['n_obs'][i]]
            for key, value in self.native_geometry_arr.items()}
            for i, t in enumerate(self.t_obs)}
        for attr, key in (('n_data', 'n_data'), ('n_obs', 'n_obs'),
                          ('data_arr', 'data'), ('indices_arr', 'indices'),
                          ('varobs_arr', 'var_obs'), ('errobs_arr', 'err_obs')):
            setattr(self, attr, jnp.asarray(payload[key]))

    def observation_mask(self):
        """Rows usable by both ocean and LWE likelihoods, without dropping IDs."""
        return (jnp.asarray(self.native_geometry_arr['valid'])
                & jnp.isfinite(self.varobs_arr) & jnp.isfinite(self.errobs_arr)
                & (self.errobs_arr > 0))

    def _misfit(self, t, X):
        if getattr(self, '_geometry_only', False):
            raise NotImplementedError('LWE geometry/projection only: joint inversion cost requires prepare_joint')
        if not hasattr(self, 'native_geometry_arr'):
            return super()._misfit(t, X)  # Disabled LWE keeps ordinary L3 behavior.
        index = jnp.where(self.t_obs_jax == t, size=1)[0][0]
        valid = self.observation_mask()[index]
        prediction = self.explicit_proj_operation(
            self.data_arr[index], self.indices_arr[index],
            X * self.model_to_observation_scale, self.varobs_arr.shape[1])
        # Sanitize before division so missing errors cannot poison autodiff.
        sigma = jnp.where(valid, self.errobs_arr[index], 1.)
        residual = prediction - jnp.where(valid, self.varobs_arr[index], 0.)
        return (1 / sigma) * jnp.where(valid & ~jnp.isnan(residual), residual, 0.)

    def __init__(self, config, State, dict_obs, Model):
        self.lwe_config, self.lwe_sources = self.resolve_config(config)
        if self.lwe_sources:
            prepared = self.prepare_joint(config, State, dict_obs, Model)
            self.__dict__.update(prepared.__dict__)
            self._misfit_jit = jit(self._misfit)
            self._misfit_reduced_jit = jit(self._misfit_reduced)
            return
        super().__init__(config, State, dict_obs, Model)


class Obsop_interp_l4(Obsop_interp):

    def __init__(self,config,State,dict_obs,Model):

        super().__init__(config,State,dict_obs,Model)

        # Date obs
        self.name_var = config.OBSOP.name_var
        self.model_to_observation_scale = self._model_to_observation_scale(Model, self.name_var).reshape(-1)
        self.name_var_obs = {}
        self.name_obs = []
        self.t_obs = [] 
        date_obs = list(dict_obs.keys())
        for t,timestamp in zip(Model.T,Model.timestamps):
            delta_t = [(timestamp - date).total_seconds() for date in date_obs]
            if len(delta_t)>0:
                
                if np.min(np.abs(delta_t))<=Model.dt/2:
                    
                    ind_obs = np.argmin(np.abs(delta_t))

                    for obs_name, sat_info in zip(dict_obs[date_obs[ind_obs]]['obs_name'], 
                                                  dict_obs[date_obs[ind_obs]]['attributes']):
                        
                        if (self.name_var in sat_info['name_var']) and ((config.OBSOP.name_obs is None) or (config.OBSOP.name_obs is not None and obs_name in config.OBSOP.name_obs)):
                            if obs_name not in self.name_obs:
                                self.name_obs.append(obs_name)
                            if t not in self.t_obs:
                                self.date_obs.append(date_obs[ind_obs])
                                self.t_obs.append(t)
    
        # For grid interpolation:
        self.interp_method = config.OBSOP.interp_method
        self.dist_min = .5*np.sqrt(State.dx**2+State.dy**2)*1e-3 # Minimum distance to consider an observation inside a model pixel

        self.DX = State.DX
        self.DY = State.DY

        self.DX = State.DX
        self.DY = State.DY
        
        # Mask land and sponge cells for observations only.
        self.mask = self.obs_mask

        # Misfit on gradients
        self.gradients = config.OBSOP.gradients
        if self.gradients:
            self.name_H += f'_L4_grad_{self.name_var}_{config.OBSOP.interp_method}'
        else:
            self.name_H += f'_L4_{self.name_var}_{config.OBSOP.interp_method}'
        
        self.t_obs = np.array(self.t_obs)
        self.t_obs_jax = jnp.array(self.t_obs)

        self._misfit_reduced_jit = jit(self._misfit_reduced)
        self._misfit_jit = jit(self._misfit)

        # Filled from the first processed date. Source coordinates are checked
        # exactly at runtime before any geometry-dependent mapping is reused.
        self._interp_source_coords = None
        self._linear_weights = None
        self._linear_outside_hull = None
        self._nearest_source_indices = None
        self._cubic_triangulation = None
        self._block_source_cells = None
        self._block_coord_valid = None
        self._block_target_tree = None

        self.flag_plot = config.EXP.flag_plot

    def process_obs(self, var_bc=None):
        
        stacked_cache = _stacked_obs_cache_path(self, 'l4')
        if (
            var_bc is None
            and not self.compute_op
            and stacked_cache is not None
            and os.path.exists(stacked_cache)
        ):
            with open(stacked_cache, 'rb') as stream:
                cached = pickle.load(stream)
            self.date_obs = cached['date_obs']
            self.t_obs = np.asarray(cached['t_obs'])
            self.t_obs_jax = jnp.asarray(self.t_obs)
            if self.gradients:
                self.varobs_grady = cached['varobs_grady']
                self.varobs_gradx = cached['varobs_gradx']
            else:
                self.varobs = cached['varobs']
            self.errobs = cached['errobs']
            self.varobs_arr = jnp.asarray(
                self.varobs_grady if self.gradients else self.varobs
            )
            self.errobs_arr = jnp.asarray(self.errobs)
            mask = (
                jnp.isnan(self.varobs_arr)
                | jnp.isnan(self.errobs_arr)
                | (self.errobs_arr < 1e-7)
                | (self.varobs_arr > 1e7)
            )
            self.varobs_arr = jnp.where(mask, 0., self.varobs_arr)
            self.errobs_arr = jnp.where(mask, 1e15, self.errobs_arr)
            return

        # Initialize dictionnaries
        if self.gradients:
            self.varobs_grady = np.zeros((self.t_obs.size, self.DX.size))
            self.varobs_gradx = np.zeros((self.t_obs.size, self.DX.size))
        else:
            self.varobs = np.zeros((self.t_obs.size, self.DX.size))
        self.errobs = np.zeros((self.t_obs.size, self.DX.size))

        #############################
        # Loop on observation dates #
        #############################
        for i,(date,t) in enumerate(zip(self.date_obs,self.t_obs)):

            print(f"Processing observations at date {date} for variable {self.name_var}...")

            sat_info_list = self.dict_obs[date]['attributes']
            obs_file_list = self.dict_obs[date]['obs_path']
            obs_name_list = self.dict_obs[date]['obs_name']

            # Concatenate obs from different sensors
            lon_obs = []
            lat_obs = []
            var_obs = []
            err_obs = []

            # The L4 cache already contains the fully interpolated and masked
            # fields.  Reuse it before reopening every source NetCDF file.
            file_L4 = (
                f"{self.path_save}/{self.name_H}_{'_'.join(self.name_obs)}_"
                f"{date.strftime('%Y%m%d_%H%M')}.pic"
            )
            cached_L4 = (
                not self.compute_op
                and self.write_op
                and os.path.exists(file_L4)
            )
            if cached_L4:
                with open(file_L4, "rb") as f:
                    var_obs_interp, err_obs_interp = pickle.load(f)
                var_obs_interp = np.asarray(var_obs_interp).copy()
                err_obs_interp = np.asarray(err_obs_interp).copy()
                if var_bc is not None and self.name_var in var_bc:
                    var_obs_interp -= var_bc[self.name_var][i]
                if self.gradients:
                    var_obs_interp_grady = np.full_like(
                        var_obs_interp, np.nan
                    )
                    var_obs_interp_gradx = np.full_like(
                        var_obs_interp, np.nan
                    )
                    var_obs_interp_grady[1:-1, 1:-1] = (
                        var_obs_interp[2:, 1:-1]
                        - var_obs_interp[:-2, 1:-1]
                    ) / (2 * self.DY[1:-1, 1:-1])
                    var_obs_interp_gradx[1:-1, 1:-1] = (
                        var_obs_interp[1:-1, 2:]
                        - var_obs_interp[1:-1, :-2]
                    ) / (2 * self.DX[1:-1, 1:-1])
                    self.varobs_grady[i] = var_obs_interp_grady.ravel()
                    self.varobs_gradx[i] = var_obs_interp_gradx.ravel()
                    self.errobs[i] = (
                        .5 * err_obs_interp
                        / np.sqrt(self.DY ** 2 + self.DX ** 2)
                    ).ravel()
                else:
                    self.varobs[i] = var_obs_interp.ravel()
                    self.errobs[i] = err_obs_interp.ravel()
                continue

            ####################
            # Merge observations
            ####################
            for sat_info, obs_file, obs_name in zip(sat_info_list, obs_file_list, obs_name_list):

                if obs_name not in self.name_obs:
                    continue

                try:
                
                    with _open_obs_file(obs_file) as ncin:

                        lon = ncin[sat_info['name_lon']].values
                        lat = ncin[sat_info['name_lat']].values

                        # Check if this observation class is wanted
                        if self.name_var not in ncin :
                            continue

                        # Observed variable
                        var = ncin[self.name_var].values

                        if lon.size != var.size and len(lon.shape)==1: # 2D regular grid
                            lon, lat = np.meshgrid(lon, lat)

                        # Observed error
                        name_err = self.name_var + '_err'
                        if name_err in ncin:
                            err = ncin[name_err].values
                        elif sat_info['sigma_noise'] is not None:
                            err = sat_info['sigma_noise'] * np.ones_like(var)
                        else:
                            err = np.ones_like(var)
                        err[np.isnan(var)] = np.nan

                        # Representativeness inflation when obs pixels are
                        # finer than the model grid (averaging ~_err_res
                        # independent samples per cell -> sqrt(N) reduction
                        # cancels into a sqrt(_err_res) inflation of the
                        # per-cell error under the i.i.d. assumption).
                        # Skip when obs are coarser than the grid: the same
                        # obs is reused across neighbouring cells, so the
                        # per-cell noise stays at the sensor noise (spatial
                        # correlation between cells is not represented here).
                        dx, dy = grid.lonlat2dxdy(lon,lat)
                        _err_res = np.nanmean(dx * dy) / np.nanmean(self.DX * self.DY)
                        if _err_res>1:
                            err *= np.sqrt(_err_res)
                                        
                        # Append to lists
                        var_obs.append(+var.flatten())
                        err_obs.append(+err.flatten())
                        lon_obs.append(+lon.flatten())
                        lat_obs.append(+lat.flatten())
                except:
                    print(f"Warning: problem reading {obs_file}")
                    continue
            
            if len(var_obs)==0:
                # remove date from list
                print(f"Warning: no observation found at date {date} for variable {self.name_var}, skipping this date.")
                self.date_obs.pop(i)
                self.t_obs = np.delete(self.t_obs, i)
                self.t_obs_jax = jnp.delete(self.t_obs_jax, i)
                self.varobs = np.delete(self.varobs, i, axis=0)
                self.errobs = np.delete(self.errobs, i, axis=0)
                continue
            
            # Concatenations of lists
            var_obs = np.concatenate(var_obs)
            err_obs = np.concatenate(err_obs)
            lon_obs = np.concatenate(lon_obs)
            lat_obs = np.concatenate(lat_obs)
            
            ################
            # Process L4 obs
            ################
            # No cache was available above: compute the interpolation.
            if not cached_L4:
                # Grid interpolation: performing spatial interpolation now
                # Loop on different obs for this date and this variable name
                _coords_obs = np.column_stack((lon_obs, lat_obs))
                same_source_grid = (
                    self._interp_source_coords is not None
                    and self._interp_source_coords.shape == _coords_obs.shape
                    and np.array_equal(
                        self._interp_source_coords,
                        _coords_obs,
                        equal_nan=True,
                    )
                )
                if not same_source_grid:
                    self._interp_source_coords = _coords_obs.copy()

                if self.interp_method=='hybrid':
                    if not same_source_grid:
                        self._nearest_source_indices = cKDTree(
                            _coords_obs
                        ).query(self.coords_geo)[1]
                        self._linear_weights, self._linear_outside_hull = (
                            _linear_barycentric_weights(
                                _coords_obs, self.coords_geo
                            )
                        )

                    source_values = np.column_stack((var_obs, err_obs))
                    interpolated = source_values[
                        self._nearest_source_indices
                    ].copy()
                    interpolated_linear = self._linear_weights @ source_values
                    interpolated_linear[self._linear_outside_hull] = np.nan
                    valid_linear = ~np.isnan(interpolated_linear)
                    interpolated[valid_linear] = interpolated_linear[valid_linear]
                    _var_obs_interp = interpolated[:, 0]
                    _err_obs_interp = interpolated[:, 1]
                
                elif self.interp_method == 'block_mean':
                    # Proper L4 swath -> grid projection: assign each obs
                    # pixel to its nearest target cell and average inside
                    # the cell. Yields a constant value within the swath
                    # for a constant input (e.g. flat noise field).
                    # Cell-wise error: obs error / sqrt(N_in_cell), assuming
                    # decorrelated per-pixel noise.
                    if self._block_target_tree is None:
                        self._block_target_tree = KDTree(self.coords_geo)
                    if not same_source_grid:
                        self._block_coord_valid = ~(
                            np.isnan(lon_obs) | np.isnan(lat_obs)
                        )
                        self._block_source_cells = np.full(
                            lon_obs.size, -1, dtype=np.intp
                        )
                        self._block_source_cells[self._block_coord_valid] = (
                            self._block_target_tree.query(
                                _coords_obs[self._block_coord_valid]
                            )[1]
                        )
                    _valid = self._block_coord_valid & ~np.isnan(var_obs)
                    _idx = self._block_source_cells[_valid]
                    _ncell = self.coords_geo.shape[0]
                    _sum_var = np.bincount(_idx, weights=var_obs[_valid],
                                           minlength=_ncell)
                    _sum_err2 = np.bincount(_idx,
                                            weights=err_obs[_valid] ** 2,
                                            minlength=_ncell)
                    _count = np.bincount(_idx, minlength=_ncell).astype(float)
                    _empty = (_count == 0)
                    _count_safe = np.where(_empty, 1.0, _count)
                    # Cell-wise error: assume correlated per-pixel noise
                    # (conservative) -> err_cell = sqrt(<err^2>) (no /sqrt(N)).
                    _var_obs_interp = np.where(_empty, np.nan,
                                               _sum_var / _count_safe)
                    _err_obs_interp = np.where(
                        _empty, np.nan,
                        np.sqrt(_sum_err2 / _count_safe))

                elif self.interp_method=='rtree': 

                    def _regrid_unstructured(lon_target, lat_target, lon, lat, var):
                        
                        # Spatial interpolation 
                        mesh = pyinterp.RTree() 
                        if len(lon_target.shape)==1:
                            lon_target, lat_target = np.meshgrid(lon_target, lat_target)
                        lons = lon.ravel()
                        lats = lat.ravel()
                        var_regridded = np.zeros((lat_target.shape[0],lon_target.shape[1])) 
                        data = np.array(var) 
                        mask = np.isnan(lons) | np.isnan(lats) | np.isnan(data) 
                        data = data[~mask]
                        mesh.packing(np.vstack((lons[~mask], lats[~mask])).T, data)
                        idw, _ = mesh.radial_basis_function(
                            np.vstack((lon_target.ravel(), lat_target.ravel())).T,
                            within=True,
                            k=4,
                            rbf='multiquadric',
                            epsilon=None,
                            smooth=0,
                            num_threads=0)
                        var_regridded[:,:] = idw.reshape(lon_target.shape) 

                        return var_regridded
                     
                    lon_target,lat_target = self.coords_geo.T
                    lon_target_grid,lat_target_grid = lon_target.reshape(self.shape_grid),lat_target.reshape(self.shape_grid)

                    _var_obs_interp = _regrid_unstructured(lon_target_grid,lat_target_grid,lon_obs,lat_obs,var_obs) 
                    _err_obs_interp = _regrid_unstructured(lon_target_grid,lat_target_grid,lon_obs,lat_obs,err_obs) 

                elif self.interp_method == 'nearest':
                    if not same_source_grid:
                        self._nearest_source_indices = cKDTree(
                            _coords_obs
                        ).query(self.coords_geo)[1]
                    interpolated = np.column_stack(
                        (var_obs, err_obs)
                    )[self._nearest_source_indices]
                    _var_obs_interp = interpolated[:, 0]
                    _err_obs_interp = interpolated[:, 1]

                elif self.interp_method == 'linear':
                    if not same_source_grid:
                        self._linear_weights, self._linear_outside_hull = (
                            _linear_barycentric_weights(
                                _coords_obs, self.coords_geo
                            )
                        )

                    interpolated = self._linear_weights @ np.column_stack(
                        (var_obs, err_obs)
                    )
                    interpolated[self._linear_outside_hull] = np.nan
                    _var_obs_interp = interpolated[:, 0]
                    _err_obs_interp = interpolated[:, 1]

                elif self.interp_method == 'cubic':
                    if not same_source_grid:
                        self._cubic_triangulation = Delaunay(_coords_obs)
                    interpolated = CloughTocher2DInterpolator(
                        self._cubic_triangulation,
                        np.column_stack((var_obs, err_obs)),
                    )(self.coords_geo)
                    _var_obs_interp = interpolated[:, 0]
                    _err_obs_interp = interpolated[:, 1]

                else:
                    _var_obs_interp = griddata(_coords_obs, var_obs, self.coords_geo, method=self.interp_method)
                    _err_obs_interp = griddata(_coords_obs, err_obs, self.coords_geo, method=self.interp_method)

                if np.all(np.isnan(_var_obs_interp)):
                    # remove date from list
                    print(f"Warning: all interpolated values are NaN at date {date} for variable {self.name_var}, skipping this date.")
                    print(self.date_obs[i], self.t_obs[i])
                    self.date_obs.pop(i)
                    self.t_obs = np.delete(self.t_obs, i)
                    self.t_obs_jax = jnp.delete(self.t_obs_jax, i)
                    self.varobs = np.delete(self.varobs, i, axis=0)
                    self.errobs = np.delete(self.errobs, i, axis=0)
                    
                    continue

                # Mask values outside obs range
                var_obs_interp = _var_obs_interp.reshape(self.shape_grid)
                err_obs_interp = _err_obs_interp.reshape(self.shape_grid)
                mask = (var_obs_interp<np.nanmin(var_obs)) | (var_obs_interp>np.nanmax(var_obs)) | (self.mask)
                var_obs_interp[mask] = np.nan
                err_obs_interp[mask] = np.nan
                
                # Save operator if asked
                if self.write_op:
                    with open(file_L4, "wb") as f:
                        pickle.dump((var_obs_interp,err_obs_interp), f)

            mask = ((var_obs_interp < np.nanmin(var_obs)) |
                    (var_obs_interp > np.nanmax(var_obs)) |
                    self.mask)
            var_obs_interp[mask] = np.nan
            err_obs_interp[mask] = np.nan

            if var_bc is not None and self.name_var in var_bc:
                var_obs_interp -= var_bc[self.name_var][i].flatten()
                
            if self.gradients:
                    # Compute gradients
                var_obs_interp_grady = np.zeros_like(var_obs_interp)*np.nan
                var_obs_interp_gradx = np.zeros_like(var_obs_interp)*np.nan
                var_obs_interp_grady[1:-1,1:-1] = (var_obs_interp[2:,1:-1] - var_obs_interp[:-2,1:-1]) / (2 * self.DY[1:-1,1:-1])
                var_obs_interp_gradx[1:-1,1:-1] = (var_obs_interp[1:-1,2:] - var_obs_interp[1:-1,:-2]) / (2 * self.DX[1:-1,1:-1])

                # Fill dictionnaries
                self.varobs_grady[i] = var_obs_interp_grady.flatten()
                self.varobs_gradx[i] = var_obs_interp_gradx.flatten()
                self.errobs[i] = (.5* err_obs_interp /  (self.DY[np.newaxis,:,:]**2 + self.DX[np.newaxis,:,:]**2)**.5).flatten()

            else:
                # Fill dictionnaries
                self.varobs[i] = var_obs_interp.flatten()
                self.errobs[i] = err_obs_interp.flatten()
            
            if self.flag_plot>1:
                lon_grid,lat_grid = self.coords_geo.T
                fig,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
                im1 = ax1.scatter(lon_obs,lat_obs,c=var_obs)
                plt.colorbar(im1,ax=ax1)
                im2 = ax2.pcolormesh(lon_grid.reshape(self.shape_grid),lat_grid.reshape(self.shape_grid), var_obs_interp)
                plt.colorbar(im2,ax=ax2)
                im3 = ax3.pcolormesh(lon_grid.reshape(self.shape_grid),lat_grid.reshape(self.shape_grid), err_obs_interp)
                plt.colorbar(im3,ax=ax3)
                fig.suptitle(date.strftime('%Y-%m-%d %H:%M'))
                plt.show()

        if stacked_cache is not None and var_bc is None:
            cached = {
                'version': 1,
                'date_obs': self.date_obs,
                't_obs': self.t_obs,
                'errobs': self.errobs,
            }
            if self.gradients:
                cached['varobs_grady'] = self.varobs_grady
                cached['varobs_gradx'] = self.varobs_gradx
            else:
                cached['varobs'] = self.varobs
            with open(stacked_cache, 'wb') as stream:
                pickle.dump(cached, stream)

        self.varobs_arr = jnp.array(
            self.varobs_grady if self.gradients else self.varobs
        )
        self.errobs_arr = jnp.array(self.errobs)

        mask = jnp.isnan(self.varobs_arr) | jnp.isnan(self.errobs_arr) | (self.errobs_arr<1e-7) | (self.varobs_arr>1e7) 
        self.varobs_arr = jnp.where(mask, 0., self.varobs_arr)
        self.errobs_arr = jnp.where(mask, 1e15, self.errobs_arr)

    
    def is_obs_time(self,t):
        """Check if t is in observation times."""
        return bool(np.any(np.isclose(t, self.t_obs)))

    def is_obs_time_jax(self, t):
        """Device-side observation mask for a rolled checkpoint scan."""
        return jnp.any(jnp.isclose(t, self.t_obs_jax))
                
    def _to_observation_grid(self, X):
        """Convert a model C-grid velocity to the scalar L4 grid."""
        ny, nx = self.DX.shape
        if self.name_var == 'U' and X.size == ny * (nx + 1):
            field = X.reshape((ny, nx + 1))
            return 0.5 * (field[:, :-1] + field[:, 1:]).reshape(-1)
        if self.name_var == 'V' and X.size == (ny + 1) * nx:
            field = X.reshape((ny + 1, nx))
            return 0.5 * (field[:-1, :] + field[1:, :]).reshape(-1)
        return X.reshape(-1)

    def _from_observation_grid(self, increment, shape):
        """Adjoint of ``_to_observation_grid`` for C-grid velocities."""
        ny, nx = self.DX.shape
        if self.name_var == 'U' and tuple(shape) == (ny, nx + 1):
            inc = increment.reshape((ny, nx))
            out = jnp.zeros((ny, nx + 1), dtype=increment.dtype)
            out = out.at[:, :-1].add(0.5 * inc)
            return out.at[:, 1:].add(0.5 * inc)
        if self.name_var == 'V' and tuple(shape) == (ny + 1, nx):
            inc = increment.reshape((ny, nx))
            out = jnp.zeros((ny + 1, nx), dtype=increment.dtype)
            out = out.at[:-1, :].add(0.5 * inc)
            return out.at[1:, :].add(0.5 * inc)
        return increment.reshape(shape)

    def misfit(self, t, State):

        X = self._to_observation_grid(self._observation_model_field(State))

        if self.gradients:
            return self._misfit_grad(t, X)
        else:
            return self._misfit(t, X)

    def scan_misfit_size(self):
        if self.gradients:
            raise NotImplementedError(
                'lax.scan does not yet support gradient-form L4 observations'
            )
        if self.varobs_arr.shape[0] == 0:
            return 0
        return int(np.prod(self.varobs_arr.shape[1:]))

    def scan_misfit(self, t, State_var):
        """Return a fixed-size, zero-masked L4 misfit for every checkpoint."""
        if self.gradients:
            raise NotImplementedError(
                'lax.scan does not yet support gradient-form L4 observations'
            )
        # A static shape guard prevents tracing an invalid gather on the
        # empty time axis. jnp.where cannot protect that gather under scan.
        if self.varobs_arr.shape[0] == 0:
            return jnp.zeros((0,), dtype=self.varobs_arr.dtype)
        field = self._observation_model_field(State_var)
        raw = self._misfit(t, self._to_observation_grid(field))
        return jnp.where(self.is_obs_time_jax(t), raw, jnp.zeros_like(raw))

    def scan_adj(self, t, adState_var, State_var, misfit):
        """Pure-pytree L4 adjoint used inside ``lax.scan``."""
        if self.gradients:
            raise NotImplementedError(
                'lax.scan does not yet support gradient-form L4 observations'
            )
        if self.varobs_arr.shape[0] == 0:
            return adState_var
        idt = jnp.where(self.t_obs_jax == t, size=1)[0]
        inverr = 1 / self.errobs_arr[idt]
        increment = jnp.where(
            jnp.isnan(inverr * misfit),
            0.,
            self.model_to_observation_scale * (inverr * misfit),
        )
        field = self._observation_model_field(State_var)
        result = dict(adState_var)
        increment = self._from_observation_grid(increment.reshape(-1), field.shape)
        if self.observation_role is not None and hasattr(self.model, 'add_observation_adjoint'):
            self.model.add_observation_adjoint(
                result, self.name_var, self.observation_role, increment)
        else:
            name = self.name_mod_var[self.name_var]
            result[name] = result[name] + increment
        return result

    def _misfit(self, t, X):

        # Get data at time t
        idt = jnp.where(self.t_obs_jax==t, size=1)[0]

        # Prognostic fields intentionally carry NaNs on land. Masking the
        # residual *after* the subtraction leaves a NaN-valued inactive branch
        # in the JAX reverse pass, which can contaminate gradients although J
        # itself is finite. Make the model and conversion arrays finite before
        # the differentiable arithmetic; invalid observations already receive
        # an effectively zero weight below.
        X = jnp.nan_to_num(jnp.asarray(X), nan=0., posinf=0., neginf=0.)
        scale = jnp.nan_to_num(
            jnp.asarray(self.model_to_observation_scale),
            nan=0., posinf=0., neginf=0.,
        )

        # Compute misfit & errors in the observed coordinate (physical SSH
        # for nl=1 physical reduced-gravity QGSW runs).
        misfit = X * scale - self.varobs_arr[idt]
        inverr = 1/self.errobs_arr[idt]
        
        res = inverr * misfit
        res = jnp.where(jnp.isnan(res), 0., res) 

        return res.flatten()

    def _misfit_reduced(self, t, grad_obs, X):

        """
        Projects a gradient in observation space back to the model state space.
        
        Parameters:
            t: Current time
            grad_obs: Gradient in observation space (adjoint variable associated with misfit).
            X: Current model state (used to ensure consistent adjoint mapping).
            
        Returns:
            Gradient in the model state space (reduced space).
        """

        # Define a wrapper for _misfit that computes the forward misfit
        def misfit_func(X):
            return self._misfit(t, X)

        # Compute the vector-Jacobian product (vjp) for the forward operation
        _, vjp_func = jax.vjp(misfit_func, X)

        # Use the vjp function to project the gradient from observation space to model state space
        grad_model_state, = vjp_func(grad_obs)

        return grad_model_state
    
    def _misfit_grad(self, t, State):

        # Initialization
        misfit = np.array([])

        mode = 'w'
        for name in self.name_var_obs[t]:

            # Get model state in the observed coordinate. This preserves the
            # physical-SSH gradient when QGSW internally stores eta.
            X = State.getvar(self.name_mod_var[name])
            if name == 'SSH':
                X = X * self.model_to_observation_scale.reshape(X.shape)

            # Compute gradients
            HX_grady = np.zeros_like(self.DY)
            HX_gradx = np.zeros_like(self.DY)
            HX_grady[1:-1,1:-1] = ((X[2:,1:-1] - X[:-2,1:-1]) / (2 * self.DY[1:-1,1:-1]))
            HX_gradx[1:-1,1:-1] = ((X[1:-1,2:] - X[1:-1,:-2]) / (2 * self.DX[1:-1,1:-1]))
            HX_grady = HX_grady[np.newaxis,:,:]
            HX_gradx = HX_gradx[np.newaxis,:,:]

            # Compute misfit & errors
            _misfit_grady = (HX_grady-self.varobs[t][name+'_grady']) 
            _misfit_gradx = (HX_gradx-self.varobs[t][name+'_gradx']) 
            _inverr = 1/self.errobs[t][name]
            _misfit_grady[np.isnan(_misfit_grady)] = 0
            _misfit_gradx[np.isnan(_misfit_gradx)] = 0
            _inverr[np.isnan(_inverr)] = 0
        
            # Save to netcdf
            dsout = xr.Dataset(
                    {
                    "misfit_grady": (('Nobs','Ny','Nx'), _misfit_grady),
                    "misfit_gradx": (('Nobs','Ny','Nx'), _misfit_gradx),
                    "inverr" : (('Nobs','Ny','Nx'), _inverr)
                    }
                    )
            dsout.to_netcdf(
                os.path.join(self.tmp_DA_path,f"misfit_L4_grad_{t.strftime('%Y%m%d_%H%M')}.nc"), 
                mode=mode, 
                group=name
                )
            dsout.close()
            mode = 'a'

            # Concatenate
            for iobs in range(_inverr.shape[0]):
                misfit = np.concatenate((misfit,(_inverr[iobs]*_misfit_grady[iobs]).flatten(),(_inverr[iobs]*_misfit_gradx[iobs]).flatten()))

        return misfit
    
    def adj(self, t, adState, State, misfit):

        if self.gradients:
            return self._adj_grad(t, adState, State, misfit)
        else:
            return self._adj(t, adState, State, misfit)

    def _adj(self, t, adState, State, misfit):

        # Get data at time t
        idt = jnp.where(self.t_obs_jax==t, size=1)[0]
        inverr = 1/self.errobs_arr[idt]
        _advar = self.model_to_observation_scale * (inverr * misfit)
        _advar = jnp.where(jnp.isnan(_advar), 0., _advar)

        field = self._observation_model_field(adState)
        increment = self._from_observation_grid(_advar.reshape(-1), field.shape)
        self._add_observation_adjoint(adState, increment)
  
    def _adj_grad(self, t, adState, State, misfit):

        for name in self.name_var_obs[t]:

            # Read misfit
            ds = xr.open_dataset(os.path.join(
                os.path.join(self.tmp_DA_path,f"misfit_L4_grad_{t.strftime('%Y%m%d_%H%M')}.nc")), 
                group=name)
            misfit_grady = ds['misfit_grady'].values
            misfit_gradx = ds['misfit_gradx'].values
            inverr = ds['inverr'].values
            ds.close()
            del ds

            # Apply R operator
            misfit_grady = R.inv(misfit_grady)
            misfit_gradx = R.inv(misfit_gradx)

            # Read adjoint variable
            advar = adState.getvar(self.name_mod_var[name])

            # Compute adjoint operation of y = H(S eta). The pointwise
            # scale S converts internal eta to physical SSH before gradients.
            scale = (self.model_to_observation_scale.reshape(advar.shape)
                     if name == 'SSH' else 1.)
            for iobs in range(inverr.shape[0]):
                grad_y = inverr[iobs,1:-1,1:-1] * inverr[iobs,1:-1,1:-1] * misfit_grady[iobs,1:-1,1:-1] / (2 * self.DY[1:-1,1:-1])
                grad_x = inverr[iobs,1:-1,1:-1] * inverr[iobs,1:-1,1:-1] * misfit_gradx[iobs,1:-1,1:-1] / (2 * self.DX[1:-1,1:-1])
                advar[2:,1:-1] += scale[2:,1:-1] * grad_y
                advar[:-2,1:-1] += -scale[:-2,1:-1] * grad_y
                advar[1:-1,2:] += scale[1:-1,2:] * grad_x
                advar[1:-1,:-2] += -scale[1:-1,:-2] * grad_x

            # Update adjoint variable
            adState.setvar(advar, self.name_mod_var[name])      

###############################################################################
#                            Multi-Operators                                  #
###############################################################################      

class LweControl:
    """Ordered observation-error blocks, separate from ocean controls.

    Projections are prepared once per evaluation. Time selection uses JAX
    indices; Python and scan layouts follow the corresponding H concatenation.
    All public corrections and transposes include errobs, but not INV.sigma_R.
    """

    def __init__(self, observations):
        self.observations = observations
        self.operators = []
        self.slices = {}
        self.size = 0

        def visit(operator):
            if hasattr(operator, 'Obsop'):
                for child in operator.Obsop:
                    visit(child)
            elif isinstance(operator, Obsop_interp_lwe_l3):
                if not operator.lwe_sources:
                    return  # Disabled LWE retains the ordinary L3 control layout.
                if getattr(operator, '_geometry_only', True):
                    raise ValueError('Joint LWE cost requires prepare_joint, not geometry/projection only')
                if id(operator) in self.slices:
                    raise ValueError('A LWE operator cannot be repeated in a joint layout')
                self.slices[id(operator)] = slice(self.size, self.size + operator.n_lwe)
                self.size += operator.n_lwe
                self.operators.append(operator)
        visit(observations)

    def metadata(self, ocean_size):
        """Stable, versioned identity of the complete observation block tree."""
        from .obs import _lwe_signature
        blocks = []
        def visit(op, path):
            if hasattr(op, 'Obsop'):
                return dict(children=[visit(child, path + [i])
                                      for i, child in enumerate(op.Obsop)])
            entry = dict(kind=type(op).__name__, path=path)
            # Include ordinary leaves: reordering them changes residual coordinates.
            arrays = {name: np.asarray(getattr(op, name)) for name in
                      ('t_obs', 'varobs_arr', 'errobs_arr', 'data_arr', 'indices_arr',
                       'model_to_observation_scale') if hasattr(op, name)}
            entry['observations'] = Obsop_interp_lwe_l3._payload_digest(arrays)
            if id(op) in self.slices:
                block = self.slices[id(op)]
                entry.update(start=int(ocean_size + block.start),
                             stop=int(ocean_size + block.stop),
                             sources=list(op.source_names),
                             geometry=op.geometry_signature,
                             projection=op.projection_signature)
                blocks.append(dict(entry))
            return entry
        tree = visit(self.observations, [])
        return dict(version=1, ocean_size=int(ocean_size), size=int(ocean_size + self.size),
                    coordinates='ocean_then_whitened_lwe', blocks=blocks,
                    signature=_lwe_signature(tree))

    def control_file_attributes(self, ocean_size, coordinates):
        import json
        if not self.operators:
            return {}
        return dict(lwe_control_layout=json.dumps(self.metadata(ocean_size), sort_keys=True),
                    control_coordinates=coordinates)

    def validate_control_file(self, attributes, values, ocean_size, coordinates):
        """Never infer compatibility from coefficient count alone."""
        import json
        encoded = attributes.get('lwe_control_layout')
        if not self.operators:
            if encoded is not None:
                raise ValueError('LWE restart cannot be loaded with LWE disabled')
            return
        try:
            metadata = json.loads(encoded)
        except (TypeError, ValueError) as error:
            raise ValueError('LWE restart lacks valid signed control layout metadata') from error
        if (metadata != self.metadata(ocean_size)
                or attributes.get('control_coordinates') != coordinates
                or np.shape(values) != (ocean_size + self.size,)
                or not np.isfinite(values).all()):
            raise ValueError('Incompatible LWE restart layout, coordinates or coefficients')

    def residual_rows(self, times, values, selected=None):
        """Unpack the scan concatenation without losing operator identities."""
        values = np.asarray(values)
        selected = np.ones(len(times), bool) if selected is None else np.asarray(selected, bool)
        result = {id(op): {} for op in self.operators}
        def visit(op, offset):
            if hasattr(op, 'Obsop'):
                for child in op.Obsop:
                    offset = visit(child, offset)
                return offset
            size = op.scan_misfit_size()
            if id(op) in result:
                for i, t in enumerate(times):
                    if selected[i] and op.is_obs_time(t):
                        result[id(op)][t] = values[i, offset:offset+size]
            return offset + size
        end = visit(self.observations, 0)
        if values.shape != (len(times), end):
            raise ValueError('LWE export residual scan layout does not align')
        return result

    @staticmethod
    def write_dataset(dataset, path):
        """Publish a complete NetCDF atomically, preserving int64 native times."""
        path = os.fspath(path)
        temporary = path + '.tmp-' + str(os.getpid())
        try:
            dataset.to_netcdf(temporary, engine='netcdf4')
            os.replace(temporary, path)
        finally:
            dataset.close()
            if os.path.exists(temporary):
                os.remove(temporary)

    def diagnostics(self, z, ocean_residuals, ocean_size=0):
        """Native, unpadded rows. Input residuals are (H(x)-y)/sigma.

        ocean_residuals maps operator IDs to checkpoint -> padded residual.
        Missing checkpoints are exported with assimilated=False and NaN residuals.
        """
        import json
        z = np.asarray(z)
        if z.shape != (self.size,) or not np.isfinite(z).all():
            raise ValueError('Invalid whitened LWE control for export')
        rows, coefficients = [], []
        for index, op in enumerate(self.operators):
            valid = op.native_geometry_arr['valid']
            block = self.slices[id(op)]
            error = np.asarray(op.project_lwe(z[block]))[valid]
            sigma = np.asarray(op.errobs_arr)[valid]
            selected = np.zeros(valid.shape, dtype=bool)
            residual = np.full(valid.shape, np.nan)
            supplied = ocean_residuals.get(id(op), {})
            usable = np.asarray(op.observation_mask())
            for i, t in enumerate(op.t_obs):
                if t in supplied:
                    values = np.asarray(supplied[t])
                    if values.shape != valid.shape[1:]:
                        raise ValueError('LWE export residual rows do not align')
                    selected[i] = usable[i]
                    residual[i] = np.where(selected[i], values, np.nan)
            normalized_error = np.divide(error, sigma, out=np.zeros_like(error),
                                         where=selected[valid])
            fields = {key: value[valid] for key, value in op.native_geometry_arr.items()
                      if key != 'valid'}
            fields.update(operator_id=np.full(error.size, index, dtype=np.int64),
                          checkpoint_seconds=np.broadcast_to(op.t_obs[:, None], valid.shape)[valid],
                          source_name=np.asarray(op.source_names)[fields['source_id']],
                          observation=np.asarray(op.varobs_arr)[valid], sigma=sigma,
                          lwe_error=error, lwe_correction=-error,
                          residual_before=residual[valid]*sigma,
                          residual_after=residual[valid]*sigma+error,
                          normalized_residual_after=residual[valid]+normalized_error,
                          assimilated=selected[valid])
            rows.append(fields)
            coeff = {key: np.asarray(value) for key, value in op.lwe_coefficients.items()}
            coeff.update(z_lwe=z[block], amplitude=np.asarray(op.lwe_sqrtq)*z[block],
                         operator_id=np.full(op.n_lwe, index, dtype=np.int64),
                         control_index=np.arange(ocean_size + block.start, ocean_size + block.stop))
            coefficients.append(coeff)
        data = {}
        for items, dimension, prefix in ((rows, 'observation_row', ''),
                                          (coefficients, 'coefficient', 'coefficient_')):
            if items:
                for key in items[0]:
                    values = np.concatenate([item[key] for item in items])
                    if values.dtype.kind == 'b':
                        values = values.astype('int8')
                    data[prefix + key] = ((dimension,), values)
        dataset = xr.Dataset(data, attrs=dict(
            lwe_control_layout=json.dumps(self.metadata(ocean_size), sort_keys=True),
            sign_convention='y = H(x) + lwe_error + noise; corrected_y = y - lwe_error',
            observation_convention='prepared sla_unfiltered - lwe, including configured MDT/boundary handling',
            residual_convention='H(x) - y; residual_after = H(x) + lwe_error - y',
            native_time_units='int64 nanoseconds since 1970-01-01 UTC',
            source_tables=json.dumps([list(op.source_names) for op in self.operators])))
        for name in ('lwe_error', 'lwe_correction', 'observation', 'sigma',
                     'residual_before', 'residual_after', 'coefficient_amplitude'):
            if name in dataset:
                dataset[name].attrs['units'] = 'm'
        return dataset

    def project(self, z):
        z = jnp.asarray(z)
        if z.shape != (self.size,):
            raise ValueError(f'Joint LWE control must have shape {(self.size,)}')
        return tuple(op.project_lwe(z[self.slices[id(op)]]) for op in self.operators)

    @staticmethod
    def _row(operator, t, values):
        if values.shape[0] == 0:
            return jnp.zeros(values.shape[1:], dtype=values.dtype)
        index = jnp.argmax(operator.t_obs_jax == t)
        return jnp.where(operator.is_obs_time_jax(t), values[index], jnp.zeros_like(values[index]))

    def correction(self, t, projected, scan=True):
        by_id = {id(op): value for op, value in zip(self.operators, projected)}

        def part(op):
            if hasattr(op, 'Obsop'):
                children = [part(child) for child in op.Obsop
                            if scan or child.is_obs_time(t)]
                return jnp.concatenate(children) if children else jnp.zeros(1)
            if id(op) in by_id:
                valid = op.observation_mask()
                sigma = jnp.where(valid, op.errobs_arr, 1.)
                return self._row(op, t, jnp.where(valid, by_id[id(op)] / sigma, 0.))
            return jnp.zeros(op.scan_misfit_size())
        return part(self.observations)

    @staticmethod
    def _size(operator, t, scan):
        if hasattr(operator, 'Obsop'):
            children = [child for child in operator.Obsop if scan or child.is_obs_time(t)]
            return sum(LweControl._size(child, t, scan) for child in children) if children else (0 if scan else 1)
        return operator.scan_misfit_size()

    def adjoint(self, t, residual, scan=True):
        result = {id(op): jnp.zeros(op.n_lwe, dtype=residual.dtype)
                  for op in self.operators}

        def visit(op, values):
            if hasattr(op, 'Obsop'):
                offset = 0
                for child in op.Obsop:
                    if scan or child.is_obs_time(t):
                        size = self._size(child, t, scan)
                        visit(child, values[offset:offset + size])
                        offset += size
            elif id(op) in self.slices:
                valid = self._row(op, t, op.observation_mask())
                sigma = self._row(op, t, op.errobs_arr)
                weighted = jnp.where(valid, values / jnp.where(valid, sigma, 1.), 0.)
                result[id(op)] = op.scan_adjoint_lwe(t, weighted)
        visit(self.observations, residual)
        return (jnp.concatenate([result[id(op)] for op in self.operators])
                if result else jnp.zeros((0,), dtype=residual.dtype))


class Obsop_multi:

    def __init__(self,config,State,dict_obs,Model):

        self.Obsop = []
        _config = config.copy()

        for _OBSOP in config.OBSOP:
            _config.OBSOP = config.OBSOP[_OBSOP]
            self.Obsop.append(Obsop(_config,State,dict_obs,Model))
        
        self.misfit_indt = {}

    def is_obs_time(self,t):

        for _Obsop in self.Obsop:
            if _Obsop.is_obs_time(t):
                return True
        return False

    def is_obs(self,t):

        for _Obsop in self.Obsop:
            if _Obsop.is_obs(t):
                return True
        return False

    def process_obs(self, var_bc=None):

        for _Obsop in self.Obsop:
            _Obsop.process_obs(var_bc)
                

    def misfit(self,t,State):

        misfit_parts = []
        
        self.misfit_indt[t] = [None for _ in self.Obsop]
        indt = 0
        for i, _Obsop in enumerate(self.Obsop):
            if _Obsop.is_obs_time(t):
                _misfit = _Obsop.misfit(t,State)
                self.misfit_indt[t][i] = slice(indt, indt+_misfit.size)
                indt += _misfit.size
                misfit_parts.append(jnp.ravel(_misfit))

        if not misfit_parts:
            print(t)
            return jnp.array([1e-7]) # To avoid problems with empty misfit
    
        return jnp.concatenate(misfit_parts)

    def scan_misfit_size(self):
        return sum(operator.scan_misfit_size() for operator in self.Obsop)

    def scan_misfit(self, t, State_var):
        """Concatenate fixed-size, masked misfits from every sub-operator."""
        return jnp.concatenate([
            operator.scan_misfit(t, State_var)
            for operator in self.Obsop
        ])

    def scan_adj(self, t, adState_var, State_var, misfit):
        """Apply every sub-operator adjoint to its fixed misfit slice."""
        result = adState_var
        offset = 0
        for operator in self.Obsop:
            size = operator.scan_misfit_size()
            result = operator.scan_adj(
                t,
                result,
                State_var,
                misfit[offset:offset + size],
            )
            offset += size
        return result

    def adj(self, t, adState, State, misfit):
    
        for i, _Obsop in enumerate(self.Obsop):
            if _Obsop.is_obs_time(t):
                _mistfit = misfit[self.misfit_indt[t][i]]
                _Obsop.adj(t, adState, State, _mistfit)
