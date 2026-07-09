#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Florian Le Guillou on June 2026.

Builds reduced-basis operators and related helper models.
"""
from .config import USE_FLOAT64
import os, sys
import numpy as np
import logging
import datetime 
import xarray as xr
import scipy
from scipy.sparse import csc_matrix
from scipy.integrate import quad
import jax.numpy as jnp 
from jax.experimental import sparse as sparse
from jax import jit
import jax


jax.config.update("jax_enable_x64", USE_FLOAT64)


def _as_name_list(name_mod_var):
    if isinstance(name_mod_var, (list, tuple, np.ndarray)):
        return list(name_mod_var)
    return [name_mod_var]


def _primary_name(name_mod_var):
    return _as_name_list(name_mod_var)[0]


def _assign_to_state_names(obj, State, value):
    for name in _as_name_list(obj.name_mod_var):
        if not obj.multi_mode:
            State[name] = value
        else:
            State[name] += value


def _ensure_adstate_names(obj, adState, zero):
    for name in _as_name_list(obj.name_mod_var):
        if adState[name] is None:
            adState[name] = zero.copy() if hasattr(zero, 'copy') else +zero


def _sum_adstate_names(obj, adState):
    adparams = None
    for name in _as_name_list(obj.name_mod_var):
        if adparams is None:
            adparams = +adState[name]
        else:
            adparams = adparams + adState[name]
    return adparams


def _clear_adstate_names(obj, adState):
    for name in _as_name_list(obj.name_mod_var):
        adState[name] *= 0.


def Basis(config, State, verbose=True, multi_mode=False, *args, **kwargs):
    """
    NAME
        Basis

    DESCRIPTION
        Main function calling subfunctions for specific Reduced Basis functions
    """
    
    if config.BASIS is None:
        return 
    
    elif config.BASIS.super is None:
        return Basis_multi(config, State, verbose=verbose)

    else:
        if verbose:
            print(config.BASIS)

        if config.BASIS.super=='BASIS_GAUSS3D':
            return Basis_gauss3d(config,State,multi_mode=multi_mode)

        elif config.BASIS.super=='BASIS_GAUSS2D':
            return Basis_gauss2d(config,State,multi_mode=multi_mode)

        elif config.BASIS.super=='BASIS_BMaux':
            return Basis_bmaux(config,State,multi_mode=multi_mode)

        elif config.BASIS.super == 'BASIS_OFFSET':
            return Basis_offset(config,State,multi_mode=multi_mode)
        
        elif config.BASIS.super == 'BASIS_HBC':
            return Basis_hbc(config,State)

        else:
            sys.exit(config.BASIS.super + ' not implemented yet')

 
###############################################################################
#                3D Gaussian (time/spatial components)                        #
###############################################################################    
# Old version of the basis class, kept for reference. The new Basis_gauss3d class is preferred.
class _Basis_gauss3d:
   
    def __init__(self, config, State, multi_mode=False):

        self.km2deg = 1./110

        self.flux = config.BASIS.flux
        self.facns = config.BASIS.facns
        self.facnlt = config.BASIS.facnlt
        self.sigma_D = config.BASIS.sigma_D
        self.sigma_T = config.BASIS.sigma_T
        self.sigma_Q = config.BASIS.sigma_Q
        self.normalize_fact = config.BASIS.normalize_fact
        self.name_mod_var = config.BASIS.name_mod_var
        self.time_spinup = config.BASIS.time_spinup
        self.fcor = config.BASIS.fcor
        self.flag_variable_Q = config.BASIS.flag_variable_Q
        self.path_sad = config.BASIS.path_sad
        self.name_var_sad = config.BASIS.name_var_sad
        self.path_background = config.BASIS.path_background
        self.var_background = config.BASIS.var_background

        # C-grid variable type (None, 'U', or 'V')
        self.c_grid_var = getattr(config.BASIS, 'c_grid_var', None)

        # Grid params
        self.ny = State.ny
        self.nx = State.nx
        self.lon_min = State.lon_min
        self.lon_max = State.lon_max
        self.lat_min = State.lat_min
        self.lat_max = State.lat_max

        if self.c_grid_var == 'U':
            self.shape_phys = (State.ny, State.nx + 1)
            lon_h = State.lon
            lat_h = State.lat
            lon_u = np.zeros((State.ny, State.nx + 1))
            lat_u = np.zeros((State.ny, State.nx + 1))
            lon_u[:, 1:State.nx] = 0.5 * (lon_h[:, :-1] + lon_h[:, 1:])
            lat_u[:, 1:State.nx] = 0.5 * (lat_h[:, :-1] + lat_h[:, 1:])
            lon_u[:, 0] = lon_h[:, 0] - 0.5 * (lon_h[:, 1] - lon_h[:, 0])
            lat_u[:, 0] = lat_h[:, 0] - 0.5 * (lat_h[:, 1] - lat_h[:, 0])
            lon_u[:, State.nx] = lon_h[:, -1] + 0.5 * (lon_h[:, -1] - lon_h[:, -2])
            lat_u[:, State.nx] = lat_h[:, -1] + 0.5 * (lat_h[:, -1] - lat_h[:, -2])
            self.lon1d = lon_u.flatten()
            self.lat1d = lat_u.flatten()
        elif self.c_grid_var == 'V':
            self.shape_phys = (State.ny + 1, State.nx)
            lon_h = State.lon
            lat_h = State.lat
            lon_v = np.zeros((State.ny + 1, State.nx))
            lat_v = np.zeros((State.ny + 1, State.nx))
            lon_v[1:State.ny, :] = 0.5 * (lon_h[:-1, :] + lon_h[1:, :])
            lat_v[1:State.ny, :] = 0.5 * (lat_h[:-1, :] + lat_h[1:, :])
            lon_v[0, :] = lon_h[0, :] - 0.5 * (lon_h[1, :] - lon_h[0, :])
            lat_v[0, :] = lat_h[0, :] - 0.5 * (lat_h[1, :] - lat_h[0, :])
            lon_v[State.ny, :] = lon_h[-1, :] + 0.5 * (lon_h[-1, :] - lon_h[-2, :])
            lat_v[State.ny, :] = lat_h[-1, :] + 0.5 * (lat_h[-1, :] - lat_h[-2, :])
            self.lon1d = lon_v.flatten()
            self.lat1d = lat_v.flatten()
        else:
            self.shape_phys = (State.ny, State.nx)
            self.lon1d = State.lon.flatten()
            self.lat1d = State.lat.flatten()

        self.nphys = self.lon1d.size

        # Gravity 
        self.g = 9.81

        # Compute geostrophic velocoties
        self.compute_velocities = config.BASIS.compute_velocities
        self.name_mod_u = config.BASIS.name_mod_u
        self.name_mod_v = config.BASIS.name_mod_v
        pad = ((1,0),(1,0))
        _f = np.pad(State.f, pad_width=pad, mode='edge')
        self.f_on_v = 0.5*(_f[:,1:] + _f[:,:-1])
        self.f_on_u = 0.5*(_f[1:,:] + _f[:-1,:])

        # Grid spacing
        self.dx = np.pad(State.DX, pad_width=pad, mode='edge')
        self.dy = np.pad(State.DY, pad_width=pad, mode='edge')
        self.dx_on_v = 0.5*(self.dx[:,1:] + self.dx[:,:-1])
        self.dy_on_u = 0.5*(self.dy[1:,:] + self.dy[:-1,:])

        # Mask
        if State.mask is not None and np.any(State.mask):
            if self.c_grid_var == 'U':
                mask_u = np.zeros((State.ny, State.nx + 1), dtype=bool)
                mask_u[:, 1:State.nx] = State.mask[:, :-1] | State.mask[:, 1:]
                self.mask1d = mask_u.ravel()
            elif self.c_grid_var == 'V':
                mask_v = np.zeros((State.ny + 1, State.nx), dtype=bool)
                mask_v[1:State.ny, :] = State.mask[:-1, :] | State.mask[1:, :]
                self.mask1d = mask_v.ravel()
            else:
                self.mask1d = State.mask.ravel()
        else:
            self.mask1d = None

        # Coastal face masks for C-grid geostrophic velocity masking
        if State.mask is not None and np.any(State.mask):
            _mask_pad = np.pad(State.mask.astype(bool), pad_width=((1,0),(1,0)), mode='edge')
            self.u_coast_mask = _mask_pad[1:, :] | _mask_pad[:-1, :]  # (ny, nx+1)
            self.v_coast_mask = _mask_pad[:, 1:] | _mask_pad[:, :-1]  # (ny+1, nx)
        else:
            self.u_coast_mask = None
            self.v_coast_mask = None

        # Time window
        if self.flux:
            self.window = mywindow_flux
        else:
            self.window = mywindow
 
        # For time normalization
        if self.normalize_fact:
            tt = np.linspace(-self.sigma_T,self.sigma_T)
            tmp = np.zeros_like(tt)
            for i in range(tt.size-1):
                tmp[i+1] = tmp[i] + self.window(tt[i]/self.sigma_T)*(tt[i+1]-tt[i])
            self.norm_fact = tmp.max()
        
        # Longitude unit
        self.lon_unit = State.lon_unit
        
        # For multi-basis
        self.multi_mode = multi_mode
         
    def set_basis(self,time,return_q=False,**kwargs):
        
        TIME_MIN = time.min()
        TIME_MAX = time.max()
        LON_MIN = self.lon_min
        LON_MAX = self.lon_max
        LAT_MIN = self.lat_min
        LAT_MAX = self.lat_max
        if (LON_MAX<LON_MIN): LON_MAX = LON_MAX+360.

        self.time = time
        
        # Coordinates in space
        dlat = self.sigma_D/self.facns*self.km2deg
        lat0 = LAT_MIN - LAT_MIN%dlat - self.sigma_D*(1-1./self.facns)*self.km2deg  # To start at a fix latitude
        lat1 = LAT_MAX + 1.5*dlat
        ENSLAT1 = np.arange(lat0, lat1, dlat)
        ENSLAT = []
        ENSLON = []
        for I in range(len(ENSLAT1)):
            dlon = self.sigma_D/self.facns/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg
            lon0 = LON_MIN - LON_MIN%dlon - self.sigma_D*(1-1./self.facns)/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg # To start at a fix longitude
            lon1 = LON_MAX + dlon * 1.5
            ENSLON1 = np.arange(lon0, lon1, dlon)
            ENSLAT = np.concatenate(([ENSLAT,np.repeat(ENSLAT1[I],len(ENSLON1))]))
            ENSLON = np.concatenate(([ENSLON,ENSLON1]))
        self.ENSLAT = ENSLAT
        self.ENSLON = ENSLON
        
        # Coordinates in time
        ENST = np.arange(-self.sigma_T*(1-1./self.facnlt),(TIME_MAX - TIME_MIN)+1.5*self.sigma_T/self.facnlt , self.sigma_T/self.facnlt)
        self.ENST = ENST
    
        self.nbasis = ENST.size * ENSLAT.size
        self.nphys = self.lon1d.size
        if self.c_grid_var == 'U':
            self.shape_phys = [self.ny, self.nx + 1]
        elif self.c_grid_var == 'V':
            self.shape_phys = [self.ny + 1, self.nx]
        else:
            self.shape_phys = [self.ny, self.nx]
        self.shape_basis = [ENST.size,ENSLAT.size]
        
        # Fill Q matrix
        if self.flag_variable_Q:
            Q = []
            sad = xr.open_dataset(self.path_sad)[self.name_var_sad['var']] 
            # Convert longitude 
            if np.sign(sad[self.name_var_sad['lon']].data.min())==-1 and self.lon_unit=='0_360':
                sad = sad.assign_coords({self.name_var_sad['lon']:((self.name_var_sad['lon'], sad[self.name_var_sad['lon']].data % 360))})
            elif (np.sign(sad[self.name_var_sad['lon']].data.min())>=0  or sad[self.name_var_sad['lon']].data.max()>180) and self.lon_unit=='-180_180':
                sad = sad.assign_coords({self.name_var_sad['lon']:((self.name_var_sad['lon'], (sad[self.name_var_sad['lon']].data + 180) % 360 - 180 ))})
            sad = sad.sortby(sad[self.name_var_sad['lon']])    
            for (lon,lat) in zip(ENSLON,ENSLAT):
                # Precompute interpolation grid once
                dlon = .5 * self.sigma_D/np.cos(lat*np.pi/180.)
                dlat = .5 * self.sigma_D
                elon = np.linspace(lon - dlon, lon + dlon, 10)
                elat = np.linspace(lat - dlat, lat + dlat, 10)
                elon2, elat2 = np.meshgrid(elon, elat)
                std_tmp_values = sad.interp({self.name_var_sad['lon']:elon2.ravel(), 
                                            self.name_var_sad['lat']:elat2.ravel()}).values
                std_tmp = np.nanmean(std_tmp_values) if not np.all(np.isnan(std_tmp_values)) else 10**-10
                Q_tmp = std_tmp / ((self.facns*self.facnlt))**.5 
                Q.append(Q_tmp) 
            # Repeat for all time centers
            Q = np.tile(Q, len(self.ENST))
        else:
            Q = self.sigma_Q / ((self.facns*self.facnlt))**.5 * np.ones((self.nbasis))


        
        Xb = np.zeros_like(Q)
        # Background
        if self.path_background is not None and os.path.exists(self.path_background):
            with xr.open_dataset(self.path_background) as ds:
                print('gauss3d np.shape(Xb)',np.shape(Xb))
                print('gauss3d np.shape(ds[self.var_background].values)',np.shape(ds[self.var_background].values))
                print(f'Load background from file: {self.path_background}') 
                Xb = ds[self.var_background].values[:len(Xb)] 

        
        print(f'lambda={self.sigma_D:.1E}',
            f'nlocs={ENSLAT.size:.1E}',
            f'tdec={self.sigma_T:.1E}',
            f'ntime={ENST.size:.1E}',
            f'Q={np.mean(Q):.1E}')
        
        print(f'reduced order: {time.size * self.nphys} --> {self.nbasis}\n reduced factor: {int(time.size * self.nphys/self.nbasis)}')

        # Compute basis components
        Gauss_xy = self._compute_component_space()
        Gauss_t, Nt = self._compute_component_time(time)
        self.Gauss_xy = Gauss_xy
        self.Gauss_t = Gauss_t
        self.Nt = Nt
        self.Nx = ENSLAT.size

        if return_q:
            return np.zeros_like(Q), Q
        

    def _compute_component_space(self):
        """
            Gaussian functions in space
        """

        data = np.empty((self.ENSLAT.size*self.lon1d.size,))
        indices = np.empty((self.ENSLAT.size*self.lon1d.size,),dtype=int)
        sizes = np.zeros((self.ENSLAT.size,),dtype=int)
        ind_tmp = 0
        for i,(lat0,lon0) in enumerate(zip(self.ENSLAT,self.ENSLON)):
            indphys = np.where(
                    (np.abs((np.mod(self.lon1d - lon0+180,360)-180) / self.km2deg * np.cos(lat0 * np.pi / 180.)) <= self.sigma_D) &
                    (np.abs((self.lat1d - lat0) / self.km2deg) <= self.sigma_D)
                    )[0]
            xx = (np.mod(self.lon1d[indphys] - lon0+180,360)-180) / self.km2deg * np.cos(lat0 * np.pi / 180.) 
            yy = (self.lat1d[indphys] - lat0) / self.km2deg
            if self.mask1d is not None:
                indmask = self.mask1d[indphys]
                indphys = indphys[~indmask]
                xx = xx[~indmask]
                yy = yy[~indmask]
            sizes[i] = indphys.size
            indices[ind_tmp:ind_tmp+indphys.size] = indphys
            data[ind_tmp:ind_tmp+indphys.size] = mywindow(xx / self.sigma_D) * mywindow(yy / self.sigma_D)
            ind_tmp += indphys.size
        indptr = np.zeros((i+2),dtype=int)
        indptr[1:] = np.cumsum(sizes)

        return csc_matrix((data, indices, indptr), shape=(self.lon1d.size, self.ENSLAT.size))
    
    def _compute_component_time(self, time):
        """
            Gaussian functions in time
        """
        Gauss_t = {}
        Nt = {}
        for t in time:
            Gauss_t[t] = np.zeros((self.ENSLAT.size*self.ENST.size))
            Nt[t] = 0
            ind_tmp = 0
            for it in range(len(self.ENST)):
                dt = t - self.ENST[it]
                if abs(dt) < self.sigma_T:
                    fact = self.window(dt / self.sigma_T) 
                    if self.normalize_fact:
                        fact /= self.norm_fact
                    if self.time_spinup is not None and t<self.time_spinup:
                        fact *= (1-self.window(t / self.time_spinup))
                    if fact!=0:   
                        Nt[t] += 1
                        Gauss_t[t][ind_tmp:ind_tmp+self.ENSLAT.size] = fact   
                ind_tmp += self.ENSLAT.size

        return Gauss_t, Nt

    def _ssh2uv(self, ssh):

        """
            Compute geostrophic velocities from SSH
        """

        _ssh = np.pad(ssh, pad_width=((1,0),(1,0)), mode='edge')

        _u = -self.g / self.f_on_u * (_ssh[1:,:] - _ssh[:-1,:]) / self.dy_on_u
        _v = self.g / self.f_on_v * (_ssh[:,1:] - _ssh[:,:-1]) / self.dx_on_v

        if self.u_coast_mask is not None:
            _u = np.where(self.u_coast_mask, 0., _u)
            _v = np.where(self.v_coast_mask, 0., _v)

        return _u, _v
    
    def _ssh2uv_adj(self, adu, adv):

        """
        Adjoint of geostrophic velocity computation.
        Uses jnp so that JAX device arrays are kept on-device (no GPU→CPU transfer).
        """

        if self.u_coast_mask is not None:
            adu = jnp.where(self.u_coast_mask, 0., adu)
            adv = jnp.where(self.v_coast_mask, 0., adv)

        # _adssh lives on padded grid: (ny+1, nx+1)
        _adssh = jnp.zeros((self.shape_phys[0] + 1, self.shape_phys[1] + 1))

        _adssh = _adssh.at[1:,:].add( -self.g / self.f_on_u * adu / self.dy_on_u)
        _adssh = _adssh.at[:-1,:].add(  self.g / self.f_on_u * adu / self.dy_on_u)
        _adssh = _adssh.at[:,1:].add(   self.g / self.f_on_v * adv / self.dx_on_v)
        _adssh = _adssh.at[:,:-1].add( -self.g / self.f_on_v * adv / self.dx_on_v)

        # map padded grid back to physical ssh grid:
        # physical ssh[i,j] == _ssh[i+1,j+1]
        adssh = _adssh[1:,1:]

        # contributions from the padded first row/col (mode='edge' duplicates edge values)
        # add the padded southern row (index 0) to physical southern row (adssh[0,:])
        adssh = adssh.at[0,:].add(_adssh[0,1:])
        # add the padded western column (index 0) to physical western column (adssh[:,0])
        adssh = adssh.at[:,0].add(_adssh[1:,0])
        # the padded corner (0,0) was duplicated as well — add it into adssh[0,0]
        adssh = adssh.at[0,0].add(_adssh[0,0])

        return adssh
    
    def operg(self, t, X, State=None):

        """
            Project to physicial space
        """
        
        phi = np.zeros(self.nphys)
        GtX = self.Gauss_t[t] * X
        ind0 = np.nonzero(self.Gauss_t[t])[0]
        if ind0.size>0:
            GtX = GtX[ind0].reshape(self.Nt[t],self.Nx)
            phi += self.Gauss_xy.dot(GtX.sum(axis=0))
        phi = phi.reshape(self.shape_phys)

        # Compute geostrophic velocities
        if self.compute_velocities:
            u, v = self._ssh2uv(phi)
            if State is not None:
                if not self.multi_mode:
                    State[self.name_mod_u] = u
                    State[self.name_mod_v] = v
                else:
                    State[self.name_mod_u] += u
                    State[self.name_mod_v] += v
    
        # Update State
        if State is not None:
            _assign_to_state_names(self, State, phi)
        else:
            if self.compute_velocities:
                return phi, u, v
            else:
                return phi

    def operg_transpose(self, t, adState):
        """
            Project to reduced space
        """

        _ensure_adstate_names(self, adState, np.zeros(self.shape_phys))
        
        if self.compute_velocities and (adState[self.name_mod_u] is None or adState[self.name_mod_v] is None):
            adState[self.name_mod_u] = np.zeros_like(self.f_on_u)
            adState[self.name_mod_v] = np.zeros_like(self.f_on_v)

        adX = np.zeros(self.nbasis)
        adparams = _sum_adstate_names(self, adState).ravel()

        if self.compute_velocities:
            adparams += self._ssh2uv_adj(adState[self.name_mod_u], adState[self.name_mod_v]).ravel()

        Gt = self.Gauss_t[t]
        ind0 = np.nonzero(Gt)[0]
        if ind0.size>0:
            Gt = Gt[ind0].reshape(self.Nt[t],self.Nx)
            adGtX = self.Gauss_xy.T.dot(adparams)
            adGtX = np.repeat(adGtX[np.newaxis,:],self.Nt[t],axis=0)
            adX[ind0] += (Gt*adGtX).ravel()

        if not self.multi_mode:
            _clear_adstate_names(self, adState)

        return adX
 
class Basis_gauss3d(_Basis_gauss3d):

    def __init__(self,config, State, multi_mode=False):
        super().__init__(config, State,multi_mode=multi_mode)

        self._operg_jit = jit(self._operg)
        self._operg_reduced_jit = jit(self._operg_reduced)
        self._ssh2uv_adj_jit = jit(self._ssh2uv_adj)
        self._ssh2uv_jit = jit(self._ssh2uv)
        
    def set_basis(self,time,return_q=False,**kwargs):
        res = super().set_basis(time,return_q=return_q,**kwargs)

        self.time = time
        self.vect_time = jnp.eye(time.size)

        self.zero_basis = jnp.zeros((self.nbasis,))
        self.zero_phys = jnp.zeros((self.nphys,))

        return res
    
    def _compute_component_space(self):
        """
            Gaussian functions in space
        """

        Gauss_2d = np.zeros((self.ENSLAT.size,self.lon1d.size))
        for i,(lat0,lon0) in enumerate(zip(self.ENSLAT,self.ENSLON)):
            indphys = np.where(
                    (np.abs((np.mod(self.lon1d - lon0+180,360)-180) / self.km2deg * np.cos(lat0 * np.pi / 180.)) <= self.sigma_D) &
                    (np.abs((self.lat1d - lat0) / self.km2deg) <= self.sigma_D)
                    )[0]
            xx = (np.mod(self.lon1d[indphys] - lon0+180,360)-180) / self.km2deg * np.cos(lat0 * np.pi / 180.) 
            yy = (self.lat1d[indphys] - lat0) / self.km2deg
            if self.mask1d is not None:
                indmask = self.mask1d[indphys]
                indphys = indphys[~indmask]
                xx = xx[~indmask]
                yy = yy[~indmask]
            Gauss_2d[i,indphys] = mywindow(xx / self.sigma_D) * mywindow(yy / self.sigma_D)
        Gauss_2d = jnp.array(Gauss_2d)
        self.Gauss_xy_T = sparse.CSR.fromdense(Gauss_2d)
        return sparse.CSR.fromdense(Gauss_2d.T)  

    def _compute_component_time(self, time):

        Gt_np = np.zeros((time.size,self.nbasis))
        ind_tmp = 0
        for it in range(len(self.ENST)):
            for _ in range(self.ENSLAT.size):
                for i,t in enumerate(time) :
                    dt = t - self.ENST[it]
                    if abs(dt) < self.sigma_T:
                        fact = self.window(dt / self.sigma_T) 
                        if self.normalize_fact:
                            fact /= self.norm_fact
                        if self.time_spinup is not None and t<self.time_spinup:
                            fact *= (1-self.window(t / self.time_spinup))
                        if fact!=0:   
                            Gt_np[i,ind_tmp:ind_tmp+1] = fact
                ind_tmp += 1
        Gt = sparse.csr_fromdense(jnp.array(Gt_np).T)

        return Gt, None
    
    def _ssh2uv(self, ssh):

        """
            Compute geostrophic velocities from SSH
        """

        _ssh = jnp.pad(ssh, pad_width=((1,0),(1,0)), mode='edge')

        _u = -self.g / self.f_on_u * (_ssh[1:,:] - _ssh[:-1,:]) / self.dy_on_u
        _v = self.g / self.f_on_v * (_ssh[:,1:] - _ssh[:,:-1]) / self.dx_on_v

        if self.u_coast_mask is not None:
            _u = jnp.where(self.u_coast_mask, 0., _u)
            _v = jnp.where(self.v_coast_mask, 0., _v)

        return _u, _v
    
    def get_Gt_value(self, t):

        idt = jnp.where(self.time == t, size=1)[0]  # Find index

        return self.Gauss_t @ self.vect_time[idt[0]] # Get corresponding value
    
    def _operg(self, t, X):

        """
            Project to physicial space
        """

        # Initialize phi
        phi = self.zero_phys.ravel()

        # Get Gt value
        Gt = self.get_Gt_value(t)
        GtX = Gt * X

        reshaped_GtX = GtX.reshape((-1, self.Nx))

        phi += self.Gauss_xy @ (reshaped_GtX.sum(axis=0))
        
        phi = phi.reshape(self.shape_phys)
        
        return phi

    def _operg_reduced(self, t, phi_2d):
        """Project a 2D physical adjoint field back to reduced space."""

        Gt = self.get_Gt_value(t)
        ad_space = self.Gauss_xy_T @ phi_2d.ravel()
        adX = Gt.reshape((-1, self.Nx)) * ad_space[None, :]

        return adX.ravel()

    def operg(self, t, X, State=None):
        
        """
            Project to physicial space
        """

        # Projection
        phi = self._operg_jit(t, X)

        # Compute geostrophic velocities
        if self.compute_velocities:
            u, v = self._ssh2uv_jit(phi)
            if State is not None:
                if not self.multi_mode:
                    State[self.name_mod_u] = u
                    State[self.name_mod_v] = v
                else:
                    State[self.name_mod_u] += u
                    State[self.name_mod_v] += v

        # Update State
        if State is not None:
            _assign_to_state_names(self, State, phi)
        else:
            if self.compute_velocities:
                return phi, u, v
            else:
                return phi
        
    def operg_transpose(self, t, adState):
        
        """
            Project to reduced space
        """

        _ensure_adstate_names(self, adState, jnp.zeros(self.shape_phys))
        if self.compute_velocities and (adState[self.name_mod_u] is None or adState[self.name_mod_v] is None):
            adState[self.name_mod_u] = jnp.zeros_like(self.f_on_u)
            adState[self.name_mod_v] = jnp.zeros_like(self.f_on_v)

        adparams = _sum_adstate_names(self, adState)
        if self.compute_velocities:
            adparams = adparams + self._ssh2uv_adj_jit(adState[self.name_mod_u], adState[self.name_mod_v])
        adX = self._operg_reduced_jit(t, adparams)
        
        if not self.multi_mode:
            _clear_adstate_names(self, adState)
            if self.compute_velocities:
                adState[self.name_mod_u] *= 0.
                adState[self.name_mod_v] *= 0.
        
        return adX


###############################################################################
#                        2D Gaussian (spatial only)                           #
###############################################################################
# Old version of the basis class, kept for reference. The new Basis_gauss3d class is preferred.
class _Basis_gauss2d:
    """Purely spatial Gaussian radial-basis functions.

    Each control coefficient multiplies one spatial Gaussian bell.
    No time dimension — the same set of coefficients is applied at every
    time step by ``operg``.
    """

    def __init__(self, config, State, multi_mode=False):

        self.km2deg = 1. / 110

        self.facns = config.BASIS.facns
        self.sigma_D = config.BASIS.sigma_D
        self.sigma_Q = config.BASIS.sigma_Q
        self.name_mod_var = config.BASIS.name_mod_var
        self.flag_variable_Q = config.BASIS.flag_variable_Q
        self.path_sad = config.BASIS.path_sad
        self.name_var_sad = config.BASIS.name_var_sad
        self.path_background = config.BASIS.path_background
        self.var_background = config.BASIS.var_background

        # C-grid variable type (None, 'U', or 'V')
        self.c_grid_var = getattr(config.BASIS, 'c_grid_var', None)

        # Grid params
        self.ny = State.ny
        self.nx = State.nx
        self.lon_min = State.lon_min
        self.lon_max = State.lon_max
        self.lat_min = State.lat_min
        self.lat_max = State.lat_max

        if self.c_grid_var == 'U':
            self.shape_phys = (State.ny, State.nx + 1)
            lon_h = State.lon
            lat_h = State.lat
            lon_u = np.zeros((State.ny, State.nx + 1))
            lat_u = np.zeros((State.ny, State.nx + 1))
            lon_u[:, 1:State.nx] = 0.5 * (lon_h[:, :-1] + lon_h[:, 1:])
            lat_u[:, 1:State.nx] = 0.5 * (lat_h[:, :-1] + lat_h[:, 1:])
            lon_u[:, 0] = lon_h[:, 0] - 0.5 * (lon_h[:, 1] - lon_h[:, 0])
            lat_u[:, 0] = lat_h[:, 0] - 0.5 * (lat_h[:, 1] - lat_h[:, 0])
            lon_u[:, State.nx] = lon_h[:, -1] + 0.5 * (lon_h[:, -1] - lon_h[:, -2])
            lat_u[:, State.nx] = lat_h[:, -1] + 0.5 * (lat_h[:, -1] - lat_h[:, -2])
            self.lon1d = lon_u.flatten()
            self.lat1d = lat_u.flatten()
        elif self.c_grid_var == 'V':
            self.shape_phys = (State.ny + 1, State.nx)
            lon_h = State.lon
            lat_h = State.lat
            lon_v = np.zeros((State.ny + 1, State.nx))
            lat_v = np.zeros((State.ny + 1, State.nx))
            lon_v[1:State.ny, :] = 0.5 * (lon_h[:-1, :] + lon_h[1:, :])
            lat_v[1:State.ny, :] = 0.5 * (lat_h[:-1, :] + lat_h[1:, :])
            lon_v[0, :] = lon_h[0, :] - 0.5 * (lon_h[1, :] - lon_h[0, :])
            lat_v[0, :] = lat_h[0, :] - 0.5 * (lat_h[1, :] - lat_h[0, :])
            lon_v[State.ny, :] = lon_h[-1, :] + 0.5 * (lon_h[-1, :] - lon_h[-2, :])
            lat_v[State.ny, :] = lat_h[-1, :] + 0.5 * (lat_h[-1, :] - lat_h[-2, :])
            self.lon1d = lon_v.flatten()
            self.lat1d = lat_v.flatten()
        else:
            self.shape_phys = (State.ny, State.nx)
            self.lon1d = State.lon.flatten()
            self.lat1d = State.lat.flatten()

        self.nphys = self.lon1d.size

        # Gravity
        self.g = 9.81

        # Geostrophic velocities
        self.compute_velocities = config.BASIS.compute_velocities
        self.name_mod_u = config.BASIS.name_mod_u
        self.name_mod_v = config.BASIS.name_mod_v
        pad = ((1, 0), (1, 0))
        _f = np.pad(State.f, pad_width=pad, mode='edge')
        self.f_on_v = 0.5 * (_f[:, 1:] + _f[:, :-1])
        self.f_on_u = 0.5 * (_f[1:, :] + _f[:-1, :])

        # Grid spacing
        self.dx = np.pad(State.DX, pad_width=pad, mode='edge')
        self.dy = np.pad(State.DY, pad_width=pad, mode='edge')
        self.dx_on_v = 0.5 * (self.dx[:, 1:] + self.dx[:, :-1])
        self.dy_on_u = 0.5 * (self.dy[1:, :] + self.dy[:-1, :])

        # Mask
        if State.mask is not None and np.any(State.mask):
            if self.c_grid_var == 'U':
                mask_u = np.zeros((State.ny, State.nx + 1), dtype=bool)
                mask_u[:, 1:State.nx] = State.mask[:, :-1] | State.mask[:, 1:]
                self.mask1d = mask_u.ravel()
            elif self.c_grid_var == 'V':
                mask_v = np.zeros((State.ny + 1, State.nx), dtype=bool)
                mask_v[1:State.ny, :] = State.mask[:-1, :] | State.mask[1:, :]
                self.mask1d = mask_v.ravel()
            else:
                self.mask1d = State.mask.ravel()
        else:
            self.mask1d = None

        # Coastal face masks for C-grid geostrophic velocity masking
        if State.mask is not None and np.any(State.mask):
            _mask_pad = np.pad(State.mask.astype(bool), pad_width=((1,0),(1,0)), mode='edge')
            self.u_coast_mask = _mask_pad[1:, :] | _mask_pad[:-1, :]  # (ny, nx+1)
            self.v_coast_mask = _mask_pad[:, 1:] | _mask_pad[:, :-1]  # (ny+1, nx)
        else:
            self.u_coast_mask = None
            self.v_coast_mask = None

        # Longitude unit
        self.lon_unit = State.lon_unit

        # For multi-basis
        self.multi_mode = multi_mode

    def set_basis(self, time, return_q=False, **kwargs):

        LON_MIN = self.lon_min
        LON_MAX = self.lon_max
        LAT_MIN = self.lat_min
        LAT_MAX = self.lat_max
        if LON_MAX < LON_MIN:
            LON_MAX = LON_MAX + 360.

        # Spatial Gaussian centres
        dlat = self.sigma_D / self.facns * self.km2deg
        lat0 = LAT_MIN - LAT_MIN % dlat - self.sigma_D * (1 - 1. / self.facns) * self.km2deg
        lat1 = LAT_MAX + 1.5 * dlat
        ENSLAT1 = np.arange(lat0, lat1, dlat)
        ENSLAT = []
        ENSLON = []
        for I in range(len(ENSLAT1)):
            dlon = self.sigma_D / self.facns / np.cos(ENSLAT1[I] * np.pi / 180.) * self.km2deg
            lon0 = LON_MIN - LON_MIN % dlon - self.sigma_D * (1 - 1. / self.facns) / np.cos(ENSLAT1[I] * np.pi / 180.) * self.km2deg
            lon1 = LON_MAX + dlon * 1.5
            ENSLON1 = np.arange(lon0, lon1, dlon)
            ENSLAT = np.concatenate(([ENSLAT, np.repeat(ENSLAT1[I], len(ENSLON1))]))
            ENSLON = np.concatenate(([ENSLON, ENSLON1]))
        self.ENSLAT = ENSLAT
        self.ENSLON = ENSLON

        self.nbasis = ENSLAT.size

        if self.c_grid_var == 'U':
            self.shape_phys = [self.ny, self.nx + 1]
        elif self.c_grid_var == 'V':
            self.shape_phys = [self.ny + 1, self.nx]
        else:
            self.shape_phys = [self.ny, self.nx]

        # Fill Q matrix
        if self.flag_variable_Q:
            Q = []
            sad = xr.open_dataset(self.path_sad)[self.name_var_sad['var']]
            if np.sign(sad[self.name_var_sad['lon']].data.min()) == -1 and self.lon_unit == '0_360':
                sad = sad.assign_coords({self.name_var_sad['lon']: ((self.name_var_sad['lon'], sad[self.name_var_sad['lon']].data % 360))})
            elif (np.sign(sad[self.name_var_sad['lon']].data.min()) >= 0 or sad[self.name_var_sad['lon']].data.max() > 180) and self.lon_unit == '-180_180':
                sad = sad.assign_coords({self.name_var_sad['lon']: ((self.name_var_sad['lon'], (sad[self.name_var_sad['lon']].data + 180) % 360 - 180))})
            sad = sad.sortby(sad[self.name_var_sad['lon']])
            for (lon, lat) in zip(ENSLON, ENSLAT):
                dlon_h = .5 * self.sigma_D / np.cos(lat * np.pi / 180.)
                dlat_h = .5 * self.sigma_D
                elon = np.linspace(lon - dlon_h, lon + dlon_h, 10)
                elat = np.linspace(lat - dlat_h, lat + dlat_h, 10)
                elon2, elat2 = np.meshgrid(elon, elat)
                std_tmp_values = sad.interp({self.name_var_sad['lon']: elon2.ravel(),
                                             self.name_var_sad['lat']: elat2.ravel()}).values
                std_tmp = np.nanmean(std_tmp_values) if not np.all(np.isnan(std_tmp_values)) else 10**-10
                Q.append(std_tmp / self.facns**.5)
            Q = np.array(Q)
        else:
            Q = self.sigma_Q / self.facns**.5 * np.ones((self.nbasis,))

        Xb = np.zeros_like(Q)
        if self.path_background is not None and os.path.exists(self.path_background):
            with xr.open_dataset(self.path_background) as ds:
                print(f'Load background from file: {self.path_background}')
                Xb = ds[self.var_background].values[:len(Xb)]

        print(f'sigma_D={self.sigma_D:.1E}',
              f'nlocs={ENSLAT.size:.1E}',
              f'Q={np.mean(Q):.1E}')
        print(f'reduced order: {time.size * self.nphys} --> {self.nbasis}\n'
              f' reduced factor: {int(time.size * self.nphys / self.nbasis)}')

        # Compute spatial basis matrix
        self.Gauss_xy = self._compute_component_space()

        if return_q:
            return Xb, Q

    def _compute_component_space(self):
        """Gaussian functions in space."""

        data = np.empty((self.ENSLAT.size * self.lon1d.size,))
        indices = np.empty((self.ENSLAT.size * self.lon1d.size,), dtype=int)
        sizes = np.zeros((self.ENSLAT.size,), dtype=int)
        ind_tmp = 0
        for i, (lat0, lon0) in enumerate(zip(self.ENSLAT, self.ENSLON)):
            indphys = np.where(
                (np.abs((np.mod(self.lon1d - lon0 + 180, 360) - 180) / self.km2deg * np.cos(lat0 * np.pi / 180.)) <= self.sigma_D) &
                (np.abs((self.lat1d - lat0) / self.km2deg) <= self.sigma_D)
            )[0]
            xx = (np.mod(self.lon1d[indphys] - lon0 + 180, 360) - 180) / self.km2deg * np.cos(lat0 * np.pi / 180.)
            yy = (self.lat1d[indphys] - lat0) / self.km2deg
            if self.mask1d is not None:
                indmask = self.mask1d[indphys]
                indphys = indphys[~indmask]
                xx = xx[~indmask]
                yy = yy[~indmask]
            sizes[i] = indphys.size
            indices[ind_tmp:ind_tmp + indphys.size] = indphys
            data[ind_tmp:ind_tmp + indphys.size] = mywindow(xx / self.sigma_D) * mywindow(yy / self.sigma_D)
            ind_tmp += indphys.size
        indptr = np.zeros((i + 2,), dtype=int)
        indptr[1:] = np.cumsum(sizes)
        return csc_matrix((data, indices, indptr), shape=(self.lon1d.size, self.ENSLAT.size))

    def _ssh2uv(self, ssh):
        _ssh = np.pad(ssh, pad_width=((1, 0), (1, 0)), mode='edge')
        _u = -self.g / self.f_on_u * (_ssh[1:, :] - _ssh[:-1, :]) / self.dy_on_u
        _v = self.g / self.f_on_v * (_ssh[:, 1:] - _ssh[:, :-1]) / self.dx_on_v
        if self.u_coast_mask is not None:
            _u = np.where(self.u_coast_mask, 0., _u)
            _v = np.where(self.v_coast_mask, 0., _v)
        return _u, _v

    def _ssh2uv_adj(self, adu, adv):
        """
        Adjoint of geostrophic velocity computation.
        Uses jnp so that JAX device arrays are kept on-device (no GPU→CPU transfer).
        """
        if self.u_coast_mask is not None:
            adu = jnp.where(self.u_coast_mask, 0., adu)
            adv = jnp.where(self.v_coast_mask, 0., adv)
        _adssh = jnp.zeros((self.shape_phys[0] + 1, self.shape_phys[1] + 1))
        _adssh = _adssh.at[1:, :].add( -self.g / self.f_on_u * adu / self.dy_on_u)
        _adssh = _adssh.at[:-1, :].add(  self.g / self.f_on_u * adu / self.dy_on_u)
        _adssh = _adssh.at[:, 1:].add(   self.g / self.f_on_v * adv / self.dx_on_v)
        _adssh = _adssh.at[:, :-1].add( -self.g / self.f_on_v * adv / self.dx_on_v)
        adssh = _adssh[1:, 1:]
        adssh = adssh.at[0, :].add(_adssh[0, 1:])
        adssh = adssh.at[:, 0].add(_adssh[1:, 0])
        adssh = adssh.at[0, 0].add(_adssh[0, 0])
        return adssh

    def operg(self, t, X, State=None):
        """Project control vector to physical space."""

        phi = self.Gauss_xy.dot(X).reshape(self.shape_phys)

        if self.compute_velocities:
            u, v = self._ssh2uv(phi)
            if State is not None:
                if not self.multi_mode:
                    State[self.name_mod_u] = u
                    State[self.name_mod_v] = v
                else:
                    State[self.name_mod_u] += u
                    State[self.name_mod_v] += v

        if State is not None:
            _assign_to_state_names(self, State, phi)
        else:
            if self.compute_velocities:
                return phi, u, v
            else:
                return phi

    def operg_transpose(self, t, adState):
        """Project adjoint physical-space field to control space."""

        _ensure_adstate_names(self, adState, np.zeros(self.shape_phys))
        if self.compute_velocities and (adState[self.name_mod_u] is None or adState[self.name_mod_v] is None):
            adState[self.name_mod_u] = np.zeros_like(self.f_on_u)
            adState[self.name_mod_v] = np.zeros_like(self.f_on_v)

        adparams = _sum_adstate_names(self, adState).ravel()
        if self.compute_velocities:
            adparams = adparams + self._ssh2uv_adj(adState[self.name_mod_u], adState[self.name_mod_v]).ravel()

        adX = self.Gauss_xy.T.dot(adparams)

        if not self.multi_mode:
            _clear_adstate_names(self, adState)
            if self.compute_velocities:
                adState[self.name_mod_u] *= 0.
                adState[self.name_mod_v] *= 0.

        return adX

class Basis_gauss2d(_Basis_gauss2d):
    """JAX-differentiable version of :class:`Basis_gauss2d`."""

    def __init__(self, config, State, multi_mode=False):
        super().__init__(config, State, multi_mode=multi_mode)
        self._operg_jit = jit(self._operg)
        self._operg_reduced_jit = jit(self._operg_reduced)
        self._ssh2uv_adj_jit = jit(self._ssh2uv_adj)
        self._ssh2uv_jit = jit(self._ssh2uv)

    def set_basis(self, time, return_q=False, **kwargs):
        res = super().set_basis(time, return_q=return_q, **kwargs)
        self.zero_basis = jnp.zeros((self.nbasis,))
        self.zero_phys = jnp.zeros((self.nphys,))
        return res

    def _compute_component_space(self):
        """Gaussian functions in space (JAX sparse CSR)."""

        Gauss_2d = np.zeros((self.ENSLAT.size, self.lon1d.size))
        for i, (lat0, lon0) in enumerate(zip(self.ENSLAT, self.ENSLON)):
            indphys = np.where(
                (np.abs((np.mod(self.lon1d - lon0 + 180, 360) - 180) / self.km2deg * np.cos(lat0 * np.pi / 180.)) <= self.sigma_D) &
                (np.abs((self.lat1d - lat0) / self.km2deg) <= self.sigma_D)
            )[0]
            xx = (np.mod(self.lon1d[indphys] - lon0 + 180, 360) - 180) / self.km2deg * np.cos(lat0 * np.pi / 180.)
            yy = (self.lat1d[indphys] - lat0) / self.km2deg
            if self.mask1d is not None:
                indmask = self.mask1d[indphys]
                indphys = indphys[~indmask]
                xx = xx[~indmask]
                yy = yy[~indmask]
            Gauss_2d[i, indphys] = mywindow(xx / self.sigma_D) * mywindow(yy / self.sigma_D)
        Gauss_2d = jnp.array(Gauss_2d)
        self.Gauss_xy_T = sparse.CSR.fromdense(Gauss_2d)
        return sparse.CSR.fromdense(Gauss_2d.T)

    def _ssh2uv(self, ssh):
        _ssh = jnp.pad(ssh, pad_width=((1, 0), (1, 0)), mode='edge')
        _u = -self.g / self.f_on_u * (_ssh[1:, :] - _ssh[:-1, :]) / self.dy_on_u
        _v = self.g / self.f_on_v * (_ssh[:, 1:] - _ssh[:, :-1]) / self.dx_on_v
        if self.u_coast_mask is not None:
            _u = jnp.where(self.u_coast_mask, 0., _u)
            _v = jnp.where(self.v_coast_mask, 0., _v)
        return _u, _v

    def _operg(self, X):
        """Forward projection (JAX traceable)."""
        return (self.Gauss_xy @ X).reshape(self.shape_phys)

    def _operg_reduced(self, phi_2d):
        """Project a 2D physical adjoint field back to reduced space."""
        return self.Gauss_xy_T @ phi_2d.ravel()

    def operg(self, t, X, State=None):
        """Project control vector to physical space."""

        phi = self._operg_jit(X)

        if self.compute_velocities:
            u, v = self._ssh2uv_jit(phi)
            if State is not None:
                if not self.multi_mode:
                    State[self.name_mod_u] = u
                    State[self.name_mod_v] = v
                else:
                    State[self.name_mod_u] += u
                    State[self.name_mod_v] += v

        if State is not None:
            _assign_to_state_names(self, State, phi)
        else:
            if self.compute_velocities:
                return phi, u, v
            else:
                return phi

    def operg_transpose(self, t, adState):
        """Project adjoint physical-space field to control space."""

        _ensure_adstate_names(self, adState, jnp.zeros(self.shape_phys))
        if self.compute_velocities and (adState[self.name_mod_u] is None or adState[self.name_mod_v] is None):
            adState[self.name_mod_u] = jnp.zeros_like(self.f_on_u)
            adState[self.name_mod_v] = jnp.zeros_like(self.f_on_v)

        adparams = _sum_adstate_names(self, adState)
        if self.compute_velocities:
            adparams = adparams + self._ssh2uv_adj_jit(adState[self.name_mod_u], adState[self.name_mod_v])

        adX = self._operg_reduced_jit(adparams)

        if not self.multi_mode:
            _clear_adstate_names(self, adState)
            if self.compute_velocities:
                adState[self.name_mod_u] *= 0.
                adState[self.name_mod_v] *= 0.

        return adX


###############################################################################
#                            Balanced Motions                                 #
###############################################################################
# Old version of the basis class, kept for reference. The new Basis_bmaux class is preferred.
class _Basis_bmaux:
   
    def __init__(self,config,State,multi_mode=False):

        self.km2deg=1./110
        
        # Internal params
        self.file_aux = config.BASIS.file_aux
        self.flux = config.BASIS.flux
        self.facns = config.BASIS.facns 
        self.facnlt = config.BASIS.facnlt
        self.npsp = config.BASIS.npsp 
        self.facpsp = config.BASIS.facpsp 
        self.lmin = config.BASIS.lmin 
        self.lmax = config.BASIS.lmax
        self.tdecmin = config.BASIS.tdecmin
        self.tdecmax = config.BASIS.tdecmax
        self.factdec = config.BASIS.factdec
        self.facQ = config.BASIS.facQ
        self.facQ_aux_path = config.BASIS.facQ_aux_path
        self.l_largescale = config.BASIS.l_largescale
        self.facQ_largescale = config.BASIS.facQ_largescale
        self.name_mod_var = config.BASIS.name_mod_var
        self.path_background = config.BASIS.path_background
        self.var_background = config.BASIS.var_background
        self.norm_time = config.BASIS.norm_time

        # C-grid variable type (None, 'U', or 'V')
        self.c_grid_var = getattr(config.BASIS, 'c_grid_var', None)
        
        # Grid params
        self.ny = State.ny
        self.nx = State.nx
        self.lon_min = State.lon_min
        self.lon_max = State.lon_max
        self.lat_min = State.lat_min
        self.lat_max = State.lat_max

        if self.c_grid_var == 'U':
            self.shape_phys = (State.ny, State.nx + 1)
            lon_h = State.lon
            lat_h = State.lat
            lon_u = np.zeros((State.ny, State.nx + 1))
            lat_u = np.zeros((State.ny, State.nx + 1))
            lon_u[:, 1:State.nx] = 0.5 * (lon_h[:, :-1] + lon_h[:, 1:])
            lat_u[:, 1:State.nx] = 0.5 * (lat_h[:, :-1] + lat_h[:, 1:])
            lon_u[:, 0] = lon_h[:, 0] - 0.5 * (lon_h[:, 1] - lon_h[:, 0])
            lat_u[:, 0] = lat_h[:, 0] - 0.5 * (lat_h[:, 1] - lat_h[:, 0])
            lon_u[:, State.nx] = lon_h[:, -1] + 0.5 * (lon_h[:, -1] - lon_h[:, -2])
            lat_u[:, State.nx] = lat_h[:, -1] + 0.5 * (lat_h[:, -1] - lat_h[:, -2])
            self.lon1d = lon_u.flatten()
            self.lat1d = lat_u.flatten()
        elif self.c_grid_var == 'V':
            self.shape_phys = (State.ny + 1, State.nx)
            lon_h = State.lon
            lat_h = State.lat
            lon_v = np.zeros((State.ny + 1, State.nx))
            lat_v = np.zeros((State.ny + 1, State.nx))
            lon_v[1:State.ny, :] = 0.5 * (lon_h[:-1, :] + lon_h[1:, :])
            lat_v[1:State.ny, :] = 0.5 * (lat_h[:-1, :] + lat_h[1:, :])
            lon_v[0, :] = lon_h[0, :] - 0.5 * (lon_h[1, :] - lon_h[0, :])
            lat_v[0, :] = lat_h[0, :] - 0.5 * (lat_h[1, :] - lat_h[0, :])
            lon_v[State.ny, :] = lon_h[-1, :] + 0.5 * (lon_h[-1, :] - lon_h[-2, :])
            lat_v[State.ny, :] = lat_h[-1, :] + 0.5 * (lat_h[-1, :] - lat_h[-2, :])
            self.lon1d = lon_v.flatten()
            self.lat1d = lat_v.flatten()
        else:
            self.shape_phys = (State.ny, State.nx)
            self.lon1d = State.lon.flatten()
            self.lat1d = State.lat.flatten()

        self.nphys = self.lon1d.size

        # Gravity 
        self.g = 9.81

        # Compute geostrophic velocoties
        self.compute_velocities = config.BASIS.compute_velocities
        self.name_mod_u = config.BASIS.name_mod_u
        self.name_mod_v = config.BASIS.name_mod_v
        pad = ((1,0),(1,0))
        _f = np.pad(State.f, pad_width=pad, mode='edge')
        self.f_on_v = 0.5*(_f[:,1:] + _f[:,:-1])
        self.f_on_u = 0.5*(_f[1:,:] + _f[:-1,:])

        # Grid spacing
        self.dx = np.pad(State.DX, pad_width=pad, mode='edge')
        self.dy = np.pad(State.DY, pad_width=pad, mode='edge')
        self.dx_on_v = 0.5*(self.dx[:,1:] + self.dx[:,:-1])
        self.dy_on_u = 0.5*(self.dy[1:,:] + self.dy[:-1,:])

        # Reference time to have fixed time coordinates
        self.delta_time_ref = (config.EXP.init_date - datetime.datetime(1950,1,1,0)).total_seconds() / 24/3600

        # Mask
        if State.mask is not None and np.any(State.mask):
            if self.c_grid_var == 'U':
                mask_u = np.zeros((State.ny, State.nx + 1), dtype=bool)
                mask_u[:, 1:State.nx] = State.mask[:, :-1] | State.mask[:, 1:]
                self.mask1d = mask_u.ravel()
            elif self.c_grid_var == 'V':
                mask_v = np.zeros((State.ny + 1, State.nx), dtype=bool)
                mask_v[1:State.ny, :] = State.mask[:-1, :] | State.mask[1:, :]
                self.mask1d = mask_v.ravel()
            else:
                self.mask1d = State.mask.ravel()
        else:
            self.mask1d = None

        # Coastal face masks for C-grid geostrophic velocity masking
        if State.mask is not None and np.any(State.mask):
            _mask_pad = np.pad(State.mask.astype(bool), pad_width=((1,0),(1,0)), mode='edge')
            self.u_coast_mask = _mask_pad[1:, :] | _mask_pad[:-1, :]  # (ny, nx+1)
            self.v_coast_mask = _mask_pad[:, 1:] | _mask_pad[:, :-1]  # (ny+1, nx)
        else:
            self.u_coast_mask = None
            self.v_coast_mask = None

        # Depth data
        if config.BASIS.file_depth is not None:
            ds = xr.open_dataset(config.BASIS.file_depth)
            lon_depth = ds[config.BASIS.name_var_depth['lon']].values
            lat_depth = ds[config.BASIS.name_var_depth['lat']].values
            var_depth = ds[config.BASIS.name_var_depth['var']].values
            finterpDEPTH = scipy.interpolate.RegularGridInterpolator((lon_depth,lat_depth),var_depth,bounds_error=False,fill_value=None)
            self.depth = -finterpDEPTH((self.lon1d,self.lat1d))
            self.depth[np.isnan(self.depth)] = 0.
            self.depth[np.isnan(self.depth)] = 0.

            self.depth1 = config.BASIS.depth1
            self.depth2 = config.BASIS.depth2
        else:
            self.depth = None
        
        # FacQ_aux file (e.g. from background error)
        if config.BASIS.file_facQaux is not None:
            self.file_facQaux = config.BASIS.file_facQaux
            self.name_var_facQaux = config.BASIS.name_var_facQaux
        else:
            self.file_facQaux = None

        # Longitude unit
        self.lon_unit = State.lon_unit

        # Dictionnaries to save wave coefficients and indexes for repeated runs
        self.path_save_tmp = config.EXP.tmp_DA_path

        # Time window
        if self.flux:
            self.window = mywindow_flux
        else:
            self.window = mywindow
        
        self.multi_mode = multi_mode

    def set_basis(self,time,return_q=False,**kwargs):

        print('Setting Basis BMaux...')

        Mutltiple_basis_exp = False
        if Mutltiple_basis_exp:  
            L_MIN = 30
            L_MAX = 1000 

            TIME_MIN = time.min()
            TIME_MAX = time.max()
            LON_MIN = self.lon_min
            LON_MAX = self.lon_max
            LAT_MIN = self.lat_min
            LAT_MAX = self.lat_max
            if (LON_MAX<LON_MIN): LON_MAX = LON_MAX+360.

            # Ensemble of pseudo-frequencies for the wavelets (spatial)
            logff_all = np.arange(
                np.log(1./L_MIN),
                np.log(1. / L_MAX) - np.log(1 + self.facpsp / self.npsp),
                -np.log(1 + self.facpsp / self.npsp))[::-1]
        
            logff = logff_all[(logff_all>=np.log(1/self.lmax)) & (logff_all<=np.log(1/self.lmin))] 
            ff = np.exp(logff)
            ff = ff[1/ff<=self.lmax]
            dff = ff[1:] - ff[:-1]

        else: 
        
            TIME_MIN = time.min()
            TIME_MAX = time.max()
            LON_MIN = self.lon_min
            LON_MAX = self.lon_max
            LAT_MIN = self.lat_min
            LAT_MAX = self.lat_max
            if (LON_MAX<LON_MIN): LON_MAX = LON_MAX+360.

            # Ensemble of pseudo-frequencies for the wavelets (spatial)
            logff = np.arange(
                np.log(1./self.lmin),
                np.log(1. / self.lmax) - np.log(1 + self.facpsp / self.npsp),
                -np.log(1 + self.facpsp / self.npsp))[::-1]
            
            ff = np.exp(logff)
            #ff = ff[1/ff<=self.lmax]
            dff = ff[1:] - ff[:-1]
        
        # Ensemble of directions for the wavelets (2D plane)
        theta = np.linspace(0, np.pi, int(np.pi * ff[0] / dff[0] * self.facpsp))[:-1]
        ntheta = len(theta)
        nf = len(ff)
        logging.info('spatial normalized wavelengths: %s', 1./np.exp(logff))
        logging.info('ntheta: %s', ntheta)

        # Global time window
        deltat = TIME_MAX - TIME_MIN

        # Auxiliary data
        aux = xr.open_dataset(self.file_aux,decode_times=False)
        if np.sign(aux['lon'].data.min())==-1 and self.lon_unit=='0_360':
            aux = aux.assign_coords({'lon':(('lon', aux['lon'].data % 360))})
        elif (np.sign(aux['lon'].data.min())>=0 or aux['lon'].data.max()>180) and self.lon_unit=='-180_180':
            aux = aux.assign_coords({'lon':(('lon', (aux['lon'].data + 180) % 360 - 180 ))})
        aux = aux.sortby(aux['lon'])    
        daStd = aux['Std']
        if 'tdec' in daStd.dims:
            _multi_scale = True
            _tdec_bins_file = aux['tdec'].values.astype(float)
            _n_tdec_bins = len(_tdec_bins_file)
            daTdec = None
        else:
            _multi_scale = False
            _n_tdec_bins = 1
            _tdec_bins_file = None
            daTdec = aux['Tdec']

        # Auxiliary for Q
        if self.file_facQaux is not None:
            auxQ = xr.open_dataset(self.file_facQaux,decode_times=False)
            if np.sign(auxQ['lon'].data.min())==-1 and self.lon_unit=='0_360':
                auxQ = auxQ.assign_coords({'lon':(('lon', auxQ['lon'].data % 360))})
            elif (np.sign(auxQ['lon'].data.min())>=0 or auxQ['lon'].data.max()>180) and self.lon_unit=='-180_180':
                auxQ = auxQ.assign_coords({'lon':(('lon', (auxQ['lon'].data + 180) % 360 - 180 ))})
            auxQ = auxQ.sortby(auxQ['lon'])    
            daFacQ = auxQ[self.name_var_facQaux['var']]

        # Wavelet space-time coordinates
        ENSLON = [None]*nf # Ensemble of longitudes of the center of each wavelets
        ENSLAT = [None]*nf # Ensemble of latitudes of the center of each wavelets
        
        DX = 1./ff*self.npsp * 0.5 # wavelet extension
        #DXG = DX / self.facns # distance (km) between the wavelets grid in space
        NP = np.empty(nf, dtype='int64') # Nomber of spatial wavelet locations for a given frequency

        for iff in range(nf):
            
            # Spatial coordinates of wavelet components
            ENSLON[iff] = []
            ENSLAT[iff] = []

            facns = self.facns
            DXG = DX / facns

            # Latitudes
            dlat = DXG[iff]*self.km2deg
            lat0 = LAT_MIN - LAT_MIN%dlat - DX[iff]*self.km2deg  # To start at a fix latitude
            lat1 = LAT_MAX + DX[iff]*self.km2deg 
            ENSLAT1 = np.arange(lat0, lat1, dlat)
            
            # Longitudes
            for I in range(len(ENSLAT1)):
                dlon = DXG[iff]/np.cos(ENSLAT1[I]*np.pi/180.) *self.km2deg
                lon0 = LON_MIN - LON_MIN%dlon - DX[iff]/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg # To start at a fix longitude
                lon1 = LON_MAX + DX[iff]/np.cos(ENSLAT1[I]*np.pi/180.)*self.km2deg 
                _ENSLON = np.arange(lon0, lon1, dlon)
                _ENSLAT = np.repeat(ENSLAT1[I],len(_ENSLON))

                if self.mask1d is None:
                    _ENSLON1 = _ENSLON
                    _ENSLAT1 = _ENSLAT
                
                else:
                    # Avoid wave component for which the state grid points are full masked
                    _ENSLON1 = []
                    _ENSLAT1 = []
                    for (lon,lat) in zip(_ENSLON,_ENSLAT):
                        indphys = np.where(
                            (np.abs((self.lon1d - lon) / self.km2deg * np.cos(lat * np.pi / 180.)) <= 1/ff[iff]) &
                            (np.abs((self.lat1d - lat) / self.km2deg) <= 1/ff[iff])
                            )[0]
                        if not np.all(self.mask1d[indphys]):
                            _ENSLON1.append(lon)
                            _ENSLAT1.append(lat)                    

                ENSLAT[iff] = np.concatenate(([ENSLAT[iff],_ENSLAT1]))
                ENSLON[iff] = np.concatenate(([ENSLON[iff],_ENSLON1]))
            NP[iff] = len(ENSLON[iff])

        # ---- Multi-scale expansion: replicate each spatial band per tdec bin ----
        if _multi_scale:
            nf_eff = nf * _n_tdec_bins
            ENSLON_eff = [ENSLON[ieff // _n_tdec_bins] for ieff in range(nf_eff)]
            ENSLAT_eff = [ENSLAT[ieff // _n_tdec_bins] for ieff in range(nf_eff)]
            NP_eff = np.array([NP[ieff // _n_tdec_bins] for ieff in range(nf_eff)], dtype='int64')
            DX_eff = np.array([DX[ieff // _n_tdec_bins] for ieff in range(nf_eff)])
            ff_eff = np.array([ff[ieff // _n_tdec_bins] for ieff in range(nf_eff)])
        else:
            nf_eff = nf
            ENSLON_eff = ENSLON
            ENSLAT_eff = ENSLAT
            NP_eff = NP
            DX_eff = DX
            ff_eff = ff

        # ---- Temporal setup for each effective band -------------------------
        enst = [None] * nf_eff
        tdec = [None] * nf_eff
        norm_fact = [None] * nf_eff
        for ieff in range(nf_eff):
            iff_o = ieff // _n_tdec_bins if _multi_scale else ieff
            k_o   = ieff %  _n_tdec_bins if _multi_scale else 0
            NP_i = int(NP_eff[ieff])
            tdec[ieff] = [None] * NP_i
            enst[ieff] = [None] * NP_i
            norm_fact[ieff] = [None] * NP_i
            for P in range(NP_i):
                dlon = DX_eff[ieff]*self.km2deg/np.cos(ENSLAT_eff[ieff][P] * np.pi / 180.)
                dlat = DX_eff[ieff]*self.km2deg
                elon = np.linspace(ENSLON_eff[ieff][P]-dlon, ENSLON_eff[ieff][P]+dlon, 10)
                elat = np.linspace(ENSLAT_eff[ieff][P]-dlat, ENSLAT_eff[ieff][P]+dlat, 10)
                elon2, elat2 = np.meshgrid(elon, elat)
                if _multi_scale:
                    tdec_val = float(_tdec_bins_file[k_o]) * self.factdec
                else:
                    tdec_tmp = daTdec.interp(f=ff_eff[ieff], lon=elon2.flatten(), lat=elat2.flatten()).values
                    tdec_val = float(np.nanmean(tdec_tmp)) if not np.all(np.isnan(tdec_tmp)) else 0.0
                    tdec_val *= self.factdec
                if tdec_val < self.tdecmin:
                    tdec_val = self.tdecmin
                if tdec_val > self.tdecmax:
                    tdec_val = self.tdecmax
                tdec[ieff][P] = tdec_val
                # Compute time integral for normalization
                if self.norm_time:
                    tt = np.linspace(-tdec_val, tdec_val)
                    tmp = np.zeros_like(tt)
                    for i in range(tt.size-1):
                        tmp[i+1] = tmp[i] + self.window(tt[i]/tdec_val)*(tt[i+1]-tt[i])
                    norm_fact[ieff][P] = tmp.max()
                else:
                    norm_fact[ieff][P] = 1
                t0 = -self.delta_time_ref % tdec_val
                enst[ieff][P] = np.arange(t0 - tdec_val/self.facnlt, deltat+tdec_val/self.facnlt, tdec_val/self.facnlt)
                
        # Harmonize the wavelet time center dimensions for all point by adding NaN if needed 
        # (we must do that for the time operator Gt to be independent from the space operator Gx)
        enst_same_dim = [None]*nf_eff
        for ieff in range(nf_eff):
            max_number_enst_ieff = np.max([enst[ieff][P].size for P in range(NP_eff[ieff])])
            enst_same_dim[ieff] = np.zeros((NP_eff[ieff], max_number_enst_ieff)) * np.nan
            for P in range(NP_eff[ieff]):
                enst_same_dim[ieff][P, :enst[ieff][P].size] = enst[ieff][P]
        
        # Fill the Q diagonal matrix (expected variance for each wavelet)   
        print('Computing Q')  

        iwave = 0
        self.iff_wavebounds = [None]*(nf_eff+1)
        Q = np.array([])
        facQ = self.facQ  # Move outside the loop for efficiency
        facQ_largescale = self.facQ_largescale 
        l_largescale = self.l_largescale 

        std = []
        facQaux = []
        facQaux = []
        for ieff in range(nf_eff):
            iff_o = ieff // _n_tdec_bins if _multi_scale else ieff
            k_o   = ieff %  _n_tdec_bins if _multi_scale else 0
            std.append([])
            std[ieff] = []
            if self.file_facQaux is not None:
                facQaux.append([])
                facQaux[ieff] = []
            for P in range(NP_eff[ieff]):
                
                dlon = DX_eff[ieff] * self.km2deg / np.cos(ENSLAT_eff[ieff][P] * np.pi / 180.0)
                dlat = DX_eff[ieff] * self.km2deg

                # Precompute interpolation grid once
                elon = np.linspace(ENSLON_eff[ieff][P] - dlon, ENSLON_eff[ieff][P] + dlon, 10)
                elat = np.linspace(ENSLAT_eff[ieff][P] - dlat, ENSLAT_eff[ieff][P] + dlat, 10)
                elon2, elat2 = np.meshgrid(elon, elat)

                if _multi_scale:
                    std_tmp_values = daStd.interp(
                        f=float(ff[iff_o]),
                        tdec=float(_tdec_bins_file[k_o]),
                        lon=elon2.ravel(),
                        lat=elat2.ravel(),
                    ).values
                else:
                    std_tmp_values = daStd.interp(f=ff_eff[ieff], lon=elon2.ravel(), lat=elat2.ravel()).values
                std_tmp = np.nanmean(std_tmp_values) if not np.all(np.isnan(std_tmp_values)) else 10**-10 
                std[ieff].append(std_tmp) 

                if self.file_facQaux is not None:
                    facQaux_tmp_values = daFacQ.interp({self.name_var_facQaux['wavenumber']:ff_eff[ieff], 
                                                        self.name_var_facQaux['lon']:elon2.ravel(), 
                                                        self.name_var_facQaux['lat']:elat2.ravel()}).values
                    facQaux_tmp = np.nanmean(facQaux_tmp_values) if not np.all(np.isnan(facQaux_tmp_values)) else 1.0
                    facQaux[ieff].append(facQaux_tmp)

        for ieff in range(nf_eff):

            self.iff_wavebounds[ieff] = iwave
            _nwavef = 0
            Qf_list = []  # Use a list instead of np.concatenate in loops

            enst_data = enst_same_dim[ieff]  # Store reference to avoid repeated access
            num_it = enst_data.shape[1]

            for it in range(num_it):
                for P in range(NP_eff[ieff]):
                    enst_value = enst_data[P, it]
                    if np.isnan(enst_value):
                        Q_tmp = 10**-10  # Small nonzero value to avoid division errors
                    else:
                        Q_tmp = self._Q_from_std(std[ieff][P], tdec[ieff][P])

                    Q_tmp *= facQ   # Multiply after NaN check

                    # Include facQaux if available
                    if self.file_facQaux is not None:
                        Q_tmp *= np.sqrt(facQaux[ieff][P])

                    # Store Q_tmp values in list for later concatenation
                    Qf_list.append(Q_tmp * np.ones(2 * ntheta))
                    _nwavef += 2 * ntheta

            # Convert list to numpy array once
            if Qf_list:
                Qf = np.concatenate(Qf_list)
                Q = np.concatenate((Q, Qf))

            iwave += _nwavef

            print(f'lambda={1/ff_eff[ieff]:.1E}',
                f'nlocs={NP_eff[ieff]:.1E}',
                f'tdec={np.mean(tdec[ieff]):.1E}',
                f'Q={np.mean(Q[self.iff_wavebounds[ieff]:iwave]):.1E}')

        self.iff_wavebounds[-1] = iwave

        Xb = np.zeros_like(Q)
        # Background
        if self.path_background is not None and os.path.exists(self.path_background):
            with xr.open_dataset(self.path_background) as ds:
                print('bmaux np.shape(Xb)',np.shape(Xb))
                print('bmaux np.shape(ds[self.var_background].values)',np.shape(ds[self.var_background].values))
                print(f'Load background from file: {self.path_background}') 
                Xb = ds[self.var_background].values[-len(Xb):] 

            

        self.DX=DX_eff
        self.ENSLON=ENSLON_eff
        self.ENSLAT=ENSLAT_eff
        self.NP=NP_eff
        self.tdec=tdec
        self.norm_fact = norm_fact
        self.enst=enst_same_dim
        self.nbasis=Q.size
        self.nf=nf_eff
        self.theta=theta
        self.ntheta=ntheta
        self.ff=ff_eff
        self.k = 2 * np.pi * ff_eff

        # Compute basis components
        print('Computing Spatial components')
        self.Gx, self.Nx = self._compute_component_space() # in space
        print('Computing Time components')
        self.Gt, self.Nt = self._compute_component_time(time) # in time
        
        print(f'reduced order: {time.size * self.nphys} --> {self.nbasis}\n reduced factor: {int(time.size * self.nphys/self.nbasis)}')
            
        if return_q:
            return Xb, Q

    def _Q_from_std(self, std_val: float, tdec_val: float) -> float:
        """Convert calibrated Std to prior variance Q.

        Legacy convention (default): Q = Std.
        Override in subclasses for self-consistent normalisation.
        """
        return float(std_val)
        
    def _compute_component_space(self):

        Gx = [None,]*self.nf
        Nx = [None,]*self.nf

        for iff in range(self.nf):

            data = np.empty((2*self.ntheta*self.NP[iff]*self.nphys,))
            indices = np.empty((2*self.ntheta*self.NP[iff]*self.nphys,),dtype=int)
            sizes = np.zeros((2*self.ntheta*self.NP[iff],),dtype=int)

            ind_tmp = 0
            iwave = 0

            for P in range(self.NP[iff]):
                # Obs selection around point P
                indphys = np.where(
                    (np.abs((self.lon1d - self.ENSLON[iff][P]) / self.km2deg * np.cos(self.ENSLAT[iff][P] * np.pi / 180.)) <= self.DX[iff]) &
                    (np.abs((self.lat1d - self.ENSLAT[iff][P]) / self.km2deg) <= self.DX[iff])
                    )[0]
                xx = (self.lon1d[indphys] - self.ENSLON[iff][P]) / self.km2deg * np.cos(self.ENSLAT[iff][P] * np.pi / 180.) 
                yy = (self.lat1d[indphys] - self.ENSLAT[iff][P]) / self.km2deg
                # Spatial tapering shape of the wavelet 
                if self.mask1d is not None:
                    indmask = self.mask1d[indphys]
                    indphys = indphys[~indmask]
                    xx = xx[~indmask]
                    yy = yy[~indmask]
                facd = np.ones((indphys.size))
                if self.depth is not None:
                    facd = (self.depth[indphys]-self.depth1)/(self.depth2-self.depth1)
                    facd[facd>1]=1.
                    facd[facd<0]=0.
                    indphys = indphys[facd>0]
                    xx = xx[facd>0]
                    yy = yy[facd>0]
                    facd = facd[facd>0]

                facs = mywindow(xx / self.DX[iff]) * mywindow(yy / self.DX[iff]) * facd

                for itheta in range(self.ntheta):
                    # Wave vector components
                    kx = self.k[iff] * np.cos(self.theta[itheta])
                    ky = self.k[iff] * np.sin(self.theta[itheta])
                    # Cosine component
                    sizes[iwave] = indphys.size
                    indices[ind_tmp:ind_tmp+indphys.size] = indphys
                    data[ind_tmp:ind_tmp+indphys.size] = np.sqrt(2) * facs * np.cos(kx*(xx)+ky*(yy))
                    ind_tmp += indphys.size
                    iwave += 1
                    # Sine component
                    sizes[iwave] = indphys.size
                    indices[ind_tmp:ind_tmp+indphys.size] = indphys
                    data[ind_tmp:ind_tmp+indphys.size] = np.sqrt(2) * facs * np.sin(kx*(xx)+ky*(yy))
                    ind_tmp += indphys.size
                    iwave += 1

            nwaves = iwave
            Nx[iff] = nwaves

            sizes = sizes[:nwaves]
            indices = indices[:ind_tmp]
            data = data[:ind_tmp]

            indptr = np.zeros((nwaves+1),dtype=int)
            indptr[1:] = np.cumsum(sizes)

            Gx[iff] = csc_matrix((data, indices, indptr), shape=(self.nphys, nwaves))

        return Gx, Nx
    
    def _compute_component_time(self, time):

        Gt = {} # Time operator that gathers the time factors for each frequency 
        Nt = {} # Number of wave times tw such as abs(tw-t)<tdec

        for t in time:

            Gt[t] = [None,]*self.nf
            Nt[t] = [0,]*self.nf

            for iff in range(self.nf):
                Gt[t][iff] = np.zeros((self.iff_wavebounds[iff+1]-self.iff_wavebounds[iff],)) * np.nan
                ind_tmp = 0
                for it in range(self.enst[iff].shape[1]):
                    for P in range(self.NP[iff]):
                        dt = t - self.enst[iff][P,it]
                        if abs(dt)>self.tdec[iff][P] or np.isnan(self.enst[iff][P,it]):
                            fact = 0
                        else:
                            fact = self.window(dt / self.tdec[iff][P]) 
                            fact /= self.norm_fact[iff][P]
                        Gt[t][iff][ind_tmp:ind_tmp+2*self.ntheta] = fact   
                        if P==0:
                            Nt[t][iff] += 1
                        ind_tmp += 2*self.ntheta
        return Gt, Nt       

    def _ssh2uv(self, ssh):

        """
            Compute geostrophic velocities from SSH
        """

        _ssh = np.pad(ssh, pad_width=((1,0),(1,0)), mode='edge')

        _u = -self.g / self.f_on_u * (_ssh[1:,:] - _ssh[:-1,:]) / self.dy_on_u
        _v = self.g / self.f_on_v * (_ssh[:,1:] - _ssh[:,:-1]) / self.dx_on_v

        if self.u_coast_mask is not None:
            _u = np.where(self.u_coast_mask, 0., _u)
            _v = np.where(self.v_coast_mask, 0., _v)

        return _u, _v
    
    def _ssh2uv_adj(self, adu, adv):

        """
        Adjoint of geostrophic velocity computation.
        Uses jnp so that JAX device arrays are kept on-device (no GPU→CPU transfer).
        """

        if self.u_coast_mask is not None:
            adu = jnp.where(self.u_coast_mask, 0., adu)
            adv = jnp.where(self.v_coast_mask, 0., adv)

        # _adssh lives on padded grid: (ny+1, nx+1)
        _adssh = jnp.zeros((self.shape_phys[0] + 1, self.shape_phys[1] + 1))

        _adssh = _adssh.at[1:,:].add( -self.g / self.f_on_u * adu / self.dy_on_u)
        _adssh = _adssh.at[:-1,:].add(  self.g / self.f_on_u * adu / self.dy_on_u)
        _adssh = _adssh.at[:,1:].add(   self.g / self.f_on_v * adv / self.dx_on_v)
        _adssh = _adssh.at[:,:-1].add( -self.g / self.f_on_v * adv / self.dx_on_v)

        # map padded grid back to physical ssh grid:
        # physical ssh[i,j] == _ssh[i+1,j+1]
        adssh = _adssh[1:,1:]

        # contributions from the padded first row/col (mode='edge' duplicates edge values)
        # add the padded southern row (index 0) to physical southern row (adssh[0,:])
        adssh = adssh.at[0,:].add(_adssh[0,1:])
        # add the padded western column (index 0) to physical western column (adssh[:,0])
        adssh = adssh.at[:,0].add(_adssh[1:,0])
        # the padded corner (0,0) was duplicated as well — add it into adssh[0,0]
        adssh = adssh.at[0,0].add(_adssh[0,0])

        return adssh
        
    def operg(self, t, X, State=None):
        
        """
            Project to physicial space
        """

        # Projection
        ssh = np.zeros(self.shape_phys).ravel()
        phi = np.zeros(self.shape_phys).ravel()
        for iff in range(self.nf):
            Xf = X[self.iff_wavebounds[iff]:self.iff_wavebounds[iff+1]]
            GtXf = self.Gt[t][iff] * Xf
            indNoNan = ~np.isnan(self.Gt[t][iff])
            if indNoNan.size>0:
                GtXf = GtXf[indNoNan].reshape(self.Nt[t][iff],self.Nx[iff])
                phi += self.Gx[iff].dot(GtXf.sum(axis=0))
        ssh = ssh.reshape(self.shape_phys)

        # Compute geostrophic velocities
        if self.compute_velocities:
            u, v = self._ssh2uv(ssh)
            if State is not None:
                if not self.multi_mode:
                    State[self.name_mod_u] = u
                    State[self.name_mod_v] = v
                else:
                    State[self.name_mod_u] += u
                    State[self.name_mod_v] += v

        # Update State
        if State is not None:
            _assign_to_state_names(self, State, ssh)
        else:
            if self.compute_velocities:
                return ssh, u, v
            else:
                return ssh

    def operg_transpose(self, t, adState):
        
        """
            Project to reduced space
        """

        _ensure_adstate_names(self, adState, np.zeros(self.shape_phys))
        if self.compute_velocities and (adState[self.name_mod_u] is None or adState[self.name_mod_v] is None):
            adState[self.name_mod_u] = np.zeros_like(self.f_on_u)
            adState[self.name_mod_v] = np.zeros_like(self.f_on_v)
            
        adX = np.zeros(self.nbasis)

        adssh = _sum_adstate_names(self, adState)

        if self.compute_velocities:
            adssh += self._ssh2uv_adj(adState[self.name_mod_u], adState[self.name_mod_v])

        for iff in range(self.nf):
            Gt = +self.Gt[t][iff]
            indNoNan = ~np.isnan(self.Gt[t][iff])
            if indNoNan.size>0:
                Gt = Gt[indNoNan].reshape(self.Nt[t][iff],self.Nx[iff])
                adGtXf = self.Gx[iff].T.dot(adssh.ravel())
                adGtXf = np.repeat(adGtXf[np.newaxis,:],self.Nt[t][iff],axis=0)
                adX[self.iff_wavebounds[iff]:self.iff_wavebounds[iff+1]][indNoNan] += (Gt*adGtXf).ravel()
        
        if not self.multi_mode:
            _clear_adstate_names(self, adState)
            if self.compute_velocities:
                adState[self.name_mod_u] *= 0.
                adState[self.name_mod_v] *= 0.
        
        return adX

class Basis_bmaux(_Basis_bmaux):

    def __init__(self, config, State, multi_mode=False):
        super().__init__(config, State, multi_mode=multi_mode)

        # JIT 
        self._operg_jit = jit(self._operg)
        self._operg_reduced_jit = jit(self._operg_reduced)
        self._ssh2uv_adj_jit = jit(self._ssh2uv_adj)
        self._ssh2uv_jit = jit(self._ssh2uv)


    def set_basis(self,time,return_q=False,**kwargs):
        res = super().set_basis(time,return_q=return_q,**kwargs)
        self.time = time
        self.vect_time = jnp.eye(time.size)

        self.zero_basis = jnp.zeros((self.nbasis,))
        self.zero_phys = jnp.zeros((self.nphys,))

        return res

    def _compute_component_space(self):

        Gx = [None,]*self.nf
        GxT = [None,]*self.nf
        Nx = [None,]*self.nf

        for iff in range(self.nf):

            data = np.empty((2*self.ntheta*self.NP[iff]*self.nphys,))
            indices = np.empty((2*self.ntheta*self.NP[iff]*self.nphys,),dtype=int)
            sizes = np.zeros((2*self.ntheta*self.NP[iff],),dtype=int)

            ind_tmp = 0
            iwave = 0

            for P in range(self.NP[iff]):
                # Obs selection around point P
                indphys = np.where(
                    (np.abs((self.lon1d - self.ENSLON[iff][P]) / self.km2deg * np.cos(self.ENSLAT[iff][P] * np.pi / 180.)) <= self.DX[iff]) &
                    (np.abs((self.lat1d - self.ENSLAT[iff][P]) / self.km2deg) <= self.DX[iff])
                    )[0]
                xx = (self.lon1d[indphys] - self.ENSLON[iff][P]) / self.km2deg * np.cos(self.ENSLAT[iff][P] * np.pi / 180.) 
                yy = (self.lat1d[indphys] - self.ENSLAT[iff][P]) / self.km2deg
                # Spatial tapering shape of the wavelet 
                if self.mask1d is not None:
                    indmask = self.mask1d[indphys]
                    indphys = indphys[~indmask]
                    xx = xx[~indmask]
                    yy = yy[~indmask]
                facd = np.ones((indphys.size))
                if self.depth is not None:
                    facd = (self.depth[indphys]-self.depth1)/(self.depth2-self.depth1)
                    facd[facd>1]=1.
                    facd[facd<0]=0.
                    indphys = indphys[facd>0]
                    xx = xx[facd>0]
                    yy = yy[facd>0]
                    facd = facd[facd>0]

                facs = mywindow(xx / self.DX[iff]) * mywindow(yy / self.DX[iff]) * facd

                for itheta in range(self.ntheta):
                    # Wave vector components
                    kx = self.k[iff] * np.cos(self.theta[itheta])
                    ky = self.k[iff] * np.sin(self.theta[itheta])
                    # Cosine component
                    sizes[iwave] = indphys.size
                    indices[ind_tmp:ind_tmp+indphys.size] = indphys
                    data[ind_tmp:ind_tmp+indphys.size] = np.sqrt(2) * facs * np.cos(kx*(xx)+ky*(yy))
                    ind_tmp += indphys.size
                    iwave += 1
                    # Sine component
                    sizes[iwave] = indphys.size
                    indices[ind_tmp:ind_tmp+indphys.size] = indphys
                    data[ind_tmp:ind_tmp+indphys.size] = np.sqrt(2) * facs * np.sin(kx*(xx)+ky*(yy))
                    ind_tmp += indphys.size
                    iwave += 1

            nwaves = iwave
            Nx[iff] = nwaves

            sizes = sizes[:nwaves]
            indices = indices[:ind_tmp]
            data = data[:ind_tmp]

            indptr = np.zeros((nwaves+1),dtype=int)
            indptr[1:] = np.cumsum(sizes)

            Gx[iff] = sparse.CSC((data, indices, indptr), shape=(self.nphys, nwaves))
            GxT[iff] = sparse.CSR((data, indices, indptr), shape=(nwaves, self.nphys))
                        

        self.GxT = GxT
        return Gx, Nx

    def _compute_component_time(self, time):

        Gt = {} # Time operator that gathers the time factors for each frequency
        
        for iff in range(self.nf):
            nbasis_f = self.iff_wavebounds[iff+1] - self.iff_wavebounds[iff]
            Gt_np = np.zeros((time.size,nbasis_f))
            ind_tmp = 0
            for it in range(self.enst[iff].shape[1]):
                for P in range(self.NP[iff]):
                    for i,t in enumerate(time) :
                        dt = t - self.enst[iff][P,it]
                        if not (abs(dt)>self.tdec[iff][P] or np.isnan(self.enst[iff][P,it])):
                            fact = self.window(dt / self.tdec[iff][P])
                            fact /= self.norm_fact[iff][P]
                            Gt_np[i,ind_tmp:ind_tmp+2*self.ntheta] = fact
                    ind_tmp += 2*self.ntheta
            Gt[iff] = sparse.csr_fromdense(jnp.array(Gt_np).T)

        return Gt, None

    def _ssh2uv(self, ssh):

        """
            Compute geostrophic velocities from SSH
        """

        _ssh = jnp.pad(ssh, pad_width=((1,0),(1,0)), mode='edge')

        _u = -self.g / self.f_on_u * (_ssh[1:,:] - _ssh[:-1,:]) / self.dy_on_u
        _v = self.g / self.f_on_v * (_ssh[:,1:] - _ssh[:,:-1]) / self.dx_on_v

        if self.u_coast_mask is not None:
            _u = jnp.where(self.u_coast_mask, 0., _u)
            _v = jnp.where(self.v_coast_mask, 0., _v)

        return _u, _v

    def get_Gt_value(self, t, iff):

        idt = jnp.where(self.time == t, size=1)[0]  # Find index

        return self.Gt[iff] @ self.vect_time[idt[0]] # Get corresponding value
    
    def _operg(self, t, X):
        """
            Project to physicial space
        """

        # Initialize phi
        phi = self.zero_phys.ravel()

        for iff in range(self.nf):

            Gt = self.get_Gt_value(t,iff)
            Xf = X[self.iff_wavebounds[iff]:self.iff_wavebounds[iff+1]]
            GtXf = Gt * Xf

            # # Use shape-safe slicing instead of boolean indexing
            Nx_val = self.Nx[iff]

            # # Dynamically reshape the sliced array
            reshaped_GtXf = GtXf.reshape((-1, Nx_val))  # Ensure reshaping works dynamically

            # Update phi
            phi += self.Gx[iff] @ reshaped_GtXf.sum(axis=0)

        # Reshape phi back to physical space shape
        phi = phi.reshape(self.shape_phys)

        return phi

    def _operg_reduced(self, t, phi_2d):
        """Project a 2D physical adjoint field back to reduced space."""

        adX = self.zero_basis
        phi_1d = phi_2d.ravel()

        for iff in range(self.nf):
            Gt = self.get_Gt_value(t, iff)
            Nx_val = self.Nx[iff]
            ad_space = self.GxT[iff] @ phi_1d
            adX_f = Gt.reshape((-1, Nx_val)) * ad_space[None, :]
            adX = adX.at[self.iff_wavebounds[iff]:self.iff_wavebounds[iff+1]].set(adX_f.ravel())

        return adX

    def operg(self, t, X, State=None):
        
        """
            Project to physicial space
        """

        # Projection
        ssh = self._operg_jit(t, X)

        # Compute geostrophic velocities
        if self.compute_velocities:
            u, v = self._ssh2uv_jit(ssh)
            if State is not None:
                if not self.multi_mode:
                    State[self.name_mod_u] = u
                    State[self.name_mod_v] = v
                else:
                    State[self.name_mod_u] += u
                    State[self.name_mod_v] += v
        
        # Update State
        if State is not None:
            _assign_to_state_names(self, State, ssh)
        else:
            if self.compute_velocities:
                return ssh, u, v
            else:
                return ssh
        
    def operg_transpose(self, t, adState):
        
        """
            Project to reduced space
        """

        _ensure_adstate_names(self, adState, jnp.zeros(self.shape_phys))
        if self.compute_velocities and (adState[self.name_mod_u] is None or adState[self.name_mod_v] is None):
            adState[self.name_mod_u] = jnp.zeros_like(self.f_on_u)
            adState[self.name_mod_v] = jnp.zeros_like(self.f_on_v)

        adssh = _sum_adstate_names(self, adState)
        if self.compute_velocities:
            adssh += self._ssh2uv_adj_jit(adState[self.name_mod_u], adState[self.name_mod_v])
        adX = self._operg_reduced_jit(t, adssh)

        if not self.multi_mode:
            _clear_adstate_names(self, adState)
            if self.compute_velocities:
                adState[self.name_mod_u] *= 0.
                adState[self.name_mod_v] *= 0.
    
        return adX


###############################################################################
#                            Internal-tides                                   #
###############################################################################   

class Basis_hbc: 

    def __init__(self,config, State):

        ##################
        ### - COMMON - ###
        ##################

        # Grid specs
        self.lon_min = State.lon_min
        self.lon_max = State.lon_max
        self.lat_min = State.lat_min
        self.lat_max = State.lat_max
        self.ny = State.ny
        self.nx = State.nx
        self.lonS = State.lon[0,:]
        self.lonN = State.lon[-1,:]
        self.latE = State.lat[:,0]
        self.latW = State.lat[:,-1]
        self.km2deg =1./110 # Kilometer to deg factor 

        # Name of controlled parameters
        self.name_params = config.BASIS.name_params 

        # Basis reduction factor
        self.facns = config.BASIS.facns # Factor for gaussian spacing in space
        self.facnlt = config.BASIS.facnlt # Factor for gaussian spacing in time

        # Tidal frequencies 
        self.Nwaves = config.BASIS.Nwaves # Number of tidal components

        # Time dependancy 
        self.time_dependant = config.BASIS.time_dependant

        ##########################################
        ### - HEIGHT BOUNDARY CONDITIONS hbc - ###
        ##########################################

        self.D_bc = config.BASIS.D_bc # Space scale of gaussian decomposition for hbc (in km)
        self.T_bc = config.BASIS.T_bc # Time scale of gaussian decomposition for hbc (in days)

        # Number of angles (computed from the normal of the border) of incoming waves
        if config.BASIS.Ntheta>0: 
            self.Ntheta = 2*(config.BASIS.Ntheta-1)+3 # We add -pi/2,0,pi/2
        else:
            self.Ntheta = 1 # Only angle 0°

        self.sigma_B_bc = config.BASIS.sigma_B_bc # Covariance sigma for hbc parameter

        self.window = mywindow

        # JIT
        self._operg_jit = jit(self._operg)
        self._operg_reduced_jit = jit(self._operg_reduced)

    def set_basis(self,time,return_q=False,**kwargs):

        """
        Set the basis for the controlled parameters of the model and calculate reduced basis functions.

        Parameters:
        -----------
        time : np.ndarray
            Array of time points.
        return_q : bool, optional
            If True, returns the covariance matrix Q and the background vector array Xb, by default False.

        Returns:
        --------
        tuple of np.ndarray
            If return_q is True, returns a tuple containing:
                - Xb : np.ndarray
                    Background vector array Xb.
                - Q : np.ndarray or None
                    Covariance matrix Q.
        
        """
        
        TIME_MIN = time.min()
        TIME_MAX = time.max()
        LON_MIN = self.lon_min
        LON_MAX = self.lon_max
        LAT_MIN = self.lat_min
        LAT_MAX = self.lat_max
        if (LON_MAX<LON_MIN): LON_MAX = LON_MAX+360.

        self.time = time 
        self.vect_time = jnp.eye(time.size)

        self.Gxy = {} # Dictionary containing gaussian basis elements for each parameters. 
        self.GxyT = {} # Transpose operators for explicit adjoint projection.
        self.shape_params = {} # Dictionary containing the shapes in the reduced space of each of the parameters.
        self.shape_params_phys = {} # Dictionary containing the shapes in the physical space of each of the parameters.
        
        #############################################
        ### SETTING UP THE REDUCED BASIS ELEMENTS ###
        #############################################

        # - In Time - #
        if self.time_dependant:
            self.set_bc_gauss_t(time, TIME_MIN, TIME_MAX) 

        # - In Space - # 
        for name in self.name_params : 

            # - X height boundary conditions - #
            if name == "hbcx":  
                self.shape_params["hbcS"], self.shape_params["hbcN"], self.shape_params_phys["hbcS"], self.shape_params_phys["hbcN"] = self.set_bc_gauss_hbcx(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
            
            # - Y height boundary conditions - #
            if name == "hbcy": 
                self.shape_params["hbcE"], self.shape_params["hbcW"], self.shape_params_phys["hbcE"], self.shape_params_phys["hbcW"] = self.set_bc_gauss_hbcy(LAT_MIN, LAT_MAX)

        ############################################
        ### REDUCED BASIS INFORMATION ATTRIBUTES ###
        ############################################

        # Dictionary with the number of parameters in reduced space 
        self.n_params = {}
        for param in self.shape_params.keys():
            if self.shape_params[param] == []:
                self.n_params[param] = 0
            else :
                self.n_params[param] = np.prod(self.shape_params[param])

        # Dictionary with the number of parameters in pysical space
        self.n_params_phys = dict(zip(self.shape_params_phys.keys(), map(np.prod, self.shape_params_phys.values()))) 
        # Total number of parameters in the reduced space
        self.nbasis = sum(self.n_params.values()) 
        # Total number of parameters in the physical space
        self.nphys = sum(self.n_params_phys.values()) 
        # Total number of parameters in the physical space (including time dimension)
        self.nphystot = 0 
        for param in self.n_params_phys.keys():
            self.nphystot += self.n_params_phys[param]*time.size
        # Setting up slice information for parameters 
        interval = 0 ; interval_phys = 0 
        self.slice_params = {} # Dictionary with the slices of parameters in the reduced space
        self.slice_params_phys = {} # Dictionary with the slices of parameters in the physical space
        for name in self.shape_params.keys():
            self.slice_params[name]=slice(interval,interval+self.n_params[name])
            self.slice_params_phys[name]=slice(interval_phys,interval_phys+self.n_params_phys[name])
            interval += self.n_params[name]; interval_phys += self.n_params_phys[name]
        # PRINTING REDUCED ORDER : #     
        print(f'reduced order: {self.nphystot} --> {self.nbasis}\nreduced factor: {int(self.nphystot/self.nbasis)}')

        #########################################
        ### COMPUTING THE COVARIANCE MATRIX Q ###
        #########################################        

        if return_q :
            if self.sigma_B_bc is not None:
                Q = np.zeros((self.nbasis,)) # Initializing
                for name in self.slice_params.keys() :

                    # Normalise so that physical prior variance ≈ sigma_B_bc²:
                    # - sqrt(facns)  : ~facns overlapping spatial Gaussians sum at each point
                    # - sqrt(facnlt) : ~facnlt overlapping time Gaussians (only when time_dependant)
                    # - sqrt(Ntheta) : compute_IT_2D sums over Ntheta angles
                    _norm = (self.facns * (self.facnlt if self.time_dependant else 1) * self.Ntheta) ** 0.5
                    if hasattr(self.sigma_B_bc,'__len__'):
                        if len(self.sigma_B_bc)==self.Nwaves:
                            # Different background values for each frequency
                            nw = self.nbc//self.Nwaves
                            for iw in range(self.Nwaves):
                                slicew = slice(iw*nw,(iw+1)*nw)
                                Q[self.slice_params[name]][slicew]=self.sigma_B_bc[iw]/_norm
                        else:
                            # Not the right number of frequency prescribed in the config file 
                            # --> we use only the first one
                            Q[self.slice_params[name]]=self.sigma_B_bc[0]/_norm
                    else:
                        Q[self.slice_params[name]]=self.sigma_B_bc/_norm

            else:
                Q = None

            Xb = np.zeros_like(Q)

            return Xb, Q
    
    def set_bc_gauss_hbcx(self, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX):

        """
        Set the height boundary conditions hbcx parameter recuced basis elements for both the South and North boundaries.

        Parameters:
        -----------
        time : np.ndarray
            Array of time points.
        LAT_MIN : float
            Minimum latitude value.
        LAT_MAX : float
            Maximum latitude value.
        LON_MIN : float
            Minimum longitude value.
        LON_MAX : float
            Maximum longitude value.
        TIME_MIN : float
            Minimum time value.
        TIME_MAX : float
            Maximum time value.

        Returns:
        --------
        tuple
            A tuple containing:
                - shapehbcS : list
                    Shape of the South boundary hbcx parameter in the reduced space.
                - shapehbcN : list
                    Shape of the North boundary hbcx parameter in the reduced space.
                - shapehbcS_phys : list
                    Shape of the South boundary hbcx parameter in the physical space.
                - shapehbcN_phys : list
                    Shape of the North boundary hbcx parameter in the physical space.

        Notes:
        ------
        - This function sets the basis elements for the height boundary conditions hbcx parameter. It computes spatial and temporal Gaussian basis functions based on specificatiions of config file and coordinates.
        - It prints the total number of hbcx parameters in the reduced space.
        """

        ###############################
        ###   - SPACE DIMENSION -   ###
        ###############################

        # - SOUTH - # 
        # Ensemble of reduced basis longitudes
        ENSLON_S = np.mod(
                np.arange(
                    LON_MIN - self.D_bc*(1-1./self.facns)/np.cos(LAT_MIN*np.pi/180.)*self.km2deg,
                    LON_MAX + 1.5*self.D_bc/self.facns/np.cos(LAT_MIN*np.pi/180.)*self.km2deg,
                    self.D_bc/self.facns/np.cos(LAT_MIN*np.pi/180.)*self.km2deg),
                360)
        # Computing reduced basis elements gaussian supports 
        bc_S_gauss = np.zeros((ENSLON_S.size,self.nx))
        for i,lon0 in enumerate(ENSLON_S):
            iobs = np.where((np.abs((np.mod(self.lonS - lon0+180,360)-180) / self.km2deg * np.cos(LAT_MIN * np.pi / 180.)) <= self.D_bc))[0] 
            xx = (np.mod(self.lonS[iobs] - lon0+180,360)-180) / self.km2deg * np.cos(LAT_MIN * np.pi / 180.)     
            bc_S_gauss[i,iobs] = mywindow(xx / self.D_bc) 
        
        # - NORTH - #
        # Ensemble of reduced basis longitudes
        ENSLON_N = np.mod(
                np.arange(
                    LON_MIN - self.D_bc*(1-1./self.facns)/np.cos(LAT_MAX*np.pi/180.)*self.km2deg,
                    LON_MAX + 1.5*self.D_bc/self.facns/np.cos(LAT_MAX*np.pi/180.)*self.km2deg,
                    self.D_bc/self.facns/np.cos(LAT_MAX*np.pi/180.)*self.km2deg),
                360)
        # Computing reduced basis elements gaussian supports 
        bc_N_gauss = np.zeros((ENSLON_N.size,self.nx))
        for i,lon0 in enumerate(ENSLON_N):
            iobs = np.where((np.abs((np.mod(self.lonN - lon0+180,360)-180) / self.km2deg * np.cos(LAT_MAX * np.pi / 180.)) <= self.D_bc))[0] 
            xx = (np.mod(self.lonN[iobs] - lon0+180,360)-180) / self.km2deg * np.cos(LAT_MAX * np.pi / 180.)     
            bc_N_gauss[i,iobs] = mywindow(xx / self.D_bc) 

        # Saving gaussian reduced basis elements 
        self.Gxy["hbcS"] = sparse.CSR.fromdense(jnp.array(bc_S_gauss.T)) # For South boundary 
        self.Gxy["hbcN"] = sparse.CSR.fromdense(jnp.array(bc_N_gauss.T)) # For North boundary
        self.GxyT["hbcS"] = sparse.CSR.fromdense(jnp.array(bc_S_gauss))
        self.GxyT["hbcN"] = sparse.CSR.fromdense(jnp.array(bc_N_gauss))

        ####################################
        ###   - BASIS ELEMENT SHAPES -   ###
        ####################################

        # - Shapes of the hbcy parameters in the reduced space.

        if self.time_dependant : # the parameters include the time dependency 

            shapehbcS = [self.Nwaves,           # - Number of tidal frequency components 
                        2,                     # - Number of controlled components (cos & sin)
                        self.Ntheta,           # - Number of angles
                        self.ENST_bc.size,     # - Number of basis timesteps
                        bc_S_gauss.shape[0]]   # - Number of basis spatial elements 
            
            shapehbcN = [self.Nwaves,           # - Number of tidal frequency components 
                        2,                     # - Number of controlled components (cos & sin)
                        self.Ntheta,           # - Number of angles
                        self.ENST_bc.size,          # - Number of basis timesteps
                        bc_N_gauss.shape[0]]   # - Number of basis spatial elements 
        
        else : # the parameters do not the time dependency 

            shapehbcS = [self.Nwaves,           # - Number of tidal frequency components 
                        2,                     # - Number of controlled components (cos & sin)
                        self.Ntheta,           # - Number of angles
                        bc_S_gauss.shape[0]]   # - Number of basis spatial elements 
            
            shapehbcN = [self.Nwaves,           # - Number of tidal frequency components 
                        2,                     # - Number of controlled components (cos & sin)
                        self.Ntheta,           # - Number of angles
                        bc_N_gauss.shape[0]]   # - Number of basis spatial elements 

        # - Shapes of the hbcy parameters in the physical space.
        shapehbcS_phys = shapehbcN_phys = [self.Nwaves,     # - Number of tidal frequency components 
                                           2,               # - Number of controlled components (cos & sin)
                                           self.Ntheta,     # - Number of angles
                                           self.nx]         # - Number of gridpoints along x axis
        
        print('nbcx:',np.prod(shapehbcS)+np.prod(shapehbcN))

        return shapehbcS, shapehbcN, shapehbcS_phys, shapehbcN_phys

    def set_bc_gauss_hbcy(self,LAT_MIN, LAT_MAX): 

        """
        Set the height boundary conditions hbcy parameter recuced basis elements for both the East and West boundaries.

        Parameters:
        -----------
        time : np.ndarray
            Array of time points.
        LAT_MIN : float
            Minimum latitude value.
        LAT_MAX : float
            Maximum latitude value.
        LON_MIN : float
            Minimum longitude value.
        LON_MAX : float
            Maximum longitude value.
        TIME_MIN : float
            Minimum time value.
        TIME_MAX : float
            Maximum time value.

        Returns:
        --------
        tuple
            A tuple containing:
                - shapehbcE : list
                    Shape of the East boundary hbcx parameter in the reduced space.
                - shapehbcW : list
                    Shape of the West boundary hbcx parameter in the reduced space.
                - shapehbcE_phys : list
                    Shape of the East boundary hbcx parameter in the physical space.
                - shapehbcW_phys : list
                    Shape of the West boundary hbcx parameter in the physical space.

        Notes:
        ------
        - This function sets the basis elements for the height boundary conditions hbcx parameter. It computes spatial and temporal Gaussian basis functions based on specificatiions of config file and coordinates.
        - It prints the total number of hbcy parameters in the reduced space.
        """
        
        #########################################
        ###   - COMPUTING SPACE DIMENSION -   ###
        #########################################

        # Ensemble of reduced basis latitudes (common for each boundaries)
        ENSLAT = np.arange(
            LAT_MIN - self.D_bc*(1-1./self.facns)*self.km2deg,
            LAT_MAX + 1.5*self.D_bc/self.facns*self.km2deg, 
            self.D_bc/self.facns*self.km2deg)

        # - EAST - #
        # Computing reduced basis elements gaussian supports 
        bc_E_gauss = np.zeros((ENSLAT.size,self.ny))
        for i,lat0 in enumerate(ENSLAT):
            iobs = np.where(np.abs((self.latE - lat0) / self.km2deg) <= self.D_bc)[0]
            yy = (self.latE[iobs] - lat0) / self.km2deg
            bc_E_gauss[i,iobs] = mywindow(yy / self.D_bc) 

        # - WEST - # 
        # Computing reduced basis elements gaussian supports 
        bc_W_gauss = np.zeros((ENSLAT.size,self.ny))
        for i,lat0 in enumerate(ENSLAT):
            iobs = np.where(np.abs((self.latW - lat0) / self.km2deg) <= self.D_bc)[0]
            yy = (self.latW[iobs] - lat0) / self.km2deg
            bc_W_gauss[i,iobs] = mywindow(yy / self.D_bc) 

        # Gaussian reduced basis elements
        self.Gxy["hbcE"] = sparse.CSR.fromdense(jnp.array(bc_E_gauss.T)) # For East boundary 
        self.Gxy["hbcW"] = sparse.CSR.fromdense(jnp.array(bc_W_gauss.T)) # For West boundary 
        self.GxyT["hbcE"] = sparse.CSR.fromdense(jnp.array(bc_E_gauss))
        self.GxyT["hbcW"] = sparse.CSR.fromdense(jnp.array(bc_W_gauss))

        ####################################
        ###   - BASIS ELEMENT SHAPES -   ###
        ####################################
        
        # Shapes of the hbcx parameters in the reduced space.
        if self.time_dependant : # the parameters include the time dependency

            shapehbcE = [self.Nwaves,               # - Number of tidal frequency components 
                        2,                         # - Number of controlled components (cos & sin)
                        self.Ntheta,               # - Number of angles
                        self.ENST_bc.size,              # - Number of basis timesteps
                        bc_E_gauss.shape[0]]       # - Number of basis spatial elements 
            
            shapehbcW = [self.Nwaves,               # - Number of tidal frequency components 
                        2,                         # - Number of controlled components (cos & sin)
                        self.Ntheta,               # - Number of angles
                        self.ENST_bc.size,              # - Number of basis timesteps
                        bc_W_gauss.shape[0]]       # - Number of basis spatial elements 
        
        else : # the parameters do not the time dependency 

            shapehbcE = [self.Nwaves,               # - Number of tidal frequency components 
                        2,                         # - Number of controlled components (cos & sin)
                        self.Ntheta,               # - Number of angles
                        bc_E_gauss.shape[0]]       # - Number of basis spatial elements 
            
            shapehbcW = [self.Nwaves,               # - Number of tidal frequency components 
                        2,                         # - Number of controlled components (cos & sin)
                        self.Ntheta,               # - Number of angles
                        bc_W_gauss.shape[0]]       # - Number of basis spatial elements

        # Shapes of the hbcx parameters in the physical space.
        shapehbcE_phys = shapehbcW_phys = [self.Nwaves,     # - Number of tidal frequency components 
                                           2,               # - Number of controlled components (cos & sin)
                                           self.Ntheta,     # - Number of angles
                                           self.ny]         # - Number of gridpoints along x axis

        print('nbcy:',np.prod(shapehbcE)+np.prod(shapehbcW))

        return shapehbcE, shapehbcW, shapehbcE_phys, shapehbcW_phys
    
    def set_bc_gauss_t(self,time, TIME_MIN, TIME_MAX):
        # Ensemble of reduced basis timesteps
        ENST_bc = np.arange(-self.T_bc*(1-1./self.facnlt),(TIME_MAX - TIME_MIN)+1.5*self.T_bc/self.facnlt , self.T_bc/self.facnlt)
        # bc_t_gauss = np.zeros((time.size,ENST_bc.size))
        # for i,time0 in enumerate(ENST_bc):
        #     iobs = np.where(abs(time-time0) < self.T_bc)
        #     bc_t_gauss[iobs,i] = mywindow(abs(time-time0)[iobs]/self.T_bc)

        # self.bc_t_gauss = bc_t_gauss

        self.ENST_bc = ENST_bc

        Gt = np.zeros((time.size,self.ENST_bc.size))

        for i,t in enumerate(time) :
            for it in range(len(self.ENST_bc)):
                dt = t - self.ENST_bc[it]
                if abs(dt) < self.T_bc:
                    fact = self.window(dt / self.T_bc) 
                    if fact!=0:   
                        Gt[i,it] = fact
        
        self.Gt = sparse.csr_fromdense(jnp.array(Gt).T)

    def get_bc_t_gauss_value(self,t):

        idt = jnp.where(self.time == t, size=1)[0]  # Find index

        return self.Gt @ self.vect_time[idt[0]] # Get corresponding value

    def _operg(self,t,X):

        """
        Perform the basis projection operation for a given time and parameter vector.

        This method projects the given parameter vector X from reduced basis onto the model grid, at the provided time t. The results can be stored in the provided State object.
                
        operg : | REDUCED SPACE >>>>>> PHYSICAL SPACE | (Model Grid) 

        Parameters:
        ----------
        t : float
            The time at which the projection is performed.
        X : ndarray
            The parameter vector to be projected.
        State : object, optional
            State object to store the parameters after projection. If not provided, the method returns the projected vector onto physical space.

        Returns:
        -------
        phi : ndarray
            The projected parameter vector if State is not provided. Otherwise, updates the State object in place.

        """

        ##############################
        ###   - INITIALIZATION -   ###
        ##############################

        # Time gaussian function
        if self.time_dependant:
            _Gt = self.get_bc_t_gauss_value(t) 

        # Variable to return  
        phi = jnp.zeros((self.nphys,))

        ##########################################
        ###   - BASIS PROJECTION OPERATION -   ###
        ##########################################

        for name in self.slice_params_phys.keys():

            _X = X[self.slice_params[name]]
            _X = _X.reshape(self.shape_params[name])

            if self.time_dependant:
                _X = (_Gt[None,None,None,:, None]*_X).sum(axis=3, keepdims=False)

            _X_t = _X.T

            Gxy_X_t = sparse.csr_matmat(self.Gxy[name],_X_t.reshape(_X_t.shape[0],-1))

            Gxy_X_t = Gxy_X_t.reshape((Gxy_X_t.shape[0],)+_X_t.shape[1:])

            Gxy_X = Gxy_X_t.T

            phi = phi.at[self.slice_params_phys[name]].set(Gxy_X.flatten())#.reshape(self.shape_params_phys[name]))
        
        return phi

    def _operg_reduced(self, t, phi_2d):
        """Project HBC physical adjoints explicitly back to reduced space."""

        if self.time_dependant:
            _Gt = self.get_bc_t_gauss_value(t)

        adX = jnp.zeros((self.nbasis,))

        for name in self.slice_params_phys.keys():
            ad_phi = phi_2d[self.slice_params_phys[name]].reshape(self.shape_params_phys[name])
            ad_phi_t = ad_phi.T

            Gt_ad_phi_t = sparse.csr_matmat(
                self.GxyT[name],
                ad_phi_t.reshape(ad_phi_t.shape[0], -1),
            )
            ad_reduced = Gt_ad_phi_t.reshape((Gt_ad_phi_t.shape[0],) + ad_phi_t.shape[1:]).T

            if self.time_dependant:
                ad_reduced = _Gt[None, None, None, :, None] * ad_reduced[:, :, :, None, :]

            adX = adX.at[self.slice_params[name]].set(ad_reduced.flatten())

        return adX

    def operg(self, t, X, State=None):
        
        """
            Project to physicial space
        """

        # Projection
        phi = self._operg_jit(t, X)

        # Update State
        if State is not None:
            for name in self.name_params:
                # - Height boundary conditions hbcx - #
                if name == "hbcx" : 
                    State['hbcx'] = jnp.concatenate((jnp.expand_dims(phi[self.slice_params_phys["hbcS"]].reshape(self.shape_params_phys["hbcS"]),axis=1),
                                                            jnp.expand_dims(phi[self.slice_params_phys["hbcN"]].reshape(self.shape_params_phys["hbcN"]),axis=1)),axis=1)
                # - Height boundary conditions hbcy - #
                elif name == "hbcy" : 
                    State['hbcy'] = jnp.concatenate((jnp.expand_dims(phi[self.slice_params_phys["hbcW"]].reshape(self.shape_params_phys["hbcW"]),axis=1),
                                                        jnp.expand_dims(phi[self.slice_params_phys["hbcE"]].reshape(self.shape_params_phys["hbcE"]),axis=1)),axis=1)
            # State.params[self.name_mod_var] = phi
        else:
            return phi
    
    def operg_transpose(self, t, adState):
        
        """
            Project to reduced space
        """
        
        # if adState.params[self.name_mod_var] is None:
        #     adState.params[self.name_mod_var] = np.zeros((self.nphys,))

        # Getting the parameters 
        # if phi is not None: # If provided through phi ndarray argument 
        #     for name in self.slice_params_phys.keys():
        #         param[name] = phi[self.slice_params_phys[name]].reshape(self.shape_params_phys[name])

        adparams = np.zeros((self.nphys))
        if adState is not None: # If provided through adState object argument 
            for name in self.name_params:
                if name == "hbcx" : 
                    # adparams["hbcS"] = adState.params[name][:,0,:,:,:].reshape(self.shape_params_phys["hbcS"])
                    # adparams["hbcN"] = adState.params[name][:,1,:,:,:].reshape(self.shape_params_phys["hbcN"])
                    adparams[self.slice_params_phys["hbcS"]] = adState[name][:,0,:,:,:].flatten()
                    adparams[self.slice_params_phys["hbcN"]] = adState[name][:,1,:,:,:].flatten()
                elif name == "hbcy" : 
                    # adparams["hbcE"] = adState.params[name][:,0,:,:,:].reshape(self.shape_params_phys["hbcE"])
                    # adparams["hbcW"] = adState.params[name][:,1,:,:,:].reshape(self.shape_params_phys["hbcW"])
                    adparams[self.slice_params_phys["hbcW"]] = adState[name][:,0,:,:,:].flatten()
                    adparams[self.slice_params_phys["hbcE"]] = adState[name][:,1,:,:,:].flatten()
                # else :
                #     param[name] = adState.params[name].reshape(self.shape_params_phys[name])
        # adparams = adparams.flatten()
        # adparams = adState.getparams(self.name_params,vect=True)

        adX = self._operg_reduced_jit(t, adparams)
        
        for _param in self.name_params : 
            adState[_param] *= 0.
        
        return adX

###############################################################################
#                                Offset                                       #
###############################################################################   
# Old version of the basis class, kept for reference. The new Basis_offset class is preferred.
class _Basis_offset:

    def __init__(self,config, State, multi_mode=False):
        
        self.name_mod_var = config.BASIS.name_mod_var
        self.shape_phys = State.params[_primary_name(self.name_mod_var)].shape
        for _name in _as_name_list(self.name_mod_var)[1:]:
            if State.params[_name].shape != self.shape_phys:
                raise ValueError(
                    f"BASIS_OFFSET shared name_mod_var entries must have the same shape; "
                    f"{_name} has {State.params[_name].shape}, expected {self.shape_phys}"
                )
        self.nphys = np.prod(self.shape_phys)
        self.ny = State.ny
        self.nx = State.nx
        self.sigma_B = config.BASIS.sigma_B
        
        if self.sigma_B == None : 
            print("Warning, please prescribe sigma_B for Basis Offset") 
        
        self.multi_mode = multi_mode
    
    def set_basis(self,time,return_q=False,**kwargs):
        self.nbasis = 1
        self.shape_basis = [1]

        # Fill Q matrix
        Q = self.sigma_B * np.ones((self.nbasis))

        if return_q:
            return np.zeros_like(Q), Q

    def operg(self,t,X,State=None):

        """
            Project to physicial space
        """

        phi = X*np.ones(self.shape_phys)

        # Update State
        if State is not None:
            _assign_to_state_names(self, State, phi)
        else:
            return phi
        
    def operg_transpose(self, t, adState):
        """
            Project to reduced space
        """
        _ensure_adstate_names(self, adState, np.zeros(self.shape_phys))
        adparams = _sum_adstate_names(self, adState)

        adX = [np.sum(adparams)]
        
        if not self.multi_mode:
            _clear_adstate_names(self, adState)
        
        return adX

class Basis_offset(_Basis_offset):
   
    def __init__(self,config, State, multi_mode=False):

        super().__init__(config, State,multi_mode=multi_mode)
    
    def operg(self,t,X,State=None):

        """
            Project to physicial space
        """

        phi = X*jnp.ones(self.shape_phys)

        # Update State
        if State is not None:
            _assign_to_state_names(self, State, phi)
        else:
            return phi
        
    def operg_transpose(self, t, adState):
        """
            Project to reduced space
        """
        _ensure_adstate_names(self, adState, jnp.zeros(self.shape_phys))
        adparams = _sum_adstate_names(self, adState)

        adX = jnp.expand_dims(jnp.sum(adparams), axis=0)
        
        if not self.multi_mode:
            _clear_adstate_names(self, adState)
        
        return adX
    
###############################################################################
#                              Multi-Basis                                    #
###############################################################################      

class Basis_multi:

    def __init__(self,config,State,verbose=True):

        self.Basis = []
        _config = config.copy()

        self.name_mod_var = []
        for _BASIS in config.BASIS:
            _config.BASIS = config.BASIS[_BASIS]

            self.Basis.append(Basis(_config,State,verbose=verbose, multi_mode=True))
            if 'name_mod_var' in _config.BASIS and _config.BASIS.name_mod_var is not None:
                for _name_mod_var in _as_name_list(_config.BASIS.name_mod_var):
                    if _name_mod_var not in self.name_mod_var:
                        self.name_mod_var.append(_name_mod_var)
                if 'compute_velocities' in _config.BASIS and _config.BASIS.compute_velocities:
                    if 'name_mod_u' in _config.BASIS and _config.BASIS.name_mod_u is not None and _config.BASIS.name_mod_u not in self.name_mod_var:
                        self.name_mod_var.append(_config.BASIS.name_mod_u)
                    if 'name_mod_v' in _config.BASIS and _config.BASIS.name_mod_v is not None and _config.BASIS.name_mod_v not in self.name_mod_var:
                        self.name_mod_var.append(_config.BASIS.name_mod_v)

        
    def set_basis(self,time,return_q=False,**kwargs):

        self.nbasis = 0
        self.slice_basis = []

        if return_q:
            Xb = np.array([])
            Q = np.array([])

        for B in self.Basis:
            _Xb,_Q = B.set_basis(time,return_q=return_q,**kwargs)
            self.slice_basis.append(slice(self.nbasis,self.nbasis+B.nbasis))
            self.nbasis += B.nbasis
            
            if return_q:
                Xb = np.concatenate((Xb,_Xb))
                Q = np.concatenate((Q,_Q))
        
        if return_q:
            return Xb,Q

    def operg(self, t, X, State=None):
        
        """
            Project to physicial space
        """

        if State is None:
            phi_parts = []

        if State is not None:
            for name_mod_var in self.name_mod_var:
                State[name_mod_var] *= 0.

        for i,B in enumerate(self.Basis):
            _X = X[self.slice_basis[i]]
            _phi = B.operg(t, _X, State=State)
            if State is None:
                phi_parts.append(jnp.ravel(_phi))
        
        if State is None:
            return jnp.concatenate(phi_parts)


    def operg_transpose(self, t, adState):
        
        """
            Project to reduced space
        """
        
        adX_parts = [B.operg_transpose(t, adState=adState) for B in self.Basis]
        adX = jnp.concatenate(adX_parts)
        
        for name_mod_var in self.name_mod_var:
            adState[name_mod_var] *= 0.

        return adX


def mywindow(x): # x must be between -1 and 1
     y  = np.cos(x*0.5*np.pi)**2
     return y
  
def mywindow_flux(x): # x must be between -1 and 1
     y = -np.pi*np.sin(x*0.5*np.pi)*np.cos(x*0.5*np.pi)
     return y

def integrand(x,f):
    y  = quad(f, -1, x)[0]
    return y

def test_operg(Basis,t=0):
        
    psi = np.random.random((Basis.nbasis,))
    phi = np.random.random((Basis.shape_phys))
    
    ps1 = np.inner(psi,Basis.operg(phi,t,transpose=True))
    ps2 = np.inner(Basis.operg(psi,t).flatten(),phi.flatten())
        
    print(f'test G[{t}]:', ps1/ps2)

