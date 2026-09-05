#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Florian Le Guillou on June 2026.

Implements the model factory and dynamical model classes.
"""

from .config import USE_FLOAT64
from importlib.machinery import SourceFileLoader 
import sys
import xarray as xr
import numpy as np
import os
from math import pi
from datetime import timedelta
import matplotlib.pylab as plt 
import pyinterp
import pyinterp.fill
import jax
import jax.numpy as jnp 
from jax import jit


jax.config.update("jax_enable_x64", USE_FLOAT64)

import warnings 
from concurrent.futures import ThreadPoolExecutor

try:
    import jaxparrow
except ImportError:
    warnings.warn('jaxparrow not installed, some jax functions will not be available')


from . import  tools as grid 
from .exp import Config as Config


def _fill_nan_nearest(arr):
    """Fill NaNs in a 2D or 3D numpy array with the nearest valid value.

    Used for tracer BC fields in Model_qg1l_jax.set_bc: QG tracer advection is
    not land-mask aware, so a 0 sitting at a NaN-filled (land or coastal
    no-data) pixel of varb would leak into adjacent ocean cells via advection
    each step, producing spurious ~0 psu / cold patches at the coast.  This
    nearest-neighbour fill gives varb a sensible ocean value at land/coastal
    pixels, mirroring what SW gets for free through its land mask.
    """
    from scipy.ndimage import distance_transform_edt
    out = np.asarray(arr, dtype=float).copy()
    if out.ndim == 2:
        slices = [out]
    elif out.ndim == 3:
        nan_masks = np.isnan(out)
        if np.all(nan_masks == nan_masks[0]):
            nan_mask = nan_masks[0]
            if not nan_mask.any():
                return out
            if nan_mask.all():
                out[:] = 0.
                return out
            ind = distance_transform_edt(
                nan_mask, return_distances=False, return_indices=True
            )
            out[:, nan_mask] = out[:, ind[0][nan_mask], ind[1][nan_mask]]
            return out
        slices = [out[k] for k in range(out.shape[0])]
    else:
        out[np.isnan(out)] = 0.
        return out
    for s in slices:
        nan_mask = np.isnan(s)
        if not nan_mask.any():
            continue
        if nan_mask.all():
            s[:] = 0.
            continue
        ind = distance_transform_edt(
            nan_mask, return_distances=False, return_indices=True)
        s[nan_mask] = s[tuple(ind)][nan_mask]
    return out


class _MultiBcFields:
    """Merge several per-variable boundary-condition sources."""

    def __init__(self, bc_fields):
        self.bc_fields = bc_fields

    def interp(self, time):
        if len(self.bc_fields) == 1:
            return self.bc_fields[0].interp(time)

        # Sources are fully loaded by _BcFields.__init__, so concurrent work
        # below is CPU-only and never reads netCDF/HDF5 handles from threads.
        with ThreadPoolExecutor(max_workers=len(self.bc_fields)) as executor:
            interpolated = list(executor.map(
                lambda bc_field: bc_field.interp(time),
                self.bc_fields,
            ))
        out = {}
        for fields in interpolated:
            out.update(fields)
        return out


class _BcFields:
    """Load and interpolate boundary conditions from an external file."""

    def __init__(self, config_bc, State=None):
        if State is None:
            from . import state
            State = state.State(config_bc._config if hasattr(config_bc, '_config') else {}, verbose=False)

        self.lon = State.lon
        self.lat = State.lat
        self.mask = State.mask

        lon_min = np.nanmin(self.lon)
        lon_max = np.nanmax(self.lon)
        lat_min = np.nanmin(self.lat)
        lat_max = np.nanmax(self.lat)

        dlon = np.nanmean(self.lon[:,1:]-self.lon[:,:-1])
        dlat = np.nanmean(self.lat[1:,:]-self.lat[:-1,:])

        _ds = xr.open_mfdataset(config_bc.file, chunks=-1)
        ds = _ds.copy()
        _ds.close()

        try:
            if np.sign(ds[config_bc.name_lon].data.min()) == -1 and State.lon_unit == '0_360':
                ds = ds.assign_coords({config_bc.name_lon: ((config_bc.name_lon, ds[config_bc.name_lon].data % 360))})
            elif np.sign(ds[config_bc.name_lon].data.min()) >= 0 and State.lon_unit == '-180_180':
                ds = ds.assign_coords({config_bc.name_lon: ((config_bc.name_lon, (ds[config_bc.name_lon].data + 180) % 360 - 180))})
            ds = ds.sortby(ds[config_bc.name_lon])
        except Exception:
            print("Warning: can't convert longitude coordinates in " + config_bc.file)

        self.name_time_bc = config_bc.name_time
        has_time = (
            config_bc.name_time is not None
            and config_bc.name_time in ds
        )
        time_bc = ds[config_bc.name_time].values if has_time else None
        lon_bc = ds[config_bc.name_lon].values
        lat_bc = ds[config_bc.name_lat].values
        dlon += np.nanmean(lon_bc[1:] - lon_bc[:-1])
        dlat += np.nanmean(lat_bc[1:] - lat_bc[:-1])
        if has_time and np.size(time_bc) > 1:
            dtime = time_bc[1] - time_bc[0]
            try:
                ds = ds.sel({
                    config_bc.name_time: slice(
                        np.datetime64(config_bc._config.EXP.init_date) - dtime,
                        np.datetime64(config_bc._config.EXP.final_date) + dtime,
                    ),
                })
            except Exception:
                ds = ds.where(
                    (ds[config_bc.name_time] >= np.datetime64(config_bc._config.EXP.init_date) - dtime)
                    & (ds[config_bc.name_time] <= np.datetime64(config_bc._config.EXP.final_date) + dtime),
                    drop=True,
                )
        if len(ds[config_bc.name_lon].shape) == 1 and len(ds[config_bc.name_lat].shape) == 1:
            ds = ds.sel({
                config_bc.name_lon: slice(lon_min - 2 * dlon, lon_max + 2 * dlon),
                config_bc.name_lat: slice(lat_min - 2 * dlat, lat_max + 2 * dlat),
            })

        self.c_grid = getattr(config_bc, 'c_grid', False)
        self.name_lon_bc = config_bc.name_lon
        self.name_lat_bc = config_bc.name_lat
        self.lon_bc = ds[config_bc.name_lon].values
        self.lat_bc = ds[config_bc.name_lat].values
        self.time_bc = ds[config_bc.name_time].values if has_time else None

        self.var = {}
        for name in config_bc.name_var:
            self.var[name] = ds[config_bc.name_var[name]].load()
        ds.close()

    def interp(self, time):
        if self.time_bc is not None and self.time_bc.size > 1:
            if len(time.shape) == 0:
                time = np.array([time])
            elif len(time.shape) == 1:
                time = np.ascontiguousarray(time)
            else:
                sys.exit('Time dimension must be 1D or 0D')

        if time.size > 1 and type(time[0]) != np.datetime64:
            time = np.array([np.datetime64(dt) for dt in time])
        elif time.size == 1 and type(time) != np.datetime64:
            time = np.array([np.datetime64(time)])

        if len(self.lon_bc.shape) == 1 and len(self.lat_bc.shape) == 1:
            var_interp = self._interp_3D(time)
        elif len(self.lon_bc.shape) == 2 and len(self.lat_bc.shape) == 2 and np.all(self.lon_bc == self.lon) and np.all(self.lat_bc == self.lat):
            var_interp = self._interp_1D(time)
        else:
            sys.exit('Boundary conditions coordinates must be 1D, 2D. If 2D, they must match the model grid coordinates')

        return var_interp

    def _prepare_source_2D(self, name, da):
        if self.name_time_bc is not None and self.name_time_bc in da.dims:
            da = da.isel({self.name_time_bc: 0})
        da = da.squeeze(drop=True)

        if self.name_lat_bc in da.dims and self.name_lon_bc in da.dims:
            da = da.transpose(self.name_lat_bc, self.name_lon_bc)

        src = np.asarray(da.values, dtype=np.float64)
        if src.ndim != 2:
            raise ValueError(
                f"Boundary condition variable {name} must be 2D after "
                f"removing singleton dimensions; got shape {src.shape}"
            )

        x_source, y_source = self._source_axes(name, da, src.shape)
        return src, x_source, y_source

    def _prepare_source_3D(self, name, da, needed):
        if da.ndim > 3:
            da = da.squeeze(drop=True)
        if self.name_time_bc is not None and self.name_time_bc in da.dims:
            da = da.isel({self.name_time_bc: needed})
            if self.name_lat_bc in da.dims and self.name_lon_bc in da.dims:
                da = da.transpose(self.name_time_bc, self.name_lat_bc, self.name_lon_bc)

        src = np.asarray(da.values, dtype=np.float64)
        if src.ndim == 2:
            src = src[np.newaxis]
        if src.ndim != 3:
            raise ValueError(
                f"Boundary condition variable {name} must be 3D after "
                f"time selection; got shape {src.shape}"
            )

        x_source, y_source = self._source_axes(name, da, src.shape[-2:])
        return src, x_source, y_source

    def _axis_to_faces(self, axis):
        axis = np.asarray(axis, dtype=np.float64)
        if axis.size < 2:
            raise ValueError('At least two coordinates are needed to infer C-grid faces')
        faces = np.empty(axis.size + 1, dtype=axis.dtype)
        faces[1:-1] = 0.5 * (axis[:-1] + axis[1:])
        faces[0] = axis[0] - 0.5 * (axis[1] - axis[0])
        faces[-1] = axis[-1] + 0.5 * (axis[-1] - axis[-2])
        return faces

    def _source_axes(self, name, da, spatial_shape):
        if self.name_lon_bc in da.coords:
            lon_bc = np.asarray(da[self.name_lon_bc].values)
        else:
            lon_bc = np.asarray(self.lon_bc)
        if self.name_lat_bc in da.coords:
            lat_bc = np.asarray(da[self.name_lat_bc].values)
        else:
            lat_bc = np.asarray(self.lat_bc)

        if lon_bc.ndim != 1 or lat_bc.ndim != 1:
            raise ValueError(
                f"Boundary condition variable {name} requires 1D source "
                f"coordinates for 3D interpolation; got lon {lon_bc.shape}, "
                f"lat {lat_bc.shape}"
            )

        ny, nx = spatial_shape
        if self.c_grid and name == 'U' and len(lon_bc) + 1 == nx and len(lat_bc) == ny:
            lon_bc = self._axis_to_faces(lon_bc)
        elif self.c_grid and name == 'V' and len(lon_bc) == nx and len(lat_bc) + 1 == ny:
            lat_bc = self._axis_to_faces(lat_bc)

        if len(lon_bc) != nx or len(lat_bc) != ny:
            raise ValueError(
                f"Boundary condition variable {name} shape {(ny, nx)} does "
                f"not match source coordinates lon {lon_bc.shape}, "
                f"lat {lat_bc.shape}"
            )

        return pyinterp.Axis(lon_bc, is_circle=True), pyinterp.Axis(lat_bc)

    def _interp_3D(self, time):
        has_time = self.time_bc is not None and self.time_bc.size > 1

        if has_time:
            tbc = self.time_bc
            i0 = np.clip(np.searchsorted(tbc, time, side='right') - 1, 0, len(tbc) - 2)
            i1 = i0 + 1
            t0f = tbc[i0].astype('datetime64[ns]').astype(np.float64)
            t1f = tbc[i1].astype('datetime64[ns]').astype(np.float64)
            ttf = time.astype('datetime64[ns]').astype(np.float64)
            denom = np.where(t1f > t0f, t1f - t0f, 1.0)
            w1 = (ttf - t0f) / denom
            below = time < tbc[0]
            above = time > tbc[-1]
            w1 = np.where(below, 0.0, w1)
            w1 = np.where(above, 1.0, w1)
            w0 = 1.0 - w1
            needed = np.unique(np.concatenate([i0, i1]))
            pos = {int(k): n for n, k in enumerate(needed)}

        var_interp = {}
        for name in self.var:
            if not self.c_grid or name not in ('U', 'V'):
                lon_t = self.lon
                lat_t = self.lat
                mask_t = self.mask
            elif name == 'U':
                lon_t = np.zeros((self.lon.shape[0], self.lon.shape[1] + 1))
                lat_t = np.zeros((self.lat.shape[0], self.lat.shape[1] + 1))
                lon_t[:, 1:-1] = (self.lon[:, 1:] + self.lon[:, :-1]) / 2.
                lon_t[:, 0] = self.lon[:, 0] - (self.lon[:, 1] - self.lon[:, 0]) / 2.
                lon_t[:, -1] = self.lon[:, -1] + (self.lon[:, -1] - self.lon[:, -2]) / 2.
                lat_t[:, 1:-1] = (self.lat[:, 1:] + self.lat[:, :-1]) / 2.
                lat_t[:, 0] = self.lat[:, 0] - (self.lat[:, 1] - self.lat[:, 0]) / 2.
                lat_t[:, -1] = self.lat[:, -1] + (self.lat[:, -1] - self.lat[:, -2]) / 2.
                mask_t = np.zeros((self.mask.shape[0], self.mask.shape[1] + 1), dtype=bool)
                mask_t[:, 1:-1] = self.mask[:, :-1] | self.mask[:, 1:]
                mask_t[:, 0] = self.mask[:, 0]
                mask_t[:, -1] = self.mask[:, -1]
            else:
                lon_t = np.zeros((self.lat.shape[0] + 1, self.lat.shape[1]))
                lat_t = np.zeros((self.lat.shape[0] + 1, self.lat.shape[1]))
                lon_t[1:-1, :] = (self.lon[1:, :] + self.lon[:-1, :]) / 2.
                lon_t[0, :] = self.lon[0, :] - (self.lon[1, :] - self.lon[0, :]) / 2.
                lon_t[-1, :] = self.lon[-1, :] + (self.lon[-1, :] - self.lon[-2, :]) / 2.
                lat_t[1:-1, :] = (self.lat[1:, :] + self.lat[:-1, :]) / 2.
                lat_t[0, :] = self.lat[0, :] - (self.lat[1, :] - self.lat[0, :]) / 2.
                lat_t[-1, :] = self.lat[-1, :] + (self.lat[-1, :] - self.lat[-2, :]) / 2.
                mask_t = np.zeros((self.mask.shape[0] + 1, self.mask.shape[1]), dtype=bool)
                mask_t[1:-1, :] = self.mask[:-1, :] | self.mask[1:, :]
                mask_t[0, :] = self.mask[0, :]
                mask_t[-1, :] = self.mask[-1, :]

            out_shape = lon_t.shape
            x_flat = lon_t.T.ravel()
            y_flat = lat_t.T.ravel()

            if not has_time:
                da = self.var[name]
                src, x_source_axis, y_source_axis = self._prepare_source_2D(name, da)
                grid_source = pyinterp.Grid2D(x_source_axis, y_source_axis, src.T)
                _slice = pyinterp.bivariate(grid_source, x_flat, y_flat, bounds_error=False).reshape(out_shape[::-1]).T
                _slice[mask_t] = np.nan
                var_interp[name] = np.repeat(_slice[np.newaxis, :, :], len(time), axis=0)
                continue
            
            da = self.var[name]
            src, x_source_axis, y_source_axis = self._prepare_source_3D(name, da, needed)

            # Land/no-data masks are normally invariant through time.
            # Reuse one nearest-neighbour index map for the entire stack,
            # instead of solving the same Gauss-Seidel fill independently for
            # every date.  Model-specific set_bc still applies its established
            # target-grid land/coast handling afterwards.
            filled = _fill_nan_nearest(src)

            nt_out = len(time)
            n_needed = src.shape[0]
            result = np.empty((nt_out,) + out_shape, dtype=np.float64)

            if nt_out > n_needed:
                spatial = np.empty((n_needed,) + out_shape, dtype=np.float64)
                for k in range(n_needed):
                    g2 = pyinterp.Grid2D(x_source_axis, y_source_axis, np.ascontiguousarray(filled[k].T))
                    spatial[k] = pyinterp.bivariate(g2, x_flat, y_flat, bounds_error=False).reshape(out_shape[::-1]).T
                for t in range(nt_out):
                    k0 = pos[int(i0[t])]
                    k1 = pos[int(i1[t])]
                    if k0 == k1 or w1[t] == 0.0:
                        result[t] = spatial[k0]
                    elif w1[t] == 1.0:
                        result[t] = spatial[k1]
                    else:
                        result[t] = w0[t] * spatial[k0] + w1[t] * spatial[k1]
                    result[t][mask_t] = np.nan
            else:
                for t in range(nt_out):
                    k0 = pos[int(i0[t])]
                    k1 = pos[int(i1[t])]
                    if k0 == k1 or w1[t] == 0.0:
                        blend = filled[k0]
                    elif w1[t] == 1.0:
                        blend = filled[k1]
                    else:
                        blend = w0[t] * filled[k0] + w1[t] * filled[k1]
                    g2 = pyinterp.Grid2D(x_source_axis, y_source_axis, np.ascontiguousarray(blend.T))
                    interp = pyinterp.bivariate(g2, x_flat, y_flat, bounds_error=False).reshape(out_shape[::-1]).T
                    interp[mask_t] = np.nan
                    result[t] = interp

            var_interp[name] = result

        return var_interp

    def _interp_1D(self, time):
        var_interp = {}
        if len(time.shape) == 0:
            time = np.array([time])
        elif len(time.shape) == 1:
            time = np.ascontiguousarray(time)
        else:
            sys.exit('Time dimension must be 1D or 0D')

        for name in self.var:
            var_data = self.var[name]
            time_bc = self.time_bc
            nt = len(time)

            if time_bc is None or np.size(time_bc) <= 1:
                if self.name_time_bc is not None and self.name_time_bc in var_data.dims:
                    var_data = var_data.isel({self.name_time_bc: 0})
                interp_data = np.repeat(var_data.values[np.newaxis, ...], nt, axis=0)
            else:
                interp_result = var_data.interp({self.name_time_bc: xr.DataArray(time, dims=[self.name_time_bc])}, method="linear")
                interp_data = interp_result.values

            if hasattr(self, 'mask'):
                mask = self.mask
                if mask.shape == interp_data.shape[1:]:
                    interp_data[:, mask] = np.nan

            var_interp[name] = interp_data

        return var_interp


def Model(config, State, verbose=True):
    """
    NAME
        Model

    DESCRIPTION
        Main function calling subclass for specific models
    """
    if config.MOD is None:
        return
    
    elif config.MOD.super is None:
        return Model_multi(config,State)

    elif config.MOD.super is not None:
        if verbose:
            print(config.MOD)
        if config.MOD.super=='MOD_DIFF':
            return Model_diffusion(config,State)
        elif config.MOD.super=='MOD_QG1L':
            return Model_qg1l(config,State)
        elif config.MOD.super=='MOD_CSW1L':
            return Model_csw1l(config,State)
        elif config.MOD.super=='MOD_QGSW':
            return Model_qgsw(config,State)
        elif config.MOD.super=='MOD_BMIT':
            return Model_bmit(config,State)
        else:
            sys.exit(config.MOD.super + ' not implemented yet')
    else:
        sys.exit('super class if not defined')
    

###############################################################################
#                            Mother Model                                     #
###############################################################################

class _ITOpenBoundaryExtensionMixin:

    def _init_it_open_boundary_extension(self, config_mod):
        """Precompute S/N/W/E sponge extension weights on h-points.

        The current implementation intentionally targets only open-boundary
        sponge bands. Coast, island, and land-adjacent sponge extensions are a
        separate future step because their interior-edge geometry is not a
        simple S/N/W/E row or column.
        """

        self.extend_it_open_boundary_sponge = bool(
            getattr(config_mod, 'extend_it_open_boundary_sponge', False))
        self._it_open_boundary_extension_active = False
        self._it_open_boundary_extension_shape = (self.ny, self.nx)
        self._it_open_boundary_extension_active_mask = jnp.zeros((self.ny, self.nx), dtype=bool)
        self._it_open_boundary_extension_wS = jnp.zeros((self.ny, self.nx))
        self._it_open_boundary_extension_wN = jnp.zeros((self.ny, self.nx))
        self._it_open_boundary_extension_wW = jnp.zeros((self.ny, self.nx))
        self._it_open_boundary_extension_wE = jnp.zeros((self.ny, self.nx))
        self._it_open_boundary_extension_iS = 0
        self._it_open_boundary_extension_iN = self.ny - 1
        self._it_open_boundary_extension_jW = 0
        self._it_open_boundary_extension_jE = self.nx - 1

        if not self.extend_it_open_boundary_sponge:
            return
        if not getattr(self, 'flag_bc_sponge', False):
            return
        mask_names = ('sponge_on_h_S', 'sponge_on_h_N',
                      'sponge_on_h_W', 'sponge_on_h_E')
        if not all(hasattr(self, name) for name in mask_names):
            return

        masks = [np.asarray(getattr(self, name), dtype=bool) for name in mask_names]
        if not any(np.any(mask) for mask in masks):
            return

        yy, xx = np.indices((self.ny, self.nx))

        def edge_weight(mask, distance):
            if not np.any(mask):
                return np.zeros((self.ny, self.nx), dtype=float)
            width = np.nanmax(np.where(mask, distance, 0.0))
            if not np.isfinite(width) or width <= 0:
                width = 1.0
            r = np.clip(distance / width, 0.0, 1.0)
            smooth = r * r * r * (r * (r * 6.0 - 15.0) + 10.0)
            return np.where(mask, 1.0 - smooth, 0.0)

        wS = edge_weight(masks[0], yy.astype(float))
        wN = edge_weight(masks[1], (self.ny - 1 - yy).astype(float))
        wW = edge_weight(masks[2], xx.astype(float))
        wE = edge_weight(masks[3], (self.nx - 1 - xx).astype(float))
        power = float(getattr(config_mod, 'bc_it_corner_weight_power', 1.0))
        if power != 1.0:
            wS, wN, wW, wE = (wS**power, wN**power, wW**power, wE**power)
        wsum = wS + wN + wW + wE
        active = wsum > 0
        wsum_safe = np.where(active, wsum, 1.0)
        wS, wN, wW, wE = (wS / wsum_safe, wN / wsum_safe,
                          wW / wsum_safe, wE / wsum_safe)

        def sponge_interior_row(mask, from_south):
            rows = np.where(np.any(mask, axis=1))[0]
            if rows.size == 0:
                return 0 if from_south else self.ny - 1
            if from_south:
                return int(rows.max())
            return int(rows.min())

        def sponge_interior_col(mask, from_west):
            cols = np.where(np.any(mask, axis=0))[0]
            if cols.size == 0:
                return 0 if from_west else self.nx - 1
            if from_west:
                return int(cols.max())
            return int(cols.min())

        self._it_open_boundary_extension_active = True
        self._it_open_boundary_extension_active_mask = jnp.asarray(active)
        self._it_open_boundary_extension_wS = jnp.asarray(wS)
        self._it_open_boundary_extension_wN = jnp.asarray(wN)
        self._it_open_boundary_extension_wW = jnp.asarray(wW)
        self._it_open_boundary_extension_wE = jnp.asarray(wE)
        self._it_open_boundary_extension_iS = sponge_interior_row(masks[0], True)
        self._it_open_boundary_extension_iN = sponge_interior_row(masks[1], False)
        self._it_open_boundary_extension_jW = sponge_interior_col(masks[2], True)
        self._it_open_boundary_extension_jE = sponge_interior_col(masks[3], False)

    def _extend_it_open_boundary_field(self, field):
        if field is None or not getattr(self, '_it_open_boundary_extension_active', False):
            return field
        field = jnp.asarray(field)
        if field.shape != getattr(self, '_it_open_boundary_extension_shape', None):
            return field

        refS = jnp.broadcast_to(field[self._it_open_boundary_extension_iS:self._it_open_boundary_extension_iS + 1, :], field.shape)
        refN = jnp.broadcast_to(field[self._it_open_boundary_extension_iN:self._it_open_boundary_extension_iN + 1, :], field.shape)
        refW = jnp.broadcast_to(field[:, self._it_open_boundary_extension_jW:self._it_open_boundary_extension_jW + 1], field.shape)
        refE = jnp.broadcast_to(field[:, self._it_open_boundary_extension_jE:self._it_open_boundary_extension_jE + 1], field.shape)
        extended = (
            self._it_open_boundary_extension_wS * refS
            + self._it_open_boundary_extension_wN * refN
            + self._it_open_boundary_extension_wW * refW
            + self._it_open_boundary_extension_wE * refE
        )
        return jnp.where(self._it_open_boundary_extension_active_mask, extended, field)

    def _extend_it_open_boundary_static_field(self, field):
        if field is None:
            return None
        arr = np.asarray(field)
        if arr.shape != (self.ny, self.nx):
            return field
        return np.asarray(self._extend_it_open_boundary_field(jnp.asarray(arr)))


class M:

    def __init__(self,config,State):
        
        # Time parameters
        self.dt = config.MOD.dtmodel
        if self.dt>0:
            self.nt = 1 + int((config.EXP.final_date - config.EXP.init_date).total_seconds()//self.dt)
        else:
            self.nt = 1
        self.T = np.arange(self.nt) * self.dt
        self.ny = State.ny
        self.nx = State.nx
        
        # Construct timestamps
        if self.dt>0:
            self.timestamps = [] 
            t = config.EXP.init_date
            while t<=config.EXP.final_date:
                self.timestamps.append(t)
                t += timedelta(seconds=self.dt)
            self.timestamps = np.asarray(self.timestamps)
        else:
            self.timestamps = np.array([config.EXP.init_date])

        # Model variables
        self.name_var = config.MOD.name_var
        if config.MOD.var_to_save is not None:
            self.var_to_save = config.MOD.var_to_save 
        else:
            self.var_to_save = []
            for name in self.name_var:
                self.var_to_save.append(self.name_var[name])

        # Load boundary conditions if configured.  `bc_file` is the legacy
        # single-source form; `bc_files` lets each model variable come from its
        # own product with its own coordinate/variable names.
        self._bc_fields = None
        bc_sources = []
        if hasattr(config.MOD, 'bc_file') and config.MOD.bc_file is not None:
            bc_sources.append(type('BC_Config', (), {
                'file': config.MOD.bc_file,
                'name_lon': config.MOD.bc_name_lon,
                'name_lat': config.MOD.bc_name_lat,
                'name_time': config.MOD.bc_name_time,
                'name_var': config.MOD.bc_name_var,
                'c_grid': config.MOD.bc_c_grid,
                '_config': config
            })())
        if hasattr(config.MOD, 'bc_files') and config.MOD.bc_files is not None:
            for name, bc_src in config.MOD.bc_files.items():
                if isinstance(bc_src, str):
                    bc_src = {'file': bc_src}
                name_var = bc_src.get('name_var', name)
                if isinstance(name_var, str):
                    name_var = {name: name_var}
                bc_sources.append(type('BC_Config', (), {
                    'file': bc_src.get('file'),
                    'name_lon': bc_src.get('name_lon', config.MOD.bc_name_lon),
                    'name_lat': bc_src.get('name_lat', config.MOD.bc_name_lat),
                    'name_time': bc_src.get('name_time', config.MOD.bc_name_time),
                    'name_var': name_var,
                    'c_grid': bc_src.get('c_grid', config.MOD.bc_c_grid),
                    '_config': config
                })())

        bc_fields = []
        for bc_config in bc_sources:
            try:
                bc_fields.append(_BcFields(bc_config, State))
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to load boundary conditions from {bc_config.file}: {e}")
        if len(bc_fields) == 1:
            self._bc_fields = bc_fields[0]
        elif len(bc_fields) > 1:
            self._bc_fields = _MultiBcFields(bc_fields)

    def _land_mask_for_shape(self, State, shape):
        if State.mask is None or len(shape) < 2:
            return None
        h_shape = State.mask.shape
        var_shape = tuple(shape[-2:])
        if var_shape == h_shape:
            return State.mask
        if var_shape == (h_shape[0], h_shape[1] + 1):
            mask = np.zeros(var_shape, dtype=bool)
            mask[:, 1:-1] = State.mask[:, :-1] | State.mask[:, 1:]
            mask[:, 0] = State.mask[:, 0]
            mask[:, -1] = State.mask[:, -1]
            return mask
        if var_shape == (h_shape[0] + 1, h_shape[1]):
            mask = np.zeros(var_shape, dtype=bool)
            mask[1:-1, :] = State.mask[:-1, :] | State.mask[1:, :]
            mask[0, :] = State.mask[0, :]
            mask[-1, :] = State.mask[-1, :]
            return mask
        return None

    def _set_land_nan(self, State, name_var=None):
        if State.mask is None:
            return
        if name_var is None:
            name_var = self.name_var.values()
        for name in name_var:
            if name not in State.var:
                continue
            arr = State.var[name]
            mask = self._land_mask_for_shape(State, arr.shape)
            if mask is None:
                continue
            if State.preserve_device_arrays and isinstance(arr, (jax.Array, jax.core.Tracer)):
                arr = jnp.asarray(arr, dtype=float)
                State.var[name] = jnp.where(jnp.asarray(mask), jnp.nan, arr)
            else:
                arr = np.asarray(arr).astype(float, copy=True)
                arr[..., mask] = np.nan
                State.var[name] = arr

    def _finite_model_array(self, State, name, fill_land='zero'):
        arr = State.var[name]
        if isinstance(arr, (jax.Array, jax.core.Tracer)):
            return jnp.nan_to_num(
                jnp.asarray(arr),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        arr = np.asarray(arr, dtype=float)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        if fill_land == 'nearest':
            return _fill_nan_nearest(arr)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def init(self, State, t0=0):

        self._set_land_nan(State)
        return
    
    def set_bc(self,time_bc,var_bc=None,**kwargs):
        """Set boundary conditions from either direct interpolation or provided data.
        
        Parameters
        ----------
        time_bc : array-like
            Timestamps for boundary conditions (np.datetime64 array)
        var_bc : dict, optional
            Pre-interpolated BC variables {name: array}. If None and self._bc_fields 
            is available, will interpolate from loaded BC file automatically.
        """
        # If var_bc not provided but we have loaded BC fields, interpolate them
        if var_bc is None and self._bc_fields is not None:
            var_bc = self._bc_fields.interp(time_bc)
        
        # Store BC data in internal dict for use in model methods
        if var_bc is not None:
            self.bc = var_bc
        else:
            self.bc = {}

    def ano_bc(self,t,State,sign):

        return
        
    def step(self,State,nstep=1,t=None):

        return 
    
    def step_tgl(self,dState,State,nstep=1,t=None):

        return

    def step_adj(self,adState,State,nstep=1,t=None):

        return
    
    def save_output(self,State,present_date,name_var=None,t=None):
        
        State.save_output(present_date,name_var)
    

###############################################################################
#                            Diffusion Models                                 #
###############################################################################
         
class Model_diffusion(M):

    @staticmethod
    def _sponge_is_configured(distance, coefficient):
        return (
            distance is not None and distance > 0
            and coefficient is not None and coefficient > 0
        )
    
    def __init__(self,config,State):

        super().__init__(config,State)
        
        self.Kdiffus = config.MOD.Kdiffus
        self.SIC_mod = config.MOD.SIC_mod
        self.dx = State.DX
        self.dy = State.DY
        self.land_mask_device = (
            None if State.mask is None else jnp.asarray(State.mask, dtype=bool))

        # Initialization 
        if (config.GRID.super == 'GRID_FROM_FILE') and (config.MOD.name_init_var is not None):
            dsin = grid.open_grid_restart(config)
            for name in self.name_var:
                if name in config.MOD.name_init_var:
                    var_init = dsin[config.MOD.name_init_var[name]]
                    if len(var_init.shape)==3:
                        var_init = var_init[0,:,:]
                    if config.GRID.subsampling is not None:
                        var_init = var_init[::config.GRID.subsampling,::config.GRID.subsampling]
                    # Keep the backing dataset open until the lazy array has
                    # been computed. Boolean assignment on Dask arrays uses
                    # advanced indexing and fails for these restart grids.
                    State.var[self.name_var[name]] = var_init.fillna(0).values
                else:
                    State.var[self.name_var[name]] = np.zeros((State.ny,State.nx))
            dsin.close()
            del dsin
        else:
            for name in self.name_var:  
                State.var[self.name_var[name]] = np.zeros((State.ny,State.nx))
        
        # Model Parameters (Flux)
        for name in self.name_var:
            State.params[self.name_var[name]] = np.zeros((State.ny,State.nx))

        # Initialize boundary condition dictionnary for each model variable
        self.bc = {}
        for _name_var_mod in self.name_var:
            self.bc[_name_var_mod] = {}
        self.init_from_bc = config.MOD.init_from_bc

        # A diffusion-model sponge is active only when both its spatial width
        # and its per-step relaxation coefficient are explicitly positive.
        self.sponge_coef = getattr(config.MOD, 'sponge_coef', None)
        self.sponge_active = self._sponge_is_configured(
            config.MOD.dist_sponge_bc, self.sponge_coef)

        # Weight map to apply BC in a smoothed way
        if self.sponge_active:
            mask = (np.zeros((State.ny, State.nx), dtype=bool)
                    if State.mask is None else np.asarray(State.mask, dtype=bool))
            Wbc = grid.compute_weight_map(
                State.lon, State.lat, mask, config.MOD.dist_sponge_bc)
        else:
            Wbc = np.zeros((State.ny,State.nx))
        self.Wbc = Wbc
        self.Wbc_device = jnp.asarray(Wbc)
        
        if config.INV is not None and config.INV.super=='INV_4DVAR' and config.INV.compute_test:
            print('Tangent test:')
            tangent_test(self,State)
            print('Adjoint test:')
            adjoint_test(self,State)

    

    def init(self, State, t0=0):

        if type(self.init_from_bc)==dict:
            for name in self.init_from_bc:
                if self.init_from_bc[name] and name in self.name_var:
                    boundary = self._apply_bc(t0, names=(name,))
                    if name in boundary:
                        State.setvar(boundary[name], self.name_var[name])
        elif self.init_from_bc:
            for name, boundary in self._apply_bc(t0).items():
                State.setvar(boundary, self.name_var[name])
        self._set_land_nan(State)

    def save_output(self,State,present_date,name_var=None,t=None):

        State0 = State.copy()
        State0.save_output(present_date, name_var=self.name_var.values())

    def set_bc(self, time_bc, var_bc=None, t_bc=None, **kwargs):
        # If var_bc not provided but we have loaded BC fields, interpolate them
        if var_bc is None and self._bc_fields is not None:
            var_bc = self._bc_fields.interp(time_bc)
        
        if var_bc is None:
            return
        
        time_keys = t_bc if t_bc is not None else time_bc
        for name in self.name_var:
            if name not in var_bc:
                continue
            for i, t in enumerate(time_keys):
                self.bc[name][t] = _fill_nan_nearest(var_bc[name][i])

    def _nearest_bc_key(self, name, time):
        keys = list(self.bc.get(name, {}).keys())
        if not keys:
            return None
        if time in self.bc[name]:
            return time
        key_array = np.asarray(keys)
        target = time
        time_is_datetime = np.issubdtype(np.asarray(time).dtype, np.datetime64)
        if np.issubdtype(key_array.dtype, np.datetime64) and not time_is_datetime:
            target = np.datetime64(self.timestamps[0]) + np.timedelta64(int(time), 's')
        elif not np.issubdtype(key_array.dtype, np.datetime64) and time_is_datetime:
            target = (np.datetime64(time) - np.datetime64(self.timestamps[0])) / np.timedelta64(1, 's')
        return keys[int(np.argmin(np.abs(key_array - target)))]

    def _apply_bc(self, time, names=None):
        """Return the nearest available BC for each requested model variable."""
        if names is None:
            names = self.name_var.keys()
        boundary = {}
        for name in names:
            key = self._nearest_bc_key(name, time)
            if key is not None:
                boundary[name] = jnp.asarray(self.bc[name][key])
        return boundary

    def _stack_bc(self, time):
        boundary = self._apply_bc(time)
        values = []
        available = []
        for name in self.name_var:
            if name in boundary:
                values.append(boundary[name])
                available.append(True)
            else:
                values.append(jnp.zeros((self.ny, self.nx)))
                available.append(False)
        return jnp.stack(values), jnp.asarray(available)

    def prepare_scan_boundary_conditions(self, times, nstep):
        """Prepare end-of-interval BC targets for a device-side scan."""
        if not getattr(self, 'sponge_active', False):
            nt = len(times)
            return (
                jnp.zeros((nt, 0), dtype=self.Wbc_device.dtype),
                jnp.zeros((nt, 0), dtype=bool),
            )
        fields = [self._stack_bc(t + nstep * self.dt) for t in np.asarray(times)]
        values, available = zip(*fields)
        return jnp.stack(values), jnp.stack(available)

    def _sponge_boundary(self, t, nstep, Xb):
        if Xb is not None:
            return Xb
        if not getattr(self, 'sponge_active', False) or t is None:
            return None
        return self._stack_bc(t + nstep * self.dt)

    def _apply_sponge(self, value, boundary, available, index):
        if not getattr(self, 'sponge_active', False) or boundary is None:
            return value
        target = boundary[index]
        relaxed = value + self.sponge_coef * self.Wbc_device * (target - value)
        return jnp.where(available[index], relaxed, value)

    def _diffuse_once(self, value):
        """Apply one diffusion step with zero normal flux across land."""
        if self.Kdiffus <= 0:
            return value
        center = value[1:-1, 1:-1]
        east = value[1:-1, 2:]
        west = value[1:-1, :-2]
        north = value[2:, 1:-1]
        south = value[:-2, 1:-1]
        land = getattr(self, 'land_mask_device', None)
        if land is not None:
            east = jnp.where(land[1:-1, 2:], center, east)
            west = jnp.where(land[1:-1, :-2], center, west)
            north = jnp.where(land[2:, 1:-1], center, north)
            south = jnp.where(land[:-2, 1:-1], center, south)
        tendency = self.dt * self.Kdiffus * (
            (east + west - 2 * center) / self.dx[1:-1, 1:-1]**2
            + (north + south - 2 * center) / self.dy[1:-1, 1:-1]**2
        )
        if land is not None:
            tendency = jnp.where(land[1:-1, 1:-1], 0.0, tendency)
        return value.at[1:-1, 1:-1].add(tendency)


    def step(self,State,nstep=1,t=None,Xb=None):

        sponge = self._sponge_boundary(t, nstep, Xb)
        boundary, available = sponge if sponge is not None else (None, None)

        # Loop on model variables
        for index, name in enumerate(self.name_var):

            # Get state variable
            var0 = jnp.asarray(State.getvar(self.name_var[name]))
            
            # Init
            var1 = +var0

            # Time propagation
            for _ in range(nstep):
                var1 = self._diffuse_once(var1)
                var1 = self._apply_sponge(var1, boundary, available, index)
            
            # Update state
            if self.name_var[name] in State.params:
                params = State.params[self.name_var[name]]
                var1 = var1 + nstep*self.dt/(3600*24) * params

            State.setvar(var1, self.name_var[name])
        


    def step_tgl(self,dState,State,nstep=1,t=None,Xb=None):

        sponge = self._sponge_boundary(t, nstep, Xb)
        boundary, available = sponge if sponge is not None else (None, None)

        # Loop on model variables
        for index, name in enumerate(self.name_var):

            # Get state variable
            var0 = jnp.asarray(dState.getvar(self.name_var[name]))
            
            # Init
            var1 = +var0
            
            # Time propagation
            for _ in range(nstep):
                var1 = self._diffuse_once(var1)
                if getattr(self, 'sponge_active', False) and boundary is not None:
                    factor = 1.0 - self.sponge_coef * self.Wbc_device
                    var1 = jnp.where(available[index], factor * var1, var1)
            

            # Update state
            if self.name_var[name] in dState.params:
                params = dState.params[self.name_var[name]]
                var1 = var1 + nstep*self.dt/(3600*24) * params

            dState.setvar(var1,self.name_var[name])
        
    def step_adj(self,adState,State,nstep=1,t=None,Xb=None):

        sponge = self._sponge_boundary(t, nstep, Xb)
        boundary, available = sponge if sponge is not None else (None, None)

        # Loop on model variables
        for index, name in enumerate(self.name_var):

            # Get state variable
            ad_output = jnp.asarray(adState.getvar(self.name_var[name]))
            ad_output = jnp.where(jnp.isnan(ad_output), 0, ad_output)
            advar1 = ad_output

            # Time propagation
            for _ in range(nstep):
                advar0 = advar1
                if getattr(self, 'sponge_active', False) and boundary is not None:
                    factor = 1.0 - self.sponge_coef * self.Wbc_device
                    advar0 = jnp.where(available[index], factor * advar0, advar0)
                if self.Kdiffus>0:
                    zero = jnp.zeros_like(advar0)
                    advar1 = jax.linear_transpose(
                        self._diffuse_once, zero)(advar0)[0]
                else:
                    advar1 = advar0
                    
                

            # Update state and parameters
            if self.name_var[name] in State.params:
                adState.params[self.name_var[name]] = (
                    adState.params[self.name_var[name]]
                    + nstep*self.dt/(3600*24) * ad_output
                )
            
            adState.setvar(advar1,self.name_var[name])


###############################################################################
#                       Quasi-Geostrophic Models                              #
###############################################################################
    
class Model_qg1l(M):

    def __init__(self,config,State):

        super().__init__(config,State)

        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

        # Model specific libraries
        if config.MOD.dir_model is None:
            dir_model = os.path.realpath(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..','models','model_qg1l'))
        else:
            dir_model = config.MOD.dir_model  
        qgm = SourceFileLoader("qgm",f'{dir_model}/jqgm.py').load_module() 
        model = getattr(qgm, config.MOD.name_class)

        # Coriolis
        if config.MOD.f0 is not None and config.MOD.constant_f:
            self.f = config.MOD.f0
        else:
            self.f = State.f
            f0 = np.nanmean(self.f)
            self.f[np.isnan(self.f)] = f0
            
        # Open Rossby Radius if provided
        if config.MOD.filec_aux is not None and os.path.exists(config.MOD.filec_aux):

            ds = xr.open_dataset(config.MOD.filec_aux)
            name_lon = config.MOD.name_var_c['lon']
            lon = ds[name_lon]
            # Convert longitude 
            if np.sign(lon.data.min())==-1 and State.lon_unit=='0_360':
                ds = ds.assign_coords({name_lon:((name_lon, lon.data % 360))})
                ds = ds.sortby(name_lon)    
            elif np.sign(lon.data.min())>=0 and State.lon_unit=='-180_180':
                ds = ds.assign_coords({name_lon:((name_lon, (lon.data + 180) % 360 - 180))})
                ds = ds.sortby(name_lon)    

            self.c = grid.interp2d(ds,
                                   config.MOD.name_var_c,
                                   State.lon,
                                   State.lat)
            
            if config.MOD.cmin is not None:
                self.c[self.c<config.MOD.cmin] = config.MOD.cmin
            
            if config.MOD.cmax is not None:
                self.c[self.c>config.MOD.cmax] = config.MOD.cmax
            
            if config.EXP.flag_plot>0:
                plt.figure()
                plt.pcolormesh(self.c)
                plt.colorbar()
                plt.title('Rossby phase velocity')
                plt.show()
                
        else:
            self.c = config.MOD.c0 * np.ones((State.ny,State.nx))
        
        # Open MDT map if provided
        if config.MOD.path_mdt is not None and os.path.exists(config.MOD.path_mdt):
                      
            ds = xr.open_dataset(config.MOD.path_mdt)
            name_lon = config.MOD.name_var_mdt['lon']
            lon = ds[name_lon]
            # Convert longitude 
            if np.sign(lon.data.min())==-1 and State.lon_unit=='0_360':
                ds = ds.assign_coords({name_lon:((name_lon, lon.data % 360))})
                ds = ds.sortby(name_lon)    
            elif np.sign(lon.data.min())>=0 and State.lon_unit=='-180_180':
                ds = ds.assign_coords({name_lon:((name_lon, (lon.data + 180) % 360 - 180))})
                ds = ds.sortby(name_lon)    

            self.mdt = grid.interp2d(ds,
                                   config.MOD.name_var_mdt,
                                   State.lon,
                                   State.lat)
            self.mdt[np.isnan(self.mdt)] = 0
        
            if config.EXP.flag_plot>0:
                plt.figure()
                plt.pcolormesh(self.mdt)
                plt.colorbar()
                plt.title('MDT')
                plt.show()

        else:
            self.mdt = None
        
        # Open Bathymetry if provided
        if config.MOD.file_bathy_aux is not None and os.path.exists(config.MOD.file_bathy_aux):

            ds = xr.open_dataset(config.MOD.file_bathy_aux)
            name_lon = config.MOD.name_var_bathy['lon']
            lon = ds[name_lon]
            # Convert longitude 
            if np.sign(lon.data.min())==-1 and State.lon_unit=='0_360':
                ds = ds.assign_coords({name_lon:((name_lon, lon.data % 360))})
            elif np.sign(lon.data.min())>=0 and State.lon_unit=='-180_180':
                ds = ds.assign_coords({name_lon:((name_lon, (lon.data + 180) % 360 - 180))})
            ds = ds.sortby(name_lon)    
            
            bathymetry = grid.interp2d(ds,
                                   config.MOD.name_var_bathy,
                                   State.lon,
                                   State.lat)

            # Get H so that we have a maximum value 
            bathymetry[State.mask] = np.nan
            H =  np.nanmean(bathymetry) # Mean depth
            self.bathymetry_PV_term = - (bathymetry - H) / H
            self.bathymetry_PV_term[self.bathymetry_PV_term>config.MOD.bathy_ratio_max] = config.MOD.bathy_ratio_max
            self.bathymetry_PV_term[self.bathymetry_PV_term<-config.MOD.bathy_ratio_max] = -config.MOD.bathy_ratio_max
            self.bathymetry_PV_term[np.isnan(self.bathymetry_PV_term)] = config.MOD.bathy_ratio_max
            
            
            if config.EXP.flag_plot>0:
                plt.figure()
                plt.pcolormesh(self.bathymetry_PV_term)
                plt.colorbar()
                plt.title('Bathymetry PV term')
                plt.show()
                
        else:
            self.bathymetry_PV_term = np.zeros((State.ny,State.nx))
        
        # CFL
        if config.MOD.cfl is not None:
            grid_spacing = min(np.nanmin(State.DX), np.nanmin(State.DY)) 
            dt = config.MOD.cfl * grid_spacing / np.nanmax(self.c)
            divisors = [i for i in range(1, 3600 + 1) if 3600 % i == 0]  # Find all divisors of one hour in seconds
            lower_divisors = [d for d in divisors if d <= dt]
            self.dt = max(lower_divisors)  # Get closest
            print('CFL condition for c=', np.nanmean(self.c))
            print('Model time-step', self.dt)
            # Time parameters
            if self.dt>0:
                self.nt = 1 + int((config.EXP.final_date - config.EXP.init_date).total_seconds()//self.dt)
                self.timestamps = [] 
                t = config.EXP.init_date
                while t<=config.EXP.final_date:
                    self.timestamps.append(t)
                    t += timedelta(seconds=self.dt)
                self.timestamps = np.asarray(self.timestamps)
            else:
                self.nt = 1
                self.timestamps = np.array([config.EXP.init_date])
            self.T = np.arange(self.nt) * self.dt

            
        # Initialize model state
        if (config.GRID.super == 'GRID_FROM_FILE') and (config.MOD.name_init_var is not None):
            dsin = grid.open_grid_restart(config, group='variables')
            for name in self.name_var:
                if name in config.MOD.name_init_var:
                    var_init = dsin[config.MOD.name_init_var[name]]
                    if len(var_init.shape)==3:
                        var_init = var_init[0,:,:]
                    if config.GRID.subsampling is not None:
                        var_init = var_init[::config.GRID.subsampling,::config.GRID.subsampling]
                    State.var[self.name_var[name]] = var_init.values
                else:
                    State.var[self.name_var[name]] = np.zeros((State.ny,State.nx))
            dsin.close()
            del dsin
        else:
            for name in self.name_var:  
                State.var[self.name_var[name]] = np.zeros((State.ny,State.nx))
                if State.mask is not None:
                    State.var[self.name_var[name]][State.mask] = np.nan

        # Initialize model Parameters (Flux on SSH and tracers)
        for name in self.name_var:
            State.params[self.name_var[name]] = np.zeros((State.ny,State.nx))

        # Initialize boundary condition dictionnary for each model variable
        self.bc = {}
        self.forcing = {}
        for _name_var_mod in self.name_var:
            self.bc[_name_var_mod] = {}
            self.forcing[_name_var_mod] = {}
        self.init_from_bc = config.MOD.init_from_bc
        self.Wbc = np.zeros((State.ny,State.nx))
        if config.MOD.dist_sponge_bc is not None and State.mask is not None:
            # Use the same compute_sponge_components(.) machinery as Model_qgsw so that the
            # sponge weight is built from independent west/east/north/south + coastal
            # contributions and behaves consistently across models.
            if getattr(config.MOD, 'use_sponge_on_coast', True):
                mask_h = State.mask.copy()
            else:
                mask_h = np.zeros((State.ny, State.nx), dtype=bool)

            wc_h, wNS_h, wWE_h = grid.compute_sponge_components(
                State.lon, State.lat, mask_h, config.MOD.dist_sponge_bc)

            self.Wbc = 1.0 - (1.0 - wc_h) * (1.0 - wNS_h) * (1.0 - wWE_h)

            if config.EXP.flag_plot>1:
                plt.figure()
                plt.pcolormesh(self.Wbc)
                plt.colorbar()
                plt.title('Wbc')
                plt.show()
        # Sponge Rayleigh-damping coefficient for tracers (like Model_bmit sponge_coef)
        self.sponge_coef = config.MOD.sponge_coef
        if config.INV is not None and config.INV.super=='INV_4DVAR':
            self.anomaly_from_bc = config.INV.anomaly_from_bc
        else:
            self.anomaly_from_bc = False

        # Tracer advection flag
        self.advect_pv = config.MOD.advect_pv
        self.advect_tracer = config.MOD.advect_tracer
        self.forcing_tracer_from_bc = config.MOD.forcing_tracer_from_bc

        # Ageostrophic velocity flag
        if 'U' in self.name_var and 'V' in self.name_var:
            self.ageo_velocities = True
        else:
            self.ageo_velocities = False
        
        # Save SSH, geostrophic velocities and cyclogeostrophic velocities
        self.save_diagnosed_variables = config.MOD.save_diagnosed_variables
        # Save control parameters, i.e. corrective fluxes
        self.save_params = config.MOD.save_params

       # Masked array for model initialization
        SSH0 = State.getvar(name_var=self.name_var['SSH'])
    
        # Model initialization
        self.qgm = model(dx=State.DX,
                         dy=State.DY,
                         dt=self.dt,
                         SSH=SSH0,
                         c=self.c,
                         upwind=config.MOD.upwind,
                         time_scheme=config.MOD.time_scheme,
                         g=State.g,
                         f=self.f,
                         Wbc=self.Wbc,
                         Kdiffus=config.MOD.Kdiffus,
                         Kdiffus_trac=config.MOD.Kdiffus_trac,
                         bc_trac=config.MOD.bc_trac,
                         advect_pv=self.advect_pv,
                         ageo_velocities=self.ageo_velocities,
                         constant_c=config.MOD.constant_c,
                         constant_f=config.MOD.constant_f,
                         solver=config.MOD.solver,
                         tile_size=config.MOD.tile_size,
                         tile_overlap=config.MOD.tile_overlap,
                         mdt=self.mdt,
                         sponge_coef=self.sponge_coef,
                         bathymetry_PV_term=self.bathymetry_PV_term,
                         formulation=config.MOD.formulation)

        self.Wbc_device = jnp.asarray(self.Wbc, dtype=self.qgm.dtype)
        self.zero_bc_device = jnp.zeros(
            (self.ny, self.nx),
            dtype=self.qgm.dtype,
        )

        # Control parameters (mirrors Model_qgsw name_params contract)
        self.name_params = config.MOD.name_params if config.MOD.name_params is not None else []
        if 'c' in self.name_params:
            # State.params['c'] holds the c-anomaly dc(x,y); the full effective
            # phase-speed is c_eff = self.qgm.c (scalar prior) + dc.
            # For GRID_FROM_FILE, warm-start dc from a previous 'c_anomaly'
            # variable saved in the init file (same pattern as 'H' in
            # Model_qgsw). Fall back to dc=0 when the variable is absent.
            if config.GRID.super == 'GRID_FROM_FILE':
                dsin = grid.open_grid_restart(config)
                if 'c_anomaly' in dsin:
                    _dc_init = dsin['c_anomaly'].values.squeeze()
                    _dc_init[np.isnan(_dc_init)] = 0.
                    State.params['c'] = _dc_init
                else:
                    State.params['c'] = np.zeros((State.ny, State.nx))
                dsin.close()
                del dsin
            else:
                State.params['c'] = np.zeros((State.ny, State.nx))

        # Model functions initialization
        self.qgm_step = self.qgm.step_jit
        self.qgm_step_tgl = self.qgm.step_tgl_jit
        self.qgm_step_adj = self.qgm.step_adj_jit

        # Tests tgl & adj
        if config.INV is not None and config.INV.super=='INV_4DVAR' and config.INV.compute_test:

            print('QG1L_JAX Tangent test:')
            tangent_test(self,State,nstep=10)
            print('QG1L_JAX Adjoint test:')
            adjoint_test(self,State,nstep=10)

    def init(self, State, t0=0):

        if self.anomaly_from_bc:
            self._set_land_nan(State)
            return
        elif type(self.init_from_bc)==dict:
            for name in self.init_from_bc:
                if self.init_from_bc[name] and t0 in self.bc[name]:
                    State.setvar(self.bc[name][t0], self.name_var[name])
        elif self.init_from_bc:
            for name in self.name_var: 
                if t0 in self.bc[name]:
                     State.setvar(self.bc[name][t0], self.name_var[name])
        self._set_land_nan(State)
    
    def save_output(self,State,present_date,name_var=None,t=None):

        State0 = State.copy()

        name_var = [self.name_var['SSH']]

        if self.advect_tracer:
            for name in self.name_var:
                if name!='SSH':
                    name_var += [self.name_var[name]]
        
        # Save SSH and geostrophic velocities
        name_var_diag = []
        if self.save_diagnosed_variables:

            # Add MDT to sla to get SSH
            if self.mdt is not None:
                ssh = State0.getvar(name_var=self.name_var['SSH']) + self.mdt
                State0.setvar(ssh, name_var='ssh')
                if name_var is not None:
                    name_var_diag += ['ssh']
            else:
                ssh = State0.getvar(name_var=self.name_var['SSH'])
            
            # Current velocities
            # Geostrophy. Recent jaxparrow versions return velocities on T points.
            try:
                ug_t, vg_t = jaxparrow.geostrophy(
                    ssh, State0.lat, State0.lon, land_mask=State0.mask
                )
            except Exception as exc:
                warnings.warn(
                    'jaxparrow geostrophy failed; falling back to Qgm.h2uv: '
                    f'{exc}'
                )
                ug, vg = self.qgm.h2uv(jnp.asarray(ssh))
                ug_t = np.zeros((State0.ny, State0.nx))
                vg_t = np.zeros((State0.ny, State0.nx))
                ug_t[1:-1, 1:-1] = np.array(
                    0.5 * (ug[1:-1, 1:-1] + ug[1:-1, 2:])
                )
                vg_t[1:-1, 1:-1] = np.array(
                    0.5 * (vg[1:-1, 1:-1] + vg[2:, 1:-1])
                )
                if State0.mask is not None:
                    ug_t[State0.mask] = np.nan
                    vg_t[State0.mask] = np.nan
            # Cyclogeostrophy. New jaxparrow exposes algorithms such as fixed_point.
            try:
                if hasattr(jaxparrow, 'fixed_point'):
                    result = jaxparrow.fixed_point(
                        State0.lat, State0.lon, ssh_t=ssh, land_mask=State0.mask
                    )
                else:
                    result = jaxparrow.cyclogeostrophy(
                        ssh, State0.lat, State0.lon, State0.mask
                    )
                uc_t = result.ucg if hasattr(result, 'ucg') else result[0]
                vc_t = result.vcg if hasattr(result, 'vcg') else result[1]
            except Exception as exc:
                warnings.warn(
                    'jaxparrow cyclogeostrophy failed; skipping '
                    f'cyclogeostrophic velocity diagnostics: {exc}'
                )
                uc_t = None
                vc_t = None
            # Set geostrophic velocities
            State0.setvar(ug_t, name_var='ug')
            State0.setvar(vg_t, name_var='vg')
            # Set cyclogeostrophic velocities
            if uc_t is not None and vc_t is not None:
                State0.setvar(uc_t, name_var='uc')
                State0.setvar(vc_t, name_var='vc')
            # Flux
            Fssh = State0.params[self.name_var['SSH']]
            State0.setvar(Fssh, name_var='Fssh')

            if name_var is not None:
                name_var_diag += ['ug', 'vg', 'Fssh']
                if uc_t is not None and vc_t is not None:
                    name_var_diag += ['uc', 'vc']

        if 'c' in self.name_params:
            State0.var['c_anomaly'] = State.params['c']
            name_var_diag += ['c_anomaly']

        State0.save_output(present_date, name_var+name_var_diag)#, save_params=self.save_params)

    def set_bc(self,time_bc,var_bc=None,t_bc=None):

        # If var_bc not provided but we have loaded BC fields, interpolate them
        if var_bc is None and self._bc_fields is not None:
            var_bc = self._bc_fields.interp(time_bc)
        
        if var_bc is None:
            return

        # Use float model times as dict keys if provided (needed by _apply_bc);
        # fall back to time_bc keys when called without float times.
        time_keys = t_bc if t_bc is not None else time_bc

        for _name_var_bc in var_bc:
            for _name_var_mod in self.name_var:
                if _name_var_bc==_name_var_mod:
                    for i,t in enumerate(time_keys):
                        var_bc_t = +var_bc[_name_var_bc][i]
                        # Fill missing values.  For SSH we keep the historical
                        # NaN->0 behaviour (SSH=0 at land is benign).  For
                        # tracers we instead fill NaNs with the nearest valid
                        # ocean value: QG tracer advection is not mask-aware,
                        # so a 0 sitting at a land / coastal-no-data pixel of
                        # varb would (i) be copied into the initial state via
                        # init_from_bc / Wbc*bc, and (ii) leak into adjacent
                        # ocean cells through advection each step, producing
                        # the spurious ~0 psu / cold patches observed along
                        # the coastline.  Nearest-ocean fill mirrors what SW
                        # gets for free via its land mask.
                        if _name_var_mod == 'SSH':
                            var_bc_t[np.isnan(var_bc_t)] = 0.
                        else:
                            var_bc_t = _fill_nan_nearest(var_bc_t)
                        # Fill bc dictionnary
                        self.bc[_name_var_mod][t] = var_bc_t
                elif _name_var_bc==f'{_name_var_mod}_params':
                    for i,t in enumerate(time_keys):
                        var_bc[_name_var_bc][i][np.isnan(var_bc[_name_var_bc][i])] = 0.
                        self.forcing[_name_var_mod][t] = var_bc[_name_var_bc][i]
        
        self.bc_time = np.array(list(self.bc['SSH'].keys()))
        self.bc_values = {
            t: jnp.asarray(self.bc['SSH'][t], dtype=self.qgm.dtype)
            for t in self.bc['SSH']
        }
        for name in self.name_var:
            if name in self.bc and len(self.bc[name].keys()) > 0:
                self.bc_values[name] = {
                    t: jnp.asarray(value, dtype=self.qgm.dtype)
                    for t, value in self.bc[name].items()
                }
            else:
                self.bc_values[name] = {
                    t: self.bc_values[t] for t in self.bc['SSH']
                }
            
    def _apply_bc(self,t0,t1):

        if 'SSH' not in self.bc:
            return self.zero_bc_device
        elif len(self.bc['SSH'].keys())==0:
            return self.zero_bc_device
        elif t0 not in self.bc_time:
            # Find closest time
            idx_closest = np.argmin(np.abs(self.bc_time-t0))
            t0 = self.bc_time[idx_closest]

        Xb = self.bc_values[t0]

        if self.advect_tracer:
            Xb = Xb[np.newaxis,:,:]
            for name in self.name_var:
                if name!='SSH' and name in self.bc and len(self.bc[name].keys())>0:
                    device_bc = self.bc_values[name]
                    if t1 in device_bc:
                        Cb = device_bc[t1]
                    else:
                        t_list = np.asarray(list(device_bc.keys()))
                        idx_closest = np.argmin(np.abs(t_list-t1))
                        new_t1 = t_list[idx_closest]
                        Cb = device_bc[new_t1]
                    Xb = jnp.concatenate((Xb, Cb[jnp.newaxis,:,:]), axis=0)
        
        return Xb
    
    def prepare_scan_boundary_conditions(self, times, nstep):
        """Stack QG boundary fields for a device-side checkpoint scan."""
        fields = []
        for t in np.asarray(times):
            t0 = t.item() if hasattr(t, 'item') else t
            t1 = int(t0 + nstep * self.dt)
            fields.append(self._apply_bc(t0, t1))
        return jnp.stack(fields)

    def step(self,State,nstep=1,t=0,Xb=None):

        # Boundary feld
        if Xb is None:
            Xb = self._apply_bc(t,int(t+nstep*self.dt))

        # c-anomaly: c_eff = prior scalar (fixed throughout 4DVar) + 2-D anomaly dc.
        # self.qgm.c is never modified here so the forward and adjoint sweeps
        # see the same operator, keeping the gradient consistent.
        c_eff = (self.qgm.c + State.params['c'].astype(self.qgm.dtype)
                 ) if 'c' in self.name_params else None

        # Get state variable(s)
        X0 = jnp.asarray(
            State.getvar(name_var=self.name_var['SSH']),
            dtype=self.qgm.dtype,
        )
        if self.advect_tracer:
            X0 = X0[np.newaxis,:,:]
            # Ageostrophic velocities
            if self.ageo_velocities:
                U = State.getvar(name_var=self.name_var['U'])[np.newaxis,:,:]
                V = State.getvar(name_var=self.name_var['V'])[np.newaxis,:,:]
                X0 = jnp.append(X0, U, axis=0)
                X0 = jnp.append(X0, V, axis=0)
            # Tracers
            for name in self.name_var:
                if name not in ['SSH', 'U', 'V']:
                    C0 = jnp.asarray(
                        State.getvar(name_var=self.name_var[name]),
                        dtype=self.qgm.dtype,
                    )[jnp.newaxis,:,:]
                    X0 = jnp.append(X0, C0, axis=0)
        
        # init
        X1 = +X0

        # Time propagation
        X1 = self.qgm_step(X1, Xb, nstep=nstep, c=c_eff)
        
        # Update state
        if self.name_var['SSH'] in State.params:
            Fssh = State.params[self.name_var['SSH']]
            if self.advect_tracer: 
                State.setvar(X1[0]+ nstep*self.dt/(3600*24) * Fssh, 
                             name_var=self.name_var['SSH'])
                for i,name in enumerate(self.name_var):
                    if name!='SSH':
                        # Tracer sponge (Rayleigh damping toward BC) is now applied
                        # per-substep inside qgm.bc() via sponge_coef; do not double-apply here.
                        X1_i = X1[i]
                        Fc = State.params[self.name_var[name]] # Forcing term for tracer or ageostrophic velocities
                        # Add Nudging to BC 
                        if self.forcing_tracer_from_bc:
                            X1_i = X1_i + nstep*self.dt/(3600*24) * (1-self.Wbc_device) * Fc * (Xb[i] - X0[i])
                        # Only forcing flux
                        else:
                            X1_i = X1_i + nstep*self.dt/(3600*24) * (1-self.Wbc_device) * Fc
                        State.setvar(X1_i, name_var=self.name_var[name])
            else:
                X1 += nstep*self.dt/(3600*24) * Fssh
                State.setvar(X1, name_var=self.name_var['SSH'])
    
    def step_tgl(self,dState,State,nstep=1,t=0,Xb=None):

        # Boundary field
        if Xb is None:
            Xb = self._apply_bc(t,int(t+nstep*self.dt))

        # c-anomaly: c_eff = prior scalar + 2-D anomaly dc
        c_eff = (self.qgm.c + State.params['c'].astype(self.qgm.dtype)
                 ) if 'c' in self.name_params else None

        # Get state variable
        dX0 = dState.getvar(name_var=self.name_var['SSH']).astype('float64')
        X0 = State.getvar(name_var=self.name_var['SSH']).astype('float64')
        if self.advect_tracer:
            dX0 = dX0[np.newaxis,:,:]
            X0 = X0[np.newaxis,:,:]
            # Ageostrophic velocities
            if self.ageo_velocities:
                U = State.getvar(name_var=self.name_var['U'])[np.newaxis,:,:]
                V = State.getvar(name_var=self.name_var['V'])[np.newaxis,:,:]
                dU = dState.getvar(name_var=self.name_var['U'])[np.newaxis,:,:]
                dV = dState.getvar(name_var=self.name_var['V'])[np.newaxis,:,:]
                X0 = np.append(X0, U, axis=0)
                X0 = np.append(X0, V, axis=0)
                dX0 = np.append(dX0, dU, axis=0)
                dX0 = np.append(dX0, dV, axis=0)
            # Tracers
            for name in self.name_var:
                if name not in ['SSH', 'U', 'V']:
                    dC0 = dState.getvar(name_var=self.name_var[name])[np.newaxis,:,:]
                    dX0 = np.append(dX0, dC0, axis=0)
                    C0 = State.getvar(name_var=self.name_var[name])[np.newaxis,:,:]
                    X0 = np.append(X0, C0, axis=0)
        
        # init
        dX1 = +dX0.astype('float64')
        X1 = +X0.astype('float64')

        # Time propagation
        if 'c' in self.name_params:
            # Differentiate jointly w.r.t. (state, c_eff) so that the
            # linearised effect of dc on h2pv is captured.
            dc = jnp.asarray(dState.params['c'], dtype=self.qgm.dtype)
            _, dX1 = jax.jvp(
                lambda X, c: self.qgm.step_jit(X, Xb, nstep=nstep, c=c),
                (X1, c_eff), (dX1, dc))
        else:
            dX1 = self.qgm_step_tgl(dX1, X1, Xb, nstep=nstep)

        # Convert to numpy and reshape
        dX1 = np.array(dX1).astype('float64')

        # Update state
        if self.name_var['SSH'] in dState.params:
            dFssh = dState.params[self.name_var['SSH']].astype('float64') # Forcing term for SSH
            if self.advect_tracer:
                dX1[0] +=  nstep*self.dt/(3600*24) * dFssh  
                dState.setvar(dX1[0], name_var=self.name_var['SSH'])
                for i,name in enumerate(self.name_var):
                    if name!='SSH':
                        dFc = dState.params[self.name_var[name]] # Forcing term for tracer or ageostrophic velocities
                        # Add Nudging to BC 
                        if self.forcing_tracer_from_bc:
                            Fc = State.params[self.name_var[name]] 
                            dX1[i] +=  nstep*self.dt/(3600*24) * (1-self.Wbc) *\
                                  (dFc * (Xb[i] - X0[i]) - Fc * dX0[i])
                        # Only forcing flux
                        else:
                            dX1[i] +=  nstep*self.dt/(3600*24) * dFc  * (1-self.Wbc)
                        dState.setvar(dX1[i], name_var=self.name_var[name])
            else:
                dX1 += nstep*self.dt/(3600*24) * dFssh  
                dState.setvar(dX1, name_var=self.name_var['SSH'])

    def step_adj(self,adState,State,nstep=1,t=0,Xb=None):

        # Boundary field
        if Xb is None:
            Xb = self._apply_bc(t,int(t+nstep*self.dt))

        # c-anomaly: c_eff = prior scalar + 2-D anomaly dc
        c_eff = (self.qgm.c + State.params['c'].astype(self.qgm.dtype)
                 ) if 'c' in self.name_params else None

        # Get state variable
        adSSH0 = jnp.asarray(
            adState.getvar(name_var=self.name_var['SSH']),
            dtype=self.qgm.dtype,
        )
        SSH0 = jnp.asarray(
            State.getvar(name_var=self.name_var['SSH']),
            dtype=self.qgm.dtype,
        )
        if self.advect_tracer:
            adX0 = adSSH0[jnp.newaxis,:,:]
            X0 = SSH0[jnp.newaxis,:,:]
            # Ageostrophic velocities
            if self.ageo_velocities:
                U = State.getvar(name_var=self.name_var['U'])[np.newaxis,:,:]
                V = State.getvar(name_var=self.name_var['V'])[np.newaxis,:,:]
                adU = adState.getvar(name_var=self.name_var['U'])[np.newaxis,:,:]
                adV = adState.getvar(name_var=self.name_var['V'])[np.newaxis,:,:]
                X0 = np.append(X0, U, axis=0)
                X0 = np.append(X0, V, axis=0)
                adX0 = np.append(adX0, adU, axis=0)
                adX0 = np.append(adX0, adV, axis=0)
            # Tracers
            for name in self.name_var:
                if name not in ['SSH', 'U', 'V']:
                    adC0 = jnp.asarray(
                        adState.getvar(name_var=self.name_var[name]),
                        dtype=self.qgm.dtype,
                    )[jnp.newaxis,:,:]
                    adX0 = jnp.append(adX0, adC0, axis=0)
                    C0 = jnp.asarray(
                        State.getvar(name_var=self.name_var[name]),
                        dtype=self.qgm.dtype,
                    )[jnp.newaxis,:,:]
                    X0 = jnp.append(X0, C0, axis=0)
        else:
            adX0 = adSSH0
            X0 = SSH0

        # Init
        X1 = +X0

        # Time propagation — differentiate jointly w.r.t. (state, c_eff).
        # Using a single vjp call with adX0 (the *incoming* adjoint) is
        # correct: it avoids the bug of passing the already-propagated adjoint
        # to the c-gradient vjp.
        if 'c' in self.name_params:
            _, vjp_fun = jax.vjp(
                lambda X, c: self.qgm.step_jit(X, Xb, nstep=nstep, c=c),
                X1, c_eff)
            adX1, adc = vjp_fun(adX0)
            adState.params['c'] = adState.params['c'] + adc
        else:
            adX1 = self.qgm_step_adj(adX0, X1, Xb, nstep=nstep)

        # Update state and parameters
        if self.name_var['SSH'] in adState.params:
            for i,name in enumerate(self.name_var):
                adparams = nstep*self.dt/(3600*24) *\
                    adState.getvar(name_var=self.name_var[name])#.astype('float64') 
                if name!='SSH':
                    adparams *= (1-self.Wbc_device)
                    if self.forcing_tracer_from_bc:
                        Fc = State.params[self.name_var[name]] 
                        adparams *=  (Xb[i] - X0[i])
                        adX1 = adX1.at[i].add(
                            -nstep*self.dt/(3600*24)
                            * (1-self.Wbc_device) * Fc * adX0[i]
                        )
                adState.params[self.name_var[name]] = (
                    adState.params[self.name_var[name]] + adparams
                )
                
        if self.advect_tracer:
            adState.setvar(adX1[0],self.name_var['SSH'])
            for i,name in enumerate(self.name_var):
                if name!='SSH':
                    adState.setvar(adX1[i],self.name_var[name])
        else:
            adState.setvar(adX1,self.name_var['SSH'])


###############################################################################
#                         Shallow Water Models                                #
###############################################################################

class Model_csw1l(_ITOpenBoundaryExtensionMixin, M):

    def __init__(self,config,State):

        super().__init__(config,State)

        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

        self.config = config

        # Model specific libraries
        if config.MOD.dir_model is None:
            dir_model = os.path.realpath(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..','models','model_sw1l'))
        else:
            dir_model = config.MOD.dir_model
        
        swm = SourceFileLoader("swm", 
                                dir_model + "/jswm.py").load_module()
        model = swm.CSWm
        
        self.time_scheme = config.MOD.time_scheme

        # grid
        self.ny = State.ny
        self.nx = State.nx
        X = State.X
        Y = State.Y
        self.DX = State.DX
        self.DY = State.DY
        
        # Coriolis 
        self.f = State.f
        self.f_on_u = (self.f[:, :-1] + self.f[:, 1:]) / 2
        self.f_on_v = (self.f[:-1, :] + self.f[1:, :]) / 2

        # Gravity
        self.g = State.g

        # Open MDT map if provided
        if config.MOD.path_mdt is not None and os.path.exists(config.MOD.path_mdt):
                      
            ds = xr.open_dataset(config.MOD.path_mdt)
            name_lon = config.MOD.name_var_mdt['lon']
            lon = ds[name_lon]
            # Convert longitude 
            if np.sign(lon.data.min())==-1 and State.lon_unit=='0_360':
                ds = ds.assign_coords({name_lon:((name_lon, lon.data % 360))})
                ds = ds.sortby(name_lon)    
            elif np.sign(lon.data.min())>=0 and State.lon_unit=='-180_180':
                ds = ds.assign_coords({name_lon:((name_lon, (lon.data + 180) % 360 - 180))})
                ds = ds.sortby(name_lon)    

            self.mdt = grid.interp2d(ds,
                                   config.MOD.name_var_mdt,
                                   State.lon,
                                   State.lat)
            self.mdt[np.isnan(self.mdt)] = 0
        
            if config.EXP.flag_plot>0:
                plt.figure()
                plt.pcolormesh(self.mdt)
                plt.colorbar()
                plt.title('MDT')
                plt.show()

        else:
            print('Warning: No MDT file provided. MDT set to zero. It might affect coupling terms computation.')
            self.mdt = np.zeros((State.ny,State.nx))

        # Open Rossby Radius if provided
        if config.MOD.filec_aux is not None and os.path.exists(config.MOD.filec_aux):

            ds = xr.open_dataset(config.MOD.filec_aux)
            name_lon = config.MOD.name_var_c['lon']
            lon = ds[name_lon]
            # Convert longitude 
            if np.sign(lon.data.min())==-1 and State.lon_unit=='0_360':
                ds = ds.assign_coords({name_lon:((name_lon, lon.data % 360))})
                ds = ds.sortby(name_lon)    
            elif np.sign(lon.data.min())>=0 and State.lon_unit=='-180_180':
                ds = ds.assign_coords({name_lon:((name_lon, (lon.data + 180) % 360 - 180))})
                ds = ds.sortby(name_lon)    

            self.c = grid.interp2d(ds,
                                   config.MOD.name_var_c,
                                   State.lon,
                                   State.lat)
            
            if config.MOD.cmin is not None:
                self.c[self.c<config.MOD.cmin] = config.MOD.cmin
            
            if config.MOD.cmax is not None:
                self.c[self.c>config.MOD.cmax] = config.MOD.cmax
            
            if config.EXP.flag_plot>0:
                plt.figure()
                plt.pcolormesh(self.c)
                plt.colorbar()
                plt.title('Rossby phase velocity')
                plt.show()
                
        else:
            self.c = config.MOD.c0 * np.ones((State.ny,State.nx))
            
        # Equivalent depth
        self.Heb = self.c**2 / self.g
        if getattr(config.MOD, 'constant_He', False):
            He0 = np.nanmean(self.c)**2 / self.g
            self.Heb = He0 * np.ones((State.ny, State.nx))
            print(f'He set to constant: He = {He0:.4f} m')

        # Open Bathymetryc if provided
        if config.MOD.file_H_aux is not None and os.path.exists(config.MOD.file_H_aux):

            ds = xr.open_dataset(config.MOD.file_H_aux)
            name_lon = config.MOD.name_var_H['lon']
            lon = ds[name_lon]
            # Convert longitude 
            if np.sign(lon.data.min())==-1 and State.lon_unit=='0_360':
                ds = ds.assign_coords({name_lon:((name_lon, lon.data % 360))})
                ds = ds.sortby(name_lon)    
            elif np.sign(lon.data.min())>=0 and State.lon_unit=='-180_180':
                ds = ds.assign_coords({name_lon:((name_lon, (lon.data + 180) % 360 - 180))})
                ds = ds.sortby(name_lon)    

            self.H = grid.interp2d(ds,
                                   config.MOD.name_var_H,
                                   State.lon,
                                   State.lat)
            
            if config.EXP.flag_plot>0:
                plt.figure()
                plt.pcolormesh(self.H)
                plt.colorbar()
                plt.title('Bathymetry')
                plt.show()

        else:
            self.H = config.MOD.H 
        
        # Boundary angles
        _ntheta = config.MOD.Ntheta
        if _ntheta == -1:
            # Auto: boundary tangential Nyquist criterion.
            # A wave entering at angle theta has along-boundary wavenumber k_t = k*sin(theta).
            # At grazing incidence k_t_max = k_max = omega_max/c_min.  With 2*Ntheta+1 angular
            # samples covering [-k_max, k_max], the Nyquist condition on a boundary of length L
            # gives:  Ntheta >= k_max * L / (2*pi) = L / lambda_min
            # where lambda_min = 2*pi*c_min / omega_max and L = max boundary length.
            _omegas = np.asarray(config.MOD.w_waves)
            _He_bdy = np.concatenate([
                self.Heb[0,:], self.Heb[-1,:], self.Heb[:,0], self.Heb[:,-1]])
            _c_min = np.sqrt(self.g * np.nanmin(_He_bdy[_He_bdy > 0]))
            _lambda_min = 2*pi * _c_min / np.max(np.abs(_omegas))
            _L_bdy = max(
                float(np.asarray(State.X).max() - np.asarray(State.X).min()),
                float(np.asarray(State.Y).max() - np.asarray(State.Y).min()))
            _ntheta = int(np.ceil(_L_bdy / _lambda_min))
        if _ntheta > 0:
            theta_p = np.arange(0, pi/2 + pi/2/_ntheta, pi/2/_ntheta)
            self.bc_theta = np.append(theta_p - pi/2, theta_p[1:])
        else:
            self.bc_theta = np.array([0])
        
        # Tide frequencies
        self.omegas = np.asarray(config.MOD.w_waves)
        
        # CFL
        if config.MOD.cfl is not None:
            grid_spacing = min(np.nanmean(State.DX), np.nanmean(State.DY)) 
            dt = config.MOD.cfl * grid_spacing / np.nanmax(self.c)
            divisors = [i for i in range(1, 3600 + 1) if 3600 % i == 0]  # Find all divisors of one hour in seconds
            lower_divisors = [d for d in divisors if d <= dt]
            self.dt = max(lower_divisors)  # Get closest
            print('CFL condition for max(c)=', np.nanmax(self.c), 'm/s and grid spacing ~', grid_spacing, 'm: dt<=', dt, 's')
            print('Model time-step', self.dt)
            # Time parameters
            if self.dt>0:
                self.nt = 1 + int((config.EXP.final_date - config.EXP.init_date).total_seconds()//self.dt)
                self.timestamps = [] 
                t = config.EXP.init_date
                while t<=config.EXP.final_date:
                    self.timestamps.append(t)
                    t += timedelta(seconds=self.dt)
                self.timestamps = np.asarray(self.timestamps)
            else:
                self.nt = 1
                self.timestamps = np.array([config.EXP.init_date])
            self.T = np.arange(self.nt) * self.dt

        # Initialize model state
        self.name_var = config.MOD.name_var
        self.var_to_save = [self.name_var['SSH']] # ssh

        if (config.GRID.super == 'GRID_FROM_FILE') and (config.MOD.name_init_var is not None):
            dsin = grid.open_grid_restart(config)
            for name in self.name_var:
                if name in config.MOD.name_init_var:
                    var_init = dsin[config.MOD.name_init_var[name]]
                    if len(var_init.shape)==3:
                        var_init = var_init[0,:,:]
                    if config.GRID.subsampling is not None:
                        var_init = var_init[::config.GRID.subsampling,::config.GRID.subsampling]
                    dsin.close()
                    del dsin
                    State.var[self.name_var[name]] = var_init.values
                else:
                    if name=='U':
                        State.var[self.name_var[name]] = np.zeros((State.ny,State.nx-1))
                    elif name=='V':
                        State.var[self.name_var[name]] = np.zeros((State.ny-1,State.nx))
                    elif name=='SSH':
                        State.var[self.name_var[name]] = np.zeros((State.ny,State.nx))
        else:
            for name in self.name_var:  
                if name=='U':
                    State.var[self.name_var[name]] = np.zeros((State.ny,State.nx-1))
                elif name=='V':
                    State.var[self.name_var[name]] = np.zeros((State.ny-1,State.nx))
                elif name=='SSH':
                    State.var[self.name_var[name]] = np.zeros((State.ny,State.nx))

        
        # Model Parameters (sponge, He and alpha coefficient)
        self.name_params = config.MOD.name_params
        if 'He_mean' in self.name_params:
            State.params['He_mean'] = np.zeros((self.ny, self.nx))
        if 'alpha' in self.name_params:
            State.params['alpha'] = np.zeros((self.ny, self.nx))
        if 'alpha_He' in self.name_params:
            State.params['alpha_He'] = np.zeros((self.ny, self.nx))
        if 'alpha_Uu' in self.name_params: 
            State.params['alpha_Uu'] = np.zeros((self.ny, self.nx))
        if 'alpha_Uz' in self.name_params: 
            State.params['alpha_Uz'] = np.zeros((self.ny, self.nx))
        if 'alpha_Up' in self.name_params: 
            State.params['alpha_Up'] = np.zeros((self.ny, self.nx))
        if 'hbc' in self.name_params:
            self.shapehbcx = [len(self.omegas), # tide frequencies
                              2, # North/South
                              2, # cos/sin
                              len(self.bc_theta), # Angles
                              State.nx # NX
                              ]
            self.shapehbcy = [len(self.omegas), # tide frequencies
                              2, # North/South
                              2, # cos/sin
                              len(self.bc_theta), # Angles
                              State.ny # NY
                              ]
            State.params['hbcx'] = np.zeros((self.shapehbcx))
            State.params['hbcy'] = np.zeros((self.shapehbcy))
        

        # Coupling terms from vertical modes
        self.flag_coupling_from_bm = config.MOD.flag_coupling_from_bm
        if self.flag_coupling_from_bm:
            if config.MOD.path_interaction_terms is not None and os.path.exists(config.MOD.path_interaction_terms):
                self._read_interaction_terms(config, State)
            else:
                self._compute_coupling_coefficients(config.MOD.path_vertical_modes)
        if self.flag_coupling_from_bm and config.MOD.path_bm is not None and os.path.exists(config.MOD.path_bm):
            self.is_bm = True
            self._compute_bm_fields(config.MOD.path_bm, config.MOD.name_var_bm, State)
        else:
            self.is_bm = False

        # Sponge layer
        self.sponge_width = config.MOD.dist_sponge_bc
        self.sponge_coef = config.MOD.sponge_coef
        self.flag_bc_sponge = config.MOD.flag_bc_sponge and self.sponge_width is not None and self.sponge_width > 0
        if self.flag_bc_sponge:
            self.Xh = State.X
            self.Yh = State.Y
            self.Xu = 0.5 * (State.X[:, :-1] + State.X[:, 1:])
            self.Yu = 0.5 * (State.Y[:, :-1] + State.Y[:, 1:])
            self.Xv = 0.5 * (State.X[:-1, :] + State.X[1:, :])
            self.Yv = 0.5 * (State.Y[:-1, :] + State.Y[1:, :])

            # Sponge profile via grid.compute_sponge_components
            lon_h = State.lon
            lat_h = State.lat
            lon_u = 0.5 * (lon_h[:, :-1] + lon_h[:, 1:])
            lat_u = 0.5 * (lat_h[:, :-1] + lat_h[:, 1:])
            lon_v = 0.5 * (lon_h[:-1, :] + lon_h[1:, :])
            lat_v = 0.5 * (lat_h[:-1, :] + lat_h[1:, :])

            if config.MOD.use_sponge_on_coast:
                mask_h = State.mask.copy()
                mask_u = mask_h[:, :-1] & mask_h[:, 1:]
                mask_v = mask_h[:-1, :] & mask_h[1:, :]
            else:
                mask_h = np.zeros_like(lon_h, dtype=bool)
                mask_u = np.zeros_like(lon_u, dtype=bool)
                mask_v = np.zeros_like(lon_v, dtype=bool)

            alpha = config.MOD.tangential_sponge_factor

            # h: isotropic sponge
            wc_h, wNS_h, wWE_h = grid.compute_sponge_components(lon_h, lat_h, mask_h, config.MOD.dist_sponge_bc)
            if config.MOD.periodic_y:
                wNS_h[:] = 0.0
            if config.MOD.periodic_x:
                wWE_h[:] = 0.0
            self.sponge_h = 1.0 - (1.0 - wc_h) * (1.0 - wNS_h) * (1.0 - wWE_h)

            # u: normal at W/E (full), tangential at N/S (reduced)
            wc_u, wNS_u, wWE_u = grid.compute_sponge_components(lon_u, lat_u, mask_u, config.MOD.dist_sponge_bc)
            if config.MOD.periodic_y:
                wNS_u[:] = 0.0
            if config.MOD.periodic_x:
                wWE_u[:] = 0.0
            self.sponge_u = 1.0 - (1.0 - wc_u) * (1.0 - alpha * wNS_u) * (1.0 - wWE_u)

            # v: normal at N/S (full), tangential at W/E (reduced)
            wc_v, wNS_v, wWE_v = grid.compute_sponge_components(lon_v, lat_v, mask_v, config.MOD.dist_sponge_bc)
            if config.MOD.periodic_y:
                wNS_v[:] = 0.0
            if config.MOD.periodic_x:
                wWE_v[:] = 0.0
            self.sponge_v = 1.0 - (1.0 - wc_v) * (1.0 - wNS_v) * (1.0 - alpha * wWE_v)

            if config.EXP.flag_plot > 0:
                _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
                im1 = ax1.pcolormesh(self.sponge_h)
                plt.colorbar(im1, ax=ax1)
                ax1.set_title('Sponge SSH')
                im2 = ax2.pcolormesh(self.sponge_u)
                plt.colorbar(im2, ax=ax2)
                ax2.set_title('Sponge U')
                im3 = ax3.pcolormesh(self.sponge_v)
                plt.colorbar(im3, ax=ax3)
                ax3.set_title('Sponge V')
                plt.show()

            if not config.MOD.periodic_y:
                self.sponge_on_h_S = (np.abs(self.Yh-self.Yh[0,:][None,:])<=config.MOD.dist_sponge_bc*1e3) 
                self.sponge_on_u_S = (np.abs(self.Yu-self.Yu[0,:][None,:])<=config.MOD.dist_sponge_bc*1e3)
                self.sponge_on_v_S = (np.abs(self.Yv-self.Yv[0,:][None,:])<=config.MOD.dist_sponge_bc*1e3)
                self.sponge_on_h_N = (np.abs(self.Yh-self.Yh[-1,:][None,:])<=config.MOD.dist_sponge_bc*1e3) 
                self.sponge_on_u_N = (np.abs(self.Yu-self.Yu[-1,:][None,:])<=config.MOD.dist_sponge_bc*1e3) 
                self.sponge_on_v_N = (np.abs(self.Yv-self.Yv[-1,:][None,:])<=config.MOD.dist_sponge_bc*1e3) 
            else:
                self.sponge_on_h_S = np.zeros_like(self.Yh, dtype=bool)
                self.sponge_on_u_S = np.zeros_like(self.Yu, dtype=bool)
                self.sponge_on_v_S = np.zeros_like(self.Yv, dtype=bool)
                self.sponge_on_h_N = np.zeros_like(self.Yh, dtype=bool)
                self.sponge_on_u_N = np.zeros_like(self.Yu, dtype=bool)
                self.sponge_on_v_N = np.zeros_like(self.Yv, dtype=bool)
            if not config.MOD.periodic_x:
                self.sponge_on_h_W = (np.abs(self.Xh-self.Xh[:,0][:,None])<=config.MOD.dist_sponge_bc*1e3) 
                self.sponge_on_u_W = (np.abs(self.Xu-self.Xu[:,0][:,None])<=config.MOD.dist_sponge_bc*1e3)
                self.sponge_on_v_W = (np.abs(self.Xv-self.Xv[:,0][:,None])<=config.MOD.dist_sponge_bc*1e3)
                self.sponge_on_h_E = (np.abs(self.Xh-self.Xh[:,-1][:,None])<=config.MOD.dist_sponge_bc*1e3) 
                self.sponge_on_u_E = (np.abs(self.Xu-self.Xu[:,-1][:,None])<=config.MOD.dist_sponge_bc*1e3)
                self.sponge_on_v_E = (np.abs(self.Xv-self.Xv[:,-1][:,None])<=config.MOD.dist_sponge_bc*1e3)
            else:
                self.sponge_on_h_W = np.zeros_like(self.Yh, dtype=bool)
                self.sponge_on_u_W = np.zeros_like(self.Yu, dtype=bool)
                self.sponge_on_v_W = np.zeros_like(self.Yv, dtype=bool)
                self.sponge_on_h_E = np.zeros_like(self.Yh, dtype=bool)
                self.sponge_on_u_E = np.zeros_like(self.Yu, dtype=bool)
                self.sponge_on_v_E = np.zeros_like(self.Yv, dtype=bool)
            
            self.weight_sponge_u = np.array(self.sponge_on_u_S, dtype=float) + np.array(self.sponge_on_u_N, dtype=float) + np.array(self.sponge_on_u_W, dtype=float) + np.array(self.sponge_on_u_E, dtype=float)
            self.weight_sponge_v = np.array(self.sponge_on_v_S, dtype=float) + np.array(self.sponge_on_v_N, dtype=float) + np.array(self.sponge_on_v_W, dtype=float) + np.array(self.sponge_on_v_E, dtype=float)
            self.weight_sponge_h = np.array(self.sponge_on_h_S, dtype=float) + np.array(self.sponge_on_h_N, dtype=float) + np.array(self.sponge_on_h_W, dtype=float) + np.array(self.sponge_on_h_E, dtype=float)
            self.weight_sponge_u[self.weight_sponge_u==0] = 1
            self.weight_sponge_v[self.weight_sponge_v==0] = 1
            self.weight_sponge_h[self.weight_sponge_h==0] = 1

            # Keep sponge exclusion separate from State.mask so model and basis
            # still include sponge cells.
            if config.MOD.mask_sponge_bc:
                sponge_mask = (self.sponge_on_h_S | self.sponge_on_h_N |
                               self.sponge_on_h_W | self.sponge_on_h_E)
                if getattr(State, 'sponge_mask', None) is None:
                    State.sponge_mask = sponge_mask.copy()
                else:
                    State.sponge_mask = np.asarray(State.sponge_mask, dtype=bool) | sponge_mask
        
        self.mask = State.mask
        self._init_it_open_boundary_extension(config.MOD)
        self.H_it_open_boundary = self._extend_it_open_boundary_static_field(self.H)

        # Model initialization
        self.swm = model(X=X,
                         Y=Y,
                         dt=self.dt,
                         omegas=self.omegas,
                         bc_theta=self.bc_theta,
                         f=self.f,
                         Heb=self.Heb,
                         periodic_x=config.MOD.periodic_x,
                         periodic_y=config.MOD.periodic_y,
                         time_scheme=self.time_scheme,
                         )
    
        # Set sponge attributes on SW model for compute_IT_2D
        if self.flag_bc_sponge:
            self.swm.flag_sponge_bc = True
            self.swm.sponge_coef = self.sponge_coef
            self.swm.sponge_u = self.sponge_u
            self.swm.sponge_v = self.sponge_v
            self.swm.sponge_h = self.sponge_h
            self.swm.sponge_on_h_S = self.sponge_on_h_S
            self.swm.sponge_on_h_N = self.sponge_on_h_N
            self.swm.sponge_on_h_W = self.sponge_on_h_W
            self.swm.sponge_on_h_E = self.sponge_on_h_E
            self.swm.sponge_on_u_S = self.sponge_on_u_S
            self.swm.sponge_on_u_N = self.sponge_on_u_N
            self.swm.sponge_on_u_W = self.sponge_on_u_W
            self.swm.sponge_on_u_E = self.sponge_on_u_E
            self.swm.sponge_on_v_S = self.sponge_on_v_S
            self.swm.sponge_on_v_N = self.sponge_on_v_N
            self.swm.sponge_on_v_W = self.sponge_on_v_W
            self.swm.sponge_on_v_E = self.sponge_on_v_E
            self.swm.weight_sponge_u = self.weight_sponge_u
            self.swm.weight_sponge_v = self.weight_sponge_v
            self.swm.weight_sponge_h = self.weight_sponge_h

        # IT wave-phase method for sponge BC
        self.swm.bc_it_method = getattr(config.MOD, 'bc_it_method', 'plane_wave_bdy')
        self.swm.bc_it_corner_weight_power = getattr(config.MOD, 'bc_it_corner_weight_power', self.swm.bc_it_corner_weight_power)

        # Compile jax-related functions
        self._jstep_jit = jit(self._jstep, static_argnames=['nstep'])
        self._jstep_tgl_jit = jit(self._jstep_tgl, static_argnames=['nstep'])
        self._jstep_adj_jit = jit(self._jstep_adj, static_argnames=['nstep'])
        self._compute_advective_terms_from_bm_jit = jit(self._compute_advective_terms_from_bm)
        self._compute_He_from_bm_jit = jit(self._compute_He_from_bm)

        # Functions related to time_scheme
        if self.time_scheme=='Euler':
            self.swm_step = self.swm.step_euler_jit
            self.swm_step_nstep = self.swm._step_euler_nstep
            self.swm_step_tgl = self.swm.step_euler_tgl_jit
            self.swm_step_adj = self.swm.step_euler_adj_jit
        elif self.time_scheme=='rk3':
            self.swm_step = self.swm.step_rk3_jit
            self.swm_step_nstep = self.swm._step_rk3_nstep
            self.swm_step_tgl = self.swm.step_rk3_tgl_jit
            self.swm_step_adj = self.swm.step_rk3_adj_jit
        elif self.time_scheme=='rk4':
            self.swm_step = self.swm.step_rk4_jit
            self.swm_step_nstep = self.swm._step_rk4_nstep
            self.swm_step_tgl = self.swm.step_rk4_tgl_jit
            self.swm_step_adj = self.swm.step_rk4_adj_jit
        else:
            raise ValueError(f"Unsupported time_scheme: {self.time_scheme}")
        
        if config.INV is not None and config.INV.super=='INV_4DVAR' and config.INV.compute_test:
            print('CSW1L Tangent test:')
            #tangent_test(self,State,nstep=10)
            print('CSW1L Adjoint test:')
            #adjoint_test(self,State,nstep=1)

    def save_output(self,State,present_date,name_var=None,t=None):

        name_var_to_save = [self.name_var['SSH'], 
                            self.name_var['U']+'_interp', 
                            self.name_var['V']+'_interp',
                            'He']
        
        u = +State.getvar(name_var=self.name_var['U'])
        v = +State.getvar(name_var=self.name_var['V'])
        u_to_save = np.zeros((State.ny,State.nx))
        v_to_save = np.zeros((State.ny,State.nx))
        u_to_save[:,1:-1] = (u[:,1:] + u[:,:-1]) * .5
        v_to_save[1:-1,:] = (v[1:,:] + v[:-1,:]) * .5
        State.var[self.name_var['U']+'_interp'] = u_to_save
        State.var[self.name_var['V']+'_interp'] = v_to_save

        # Get BM field
        if self.is_bm:
            h_bm = self.ssh_bm_data[t]
        else:
            h_bm = None
        
        # Get parameters
        if 'He_mean' in self.name_params:
            He_mean = +State.params['He_mean'] 
        else:
            He_mean = jnp.zeros((self.ny,self.nx))
        if 'alpha' in self.name_params:
            alpha = +State.params['alpha'] 
        else:
            alpha = jnp.zeros((self.ny,self.nx))
        
        He2d = self._compute_He_from_bm(He_mean, alpha, h_bm)

        State.var['He'] = He2d + self.Heb

        if self.flag_coupling_from_bm:
            if 'He_mean' in self.name_params:
                State.var['He_mean'] = State.params['He_mean']
                name_var_to_save += ['He_mean']
            if self.is_bm:
                State.var['ssh_bm'] = h_bm
                name_var_to_save += ['ssh_bm']
            if 'alpha' in self.name_params:
                State.var['alpha'] = State.params['alpha']
                name_var_to_save += ['alpha']
            if 'alpha_He' in self.name_params:  
                State.var['alpha_He'] = State.params['alpha_He']
                name_var_to_save += ['alpha_He']
            if 'alpha_Uu' in self.name_params: 
                State.var['alpha_Uu'] = State.params['alpha_Uu']
                name_var_to_save += ['alpha_Uu']
            if 'alpha_Up' in self.name_params: 
                State.var['alpha_Up'] = State.params['alpha_Up']
                name_var_to_save += ['alpha_Up']
            if 'alpha_Uz' in self.name_params: 
                State.var['alpha_Uz'] = State.params['alpha_Uz']
                name_var_to_save += ['alpha_Uz']

        State.save_output(present_date,
                          name_var=name_var_to_save)

    def _compute_coupling_coefficients(self, path_vertical_modes):

        # Open dataset with vertical mode structures
        ds = xr.open_dataset(path_vertical_modes)
        phi1  = ds.phi.sel(mode=1)      # (s_rho, y_rho)
        phi1 = phi1/phi1[-1]
        phi1p = ds.dphidz.sel(mode=1)   # (s_w,  y_rho)  → given derivative!
        c1    = ds.c.sel(mode=1)        # (y_rho)
        N2    = ds.N2                   # (y_rho, s_w)
        z_r   = ds.z_r                  # (s_rho, y_rho)
        z_w = ds.z_w
        dz_r  = ds.dz                   # (s_rho, y_rho)
        s_w = ds.s_w

        #########################################################################
        # Compute U11 coupling term for mode 1
        #########################################################################
        integrand = phi1/phi1[-1] * (phi1**2) * dz_r  # multiply by layer thickness
        U11u = (1/self.H) * integrand.sum(dim="s_rho")   # integral in z
        self.U11u = jnp.array(U11u.values)  
        plt.figure()
        plt.pcolormesh(self.U11u)
        plt.colorbar()
        plt.title('U11u')
        plt.show()

        #########################################################################
        # Compute U11p coupling term for mode 1
        #########################################################################
        φ = -(c1**2 / N2)* phi1p          # (s_w, y_rho)
        phi1_w = phi1.interp(s_rho=s_w)   # now on (s_w, y_rho)
        phi1_w[0] = phi1_w[1]
        phi1_w[-1] = phi1_w[-2]

        integrand = (
            phi1_w/phi1_w[-1]
            * (N2 / c1**2)
            * φ**2
        )   # dims: (s_w, y_rho)
        # thickness between w-levels: same size as s_w except top/bottom.
        dz_w = z_w.diff("s_w")                   # (s_w_minus1, y_rho)
        # pad to same length as s_w (xarray aligns automatically)
        dz_w = dz_w.reindex(s_w=s_w, fill_value=np.nan)
        U11p = (1/self.H) * (integrand * dz_w).sum(dim="s_w")   # integral in z
        self.U11p = jnp.array(U11p.values)
        plt.figure()
        plt.pcolormesh(self.U11p)
        plt.colorbar()
        plt.title('U11p')
        plt.show()

        #########################################################################
        # Compute U11z coupling term for mode 1
        #########################################################################
        self.U11z = -self.U11p
        plt.figure()
        plt.pcolormesh(self.U11z)
        plt.colorbar()
        plt.title('U11z')
        plt.show()

        #########################################################################
        # Compute dHe coupling term for mode 1
        #########################################################################
        # forward and backward slopes
        slope_f = (phi1.shift(s_rho=-1) - phi1) / (z_r.shift(s_rho=-1) - z_r)
        slope_b = (phi1 - phi1.shift(s_rho=1)) / (z_r - z_r.shift(s_rho=1))

        # central 2nd derivative
        phi1pp = 2 * (slope_f - slope_b) / (z_r.shift(s_rho=-1) - z_r.shift(s_rho=1))

        # fill top/bottom NaNs (simple, safe)
        phi1pp = phi1pp.fillna(0)

        # vertical interpolation from s_w → s_rho
        phi1p_rho = phi1p.interp(s_w=ds.s_rho)

        N2_rho = N2.interp(s_w=ds.s_rho)

        c1_rho = c1.broadcast_like(phi1p_rho)

        A = - (c1_rho**2 / N2_rho) * phi1p_rho
        A2 = A**2

        integrand = phi1pp * A2
        dHe = (1/self.H) * (integrand * dz_r).sum("s_rho")
        self.dHe = jnp.array(dHe.values) 

        plt.figure()
        plt.pcolormesh(self.dHe)
        plt.colorbar()
        plt.title('dHe')
        plt.show()

    def _read_interaction_terms(self, config, State):
        """Read interaction terms (U11u, U11p, U11z) from a netcdf file
        instead of computing them from vertical modes."""

        ds = xr.open_dataset(config.MOD.path_interaction_terms)
        name_lon = config.MOD.name_var_interaction_terms['lon']
        lon = ds[name_lon]
        # Convert longitude
        if np.sign(lon.data.min()) == -1 and State.lon_unit == '0_360':
            ds = ds.assign_coords({name_lon: ((name_lon, lon.data % 360))})
            ds = ds.sortby(name_lon)
        elif np.sign(lon.data.min()) >= 0 and State.lon_unit == '-180_180':
            ds = ds.assign_coords({name_lon: ((name_lon, (lon.data + 180) % 360 - 180))})
            ds = ds.sortby(name_lon)

        name_vars = config.MOD.name_var_interaction_terms

        def _read_field(var_key, ds, name_vars, State, divisor):
            """Read and interpolate a field if variable name is defined and exists in the dataset."""
            if name_vars.get(var_key) and name_vars[var_key] in ds:
                return jnp.array(grid.interp2d(
                    ds, {'lon': name_vars['lon'], 'lat': name_vars['lat'], 'var': name_vars[var_key]},
                    State.lon, State.lat)) / divisor
            else:
                print(f'Warning: {var_key} not found or not defined. Setting to zero.')
                return jnp.zeros((State.ny, State.nx))

        # Read U11u
        self.U11u = _read_field('U11_u', ds, name_vars, State, self.H)

        # Read U11p
        self.U11p = _read_field('U11_p', ds, name_vars, State, self.H)

        # Read U11z
        self.U11z = _read_field('U11_z', ds, name_vars, State, self.H)

        # Read dc2 (correction to c^2) and convert to dHe = dc2 / g
        self.dHe = _read_field('dc2', ds, name_vars, State, self.H * self.g)

        if config.EXP.flag_plot > 0:
            fig, axes = plt.subplots(1, 4, figsize=(20, 4))
            for ax, data, title in zip(axes,
                                       [self.U11u, self.U11p, self.U11z, self.dHe],
                                       ['U11u', 'U11p', 'U11z', 'dHe']):
                im = ax.pcolormesh(data)
                plt.colorbar(im, ax=ax)
                ax.set_title(title)
            plt.suptitle('Interaction terms (from file)')
            plt.show()

    def _compute_bm_fields(self, path_bm, name_var_bm, State):

        dsbm = xr.open_mfdataset(path_bm, chunks=-1)
        name_lon_bm = name_var_bm['lon']
        name_lat_bm = name_var_bm['lat']

        # 0) Select time range covering self.timestamps (with 1-step margin)
        time_bm = dsbm['time'].values
        t_min = np.datetime64(self.timestamps.min())
        t_max = np.datetime64(self.timestamps.max())
        idx = np.where((time_bm >= t_min) & (time_bm <= t_max))[0]
        if len(idx) > 0:
            i0 = max(idx[0] - 1, 0)
            i1 = min(idx[-1] + 1, len(time_bm) - 1)
            dsbm = dsbm.isel(time=slice(i0, i1 + 1))

        # 1) Spatial interpolation onto model grid
        lon_bm = dsbm[name_lon_bm]
        if len(lon_bm.shape) == 1:
            # 1D coordinates: select subdomain then use xarray interp (fast)
            margin = 0.5
            lon_min, lon_max = float(State.lon.min()) - margin, float(State.lon.max()) + margin
            lat_min, lat_max = float(State.lat.min()) - margin, float(State.lat.max()) + margin
            dsbm = dsbm.sel({name_lon_bm: slice(lon_min, lon_max),
                             name_lat_bm: slice(lat_min, lat_max)})
            dsbm = dsbm.interp({name_lon_bm: State.lon[0, :],
                                 name_lat_bm: State.lat[:, 0]},
                                method='linear')
            h_all = dsbm[name_var_bm['ssh_bm']].values  # (nt_bm, ny, nx)
        else:
            # 2D coordinates: build the triangulation once, then reuse the
            # target barycentric weights for every BM time slice. This avoids
            # scipy.griddata rebuilding the same Delaunay geometry in a loop.
            from scipy.spatial import Delaunay
            lon_bm_2d = dsbm[name_lon_bm].values
            lat_bm_2d = dsbm[name_lat_bm].values
            if len(lon_bm_2d.shape) == 1:
                lon_bm_2d, lat_bm_2d = np.meshgrid(lon_bm_2d, lat_bm_2d)
            nt_bm = dsbm.sizes['time']
            h_all = np.full((nt_bm, self.ny, self.nx), np.nan)
            valid = np.isfinite(lon_bm_2d) & np.isfinite(lat_bm_2d)
            if valid.any():
                points = np.column_stack((lon_bm_2d[valid], lat_bm_2d[valid]))
                tri = Delaunay(points)
                targets = np.column_stack((State.lon.ravel(), State.lat.ravel()))
                simplex = tri.find_simplex(targets)
                inside = simplex >= 0
                vertices = np.zeros((targets.shape[0], tri.ndim + 1), dtype=int)
                weights = np.zeros((targets.shape[0], tri.ndim + 1), dtype=float)
                if inside.any():
                    transform = tri.transform[simplex[inside]]
                    delta = targets[inside] - transform[:, tri.ndim]
                    bary = np.einsum('nij,nj->ni', transform[:, :tri.ndim], delta)
                    weights[inside, :-1] = bary
                    weights[inside, -1] = 1. - bary.sum(axis=1)
                    vertices[inside] = tri.simplices[simplex[inside]]

                values_all = dsbm[name_var_bm['ssh_bm']].values.reshape(nt_bm, -1)[:, valid.ravel()]
                for it in range(nt_bm):
                    out = np.full(targets.shape[0], np.nan)
                    if inside.any():
                        vals = values_all[it, vertices[inside]]
                        out[inside] = np.sum(vals * weights[inside], axis=1)
                    h_all[it] = out.reshape(self.ny, self.nx)

        # 2) Compute geostrophic velocities on all BM timesteps
        u_all = np.zeros_like(h_all)
        v_all = np.zeros_like(h_all)
        u_all[:, 1:-1, 1:] = - self.g / self.f[None, 1:-1, 1:] * \
            (h_all[:, 2:, :-1] + h_all[:, 2:, 1:] - h_all[:, :-2, 1:] - h_all[:, :-2, :-1]) / (4 * self.DY[None, 1:-1, 1:])
        v_all[:, 1:, 1:-1] = self.g / self.f[None, 1:, 1:-1] * \
            (h_all[:, 1:, 2:] + h_all[:, :-1, 2:] - h_all[:, :-1, :-2] - h_all[:, 1:, :-2]) / (4 * self.DX[None, 1:, 1:-1])

        # 3) Build xarray dataset for time interpolation
        time_bm = dsbm['time'].values
        ds_bm_uv = xr.Dataset({
            'ssh': (['time', 'y', 'x'], h_all),
            'u':   (['time', 'y', 'x'], u_all),
            'v':   (['time', 'y', 'x'], v_all),
        }, coords={'time': time_bm})
        ds_bm_uv = ds_bm_uv.interp(time=self.timestamps, method='linear')

        # 4) Store per-timestep fields
        self.ssh_bm_data = {}
        self.u_bm_data = {}
        self.v_bm_data = {}
        for t, date in zip(self.T, self.timestamps):
            self.ssh_bm_data[t] = ds_bm_uv['ssh'].sel(time=date).values
            self.u_bm_data[t] = ds_bm_uv['u'].sel(time=date).values
            self.v_bm_data[t] = ds_bm_uv['v'].sel(time=date).values
        
        fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(20,5))
        im1 = ax1.pcolormesh(self.ssh_bm_data[0])
        plt.colorbar(im1, ax=ax1)
        ax1.set_title('SSH BM at t=0s')
        im2 = ax2.pcolormesh(self.u_bm_data[0])
        plt.colorbar(im2, ax=ax2)
        ax2.set_title('U BM at t=0s')
        im3 = ax3.pcolormesh(self.v_bm_data[0])
        plt.colorbar(im3, ax=ax3)
        ax3.set_title('V BM at t=0s')
        plt.show()

    def _compute_He_from_bm(self, He, alpha_He, h_bm):
        
        """ Compute equivalent depth with coupling term
        --------------
        Inputs:
        He        : IT equivalent depth control parameters (without units)
        h_bm      : BM sea surface height (m) 
        --------------
        Outputs:
        He_tot    : Total equivalent depth with coupling term (m)
        """

        if He is not None:
            if self.flag_coupling_from_bm and alpha_He is not None and h_bm is not None:
                He2d =  He + (.5 + alpha_He) * self.dHe * (h_bm - self.mdt)
            else:
                He2d = He
        else:
            He2d = jnp.zeros((self.ny,self.nx))
        
        He2d = jnp.where(self.mask, 0., He2d)
        
        return He2d

    def _compute_it_open_boundary_He_total(self, He, alpha_He, h_bm):

        Heb = self._extend_it_open_boundary_field(self.Heb)
        He_bc = self._extend_it_open_boundary_field(He)
        alpha_He_bc = self._extend_it_open_boundary_field(alpha_He)
        return Heb + self._compute_He_from_bm(He_bc, alpha_He_bc, h_bm)

    def _compute_advective_terms_from_bm(self, alpha_Uu, alpha_Up, alpha_Uz, u_bm, v_bm):

        """ Compute advective terms with coupling term
        --------------
        Inputs:
        alpha     : IT advective control parameters (without units)
        h_bm      : BM sea surface height (m) 
        --------------
        Outputs:
        u11, v11  : advective coupling terms for u and v (m/s)
        u11p, v11p: advective coupling terms for u and v (m/s)
        """

        if self.flag_coupling_from_bm and u_bm is not None and v_bm is not None:
            if alpha_Uu is not None:
                u11u = ((.5 - alpha_Uu) + (.5 + alpha_Uu) * self.U11u) * u_bm 
                v11u = ((.5 - alpha_Uu) + (.5 + alpha_Uu) * self.U11u) * v_bm 
            else:
                u11u = None
                v11u = None
            if alpha_Uz is not None:
                u11z = (.5 + alpha_Uz) * self.U11z * u_bm 
                v11z = (.5 + alpha_Uz) * self.U11z * v_bm 
            else:
                u11z = None
                v11z = None
            if alpha_Up is not None:
                u11p = ((.5 - alpha_Up) + (.5 + alpha_Up) * self.U11p) * u_bm 
                v11p = ((.5 - alpha_Up) + (.5 + alpha_Up) * self.U11p) * v_bm 
            else:
                u11p = None
                v11p = None
        else:
            u11u = None
            v11u = None
            u11z = None
            v11z = None
            u11p = None
            v11p = None
        
        return u11u, v11u, u11p, v11p, u11z, v11z

    def _jstep(self, t, u, v, h, He_mean, alpha_He, alpha_Uu, alpha_Up, alpha_Uz, h_SN, h_WE, h_bm, u_bm, v_bm, nstep=1):

        # Compute equivalent depth anomaly from control parameters (once, before scan)
        He2d = self._compute_He_from_bm(He_mean, alpha_He, h_bm)
        He_total = self._compute_it_open_boundary_He_total(He_mean, alpha_He, h_bm)

        # Compute advective terms from control parameters (once, before scan)
        u11u, v11u, u11p, v11p, u11z, v11z = self._compute_advective_terms_from_bm(alpha_Uu, alpha_Up, alpha_Uz, u_bm, v_bm)

        # Multi-step forward integration (scan + checkpoint inside swm)
        u1, v1, h1 = self.swm_step_nstep(
            u, v, h, He2d,
            u11u=u11u, v11u=v11u, u11z=u11z, v11z=v11z, u11p=u11p, v11p=v11p,
            nstep=nstep, t=t, He_total=He_total, h_SN=h_SN, h_WE=h_WE)
    
        return u1, v1, h1

    def step(self,State,nstep=1,t=0):

        # Get state variable
        u = +State.getvar(name_var=self.name_var['U'])
        v = +State.getvar(name_var=self.name_var['V'])
        h = +State.getvar(name_var=self.name_var['SSH'])

        # Get BM field
        if self.is_bm:
            h_bm = self.ssh_bm_data[t]
            u_bm = self.u_bm_data[t]
            v_bm = self.v_bm_data[t]
        else:
            h_bm = None
            u_bm = None
            v_bm = None
        
        # Get parameters
        if 'He_mean' in self.name_params:
            He_mean = +State.params['He_mean'] 
        else:
            He_mean = jnp.zeros((self.ny,self.nx))
        if 'alpha_He' in self.name_params:
            alpha_He = +State.params['alpha_He']
        elif 'alpha' in self.name_params:
            alpha_He = +State.params['alpha']
        else:
            alpha_He = None
        if 'alpha_Uu' in self.name_params:
            alpha_Uu = +State.params['alpha_Uu']
        elif 'alpha' in self.name_params:
            alpha_Uu = +State.params['alpha']
        else:
            alpha_Uu = None
        if 'alpha_Uz' in self.name_params:
            alpha_Uz = +State.params['alpha_Uz']
        elif 'alpha' in self.name_params:
            alpha_Uz = +State.params['alpha']
        else:
            alpha_Uz = None
        if 'alpha_Up' in self.name_params:
            alpha_Up = +State.params['alpha_Up']
        elif 'alpha' in self.name_params:
            alpha_Up = +State.params['alpha']
        else:
            alpha_Up = None
        if 'hbc' in self.name_params:
            h_SN = +State.params['hbcx']
            h_WE = +State.params['hbcy']
        else:
            h_SN = None
            h_WE = None
            
        # Time stepping (scan + checkpoint inside swm model)
        u, v, h = self._jstep_jit(t, u, v, h, He_mean, alpha_He, alpha_Uu, alpha_Up, alpha_Uz, h_SN, h_WE, h_bm, u_bm, v_bm, nstep=nstep)
        
        # Update state
        State.setvar([u,v,h],[
            self.name_var['U'],
            self.name_var['V'],
            self.name_var['SSH']])
        
    def step_tgl(self,dState,State,nstep=1,t=0):
        
        # Get state variable
        du = dState.getvar(name_var=self.name_var['U'])
        dv = dState.getvar(name_var=self.name_var['V'])
        dh = dState.getvar(name_var=self.name_var['SSH'])
        u = State.getvar(name_var=self.name_var['U'])
        v = State.getvar(name_var=self.name_var['V'])
        h = State.getvar(name_var=self.name_var['SSH'])

        # Get BM field
        if self.is_bm:
            h_bm = self.ssh_bm_data[t]
            u_bm = self.u_bm_data[t]
            v_bm = self.v_bm_data[t]
        else:
            h_bm = None
            u_bm = None
            v_bm = None
        
        # Get parameters
        if 'He_mean' in self.name_params:
            dHe_mean = +dState.params['He_mean'] 
            He_mean = +State.params['He_mean'] 
        else:
            He_mean = dHe_mean = jnp.zeros((self.ny,self.nx))
        if 'alpha_He' in self.name_params:
            dalpha_He = +dState.params['alpha_He']
            alpha_He = +State.params['alpha_He']
        elif 'alpha' in self.name_params:
            dalpha_He = +dState.params['alpha']
            alpha_He = +State.params['alpha']
        else:
            alpha_He = dalpha_He = None
        if 'alpha_Uu' in self.name_params:
            dalpha_Uu = +dState.params['alpha_Uu']
            alpha_Uu = +State.params['alpha_Uu']
        elif 'alpha' in self.name_params:
            dalpha_Uu = +dState.params['alpha']
            alpha_Uu = +State.params['alpha']
        else:
            alpha_Uu = dalpha_Uu = None
        if 'alpha_Uz' in self.name_params:
            dalpha_Uz = +dState.params['alpha_Uz']
            alpha_Uz = +State.params['alpha_Uz']
        elif 'alpha' in self.name_params:
            dalpha_Uz = +dState.params['alpha']
            alpha_Uz = +State.params['alpha']
        else:
            alpha_Uz = dalpha_Uz = None
        if 'alpha_Up' in self.name_params:
            dalpha_Up = +dState.params['alpha_Up']
            alpha_Up = +State.params['alpha_Up']
        elif 'alpha' in self.name_params:
            dalpha_Up = +dState.params['alpha']
            alpha_Up = +State.params['alpha']
        else:
            alpha_Up = dalpha_Up = None
        if 'hbc' in self.name_params:
            dh_SN = dState.params['hbcx']
            dh_WE = dState.params['hbcy']
            h_SN = State.params['hbcx']
            h_WE = State.params['hbcy']
        else:
            dh_SN = dh_WE = None
            h_SN = h_WE = None
            
        # Time stepping (JVP through _jstep with scan + checkpoint inside swm model)
        du, dv, dh = self._jstep_tgl_jit(
            t, du, dv, dh, dHe_mean, dalpha_He, dalpha_Uu, dalpha_Up, dalpha_Uz, dh_SN, dh_WE,
            u, v, h, He_mean, alpha_He, alpha_Uu, alpha_Up, alpha_Uz, h_SN, h_WE, h_bm, u_bm, v_bm, nstep=nstep)
        
        # Update state
        dState.setvar([du,dv,dh],[
            self.name_var['U'],
            self.name_var['V'],
            self.name_var['SSH']])
        
    def _jstep_tgl(self, t, 
                   du, dv, dh, dHe_mean, dalpha_He, dalpha_Uu, dalpha_Up, dalpha_Uz, dh_SN, dh_WE,
                   u, v, h, He_mean, alpha_He, alpha_Uu, alpha_Up, alpha_Uz, h_SN, h_WE, h_bm, u_bm, v_bm, nstep=1):
        
        def wrapped_jstep(x):
            u, v, h, He_mean, alpha_He, alpha_Uu, alpha_Up, alpha_Uz, h_SN, h_WE = x
            return self._jstep(t, u, v, h, He_mean, alpha_He, alpha_Uu, alpha_Up, alpha_Uz, h_SN, h_WE, h_bm, u_bm, v_bm, nstep=nstep)
        primals = ((u, v, h, He_mean, alpha_He, alpha_Uu, alpha_Up, alpha_Uz, h_SN, h_WE),)
        tangents = ((du, dv, dh, dHe_mean, dalpha_He, dalpha_Uu, dalpha_Up, dalpha_Uz, dh_SN, dh_WE),)

        _, dy = jax.jvp(wrapped_jstep, primals, tangents)

        return dy  # returns (du, dv, dh)
    
    def step_adj(self, adState, State, nstep=1, t=0):
        
        # Get state variable
        adu = adState.getvar(name_var=self.name_var['U'])
        adv = adState.getvar(name_var=self.name_var['V'])
        adh = adState.getvar(name_var=self.name_var['SSH'])
        u = State.getvar(name_var=self.name_var['U'])
        v = State.getvar(name_var=self.name_var['V'])
        h = State.getvar(name_var=self.name_var['SSH'])    
        
        # Get BM field
        if self.is_bm:
            h_bm = self.ssh_bm_data[t]
            u_bm = self.u_bm_data[t]
            v_bm = self.v_bm_data[t]
        else:
            h_bm = None
            u_bm = None
            v_bm = None
        
        # Get parameters
        if 'He_mean' in self.name_params:
            adHe_mean = +adState.params['He_mean'] 
            He_mean = +State.params['He_mean'] 
        else:
            He_mean = adHe_mean = jnp.zeros((self.ny,self.nx))
        if 'alpha_He' in self.name_params:
            adalpha_He = +adState.params['alpha_He']
            alpha_He = +State.params['alpha_He']
        elif 'alpha' in self.name_params:
            adalpha_He = +adState.params['alpha']
            alpha_He = +State.params['alpha']
        else:
            alpha_He = adalpha_He = None
        if 'alpha_Uu' in self.name_params:
            adalpha_Uu = +adState.params['alpha_Uu']
            alpha_Uu = +State.params['alpha_Uu']
        elif 'alpha' in self.name_params:
            adalpha_Uu = +adState.params['alpha']
            alpha_Uu = +State.params['alpha']
        else:
            alpha_Uu = adalpha_Uu = None
        if 'alpha_Uz' in self.name_params:
            adalpha_Uz = +adState.params['alpha_Uz']
            alpha_Uz = +State.params['alpha_Uz']
        elif 'alpha' in self.name_params:
            adalpha_Uz = +adState.params['alpha']
            alpha_Uz = +State.params['alpha']
        else:
            alpha_Uz = adalpha_Uz = None
        if 'alpha_Up' in self.name_params:
            adalpha_Up = +adState.params['alpha_Up']
            alpha_Up = +State.params['alpha_Up']
        elif 'alpha' in self.name_params:
            adalpha_Up = +adState.params['alpha']
            alpha_Up = +State.params['alpha']
        else:
            alpha_Up = adalpha_Up = None
        if 'hbc' in self.name_params:
            adh_SN = adState.params['hbcx']
            adh_WE = adState.params['hbcy']
            h_SN = State.params['hbcx']
            h_WE = State.params['hbcy']
        else:
            adh_SN = adh_WE = None
            h_SN = h_WE = None
        

        # Forward trajectory (using JIT'd single-step for fast execution)
        u_list = [u]; v_list = [v]; h_list = [h]
        for it in range(nstep):
            u, v, h = self._jstep_jit(t+it*self.dt, u, v, h, He_mean, alpha_He, alpha_Uu, alpha_Up, alpha_Uz, h_SN, h_WE, h_bm, u_bm, v_bm, nstep=1)
            u_list.append(u); v_list.append(v); h_list.append(h)
            
        # Reverse-time adjoint loop
        for it in reversed(range(nstep)):
            u = u_list[it]
            v = v_list[it]
            h = h_list[it]
            adu, adv, adh, adHe_mean, adalpha_He, adalpha_Uu, adalpha_Up, adalpha_Uz, adh_SN, adh_WE = self._jstep_adj_jit(t+it*self.dt, 
                                                                            adu, adv, adh, adHe_mean, adalpha_He, adalpha_Uu, adalpha_Up, adalpha_Uz, adh_SN, adh_WE,
                                                                            u, v, h, He_mean, alpha_He, alpha_Uu, alpha_Up, alpha_Uz, h_SN, h_WE, h_bm, u_bm, v_bm)

        if 'He_mean' in self.name_params:
            adState.params['He_mean'] = adHe_mean
        adalpha = 0.
        if 'alpha_He' in self.name_params:
            adState.params['alpha_He'] = adalpha_He
        elif 'alpha' in self.name_params:
            adalpha += adalpha_He
        if 'alpha_Uu' in self.name_params:
            adState.params['alpha_Uu'] = adalpha_Uu
        elif 'alpha' in self.name_params:
            adalpha += adalpha_Uu
        if 'alpha_Up' in self.name_params:
            adState.params['alpha_Up'] = adalpha_Up
        elif 'alpha' in self.name_params:
            adalpha += adalpha_Up
        if 'alpha_Uz' in self.name_params:
            adState.params['alpha_Uz'] = adalpha_Uz
        elif 'alpha' in self.name_params:
            adalpha += adalpha_Uz
        if 'alpha' in self.name_params:
            adState.params['alpha'] = adalpha
        if 'hbc' in self.name_params:
            adState.params['hbcx'] = adh_SN
            adState.params['hbcy'] = adh_WE
        
        # Update state and parameters
        adState.setvar(adu,self.name_var['U'])
        adState.setvar(adv,self.name_var['V'])
        adState.setvar(adh,self.name_var['SSH'])

    def _jstep_adj(self, t, 
                  adu, adv, adh, adHe_mean, adalpha_He, adalpha_Uu, adalpha_Up, adalpha_Uz, adh_SN, adh_WE,
                  u, v, h, He_mean, alpha_He, alpha_Uu, alpha_Up, alpha_Uz, h_SN, h_WE, h_bm, u_bm, v_bm, nstep=1):

        def wrapped_jstep(x):
            u, v, h, He_mean, alpha_He, alpha_Uu, alpha_Up, alpha_Uz, h_SN, h_WE = x
            return self._jstep(t, u, v, h, He_mean, alpha_He, alpha_Uu, alpha_Up, alpha_Uz, h_SN, h_WE, h_bm, u_bm, v_bm, nstep=nstep)
        primals = ((u, v, h, He_mean, alpha_He, alpha_Uu, alpha_Up, alpha_Uz, h_SN, h_WE),)
        cotangents = (adu, adv, adh)  

        _, vjp_fn = jax.vjp(wrapped_jstep, *primals)
        adjoints = vjp_fn(cotangents)

        adu, adv, adh, _adHe_mean, _adalpha_He, _adalpha_Uu, _adalpha_Up, _adalpha_Uz, _adhSN, _adhWE = adjoints[0]

        if adHe_mean is not None:
            adHe_mean += _adHe_mean
        if adalpha_He is not None:
            adalpha_He += _adalpha_He
        if adalpha_Uu is not None:
            adalpha_Uu += _adalpha_Uu
        if adalpha_Up is not None:
            adalpha_Up += _adalpha_Up
        if adalpha_Uz is not None:
            adalpha_Uz += _adalpha_Uz
        if adh_SN is not None:
            adh_SN += _adhSN
        if adh_WE is not None:
            adh_WE += _adhWE

        return adu, adv, adh, adHe_mean, adalpha_He, adalpha_Uu, adalpha_Up, adalpha_Uz, adh_SN, adh_WE


###############################################################################
#                             QG-SW Models                                    #
###############################################################################

class Model_qgsw(M):

    def __init__(self,config,State):

        super().__init__(config,State)

        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

        # Model specific libraries
        if config.MOD.dir_model is None:
            dir_model = os.path.realpath(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..','models','model_qgsw'))
        else:
            dir_model = config.MOD.dir_model  
        SourceFileLoader("tools", dir_model+"/tools.py").load_module() 
        SourceFileLoader("helmholtz", dir_model+"/helmholtz.py").load_module() 
        SourceFileLoader("helmholtz_multigrid", dir_model+"/helmholtz_multigrid.py").load_module() 
        SourceFileLoader("masks", dir_model+"/masks.py").load_module() 
        SourceFileLoader("flux", dir_model+"/flux.py").load_module()
        SourceFileLoader("finite_diff", dir_model+"/finite_diff.py").load_module()
        SourceFileLoader("reconstruction", dir_model+"/reconstruction.py").load_module() 
        if config.MOD.name_class.lower()=='qg':
            SourceFileLoader('sw',f'{dir_model}/sw.py').load_module() 
        sw = SourceFileLoader('sw',f'{dir_model}/sw.py').load_module() 
        qg = SourceFileLoader('qg',f'{dir_model}/qg.py').load_module() 
        model_sw = getattr(sw, 'SW')
        model_qg = getattr(qg, 'QG')

        if USE_FLOAT64: 
            self.dtype = jnp.float64
        else:
            self.dtype = jnp.float32

        self.nl = config.MOD.nl
        self._configured_name_params = config.MOD.name_params if config.MOD.name_params is not None else []
        self._controls_H = 'H' in self._configured_name_params

        if config.MOD.name_class.lower()=='qg':
            model = model_qg
        else:
            model = model_sw
        self._is_qg_class = (config.MOD.name_class.lower() == 'qg')

        # Coriolis
        if config.MOD.f0 is not None and config.MOD.constant_f:
            self.f = config.MOD.f0 * np.ones((State.ny,State.nx))
        else:
            self.f = State.f
            f0 = np.nanmean(self.f)
            self.f[np.isnan(self.f)] = f0
        pad = ((1,0),(1,0))
        _f = np.pad(self.f, pad_width=pad, mode='edge')
        self.f_on_v = 0.5*(_f[:,1:] + _f[:,:-1])
        self.f_on_u = 0.5*(_f[1:,:] + _f[:-1,:])
        
        # Gravity 
        self.g = 9.81

        # Grid spacing
        self.dx = np.pad(State.DX, pad_width=pad, mode='edge')
        self.dy = np.pad(State.DY, pad_width=pad, mode='edge')
        self.dx_on_v = 0.5*(self.dx[:,1:] + self.dx[:,:-1])
        self.dy_on_v = 0.5*(self.dy[:,1:] + self.dy[:,:-1])
        self.dy_on_u = 0.5*(self.dy[1:,:] + self.dy[:-1,:])
        self.dx_on_u = 0.5*(self.dx[1:,:] + self.dx[:-1,:])
        self.Xh = State.X
        self.Yh = State.Y
        self.Xu, self.Yu = grid.dxdy2xy(self.dx_on_u, self.dy_on_u)
        self.Xv, self.Yv = grid.dxdy2xy(self.dx_on_v, self.dy_on_v)

        # Open MDT map if provided
        if config.MOD.path_mdt is not None and os.path.exists(config.MOD.path_mdt):
                      
            ds = xr.open_dataset(config.MOD.path_mdt)
            name_lon = config.MOD.name_var_mdt['lon']
            lon = ds[name_lon]
            # Convert longitude 
            if np.sign(lon.data.min())==-1 and State.lon_unit=='0_360':
                ds = ds.assign_coords({name_lon:((name_lon, lon.data % 360))})
            elif np.sign(lon.data.min())>=0 and State.lon_unit=='-180_180':
                ds = ds.assign_coords({name_lon:((name_lon, (lon.data + 180) % 360 - 180))})
            ds = ds.sortby(name_lon)    

            mdt = grid.interp2d(ds,
                                   config.MOD.name_var_mdt,
                                   State.lon,
                                   State.lat)

            # Honour the model mask on the h-grid first: griddata may extrapolate
            # non-NaN values onto land cells, and near-coast ocean cells can be
            # contaminated by land-NaN triangles in the Delaunay tessellation.
            # Force land to NaN, then fill with the nearest valid ocean value so
            # the coast–ocean transition is smooth.
            # Convention: State.mask == True → land, False → ocean.
            mdt[State.mask] = np.nan
            mdt = _fill_nan_nearest(mdt)
            # mdt is now smooth everywhere (land filled with nearest ocean value).

            if 'var_u' in config.MOD.name_var_mdt and 'var_v' in config.MOD.name_var_mdt:
                # Velocities are provided on the h-grid (ny, nx): apply the same
                # mask/fill treatment, then stagger to the u/v grids using the
                # same padding convention as f_on_u / f_on_v so that the stored
                # arrays have shapes (ny, nx+1) and (ny+1, nx) respectively.
                _name_vars_u = {
                    'lon': config.MOD.name_var_mdt['lon'],
                    'lat': config.MOD.name_var_mdt['lat'],
                    'var': config.MOD.name_var_mdt['var_u'],
                }
                _name_vars_v = {
                    'lon': config.MOD.name_var_mdt['lon'],
                    'lat': config.MOD.name_var_mdt['lat'],
                    'var': config.MOD.name_var_mdt['var_v'],
                }
                mdu = grid.interp2d(ds,
                                   _name_vars_u,
                                   State.lon,
                                   State.lat)
                mdv = grid.interp2d(ds,
                                   _name_vars_v,
                                   State.lon,
                                   State.lat)
                mdu[State.mask] = np.nan
                mdu = _fill_nan_nearest(mdu)
                mdv[State.mask] = np.nan
                mdv = _fill_nan_nearest(mdv)
                # Stagger h-grid → u-grid (ny, nx+1) and v-grid (ny+1, nx)
                _pad = ((1, 0), (1, 0))
                _mdu = np.pad(mdu, pad_width=_pad, mode='edge')
                mdu = 0.5 * (_mdu[1:, :] + _mdu[:-1, :])   # (ny, nx+1)
                _mdv = np.pad(mdv, pad_width=_pad, mode='edge')
                mdv = 0.5 * (_mdv[:, 1:] + _mdv[:, :-1])   # (ny+1, nx)
            else:
                # Derive geostrophic velocities from the smooth filled mdt
                # (before zeroing land) so that no artificial MDT→0 step at the
                # coast amplifies into huge geostrophic velocities.
                # ssh2uv returns (ny, nx+1) and (ny+1, nx) directly.
                mdu, mdv = self.ssh2uv(mdt)

            # Zero land cells in mdt, and zero any staggered u/v face that
            # borders a land h-cell (same padding convention as ssh2uv).
            # This removes spurious velocities at coast-facing faces without
            # affecting the interior ocean signal.
            mdt[State.mask] = 0.
            _mask_pad = np.pad(State.mask.astype(bool), pad_width=((1, 0), (1, 0)), mode='edge')
            mdu[_mask_pad[1:, :] | _mask_pad[:-1, :]] = 0.  # (ny, nx+1)
            mdv[_mask_pad[:, 1:] | _mask_pad[:, :-1]] = 0.  # (ny+1, nx)

            # Safety: interpolation/extrapolation of external MDT/current files can
            # still leave non-finite values (NaN/Inf). Enforce finite targets before
            # they are injected into the model state and sponge relaxation.
            _nbad_mdt = int(np.size(mdt) - np.count_nonzero(np.isfinite(mdt)))
            _nbad_mdu = int(np.size(mdu) - np.count_nonzero(np.isfinite(mdu)))
            _nbad_mdv = int(np.size(mdv) - np.count_nonzero(np.isfinite(mdv)))
            if (_nbad_mdt + _nbad_mdu + _nbad_mdv) > 0:
                print('Warning: non-finite MDT background values found '
                      f'(mdt={_nbad_mdt}, mdu={_nbad_mdu}, mdv={_nbad_mdv}); '
                      'replacing with 0.')
            mdt = np.nan_to_num(mdt, nan=0.0, posinf=0.0, neginf=0.0)
            mdu = np.nan_to_num(mdu, nan=0.0, posinf=0.0, neginf=0.0)
            mdv = np.nan_to_num(mdv, nan=0.0, posinf=0.0, neginf=0.0)

        
            if config.EXP.flag_plot>0:
                fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(15, 5))

                im1 = ax1.pcolormesh(mdt)
                plt.colorbar(im1, ax=ax1)
                ax1.set_title('MDT')

                im2 = ax2.pcolormesh(mdu)
                plt.colorbar(im2, ax=ax2)
                ax2.set_title('MDU')

                im3 = ax3.pcolormesh(mdv)
                plt.colorbar(im3, ax=ax3)
                ax3.set_title('MDV')

                plt.show()

            self.mdu = jnp.expand_dims(mdu.astype(self.dtype).T, axis=(0,1))
            self.mdv = jnp.expand_dims(mdv.astype(self.dtype).T, axis=(0,1))
            self.mdt = jnp.expand_dims(mdt.astype(self.dtype).T, axis=(0,1))

        else:
            self.mdt = self.mdu = self.mdv = None

        # Open Rossby Radius if provided
        if config.MOD.filec_aux is not None and os.path.exists(config.MOD.filec_aux):

            ds = xr.open_dataset(config.MOD.filec_aux)
            name_lon = config.MOD.name_var_c['lon']
            lon = ds[name_lon]
            # Convert longitude 
            if np.sign(lon.data.min())==-1 and State.lon_unit=='0_360':
                ds = ds.assign_coords({name_lon:((name_lon, lon.data % 360))})
            elif np.sign(lon.data.min())>=0 and State.lon_unit=='-180_180':
                ds = ds.assign_coords({name_lon:((name_lon, (lon.data + 180) % 360 - 180))})
            ds = ds.sortby(name_lon)    

            self.c = grid.interp2d(ds,
                                   config.MOD.name_var_c,
                                   State.lon,
                                   State.lat)
            
            # Replace NaN before cmin/cmax clamping (NaN < x is False,
            # so NaN would survive the clamp and propagate to H as 0,
            # creating sharp unphysical gradients that destabilise the model).
            c_fill = np.nanmean(self.c)
            if np.isnan(c_fill):
                # All interpolated values are NaN — tile is likely outside the
                # aux file domain.  Use cmin first, cmax second as fallback so
                # that H stays finite and the model can proceed.
                if config.MOD.cmin is not None:
                    c_fill = config.MOD.cmin
                    print(
                        f'WARNING: all Rossby phase velocity values are NaN '
                        f'after interpolation (tile may lie outside the aux '
                        f'file domain). Falling back to cmin={c_fill} m/s.'
                    )
                elif config.MOD.cmax is not None:
                    c_fill = config.MOD.cmax
                    print(
                        f'WARNING: all Rossby phase velocity values are NaN '
                        f'after interpolation (tile may lie outside the aux '
                        f'file domain). Falling back to cmax={c_fill} m/s.'
                    )
                else:
                    print(
                        f'WARNING: all Rossby phase velocity values are NaN '
                        f'after interpolation and no cmin/cmax fallback is '
                        f'defined. H will be NaN and the model will likely fail.'
                    )
            self.c[np.isnan(self.c)] = c_fill

            if config.MOD.cmin is not None:
                self.c[self.c<config.MOD.cmin] = config.MOD.cmin
            
            if config.MOD.cmax is not None:
                self.c[self.c>config.MOD.cmax] = config.MOD.cmax
            
            if config.EXP.flag_plot>0:
                plt.figure()
                plt.pcolormesh(self.c)
                plt.colorbar()
                plt.title('Rossby phase velocity')
                plt.show()
                
        else:
            self.c = config.MOD.c0 * np.ones((State.ny,State.nx))
        
        # Reduced gravity
        if config.MOD.nl==1:
            if config.MOD.g_prime is None:
                g_prime = np.array([[[9.81]]])
            else:
                g_prime = config.MOD.g_prime
                if not hasattr(g_prime,'shape'):
                    g_prime = np.array([[[g_prime]]])
                elif len(g_prime.shape)==2:
                    g_prime = np.expand_dims(g_prime, axis=0)
                elif len(g_prime.shape)==1:
                    g_prime = np.expand_dims(g_prime, axis=(0,1))
                    
            if config.MOD.H is None:
                # self.c is interpolated onto State.lon/State.lat, so its shape
                # must be (State.ny, State.nx). The .T below converts to (nx, ny),
                # which is then nl-expanded to (1, nx, ny) — the layout sw.py expects.
                # A silent (nx, ny)-vs-(ny, nx) mismatch here would feed a diagonally
                # flipped H field to the SW core, degrading forecast skill without
                # crashing the model. Fail loudly instead.
                assert self.c.shape == (State.ny, State.nx), \
                    f'self.c has shape {self.c.shape}, expected (ny, nx)=' \
                    f'({State.ny}, {State.nx}). The .T applied below assumes ' \
                    f'(ny, nx); a mismatch will silently transpose H.'
                H = np.expand_dims(self.c.T**2/g_prime[0,0,0], axis=0) 
                H0 = np.array([[[np.nanmean(self.c)**2/g_prime[0,0,0]]]])
                H[np.isnan(H)] = 0.
                if config.EXP.flag_plot>0:
                    plt.figure()
                    plt.pcolormesh(H[0,:,:].T)
                    plt.colorbar()
                    plt.title('Equivalent Height')
                    plt.show()
                if config.MOD.name_class.lower()=='qg' or getattr(config.MOD, 'constant_H', False):
                    H = H0 # QG needs constant H; constant_H flag forces uniform H for SW too
                    print(f'H set to constant: H = {H0[0,0,0]:.4f} m')
            else:
                H = H0 = config.MOD.H
                if not hasattr(H,'shape'):
                    H = H0 = np.array([[[H]]])
                elif len(H.shape)==2:
                    H = H0 = np.expand_dims(H, axis=0)   
                elif len(H.shape)==1:
                    H = H0 = np.expand_dims(H, axis=(0,1))
        else:
            g_prime = config.MOD.g_prime
            if not hasattr(g_prime, 'shape'):
                g_prime = np.array(g_prime)
            if len(g_prime.shape) == 1:
                g_prime = g_prime.reshape(-1, 1, 1)
            H = config.MOD.H
            if not hasattr(H, 'shape'):
                H = np.array(H)
            if len(H.shape) == 1:
                H = H.reshape(-1, 1, 1)
            elif len(H.shape) == 2:
                H = np.expand_dims(H, axis=0)
        
        H_from_init = self._H_from_init_file(config, State)
        if H_from_init is not None:
            H = H_from_init

        self.H0 = H
        self._H_from_init_file_used = H_from_init is not None
        self.H_max_bound = getattr(config.MOD, 'H_max', None)
        H_floor = getattr(config.MOD, 'H_floor', None)
        if H_floor is None:
            if getattr(config.MOD, 'cmin', None) is not None and config.MOD.nl == 1:
                H_floor = float(config.MOD.cmin)**2 / float(g_prime[0, 0, 0])
            else:
                H_floor = 0.
        self.H_floor = self._as_H_model_array(H_floor, State, 'H_floor')
        self.H_floor = self._zero_H_floor_on_mask(self.H_floor, State)
        self._validate_H_floor(self.H0, self.H_floor)
        
        # CFL
        if config.MOD.cfl is not None:
            grid_spacing = min(np.nanmin(State.DX), np.nanmin(State.DY)) 
            # For nl>1 the fastest wave is the barotropic mode sqrt(g*sum(H))
            if self.nl > 1:
                H_arr = np.asarray(config.MOD.H)
                g_arr = np.asarray(config.MOD.g_prime)
                c_max = float(np.sqrt(g_arr[0] * H_arr.sum()))
                print(f'Barotropic wave speed c_max = {c_max:.2f} m/s')
            else:
                c_max = float(np.nanmax(self.c))
            dt = config.MOD.cfl * grid_spacing / c_max
            divisors = [i for i in range(1, 3600 + 1) if 3600 % i == 0]  # Find all divisors of one hour in seconds
            lower_divisors = [d for d in divisors if d <= dt]
            self.dt = max(lower_divisors)  # Get closest
            print(f'CFL condition for c= {c_max}')
            print('Model time-step', self.dt)
            # Time parameters
            if self.dt>0:
                self.nt = 1 + int((config.EXP.final_date - config.EXP.init_date).total_seconds()//self.dt)
                self.timestamps = [] 
                t = config.EXP.init_date
                while t<=config.EXP.final_date:
                    self.timestamps.append(t)
                    t += timedelta(seconds=self.dt)
                self.timestamps = np.asarray(self.timestamps)
            else:
                self.nt = 1
                self.timestamps = np.array([config.EXP.init_date])
            self.T = np.arange(self.nt) * self.dt
    
        # Initialize model state
        if (config.GRID.super == 'GRID_FROM_FILE'):
            dsin = grid.open_grid_restart(config)
            for name in self.name_var:
                if config.MOD.name_init_var is not None and name in config.MOD.name_init_var:
                    name_init_var = config.MOD.name_init_var[name]
                else:
                    name_init_var = config.MOD.name_var[name]
                if name_init_var in dsin:
                    var_init = dsin[name_init_var]
                    if len(var_init.shape)==3:
                        var_init = var_init[0,:,:]
                    if config.GRID.subsampling is not None:
                        var_init = var_init[::config.GRID.subsampling,::config.GRID.subsampling]
                    arr = var_init.values.copy()
                    if name not in ('SSH', 'U', 'V') and np.any(np.isnan(arr)):
                        # Tracer: extrapolate NaN land cells from nearest ocean value
                        # (mirrors set_bc treatment in Model_qg1l_jax via _fill_nan_nearest)
                        arr = _fill_nan_nearest(arr)
                    else:
                        arr[np.isnan(arr)] = 0.
                    State.var[self.name_var[name]] = arr
                else:
                    if name=='U':
                        State.var[self.name_var[name]] = jnp.zeros((State.ny,State.nx+1), dtype=self.dtype)
                    elif name=='V':
                        State.var[self.name_var[name]] = jnp.zeros((State.ny+1,State.nx), dtype=self.dtype)
                    else:  # SSH or passive tracer — h-grid (ny, nx)
                        arr = np.zeros((State.ny, State.nx))
                        if State.mask is not None:
                            arr[State.mask] = np.nan
                        State.var[self.name_var[name]] = jnp.asarray(arr, dtype=self.dtype)
            dsin.close()
            del dsin   
        else:
            for name in self.name_var:  
                if name=='U':
                    State.var[self.name_var[name]] = jnp.zeros((State.ny,State.nx+1), dtype=self.dtype)
                elif name=='V':
                    State.var[self.name_var[name]] = jnp.zeros((State.ny+1,State.nx), dtype=self.dtype)
                else:  # SSH or passive tracer — h-grid (ny, nx)
                    arr = np.zeros((State.ny, State.nx))
                    if State.mask is not None:
                        arr[State.mask] = np.nan
                    State.var[self.name_var[name]] = jnp.asarray(arr, dtype=self.dtype)

        # For nl>1, also initialize per-layer storage in State
        if self.nl > 1:
            State.var['u_layers'] = jnp.zeros((self.nl, State.ny, State.nx+1), dtype=self.dtype)
            State.var['v_layers'] = jnp.zeros((self.nl, State.ny+1, State.nx), dtype=self.dtype)
            State.var['h_layers'] = jnp.zeros((self.nl, State.ny, State.nx), dtype=self.dtype)

        # Initialize boundary condition dictionnary for each model variable
        self.bc = {}
        self.forcing = {}
        for _name_var_mod in self.name_var:
            self.bc[_name_var_mod] = {}
            self.forcing[_name_var_mod] = {}
        self.init_from_bc = config.MOD.init_from_bc

        # Sponge layer
        self.sponge_width = config.MOD.dist_sponge_bc  # km
        self.sponge_coef = 0.0 if self._is_qg_class else config.MOD.sponge_coef

        if self.sponge_width is not None and self.sponge_width>0:
            lon_h = State.lon
            lat_h = State.lat
            lon_u = np.zeros((State.ny, State.nx+1))
            lat_u = np.zeros((State.ny, State.nx+1))
            lon_u[:, 1:State.nx] = 0.5 * (lon_h[:, :-1] + lon_h[:, 1:])
            lat_u[:, 1:State.nx] = 0.5 * (lat_h[:, :-1] + lat_h[:, 1:])
            lon_v = np.zeros((State.ny+1, State.nx))
            lat_v = np.zeros((State.ny+1, State.nx))
            lon_v[1:State.ny, :] = 0.5 * (lon_h[:-1, :] + lon_h[1:, :])
            lat_v[1:State.ny, :] = 0.5 * (lat_h[:-1, :] + lat_h[1:, :])
            # West boundary
            lon_u[:, 0] = lon_h[:, 0] - 0.5 * (lon_h[:, 1] - lon_h[:, 0])
            lat_u[:, 0] = lat_h[:, 0] - 0.5 * (lat_h[:, 1] - lat_h[:, 0])
            # East boundary
            lon_u[:, State.nx] = lon_h[:, -1] + 0.5 * (lon_h[:, -1] - lon_h[:, -2])
            lat_u[:, State.nx] = lat_h[:, -1] + 0.5 * (lat_h[:, -1] - lat_h[:, -2])
            # South boundary
            lon_v[0, :] = lon_h[0, :] - 0.5 * (lon_h[1, :] - lon_h[0, :])
            lat_v[0, :] = lat_h[0, :] - 0.5 * (lat_h[1, :] - lat_h[0, :])
            # North boundary
            lon_v[State.ny, :] = lon_h[-1, :] + 0.5 * (lon_h[-1, :] - lon_h[-2, :])
            lat_v[State.ny, :] = lat_h[-1, :] + 0.5 * (lat_h[-1, :] - lat_h[-2, :])
            
            if config.MOD.use_sponge_on_coast:
                mask_h = State.mask.copy()
                mask_u = np.zeros((State.ny, State.nx+1), dtype=bool)
                mask_u[:, 1:State.nx] = mask_h[:, :-1] | mask_h[:, 1:]
                mask_u[:, 0] = mask_h[:, 0]
                mask_u[:, State.nx] = mask_h[:, -1]
                mask_v = np.zeros((State.ny+1, State.nx), dtype=bool)
                mask_v[1:State.ny, :] = mask_h[:-1, :] | mask_h[1:, :]
                mask_v[0, :] = mask_h[0, :]
                mask_v[State.ny, :] = mask_h[-1, :]
                
            else:
                mask_h = np.zeros((State.ny, State.nx), dtype=bool)
                mask_u = np.zeros((State.ny, State.nx+1), dtype=bool)
                mask_v = np.zeros((State.ny+1, State.nx), dtype=bool)

            alpha = config.MOD.tangential_sponge_factor

            # h: isotropic sponge everywhere
            wc_h, wNS_h, wWE_h = grid.compute_sponge_components(lon_h, lat_h, mask_h, config.MOD.dist_sponge_bc)
            sponge_h = 1.0 - (1.0 - wc_h) * (1.0 - wNS_h) * (1.0 - wWE_h)

            # u: normal at W/E (full), tangential at N/S (reduced)
            wc_u, wNS_u, wWE_u = grid.compute_sponge_components(lon_u, lat_u, mask_u, config.MOD.dist_sponge_bc)
            sponge_u = 1.0 - (1.0 - wc_u) * (1.0 - alpha * wNS_u) * (1.0 - wWE_u)

            # v: normal at N/S (full), tangential at W/E (reduced)
            wc_v, wNS_v, wWE_v = grid.compute_sponge_components(lon_v, lat_v, mask_v, config.MOD.dist_sponge_bc)
            sponge_v = 1.0 - (1.0 - wc_v) * (1.0 - wNS_v) * (1.0 - alpha * wWE_v)

            _, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(15, 5))
            im1 = ax1.pcolormesh(sponge_h)
            plt.colorbar(im1, ax=ax1)
            ax1.set_title('Sponge SSH')
            im2 = ax2.pcolormesh(sponge_u)
            plt.colorbar(im2, ax=ax2)
            ax2.set_title('Sponge U')
            im3 = ax3.pcolormesh(sponge_v)
            plt.colorbar(im3, ax=ax3)
            ax3.set_title('Sponge V')
            plt.show()
        
        else:
            sponge_u = np.zeros((State.ny, State.nx+1))
            sponge_v = np.zeros((State.ny+1, State.nx))
            sponge_h = np.zeros((State.ny, State.nx))

        self.sponge_u = sponge_u 
        self.sponge_v = sponge_v 
        self.sponge_h = sponge_h 


        # Model initialization
        # QG spectral operators assume uniform spacing scalars; SW keeps full 2-D grids.
        _dx_param = float(np.nanmean(State.DX)) if config.MOD.name_class.lower() == 'qg' else State.DX.T
        _dy_param = float(np.nanmean(State.DY)) if config.MOD.name_class.lower() == 'qg' else State.DY.T
        params = {
            "nx": State.nx,
            "ny": State.ny,
            "nl": config.MOD.nl,
            "dx": _dx_param,
            "dy": _dy_param,
            "H": H,
            "g_prime": g_prime,
            "f": np.pad(self.f.T, ((0, 1), (0, 1)), mode='edge'),
            "taux": 0.,
            "tauy": 0.,
            "bottom_drag_coef": config.MOD.bottom_drag_coef,
            "rho_water": getattr(config.MOD, 'rho_water', 1025.0),
            "h_wind": getattr(config.MOD, 'h_wind', None),
            "dtype": self.dtype,
            "mask": (1-State.mask.astype(int).T),
            "compile": True,
            "slip_coef": config.MOD.slip_coef,
            "visc_coef": config.MOD.visc_coef,
            "diff_coef": config.MOD.diff_coef,
            "dt": self.dt,
            "barotropic_filter": False,
            'barotropic_filter_spectral': False,
            'sponge_coef': self.sponge_coef,
            'forcing_momentum': getattr(config.MOD, 'forcing_momentum', 'direct'),
            'H_min': None if self._controls_H else getattr(config.MOD, 'H_min', None),
            'H_max': getattr(config.MOD, 'H_max', None),
            'diff_coef_trac': getattr(config.MOD, 'diff_coef_trac', 0.),
            'time_scheme':    getattr(config.MOD, 'time_scheme',    'rk3'),
            'h_adv_scheme':      getattr(config.MOD, 'h_adv_scheme',      'weno'),
            'mom_adv_scheme':    getattr(config.MOD, 'mom_adv_scheme',    'weno'),
            'tracer_adv_scheme': getattr(config.MOD, 'tracer_adv_scheme', 'weno'),
            'solver':            getattr(config.MOD, 'solver',            'dst_cmm'),
        }

        self.model = model(params)

        # Build auxiliary QG projector for QG-balanced sponge target (SW mode only)
        self.model_proj = None
        self.qg_balanced_sponge_bc = False
        if not self._is_qg_class and getattr(config.MOD, 'qg_balanced_sponge_bc', False):
            # Validate preconditions
            _name_params = config.MOD.name_params if config.MOD.name_params is not None else []
            if 'bc' in _name_params:
                raise ValueError(
                    'qg_balanced_sponge_bc is not compatible with the "bc" control parameter. '
                    'Use external boundary conditions only when using QG-projected sponge targets.'
                )
            if config.MOD.dist_sponge_bc is None or config.MOD.dist_sponge_bc <= 0 or config.MOD.sponge_coef == 0:
                print('Warning: qg_balanced_sponge_bc=True but sponge is inactive '
                      '(dist_sponge_bc=None/≤0 or sponge_coef==0); disabling QG-projected sponge targets.')
            else:
                # Build auxiliary QG projector by cloning the SW params and
                # overriding only what the QG elliptic solver needs.
                qg_projector_params = dict(params)
                qg_projector_params['H'] = self.H0  # spatially constant H for QG solver
                # QG spectral solver assumes uniform spacing — collapse 2-D
                # dx/dy to their domain mean if needed.
                qg_projector_params['dx'] = float(np.nanmean(State.DX))
                qg_projector_params['dy'] = float(np.nanmean(State.DY))
                # The sponge projector only builds a balanced target. In domains
                # with land, the capacitance-matrix irregular-boundary correction
                # can become ill-conditioned and seed NaNs into the target, so use
                # the robust masked DST solve for this auxiliary projection.
                qg_projector_params['solver'] = 'dst'
                qg_projector_params['compile'] = False
                qg_projector_params['sponge_coef'] = 0.0  # projector is not a time-stepper
                qg_projector_params['visc_coef'] = 0.0
                qg_projector_params['diff_coef'] = 0.0
                qg_projector_params['diff_coef_trac'] = 0.0
                try:
                    self.model_proj = model_qg(qg_projector_params)
                    self.qg_balanced_sponge_bc = True
                    print('Initialized auxiliary QG projector for balanced sponge targets')
                except Exception as e:
                    print(f'Warning: failed to initialize QG projector: {e}; '
                          'falling back to external boundary field sponge targets.')
                    self.model_proj = None
                    self.qg_balanced_sponge_bc = False

        if self.mdt is not None:
            self.mdu = jnp.nan_to_num(self.mdu, nan=0.0, posinf=0.0, neginf=0.0)
            self.mdv = jnp.nan_to_num(self.mdv, nan=0.0, posinf=0.0, neginf=0.0)
            self.mdt = jnp.nan_to_num(self.mdt, nan=0.0, posinf=0.0, neginf=0.0)
            self.mdu = jnp.where(self.model.masks.u > 0.5, self.mdu, 0.0)
            self.mdv = jnp.where(self.model.masks.v > 0.5, self.mdv, 0.0)
            self.mdt = jnp.where(self.model.masks.h > 0.5, self.mdt, 0.0)

        # Sponge masks: (1, 1, nx, ny) — broadcasts across all layers.
        # Layer 0 is nudged toward surface BC; deep layers toward zero (their BC is 0).
        self.model.sponge_u = jnp.where(
            self.model.masks.u > 0.5,
            jnp.expand_dims(jnp.asarray(self.sponge_u.T.astype(self.dtype)), axis=(0,1)),
            0.0,
        )
        self.model.sponge_v = jnp.where(
            self.model.masks.v > 0.5,
            jnp.expand_dims(jnp.asarray(self.sponge_v.T.astype(self.dtype)), axis=(0,1)),
            0.0,
        )
        self.model.sponge_h = jnp.where(
            self.model.masks.h > 0.5,
            jnp.expand_dims(jnp.asarray(self.sponge_h.T.astype(self.dtype)), axis=(0,1)),
            0.0,
        )

        # Exclude sponge cells from observation assimilation without changing
        # State.mask, which is also used by model masks and basis support.
        if getattr(config.MOD, 'mask_sponge_bc', False):
            sponge_mask = (self.sponge_h > 0)
            if getattr(State, 'sponge_mask', None) is None:
                State.sponge_mask = sponge_mask.copy()
            else:
                State.sponge_mask = np.asarray(State.sponge_mask, dtype=bool) | sponge_mask

        # Maximum number of model steps per JIT call (limits GPU memory)
        self.max_nstep = getattr(config.MOD, 'max_nstep', 240)

        # Model functions initialization
        self.model_step = self.model.step
        self.model_step_tgl = self.model.step_tgl
        self.model_step_adj = self.model.step_adj
        
        flag_compile = True
        if flag_compile:
            self.jstep_jit = jax.jit(self.jstep, static_argnames=['nstep'])
            self.jstep_tgl_jit = jax.jit(self.jstep_tgl, static_argnames=['nstep'])
            self.jstep_adj_jit = jax.jit(self.jstep_adj, static_argnames=['nstep'])
            self.jstep_core_jit = jax.jit(self.jstep_core, static_argnames=['nstep'])
        else:
            self.jstep_jit = self.jstep
            self.jstep_tgl_jit = self.jstep_tgl
            self.jstep_adj_jit = self.jstep_adj
            self.jstep_core_jit = self.jstep_core

        # Tracer detection — config flag takes precedence if explicitly set to False
        self.tracer_names = [k for k in self.name_var if k not in ('U', 'V', 'SSH')]
        _cfg_advect = getattr(config.MOD, 'advect_tracer', None)
        self.advect_tracer = (len(self.tracer_names) > 0) if _cfg_advect is None else bool(_cfg_advect)
        self.n_trac = len(self.tracer_names)

        if self.advect_tracer:
            if flag_compile:
                self.jstep_core_trac_jit = jax.jit(self.jstep_core_trac, static_argnames=['nstep'])
                self.jstep_tgl_trac_jit  = jax.jit(self.jstep_tgl_trac,  static_argnames=['nstep'])
                self.jstep_adj_trac_jit  = jax.jit(self.jstep_adj_trac,  static_argnames=['nstep'])
            else:
                self.jstep_core_trac_jit = self.jstep_core_trac
                self.jstep_tgl_trac_jit  = self.jstep_tgl_trac
                self.jstep_adj_trac_jit  = self.jstep_adj_trac

        # Control parameters
        self.name_params = self._configured_name_params
        if 'H' in self.name_params:
            if (config.GRID.super == 'GRID_FROM_FILE'):
                dsin = grid.open_grid_restart(config)
                if self._H_from_init_file_used:
                    # Saved H is already the total equivalent depth from the
                    # previous run. Start new increments from zero around it.
                    State.params['H'] = np.zeros((State.ny,State.nx))
                elif 'H_control' in dsin:
                    State.params['H'] = dsin['H_control'].values.squeeze()
                    State.params['H'][np.isnan(State.params['H'])] = 0.
                else:
                    dsin.close()
                    raise ValueError(
                        'Controlled H restarts must provide either H, the total '
                        'equivalent depth, or H_control, the dimensionless equivalent-depth '
                        'control.'
                    )
                dsin.close()
                del dsin
            else:
                State.params['H'] = np.zeros((State.ny,State.nx))

        if 'h_wind' in self.name_params:
            self.h_wind = getattr(config.MOD, 'h_wind', None)
            if (config.GRID.super == 'GRID_FROM_FILE'):
                dsin = grid.open_grid_restart(config)
                if 'h_wind' in dsin:
                    State.params['h_wind'] = dsin['h_wind'].values.squeeze()
                    State.params['h_wind'][np.isnan(State.params['h_wind'])] = 0.
                else:
                    State.params['h_wind'] = np.zeros((State.ny,State.nx))
                dsin.close()
                del dsin
            else:
                State.params['h_wind'] = np.zeros((State.ny,State.nx))

        if 'wind_strength' in self.name_params:
            if (config.GRID.super == 'GRID_FROM_FILE'):
                dsin = grid.open_grid_restart(config)
                if 'wind_strength' in dsin:
                    State.params['wind_strength'] = dsin['wind_strength'].values.squeeze()
                    State.params['wind_strength'][np.isnan(State.params['wind_strength'])] = 0.
                else:
                    State.params['wind_strength'] = np.zeros((State.ny,State.nx))
                dsin.close()
                del dsin
            else:
                State.params['wind_strength'] = np.zeros((State.ny, State.nx))

        if 'bc' in self.name_params:
            State.params['u_b'] = np.zeros((State.ny, State.nx + 1))
            State.params['v_b'] = np.zeros((State.ny + 1, State.nx))
            State.params['h_b'] = np.zeros((State.ny, State.nx))

        for name in self.name_var:
            State.params[self.name_var[name]] = np.zeros_like(State.var[self.name_var[name]])

        # Wind forcing (must be loaded after self.timestamps and self.dt are set)
        self._load_wind_forcing(config, State)

        # Tests tgl & adj
        if config.INV is not None and config.INV.super=='INV_4DVAR' and config.INV.compute_test:
            print('Tangent test:')
            #tangent_test(self,State,nstep=10)#, ampl=1e-3)
            print('Adjoint test:')
            #self.adjoint_test_jstep(nstep=10)
            adjoint_test(self,State,nstep=10)#, ampl=1e-3)
    
    def _H_from_init_file(self, config, State):
        if config.GRID.super != 'GRID_FROM_FILE':
            return None
        path_init = getattr(config.GRID, 'path_init_grid', None)
        if path_init is None:
            return None
        if (not os.path.exists(path_init)
                and not getattr(config.EXP, 'saveoutputs_zarr', False)):
            return None

        candidate_names = []
        if config.MOD.name_init_var is not None and 'H' in config.MOD.name_init_var:
            candidate_names.append(config.MOD.name_init_var['H'])
        candidate_names.append('H')

        dsin = grid.open_grid_restart(config)
        try:
            for name in candidate_names:
                if name not in dsin:
                    continue
                da = dsin[name].squeeze(drop=True)
                if config.EXP.name_time in da.dims:
                    da = da.isel({config.EXP.name_time: 0})
                if getattr(config.GRID, 'subsampling', None) is not None and da.ndim >= 2:
                    da = da[..., ::config.GRID.subsampling, ::config.GRID.subsampling]
                arr = np.asarray(da.values, dtype=float)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                H = self._as_H_model_array(arr, State, name)
                if self._is_qg_class and not (H.shape == (1, 1, 1) or np.nanstd(H) == 0.):
                    H = np.array([[[np.nanmean(H)]]])
                    print(f'H read from {path_init}:{name} and averaged for QG: H = {H[0,0,0]:.4f} m')
                else:
                    print(f'H read from {path_init}:{name}')
                return H
        finally:
            dsin.close()
        return None

    def _as_H_model_array(self, value, State, name):
        arr = np.asarray(value)
        if arr.ndim == 0:
            return arr.reshape(1, 1, 1)
        if arr.shape == (State.ny, State.nx):
            return np.expand_dims(arr.T, axis=0)
        if arr.ndim == 1:
            return arr.reshape(-1, 1, 1)
        if arr.ndim == 2:
            if arr.shape == (State.nx, State.ny):
                return np.expand_dims(arr, axis=0)
            return np.expand_dims(arr, axis=0)
        if arr.ndim == 3:
            if arr.shape[-2:] == (State.ny, State.nx):
                return np.swapaxes(arr, -1, -2)
            return arr
        raise ValueError(
            f'{name} must be scalar, 1-D layers, 2-D, or 3-D; got shape {arr.shape}'
        )

    def _zero_H_floor_on_mask(self, H_floor, State):
        if State.mask is None:
            return H_floor
        mask = np.expand_dims(np.asarray(State.mask, dtype=bool).T, axis=0)
        shape = np.broadcast_shapes(np.shape(H_floor), np.shape(mask))
        H_floor_b = np.broadcast_to(H_floor, shape).copy()
        mask_b = np.broadcast_to(mask, shape)
        H_floor_b[mask_b] = 0.
        return H_floor_b

    def _validate_H_floor(self, H_ref, H_floor):
        if np.nanmin(H_floor) < 0:
            raise ValueError('H_floor must be non-negative.')
        try:
            shape = np.broadcast_shapes(np.shape(H_ref), np.shape(H_floor))
            H_ref_b = np.broadcast_to(H_ref, shape)
            H_floor_b = np.broadcast_to(H_floor, shape)
        except ValueError as exc:
            raise ValueError(
                f'H_floor shape {np.shape(H_floor)} is not broadcastable with '
                f'Reference Equivalent Depth shape {np.shape(H_ref)}.'
            ) from exc
        if np.nanmin(H_ref_b - H_floor_b) < -1e-12:
            raise ValueError(
                'Reference Equivalent Depth must be greater than or equal to H_floor.'
            )

    def _H_control_to_sw(self, H_control):
        H_control = jnp.asarray(H_control, dtype=self.dtype)
        if H_control.ndim == 2:
            return jnp.expand_dims(H_control.T, axis=0)
        if H_control.ndim == 3:
            return H_control
        if H_control.ndim == 4 and H_control.shape[0] == 1:
            return H_control[0]
        raise ValueError(
            f'H_control must have shape (ny, nx), (nl, nx, ny), or '
            f'(1, nl, nx, ny); got {H_control.shape}.'
        )

    def _H_control_to_total(self, H_control, apply_bounds=True):
        H_control_sw = self._H_control_to_sw(H_control)
        H_ref = jnp.asarray(self.H0, dtype=self.dtype)
        H_floor = jnp.asarray(self.H_floor, dtype=self.dtype)
        H_total = H_floor + (H_ref - H_floor) * jnp.exp(H_control_sw)
        if apply_bounds and self.H_max_bound is not None:
            H_total = jnp.minimum(H_total, self.H_max_bound)
        return H_total

    def _H_control_to_model_increment(self, H_control):
        if H_control is None:
            return None
        H_ref = jnp.asarray(self.H0, dtype=self.dtype)
        return self._H_control_to_total(H_control) - H_ref

    def _H_control_to_total_state(self, H_control):
        H_total = self._H_control_to_total(H_control)
        if H_total.shape[0] != 1:
            raise NotImplementedError('Saving controlled H for nl > 1 is not implemented.')
        return np.array(H_total[0].T)

    def _load_wind_forcing(self, config, State):
        """
        Read a NetCDF wind file, convert (u10, v10) to wind stress using the
        bulk formula  tau = rho_air * Cd * |U10| * U10,  and precompute
        taux / tauy at every model timestamp on the u- and v-grids.

        Config keys used (all optional):
          config.MOD.path_wind       – path to the .nc file (None → no wind)
          config.MOD.name_var_wind   – dict with keys 'lon','lat','time','u10','v10'
                                       (defaults to ERA5 conventions)
          config.MOD.rho_air         – air density in kg/m³  (default 1.25)
          config.MOD.Cd_wind         – drag coefficient      (default 1.5e-3)

        Stores:
          self.taux_wind  – ndarray (nt, ny, nx+1)  wind stress on u-grid
          self.tauy_wind  – ndarray (nt, ny+1, nx)  wind stress on v-grid
        """
        self.taux_wind = None
        self.tauy_wind = None

        path_wind = getattr(config.MOD, 'path_wind', None)
        if path_wind is None or not os.path.exists(path_wind):
            return

        rho_air = getattr(config.MOD, 'rho_air', 1.225)
        Cd      = getattr(config.MOD, 'Cd_wind', 1.3e-3)

        nv = getattr(config.MOD, 'name_var_wind',
                     {'lon': 'longitude', 'lat': 'latitude', 'time': 'time',
                      'u10': 'u10', 'v10': 'v10'})

        ds = xr.open_mfdataset(path_wind, chunks=-1)

        # --- longitude convention ---
        lon_vals = ds[nv['lon']].values
        if lon_vals.min() < 0 and State.lon_unit == '0_360':
            ds = ds.assign_coords({nv['lon']: (nv['lon'], lon_vals % 360)})
        elif lon_vals.min() >= 0 and State.lon_unit == '-180_180':
            ds = ds.assign_coords({nv['lon']: (nv['lon'], (lon_vals + 180) % 360 - 180)})
        ds = ds.sortby(nv['lon'])

        # --- compute wind stress on the wind file's native grid ---
        u10 = ds[nv['u10']].values   # (ntime, nlat, nlon)
        v10 = ds[nv['v10']].values
        wspd   = np.sqrt(u10**2 + v10**2)

        if getattr(config.MOD, 'Cd_wind_formula', None) is not None and config.MOD.Cd_wind_formula.lower()=='large & pond':
            Cd = wspd*0
            Cd[wspd<5] = 1.1 * 1e-3
            Cd[wspd>=5] = (0.49 + 0.065*wspd[wspd>=5]) * 1e-3
            Cd[wspd>=25] = 2.0 * 1e-3

        taux_w = rho_air * Cd * wspd * u10
        tauy_w = rho_air * Cd * wspd * v10

        wind_lons  = ds[nv['lon']].values
        wind_lats  = ds[nv['lat']].values
        wind_times = ds[nv['time']].values.astype('datetime64[s]')
        ds.close()

        # --- wind update cadence ---
        wind_dt = getattr(config.MOD, 'wind_timestep', 3600)  # seconds
        wind_dt = max(wind_dt, self.dt)  # at least one model step
        total_seconds = (config.EXP.final_date - config.EXP.init_date).total_seconds()
        nt_wind = 1 + int(total_seconds // wind_dt)
        self.wind_dt = wind_dt

        # --- spatial interpolation to model h-grid ---
        from scipy.interpolate import RegularGridInterpolator
        taux_hgrid = np.zeros((nt_wind, State.ny, State.nx))
        tauy_hgrid = np.zeros((nt_wind, State.ny, State.nx))

        # wind sample timestamps as datetime64[s]
        wind_sample_times = np.array(
            [config.EXP.init_date + timedelta(seconds=i * wind_dt) for i in range(nt_wind)],
            dtype='datetime64[s]')

        # ensure lats are monotonically increasing for RegularGridInterpolator
        if wind_lats[0] > wind_lats[-1]:
            wind_lats  = wind_lats[::-1]
            taux_w     = taux_w[:, ::-1, :]
            tauy_w     = tauy_w[:, ::-1, :]

        for it in range(nt_wind):
            # --- temporal linear interpolation ---
            t_model = wind_sample_times[it]
            idx_r = int(np.searchsorted(wind_times, t_model))
            idx_r = np.clip(idx_r, 1, len(wind_times) - 1)
            idx_l = idx_r - 1

            dt_wind   = float((wind_times[idx_r] - wind_times[idx_l]).astype('float64'))
            dt_interp = float((t_model          - wind_times[idx_l]).astype('float64'))
            alpha = (dt_interp / dt_wind) if dt_wind > 0 else 0.
            alpha = np.clip(alpha, 0., 1.)

            taux_t = (1 - alpha) * taux_w[idx_l] + alpha * taux_w[idx_r]
            tauy_t = (1 - alpha) * tauy_w[idx_l] + alpha * tauy_w[idx_r]

            # --- spatial interpolation ---
            pts = np.stack([State.lat.ravel(), State.lon.ravel()], axis=-1)

            fi_taux = RegularGridInterpolator(
                (wind_lats, wind_lons), taux_t,
                method='linear', bounds_error=False, fill_value=0.)
            fi_tauy = RegularGridInterpolator(
                (wind_lats, wind_lons), tauy_t,
                method='linear', bounds_error=False, fill_value=0.)

            taux_hgrid[it] = fi_taux(pts).reshape(State.ny, State.nx)
            tauy_hgrid[it] = fi_tauy(pts).reshape(State.ny, State.nx)

        # --- interpolate from h-grid to u-grid (ny, nx+1) and v-grid (ny+1, nx) ---
        taux_u = np.zeros((nt_wind, State.ny,     State.nx + 1))
        tauy_v = np.zeros((nt_wind, State.ny + 1, State.nx    ))

        taux_u[:, :, 1:State.nx] = 0.5 * (taux_hgrid[:, :, :-1] + taux_hgrid[:, :, 1:])
        taux_u[:, :, 0]          = taux_hgrid[:, :,  0]
        taux_u[:, :, State.nx]   = taux_hgrid[:, :, -1]

        tauy_v[:, 1:State.ny, :] = 0.5 * (tauy_hgrid[:, :-1, :] + tauy_hgrid[:, 1:, :])
        tauy_v[:, 0,          :] = tauy_hgrid[:,  0, :]
        tauy_v[:, State.ny,   :] = tauy_hgrid[:, -1, :]

        self.taux_wind = taux_u   # (nt_wind, ny, nx+1)
        self.tauy_wind = tauy_v   # (nt_wind, ny+1, nx)
        print(f'  - Wind forcing loaded: {path_wind}  ({nt_wind} wind snapshots, wind_timestep={wind_dt}s)')
        print(f'    max |taux|={np.nanmax(np.abs(taux_u)):.3e}  '
              f'max |tauy|={np.nanmax(np.abs(tauy_v)):.3e}')

    def _get_wind_stress(self, t):
        """
        Return (taux, tauy) as jnp arrays for time t (seconds from init).
        Returns (None, None) when no wind file was loaded.
        """
        if self.taux_wind is None:
            return None, None
        it = int(round(t / self.wind_dt)) if self.wind_dt > 0 else 0
        it = int(np.clip(it, 0, len(self.taux_wind) - 1))
        taux = (1 - self.sponge_u) * jnp.asarray(self.taux_wind[it], dtype=self.dtype)  # (ny, nx+1)
        tauy = (1 - self.sponge_v) * jnp.asarray(self.tauy_wind[it], dtype=self.dtype)  # (ny+1, nx)
        return taux, tauy

    def _sync_layers_from_surface(self, State):
        """Project 2D surface fields into per-layer arrays (layer 0 gets the perturbation)."""
        ssh = jnp.asarray(State.getvar(name_var=self.name_var['SSH']), dtype=self.dtype)
        u   = jnp.asarray(State.getvar(name_var=self.name_var['U']),   dtype=self.dtype)
        v   = jnp.asarray(State.getvar(name_var=self.name_var['V']),   dtype=self.dtype)
        State.var['h_layers'] = jnp.zeros((self.nl, self.ny, self.nx), dtype=self.dtype).at[0].set(ssh)
        State.var['u_layers'] = jnp.zeros((self.nl, self.ny, self.nx+1), dtype=self.dtype).at[0].set(u)
        State.var['v_layers'] = jnp.zeros((self.nl, self.ny+1, self.nx), dtype=self.dtype).at[0].set(v)

    def _diagnose_surface(self, State):
        """Set 2D surface SSH/U/V from per-layer arrays."""
        State.setvar(State.var['h_layers'].sum(axis=0), name_var=self.name_var['SSH'])
        State.setvar(State.var['u_layers'][0],          name_var=self.name_var['U'])
        State.setvar(State.var['v_layers'][0],          name_var=self.name_var['V'])

    def init(self, State, t0=0):

        if type(self.init_from_bc)==dict:
            if 'SSH' in self.init_from_bc:
                u0 = self.bc['U'][t0]
                v0 = self.bc['V'][t0]
                ssh0 = self.bc['SSH'][t0]
                State.setvar(u0, self.name_var['U'])
                State.setvar(v0, self.name_var['V'])
                State.setvar(ssh0, self.name_var['SSH'])
            for name in self.init_from_bc:
                if self.init_from_bc[name] and t0 in self.bc[name]:
                    State.setvar(self.bc[name][t0], self.name_var[name])
        elif self.init_from_bc:
            for name in self.name_var: 
                if t0 in self.bc[name]:
                     State.setvar(self.bc[name][t0], self.name_var[name])

        # In QG mode the velocity field must be geostrophically balanced with SSH.
        # BC files typically only contain SSH, so u/v would otherwise be zero.
        if self._is_qg_class:
            ssh = State.getvar(name_var=self.name_var['SSH'])
            u0, v0 = self.ssh2uv(ssh)
            State.setvar(u0, self.name_var['U'])
            State.setvar(v0, self.name_var['V'])

        # For nl>1, project surface fields into per-layer arrays
        if self.nl > 1:
            self._sync_layers_from_surface(State)

        self._set_land_nan(State)

    def save_output(self,State,present_date,name_var=None,t=None):

        State0 = State.copy()

        _name_var = [self.name_var['U'], self.name_var['V'], self.name_var['SSH']] if name_var is None else name_var

        if name_var is None and self.advect_tracer:
            for name in self.tracer_names:
                _name_var.append(self.name_var[name])

        if 'H' in self.name_params:
            State0.var['H'] = self._H_control_to_total_state(State.params['H'])
            State0.var['H_control'] = +State.params['H']
            _name_var += ['H', 'H_control']

        if self.mdt is not None:
            State0.var['mdt'] = np.array(self.mdt[0, 0]).T
            State0.var['mdu'] = np.array(self.mdu[0, 0]).T
            State0.var['mdv'] = np.array(self.mdv[0, 0]).T
            _name_var += ['mdt', 'mdu', 'mdv']
        
        if 'h_wind' in self.name_params:
            State0.var['h_wind'] = +State.params['h_wind']
            _name_var += ['h_wind']
        
        if 'wind_strength' in self.name_params:
            State0.var['wind_strength'] = State.params['wind_strength']
            _name_var += ['wind_strength']
    
        State0.save_output(present_date, name_var=_name_var)
    
    def set_bc(self,time_bc,var_bc=None,t_bc=None,**kwargs):

        # If var_bc not provided but we have loaded BC fields, interpolate them
        if var_bc is None and self._bc_fields is not None:
            var_bc = self._bc_fields.interp(time_bc)
        
        if var_bc is None:
            return

        # Use float model times as dict keys if provided (needed by init and
        # _apply_bc); fall back to datetime keys when called directly.
        time_keys = t_bc if t_bc is not None else time_bc

        for i,t in enumerate(time_keys):
            ssh_bc_t = +var_bc['SSH'][i]
            # Remove nan
            ssh_bc_t[np.isnan(ssh_bc_t)] = 0.

            if 'U' in var_bc and 'V' in var_bc:
                u = +var_bc['U'][i]
                v = +var_bc['V'][i]
                # Remove nan
                u[np.isnan(u)] = 0.
                v[np.isnan(v)] = 0.
            else:
                u, v = self.ssh2uv(ssh_bc_t)
                if 'U' in var_bc:
                    u = +var_bc['U'][i]
                    u[np.isnan(u)] = 0.
                if 'V' in var_bc:
                    v = +var_bc['V'][i]
                    v[np.isnan(v)] = 0.

            # Fill bc dictionnary
            self.bc['U'][t] = u
            self.bc['V'][t] = v
            self.bc['SSH'][t] = ssh_bc_t

        # Tracer BCs
        for name in self.tracer_names:
            if name in var_bc:
                for i, t in enumerate(time_keys):
                    c_bc_t = +var_bc[name][i]
                    # Extrapolate NaN land cells from nearest ocean value
                    # (mirrors Model_qg1l_jax.set_bc — prevents spurious ~0 patches
                    # from leaking into the ocean via advection/sponge relaxation)
                    if np.any(np.isnan(c_bc_t)):
                        c_bc_t = _fill_nan_nearest(c_bc_t)
                    self.bc[name][t] = c_bc_t

        self.bc_time = np.asarray(time_keys)

    def _apply_bc(self,t0,t1):
        
        u_b = jnp.zeros((self.ny,self.nx+1))
        v_b = jnp.zeros((self.ny+1,self.nx))
        ssh_b = jnp.zeros((self.ny,self.nx))

        if 'SSH' not in self.bc:
            return u_b,v_b,ssh_b
        elif len(self.bc['SSH'].keys())==0:
             return u_b,v_b,ssh_b
        elif t0 not in self.bc_time:
            # Find closest time
            idx_closest = np.argmin(np.abs(self.bc_time-t0))
            t0 = self.bc_time[idx_closest]

        u_b = self.bc['U'][t0]
        v_b = self.bc['V'][t0]
        ssh_b = self.bc['SSH'][t0]

        return u_b, v_b, ssh_b
    
    def _apply_bc_tracer(self, t0, t1):
        """Return stacked tracer BC array of shape (n_trac, ny, nx)."""
        if not self.advect_tracer:
            return None
        c_b_list = []
        for name in self.tracer_names:
            cb_i = np.zeros((self.ny, self.nx), dtype='float32')
            if name in self.bc and len(self.bc[name]) > 0:
                if t0 in self.bc[name]:
                    cb_i = self.bc[name][t0]
                elif hasattr(self, 'bc_time') and len(self.bc_time) > 0:
                    idx = np.argmin(np.abs(self.bc_time - t0))
                    cb_i = self.bc[name][self.bc_time[idx]]
            c_b_list.append(jnp.asarray(cb_i, dtype=self.dtype))
        return jnp.stack(c_b_list, axis=0)  # (n_trac, ny, nx)
    
    def prepare_scan_boundary_conditions(self, times, nstep):
        """Pre-stack QGSW forcing for each checkpoint and internal chunk."""
        chunk_specs = []
        step_done = 0
        while step_done < nstep:
            n_chunk = (
                min(nstep - step_done, self.max_nstep)
                if self.max_nstep > 0
                else nstep - step_done
            )
            chunk_specs.append((step_done, n_chunk))
            step_done += n_chunk

        intervals = []
        for interval_time in np.asarray(times):
            t0 = interval_time.item() if hasattr(interval_time, 'item') else interval_time
            chunks = []
            for step_offset, n_chunk in chunk_specs:
                t_chunk = t0 + step_offset * self.dt
                t_end = t_chunk + n_chunk * self.dt
                u_b, v_b, h_b = self._apply_bc(t_chunk, t_end)
                c_b = (
                    self._apply_bc_tracer(t_chunk, t_end)
                    if self.advect_tracer
                    else jnp.zeros((0, self.ny, self.nx), dtype=self.dtype)
                )
                taux, tauy = self._get_wind_stress(t_chunk)
                if taux is None:
                    constant_taux = jnp.asarray(
                        getattr(self.model, 'taux', 0.0),
                        dtype=self.dtype,
                    )
                    taux = jnp.zeros(
                        (self.ny, self.nx + 1), dtype=self.dtype
                    )
                    if constant_taux.ndim == 0:
                        taux = jnp.full_like(taux, constant_taux)
                    else:
                        taux = taux.at[:, 1:-1].set(constant_taux.T)
                if tauy is None:
                    constant_tauy = jnp.asarray(
                        getattr(self.model, 'tauy', 0.0),
                        dtype=self.dtype,
                    )
                    tauy = jnp.zeros(
                        (self.ny + 1, self.nx), dtype=self.dtype
                    )
                    if constant_tauy.ndim == 0:
                        tauy = jnp.full_like(tauy, constant_tauy)
                    else:
                        tauy = tauy.at[1:-1, :].set(constant_tauy.T)
                chunks.append(tuple(
                    jnp.asarray(value, dtype=self.dtype)
                    for value in (u_b, v_b, h_b, c_b, taux, tauy)
                ))
            intervals.append(tuple(
                jnp.stack(values, axis=0)
                for values in zip(*chunks)
            ))

        return tuple(
            jnp.stack(values, axis=0)
            for values in zip(*intervals)
        )

    def _chunk_forcing(self, forcing, chunk_index, t_chunk, n_chunk):
        """Select pre-stacked scan forcing or use the historical lookup."""
        if forcing is not None:
            return tuple(value[chunk_index] for value in forcing)

        t_end = t_chunk + n_chunk * self.dt
        u_b, v_b, h_b = self._apply_bc(t_chunk, t_end)
        c_b = (
            self._apply_bc_tracer(t_chunk, t_end)
            if self.advect_tracer
            else None
        )
        taux, tauy = self._get_wind_stress(t_chunk)
        return u_b, v_b, h_b, c_b, taux, tauy

    def ssh2uv(self, ssh):
        """Geostrophic SSH → (u, v) model."""
        if self._is_qg_class:
            # Delegate to the model's G operator so that the same geostrophic
            # balance formula, f0, dx/dy scalars, and ocean masks are used.
            # Fill NaN (land), convert to SW (1, 1, nx, ny); g_prime (nl,1,1)
            # broadcasts to (1, nl, nx, ny).
            ssh_sw = jnp.array(
                np.where(np.isnan(ssh), 0.0, ssh).T
            )[np.newaxis, np.newaxis]
            p_i = self.model.g_prime.astype(self.model.dtype) * ssh_sw
            p   = self.model.hgrid_pressure_to_wgrid(p_i)
            u_sw, v_sw, _ = self.model.G(p, p_i=p_i)
            # G returns u_stored = u_phys * dx_ugrid, v_stored = v_phys * dy_vgrid
            # (same internal convention as set_input_uvh / get_physical_uvh).
            # Divide to recover physical velocities for State convention.
            u_phys = u_sw / self.model.dx_ugrid
            v_phys = v_sw / self.model.dy_vgrid
            # Keep tangential velocities zero on the remaining outer edges.
            u_phys = u_phys.at[..., :, -1].set(0.0)   # north tangential
            v_phys = v_phys.at[..., -1, :].set(0.0)   # east  tangential
            # Convert SW (1, nl, nx+1, ny) → State (ny, nx+1), likewise v.
            return np.array(u_phys[0, 0].T), np.array(v_phys[0, 0].T)
        else:
            # SW mode: finite-difference geostrophy with ocean mask.
            _ssh = np.where(np.isnan(ssh), 0.0, ssh)
            _ssh = np.pad(_ssh, pad_width=((1,0),(1,0)), mode='edge')
            _u = -self.g / self.f_on_u * np.diff(_ssh, axis=0) / self.dy_on_u
            _v = self.g / self.f_on_v * np.diff(_ssh, axis=1) / self.dx_on_v
            if hasattr(self, 'model'):
                _u = np.where(self.model.masks.u[0, 0].T, _u, 0.0)
                _v = np.where(self.model.masks.v[0, 0].T, _v, 0.0)
            return _u, _v

    def ssh2uv_tgl(self, ssh, dssh):
        """Tangent-linear model using JAX forward-mode differentiation."""
        _, (dug, dvg) = jax.jvp(
            lambda s: self.ssh2uv(s),
            (ssh,),
            (dssh,)
        )
        return dug, dvg

    def ssh2uv_adj(self, ssh, adug, advg):
        """Adjoint model using JAX reverse-mode differentiation."""
        _, vjp_fun = jax.vjp(lambda s: self.ssh2uv(s), ssh)
        (adssh,) = vjp_fun((adug, advg))
        return adssh
    
    def h_on_u(self, h, pad=((1,0),(1,0))):
        """Interpolate h to u points."""
        _h = jnp.pad(h, pad_width=pad, mode='edge')
        h_on_u = 0.5*(_h[1:,:] + _h[:-1,:])
        return h_on_u
    
    def h_on_v(self, h, pad=((1,0),(1,0))):
        """Interpolate h to v points."""
        _h = jnp.pad(h, pad_width=pad, mode='edge')
        h_on_v = 0.5*(_h[:,1:] + _h[:,:-1])
        return h_on_v

    def _qg_balanced_sponge_target(self, u, v, h, u_b_sw, v_b_sw, h_b_sw):
        """Return physical sponge targets from the auxiliary QG projector."""
        mp = self.model_proj

        if self.nl == 1:
            u_b_phys = jnp.expand_dims(u_b_sw, axis=(0, 1))
            v_b_phys = jnp.expand_dims(v_b_sw, axis=(0, 1))
            h_b_phys = jnp.expand_dims(h_b_sw, axis=(0, 1))
        else:
            u_b_phys = jnp.expand_dims(u_b_sw, axis=0)
            v_b_phys = jnp.expand_dims(v_b_sw, axis=0)
            h_b_phys = jnp.expand_dims(h_b_sw, axis=0)

        u_clean = jnp.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        v_clean = jnp.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        h_clean = jnp.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        u_b_phys = jnp.nan_to_num(u_b_phys, nan=0.0, posinf=0.0, neginf=0.0)
        v_b_phys = jnp.nan_to_num(v_b_phys, nan=0.0, posinf=0.0, neginf=0.0)
        h_b_phys = jnp.nan_to_num(h_b_phys, nan=0.0, posinf=0.0, neginf=0.0)

        # Use the projector's standard input path so land/coast masks are applied
        # before Q(u, v, h) forms coastal finite differences.
        u_in, v_in, h_in = mp.set_input_uvh(u_clean, v_clean, h_clean)
        _, _, h_b_in = mp.set_input_uvh(u_b_phys, v_b_phys, h_b_phys)

        pb, pb_i, qb = mp._compute_qg_background(h_b_in)
        u_proj, v_proj, h_proj = mp.project_qg(
            u_in, v_in, h_in, pb=pb, pb_i=pb_i, qb=qb,
        )
        u_proj_phys, v_proj_phys, h_proj_phys = mp.get_physical_uvh(
            u_proj, v_proj, h_proj, numpy=False,
        )

        max_factor = 2.0
        u_scale = jnp.maximum(jnp.maximum(jnp.max(jnp.abs(u_clean)), jnp.max(jnp.abs(u_b_phys))), 1.0)
        v_scale = jnp.maximum(jnp.maximum(jnp.max(jnp.abs(v_clean)), jnp.max(jnp.abs(v_b_phys))), 1.0)
        h_scale = jnp.maximum(jnp.maximum(jnp.max(jnp.abs(h_clean)), jnp.max(jnp.abs(h_b_phys))), 1.0)

        u_ok = jnp.isfinite(u_proj_phys) & (jnp.abs(u_proj_phys) <= max_factor * u_scale)
        v_ok = jnp.isfinite(v_proj_phys) & (jnp.abs(v_proj_phys) <= max_factor * v_scale)
        h_ok = jnp.isfinite(h_proj_phys) & (jnp.abs(h_proj_phys) <= max_factor * h_scale)
        u_proj_phys = jnp.nan_to_num(u_proj_phys, nan=0.0, posinf=0.0, neginf=0.0)
        v_proj_phys = jnp.nan_to_num(v_proj_phys, nan=0.0, posinf=0.0, neginf=0.0)
        h_proj_phys = jnp.nan_to_num(h_proj_phys, nan=0.0, posinf=0.0, neginf=0.0)
        u_target = jnp.where(u_ok, u_proj_phys, u_b_phys)
        v_target = jnp.where(v_ok, v_proj_phys, v_b_phys)
        h_target = jnp.where(h_ok, h_proj_phys, h_b_phys)

        if self.nl == 1:
            return u_target[0, 0], v_target[0, 0], h_target[0, 0]
        return u_target[0], v_target[0], h_target[0]
    
    def jstep_core(self, t, u0, v0, h0, H, Fu, Fv, Fh, u_b, v_b, h_b,
                taux=None, tauy=None, h_wind=None, wind_strength=None, nstep=1):
        """
        taux: wind stress on u-grid, shape (ny, nx+1) in State convention, or None.
        tauy: wind stress on v-grid, shape (ny+1, nx) in State convention, or None.
        They are converted to SW model convention (nx-1, ny) and (nx, ny-1) internally.
        Fu/Fv/Fh: 2D forcing in State convention — converted to per-layer for nl>1.
        u_b/v_b/h_b: 2D BCs in State convention — converted to per-layer for nl>1.
        h_wind: mixed-layer depth perturbation (nx, ny) in SW convention, or None.
        """
        u, v, h = u0, v0, h0

        # Add MDT (surface only for nl>1)
        if self.mdt is not None:
            if self.nl > 1:
                h = h.at[:, 0:1, :, :].add(self.mdt)
                u = u.at[:, 0:1, :, :].add(self.mdu)
                v = v.at[:, 0:1, :, :].add(self.mdv)
            else:
                h = h + self.mdt
                u = u + self.mdu
                v = v + self.mdv

        # Convert wind stress from State (ny, nx+1/ny+1) to SW model (nx-1/nx, ny/ny-1)
        # SW model expects taux on interior u-faces: (nx-1, ny)
        #                  tauy on interior v-faces: (nx, ny-1)
        _taux = taux[:, 1:-1].T if taux is not None else None   # (nx-1, ny)
        _tauy = tauy[1:-1, :].T if tauy is not None else None   # (nx, ny-1)

        # Convert BCs and forcing from State convention to SW convention
        if self.nl > 1:
            # Per-layer: put 2D BC/forcing in layer 0, zeros in other layers
            u_b_sw = jnp.zeros((self.nl, self.nx+1, self.ny), dtype=self.dtype).at[0].set(u_b.T)
            v_b_sw = jnp.zeros((self.nl, self.nx, self.ny+1), dtype=self.dtype).at[0].set(v_b.T)
            h_b_sw = jnp.zeros((self.nl, self.nx, self.ny),   dtype=self.dtype).at[0].set(h_b.T)
            Fu_sw = jnp.zeros((self.nl, self.nx+1, self.ny), dtype=self.dtype).at[0].set(Fu.T) / (3600 * 24)
            Fv_sw = jnp.zeros((self.nl, self.nx, self.ny+1), dtype=self.dtype).at[0].set(Fv.T) / (3600 * 24)
            Fh_sw = jnp.zeros((self.nl, self.nx, self.ny),   dtype=self.dtype).at[0].set(Fh.T) / (3600 * 24)
        else:
            u_b_sw = u_b.T
            v_b_sw = v_b.T
            h_b_sw = h_b.T
            Fu_sw = Fu.T / (3600 * 24)
            Fv_sw = Fv.T / (3600 * 24)
            Fh_sw = Fh.T / (3600 * 24)

        # Add MDT to sponge BC target: inside model_step the state already has MDT
        # added, so the sponge must relax toward (anomaly_BC + MDT), not anomaly_BC.
        if self.mdt is not None:
            if self.nl > 1:
                u_b_sw = u_b_sw.at[0].add(self.mdu[0, 0])
                v_b_sw = v_b_sw.at[0].add(self.mdv[0, 0])
                h_b_sw = h_b_sw.at[0].add(self.mdt[0, 0])
            else:
                u_b_sw = u_b_sw + self.mdu[0, 0]
                v_b_sw = v_b_sw + self.mdv[0, 0]
                h_b_sw = h_b_sw + self.mdt[0, 0]

        # QG-balanced sponge target: replace the external sponge target with a
        # masked, QG-projected field from the current MDT-included state.
        if self.qg_balanced_sponge_bc and self.model_proj is not None:
            u_b_sw, v_b_sw, h_b_sw = self._qg_balanced_sponge_target(
                u, v, h, u_b_sw, v_b_sw, h_b_sw,
            )

        H_model = self._H_control_to_model_increment(H)

        # Step
        if self._is_qg_class:
            u1, v1, h1 = self.model_step(
                u, v, h, H=H_model, nstep=nstep,
                u_b=u_b_sw, v_b=v_b_sw, h_b=h_b_sw,
                Fu=Fu_sw, Fv=Fv_sw, Fh=Fh_sw,
            )
        else:
            u1, v1, h1 = self.model_step(
                u, v, h, H=H_model, nstep=nstep,
                u_b=u_b_sw, v_b=v_b_sw, h_b=h_b_sw,
                Fu=Fu_sw, Fv=Fv_sw, Fh=Fh_sw,
                taux=_taux, tauy=_tauy,
                h_wind=h_wind,
                wind_strength=wind_strength,
            )

        # Remove MDT (surface only for nl>1)
        if self.mdt is not None:
            if self.nl > 1:
                h1 = h1.at[:, 0:1, :, :].add(-self.mdt)
                u1 = u1.at[:, 0:1, :, :].add(-self.mdu)
                v1 = v1.at[:, 0:1, :, :].add(-self.mdv)
            else:
                h1 = h1 - self.mdt
                u1 = u1 - self.mdu
                v1 = v1 - self.mdv

        return u1, v1, h1
    
    def jstep_core_trac(self, t, u0, v0, h0, c0, H, Fu, Fv, Fh, Fc,
                        u_b, v_b, h_b, c_b,
                        taux=None, tauy=None, h_wind=None, wind_strength=None, nstep=1):
        """
        Joint single-step for (u, v, h) + passive tracer c.

        c0 : (n_trac, ny, nx)  tracers in State convention (ny, nx per tracer)
        c_b: (n_trac, ny, nx)  tracer boundary condition
        Fc : (n_trac, ny, nx)  per-tracer external forcing (same convention as Fh)
        Returns (u1, v1, h1, c1) where c1 has shape (n_trac, ny, nx).
        """
        u, v, h = u0, v0, h0

        # Add MDT (surface only for nl>1)
        if self.mdt is not None:
            if self.nl > 1:
                h = h.at[:, 0:1, :, :].add(self.mdt)
                u = u.at[:, 0:1, :, :].add(self.mdu)
                v = v.at[:, 0:1, :, :].add(self.mdv)
            else:
                h = h + self.mdt
                u = u + self.mdu
                v = v + self.mdv

        # Wind stress: State (ny, nx+1/ny+1) → SW model (nx-1, ny/ny-1)
        _taux = taux[:, 1:-1].T if taux is not None else None
        _tauy = tauy[1:-1, :].T if tauy is not None else None

        # Convert BCs and forcing State → SW convention
        # Tracer: (n_trac, ny, nx) → (1, n_trac, nx, ny)
        c_sw   = jnp.expand_dims(c0.transpose(0, 2, 1), axis=0)
        c_b_sw = jnp.expand_dims(c_b.transpose(0, 2, 1), axis=0)
        Fc_sw  = jnp.expand_dims(Fc.transpose(0, 2, 1), axis=0) / (3600 * 24)

        if self.nl > 1:
            u_b_sw = jnp.zeros((self.nl, self.nx+1, self.ny), dtype=self.dtype).at[0].set(u_b.T)
            v_b_sw = jnp.zeros((self.nl, self.nx, self.ny+1), dtype=self.dtype).at[0].set(v_b.T)
            h_b_sw = jnp.zeros((self.nl, self.nx, self.ny),   dtype=self.dtype).at[0].set(h_b.T)
            Fu_sw = jnp.zeros((self.nl, self.nx+1, self.ny), dtype=self.dtype).at[0].set(Fu.T) / (3600 * 24)
            Fv_sw = jnp.zeros((self.nl, self.nx, self.ny+1), dtype=self.dtype).at[0].set(Fv.T) / (3600 * 24)
            Fh_sw = jnp.zeros((self.nl, self.nx, self.ny),   dtype=self.dtype).at[0].set(Fh.T) / (3600 * 24)
        else:
            u_b_sw = u_b.T
            v_b_sw = v_b.T
            h_b_sw = h_b.T
            Fu_sw = Fu.T / (3600 * 24)
            Fv_sw = Fv.T / (3600 * 24)
            Fh_sw = Fh.T / (3600 * 24)

        # Add MDT to sponge BC target: inside model.step_with_tracer the state
        # already has MDT added, so the sponge must relax toward
        # (anomaly_BC + MDT), not anomaly_BC.
        if self.mdt is not None:
            if self.nl > 1:
                u_b_sw = u_b_sw.at[0].add(self.mdu[0, 0])
                v_b_sw = v_b_sw.at[0].add(self.mdv[0, 0])
                h_b_sw = h_b_sw.at[0].add(self.mdt[0, 0])
            else:
                u_b_sw = u_b_sw + self.mdu[0, 0]
                v_b_sw = v_b_sw + self.mdv[0, 0]
                h_b_sw = h_b_sw + self.mdt[0, 0]

        # QG-balanced sponge target — same logic as in jstep_core.
        if self.qg_balanced_sponge_bc and self.model_proj is not None:
            u_b_sw, v_b_sw, h_b_sw = self._qg_balanced_sponge_target(
                u, v, h, u_b_sw, v_b_sw, h_b_sw,
            )

        H_model = self._H_control_to_model_increment(H)

        # Joint step (u, v, h, c)
        u1, v1, h1, c1 = self.model.step_with_tracer(
            u, v, h, c_sw,
            H=H_model, nstep=nstep,
            u_b=u_b_sw, v_b=v_b_sw, h_b=h_b_sw, c_b=c_b_sw,
            Fu=Fu_sw, Fv=Fv_sw, Fh=Fh_sw, Fc=Fc_sw,
            taux=_taux, tauy=_tauy,
            h_wind=h_wind,
            wind_strength=wind_strength,
        )

        # Remove MDT (surface only for nl>1)
        if self.mdt is not None:
            if self.nl > 1:
                h1 = h1.at[:, 0:1, :, :].add(-self.mdt)
                u1 = u1.at[:, 0:1, :, :].add(-self.mdu)
                v1 = v1.at[:, 0:1, :, :].add(-self.mdv)
            else:
                h1 = h1 - self.mdt
                u1 = u1 - self.mdu
                v1 = v1 - self.mdv

        # c1: (1, n_trac, nx, ny) → (n_trac, ny, nx)
        c1_state = c1[0].transpose(0, 2, 1)

        return u1, v1, h1, c1_state
    
    def jstep(self, t, u0, v0, h0, H, Fu, Fv, Fh, u_b, v_b, h_b,
             taux=None, tauy=None, h_wind=None, wind_strength=None, nstep=1):
        return self.jstep_core(
            t, u0, v0, h0, H, Fu, Fv, Fh,
            u_b, v_b, h_b,
            taux=taux, tauy=tauy,
            h_wind=h_wind,
            wind_strength=wind_strength,
            nstep=nstep,
        )
    
    def jstep_tgl(self, t, du0, dv0, dh0, dH, dFu, dFv, dFh, 
                u0, v0, h0, H, Fu, Fv, Fh, 
                u_b, v_b, h_b, taux=None, tauy=None,
                h_wind=None, dh_wind=None,
                wind_strength=None, dwind_strength=None,
                du_b=None, dv_b=None, dh_b=None, nstep=1):

        # Define a partial function that fixes constant parameters
        # taux/tauy are prescribed forcings: closed over, not differentiated
        # u_b/v_b/h_b are now in the differentiable tuple
        f = lambda u, v, h, H, Fu, Fv, Fh, hw, ws, ub, vb, hb: self.jstep_core_jit(
            t, u, v, h, H, Fu, Fv, Fh, ub, vb, hb,
            taux=taux, tauy=tauy,
            h_wind=hw,
            wind_strength=ws,
            nstep=nstep,
        )

        # Default zero tangents for BCs when not optimized
        if du_b is None:
            du_b = jnp.zeros_like(u_b)
        if dv_b is None:
            dv_b = jnp.zeros_like(v_b)
        if dh_b is None:
            dh_b = jnp.zeros_like(h_b)

        # JVP (forward mode)
        (u1, v1, h1), (du1, dv1, dh1) = jax.jvp(
            f,
            (u0, v0, h0, H, Fu, Fv, Fh, h_wind, wind_strength, u_b, v_b, h_b),
            (du0, dv0, dh0, dH, dFu, dFv, dFh, dh_wind, dwind_strength, du_b, dv_b, dh_b)
        )

        return du1, dv1, dh1
    
    def jstep_adj(self, t, adu1, adv1, adh1, adH, adFu, adFv, adFh,
                u0, v0, h0, H, Fu, Fv, Fh, u_b, v_b, h_b,
                taux=None, tauy=None,
                h_wind=None, adh_wind=None,
                wind_strength=None, adwind_strength=None,
                adu_b=None, adv_b=None, adh_b=None, nstep=1):

        # taux/tauy are prescribed forcings: closed over, not differentiated
        # u_b/v_b/h_b are now in the differentiable tuple
        f = lambda u, v, h, H, Fu, Fv, Fh, hw, ws, ub, vb, hb: self.jstep_core_jit(
            t, u, v, h, H, Fu, Fv, Fh, ub, vb, hb,
            taux=taux, tauy=tauy,
            h_wind=hw,
            wind_strength=ws,
            nstep=nstep,
        )

        # Build the VJP function
        (u1, v1, h1), vjp_fun = jax.vjp(f, u0, v0, h0, H, Fu, Fv, Fh, h_wind, wind_strength, u_b, v_b, h_b)

        # Apply adjoints at output
        (adu0, adv0, adh0, _adH, _adFu, _adFv, _adFh,
         _adh_wind, _adwind_strength, _adu_b, _adv_b, _adh_b) = vjp_fun((adu1, adv1, adh1))
        if adH is not None:
            adH += _adH
        adFu += _adFu
        adFv += _adFv
        adFh += _adFh
        if adh_wind is not None:
            adh_wind += _adh_wind
        if adwind_strength is not None:
            adwind_strength += _adwind_strength
        if adu_b is not None:
            adu_b += _adu_b
        if adv_b is not None:
            adv_b += _adv_b
        if adh_b is not None:
            adh_b += _adh_b
        
        return adu0, adv0, adh0, adH, adFu, adFv, adFh, adh_wind, adwind_strength, adu_b, adv_b, adh_b

    def jstep_tgl_trac(self, t, du0, dv0, dh0, dc0, dH, dFu, dFv, dFh, dFc,
                       u0, v0, h0, c0, H, Fu, Fv, Fh, Fc,
                       u_b, v_b, h_b, c_b,
                       taux=None, tauy=None,
                       h_wind=None, dh_wind=None,
                       wind_strength=None, dwind_strength=None,
                       du_b=None, dv_b=None, dh_b=None, dc_b=None, nstep=1):
        """TGL for joint (u, v, h, c) step via JAX forward-mode AD."""
        f = lambda u, v, h, c, H, Fu, Fv, Fh, Fc, hw, ws, ub, vb, hb, cb: \
            self.jstep_core_trac_jit(
                t, u, v, h, c, H, Fu, Fv, Fh, Fc, ub, vb, hb, cb,
                taux=taux, tauy=tauy, h_wind=hw, wind_strength=ws, nstep=nstep)

        if du_b is None: du_b = jnp.zeros_like(u_b)
        if dv_b is None: dv_b = jnp.zeros_like(v_b)
        if dh_b is None: dh_b = jnp.zeros_like(h_b)
        if dc_b is None: dc_b = jnp.zeros_like(c_b)
        if dFc is None:  dFc  = jnp.zeros_like(Fc)

        (u1, v1, h1, c1), (du1, dv1, dh1, dc1) = jax.jvp(
            f,
            (u0, v0, h0, c0, H, Fu, Fv, Fh, Fc, h_wind, wind_strength, u_b, v_b, h_b, c_b),
            (du0, dv0, dh0, dc0, dH, dFu, dFv, dFh, dFc, dh_wind, dwind_strength, du_b, dv_b, dh_b, dc_b),
        )
        return du1, dv1, dh1, dc1

    def jstep_adj_trac(self, t, adu1, adv1, adh1, adc1, adH, adFu, adFv, adFh, adFc,
                       u0, v0, h0, c0, H, Fu, Fv, Fh, Fc,
                       u_b, v_b, h_b, c_b,
                       taux=None, tauy=None,
                       h_wind=None, adh_wind=None,
                       wind_strength=None, adwind_strength=None,
                       adu_b=None, adv_b=None, adh_b=None, adc_b=None, nstep=1):
        """ADJ for joint (u, v, h, c) step via JAX reverse-mode AD."""
        f = lambda u, v, h, c, H, Fu, Fv, Fh, Fc, hw, ws, ub, vb, hb, cb: \
            self.jstep_core_trac_jit(
                t, u, v, h, c, H, Fu, Fv, Fh, Fc, ub, vb, hb, cb,
                taux=taux, tauy=tauy, h_wind=hw, wind_strength=ws, nstep=nstep)

        (u1, v1, h1, c1), vjp_fun = jax.vjp(
            f, u0, v0, h0, c0, H, Fu, Fv, Fh, Fc, h_wind, wind_strength, u_b, v_b, h_b, c_b)

        (adu0, adv0, adh0, adc0,
         _adH, _adFu, _adFv, _adFh, _adFc,
         _adh_wind, _adwind_strength,
         _adu_b, _adv_b, _adh_b, _adc_b) = vjp_fun((adu1, adv1, adh1, adc1))

        if adH is not None:            adH            += _adH
        if adFc is not None:           adFc           += _adFc
        if adh_wind is not None:       adh_wind       += _adh_wind
        if adwind_strength is not None: adwind_strength += _adwind_strength
        if adu_b is not None:          adu_b          += _adu_b
        if adv_b is not None:          adv_b          += _adv_b
        if adh_b is not None:          adh_b          += _adh_b
        if adc_b is not None:          adc_b          += _adc_b
        adFu += _adFu
        adFv += _adFv
        adFh += _adFh

        return (adu0, adv0, adh0, adc0,
                adH, adFu, adFv, adFh, adFc,
                adh_wind, adwind_strength,
                adu_b, adv_b, adh_b, adc_b)

    def step(self,State,nstep=1,t=0,Xb=None):

        if self.nl > 1:
            # Read per-layer state and convert to SW convention (1, nl, nx, ny)
            u_layers = np.nan_to_num(np.asarray(State.var['u_layers'], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            v_layers = np.nan_to_num(np.asarray(State.var['v_layers'], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            h_layers = np.nan_to_num(np.asarray(State.var['h_layers'], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            u = jnp.expand_dims(u_layers.astype(self.dtype).transpose(0, 2, 1), axis=0)
            v = jnp.expand_dims(v_layers.astype(self.dtype).transpose(0, 2, 1), axis=0)
            h = jnp.expand_dims(h_layers.astype(self.dtype).transpose(0, 2, 1), axis=0)
        else:
            h0 = self._finite_model_array(State, self.name_var['SSH'])
            u0 = self._finite_model_array(State, self.name_var['U'])
            v0 = self._finite_model_array(State, self.name_var['V'])
            h = jnp.expand_dims(jnp.asarray(h0, dtype=self.dtype).T, axis=(0,1))
            u = jnp.expand_dims(jnp.asarray(u0, dtype=self.dtype).T, axis=(0,1))
            v = jnp.expand_dims(jnp.asarray(v0, dtype=self.dtype).T, axis=(0,1))
        
        # Get parameters (2D forcing from Basis)
        Fu = State.params[self.name_var['U']]
        Fv = State.params[self.name_var['V']]
        Fh = State.params[self.name_var['SSH']]
        if 'H' in self.name_params:
            H = State.params['H']
            H = jnp.expand_dims(H.astype(self.dtype).T, axis=0)
        else:
            H = None
        if 'h_wind' in self.name_params:
            h_wind = State.params['h_wind']
            h_wind = h_wind.astype(self.dtype).T  # (nx, ny)
        else:
            h_wind = None
        if 'wind_strength' in self.name_params:
            wind_strength = State.params['wind_strength'].astype(self.dtype).T  # (nx, ny)
        else:
            wind_strength = None

        # BC perturbation from params
        if 'bc' in self.name_params:
            bc_du = State.params['u_b'].astype(self.dtype)
            bc_dv = State.params['v_b'].astype(self.dtype)
            bc_dh = State.params['h_b'].astype(self.dtype)
        else:
            bc_du = bc_dv = bc_dh = None

        # Tracer setup
        if self.advect_tracer:
            c = jnp.stack(
                [jnp.asarray(self._finite_model_array(State, self.name_var[n], fill_land='nearest'), dtype=self.dtype)
                 for n in self.tracer_names], axis=0)  # (n_trac, ny, nx)
            Fc = jnp.stack(
                [jnp.asarray(State.params[self.name_var[n]], dtype=self.dtype)
                 for n in self.tracer_names], axis=0)  # (n_trac, ny, nx)

        # Sub-stepping loop (limits lax.scan length for memory efficiency)
        step_done = 0
        while step_done < nstep:
            n_chunk = min(nstep - step_done, self.max_nstep) if self.max_nstep > 0 else (nstep - step_done)
            t_chunk = t + step_done * self.dt
            u_b, v_b, h_b, c_b, taux, tauy = self._chunk_forcing(
                Xb, step_done // self.max_nstep if self.max_nstep > 0 else 0,
                t_chunk, n_chunk,
            )
            if bc_du is not None:
                u_b = u_b + bc_du
                v_b = v_b + bc_dv
                h_b = h_b + bc_dh
            if self.advect_tracer:
                u, v, h, c = self.jstep_core_trac_jit(
                    t_chunk, u, v, h, c, H, Fu, Fv, Fh, Fc,
                    u_b, v_b, h_b, c_b,
                    taux=taux, tauy=tauy, h_wind=h_wind, wind_strength=wind_strength,
                    nstep=n_chunk)
            else:
                u, v, h = self.jstep_jit(t_chunk, u, v, h, H, Fu, Fv, Fh, u_b, v_b, h_b,
                                          taux=taux, tauy=tauy, h_wind=h_wind, wind_strength=wind_strength, nstep=n_chunk)
            step_done += n_chunk



        if self.nl > 1:
            # Convert back: (1, nl, nx, ny) → (nl, ny, nx)
            State.var['h_layers'] = h[0].transpose(0, 2, 1)
            State.var['u_layers'] = u[0].transpose(0, 2, 1)
            State.var['v_layers'] = v[0].transpose(0, 2, 1)
            # Diagnose 2D surface fields
            self._diagnose_surface(State)
        else:
            u = u[0,0].T
            v = v[0,0].T
            h = h[0,0].T
            State.setvar(u, name_var=self.name_var['U'])
            State.setvar(v, name_var=self.name_var['V'])
            State.setvar(h, name_var=self.name_var['SSH'])

        # Write tracer fields back to State
        if self.advect_tracer:
            for i, name in enumerate(self.tracer_names):
                State.setvar(c[i], name_var=self.name_var[name])

        self._set_land_nan(State)

    def step_tgl(self,dState,State,nstep=1,t=0,Xb=None):

        if self.nl > 1:
            # Per-layer perturbation
            du_layers = np.nan_to_num(np.asarray(dState.var['u_layers'], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            dv_layers = np.nan_to_num(np.asarray(dState.var['v_layers'], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            dh_layers = np.nan_to_num(np.asarray(dState.var['h_layers'], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            du = jnp.expand_dims(du_layers.astype(self.dtype).transpose(0, 2, 1), axis=0)
            dv = jnp.expand_dims(dv_layers.astype(self.dtype).transpose(0, 2, 1), axis=0)
            dh = jnp.expand_dims(dh_layers.astype(self.dtype).transpose(0, 2, 1), axis=0)
            # Per-layer forward trajectory
            u_layers = np.nan_to_num(np.asarray(State.var['u_layers'], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            v_layers = np.nan_to_num(np.asarray(State.var['v_layers'], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            h_layers = np.nan_to_num(np.asarray(State.var['h_layers'], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            u = jnp.expand_dims(u_layers.astype(self.dtype).transpose(0, 2, 1), axis=0)
            v = jnp.expand_dims(v_layers.astype(self.dtype).transpose(0, 2, 1), axis=0)
            h = jnp.expand_dims(h_layers.astype(self.dtype).transpose(0, 2, 1), axis=0)
        else:
            dh = self._finite_model_array(dState, self.name_var['SSH'])
            du = self._finite_model_array(dState, self.name_var['U'])
            dv = self._finite_model_array(dState, self.name_var['V'])
            h = self._finite_model_array(State, self.name_var['SSH'])
            u = self._finite_model_array(State, self.name_var['U'])
            v = self._finite_model_array(State, self.name_var['V'])
            du = jnp.expand_dims(du.astype(self.dtype).T, axis=(0,1))
            dv = jnp.expand_dims(dv.astype(self.dtype).T, axis=(0,1))
            dh = jnp.expand_dims(dh.astype(self.dtype).T, axis=(0,1))
            u = jnp.expand_dims(u.astype(self.dtype).T, axis=(0,1))
            v = jnp.expand_dims(v.astype(self.dtype).T, axis=(0,1))
            h = jnp.expand_dims(h.astype(self.dtype).T, axis=(0,1))

        # Get parameters (2D forcing — per-layer conversion in jstep_core via AD)
        Fu = State.params[self.name_var['U']]
        Fv = State.params[self.name_var['V']]
        Fh = State.params[self.name_var['SSH']]
        dFu = dState.params[self.name_var['U']]
        dFv = dState.params[self.name_var['V']]
        dFh = dState.params[self.name_var['SSH']]
        if 'H' in self.name_params:
            H = State.params['H']
            dH = dState.params['H']
            H = jnp.expand_dims(H.astype(self.dtype ).T, axis=0)
            dH = jnp.expand_dims(dH.astype(self.dtype).T, axis=0)
        else:
            H = dH = None
        if 'h_wind' in self.name_params:
            h_wind = State.params['h_wind'].astype(self.dtype).T    # (nx, ny)
            dh_wind = dState.params['h_wind'].astype(self.dtype).T  # (nx, ny)
        else:
            h_wind = dh_wind = None
        if 'wind_strength' in self.name_params:
            wind_strength = State.params['wind_strength'].astype(self.dtype).T    # (nx, ny)
            dwind_strength = dState.params['wind_strength'].astype(self.dtype).T  # (nx, ny)
        else:
            wind_strength = dwind_strength = None
        if 'bc' in self.name_params:
            bc_du = State.params['u_b'].astype(self.dtype)
            bc_dv = State.params['v_b'].astype(self.dtype)
            bc_dh = State.params['h_b'].astype(self.dtype)
            dbc_du = dState.params['u_b'].astype(self.dtype)
            dbc_dv = dState.params['v_b'].astype(self.dtype)
            dbc_dh = dState.params['h_b'].astype(self.dtype)
        else:
            bc_du = bc_dv = bc_dh = None
            dbc_du = dbc_dv = dbc_dh = None
        
        # Tracer setup
        if self.advect_tracer:
            c = jnp.stack(
                [jnp.asarray(self._finite_model_array(State, self.name_var[n], fill_land='nearest'), dtype=self.dtype)
                 for n in self.tracer_names], axis=0)
            Fc = jnp.stack(
                [jnp.asarray(State.params[self.name_var[n]], dtype=self.dtype)
                 for n in self.tracer_names], axis=0)
            dc = jnp.stack(
                [jnp.asarray(self._finite_model_array(dState, self.name_var[n]), dtype=self.dtype)
                 for n in self.tracer_names], axis=0)
            dFc = jnp.stack(
                [jnp.asarray(dState.params[self.name_var[n]], dtype=self.dtype)
                 for n in self.tracer_names], axis=0)

        # Sub-stepping loop (limits lax.scan length for memory efficiency)
        step_done = 0
        while step_done < nstep:
            n_chunk = min(nstep - step_done, self.max_nstep) if self.max_nstep > 0 else (nstep - step_done)
            t_chunk = t + step_done * self.dt
            u_b, v_b, h_b, c_b, taux, tauy = self._chunk_forcing(
                Xb, step_done // self.max_nstep if self.max_nstep > 0 else 0,
                t_chunk, n_chunk,
            )
            if bc_du is not None:
                u_b = u_b + bc_du
                v_b = v_b + bc_dv
                h_b = h_b + bc_dh
            if self.advect_tracer:
                # Propagate tangent (JVP includes forward internally)
                du, dv, dh, dc = self.jstep_tgl_trac_jit(
                    t_chunk, du, dv, dh, dc, dH, dFu, dFv, dFh, dFc,
                    u, v, h, c, H, Fu, Fv, Fh, Fc,
                    u_b, v_b, h_b, c_b,
                    taux=taux, tauy=tauy,
                    h_wind=h_wind, dh_wind=dh_wind,
                    wind_strength=wind_strength, dwind_strength=dwind_strength,
                    du_b=dbc_du, dv_b=dbc_dv, dh_b=dbc_dh,
                    nstep=n_chunk)
                # Propagate forward state for next chunk
                u, v, h, c = self.jstep_core_trac_jit(
                    t_chunk, u, v, h, c, H, Fu, Fv, Fh, Fc,
                    u_b, v_b, h_b, c_b,
                    taux=taux, tauy=tauy, h_wind=h_wind, wind_strength=wind_strength,
                    nstep=n_chunk)
            else:
                # Propagate tangent (JVP includes forward internally)
                du, dv, dh = self.jstep_tgl_jit(t_chunk, du, dv, dh, dH, dFu, dFv, dFh,
                                                u, v, h, H, Fu, Fv, Fh, u_b, v_b, h_b,
                                                taux=taux, tauy=tauy,
                                                h_wind=h_wind, dh_wind=dh_wind,
                                                wind_strength=wind_strength, dwind_strength=dwind_strength,
                                                du_b=dbc_du, dv_b=dbc_dv, dh_b=dbc_dh,
                                                nstep=n_chunk)
                # Propagate forward state for next chunk
                u, v, h = self.jstep_jit(t_chunk, u, v, h, H, Fu, Fv, Fh, u_b, v_b, h_b,
                                          taux=taux, tauy=tauy, h_wind=h_wind, wind_strength=wind_strength, nstep=n_chunk)
            step_done += n_chunk

        if self.nl > 1:
            # Convert back: (1, nl, nx, ny) → (nl, ny, nx)
            dState.var['h_layers'] = dh[0].transpose(0, 2, 1)
            dState.var['u_layers'] = du[0].transpose(0, 2, 1)
            dState.var['v_layers'] = dv[0].transpose(0, 2, 1)
            # Diagnose surface TLM
            dState.setvar(dState.var['h_layers'].sum(axis=0), name_var=self.name_var['SSH'])
            dState.setvar(dState.var['u_layers'][0],          name_var=self.name_var['U'])
            dState.setvar(dState.var['v_layers'][0],          name_var=self.name_var['V'])
        else:
            du = du[0,0].T
            dv = dv[0,0].T
            dh = dh[0,0].T
            dState.setvar(du, name_var=self.name_var['U'])
            dState.setvar(dv, name_var=self.name_var['V'])
            dState.setvar(dh, name_var=self.name_var['SSH'])

        # Write tracer perturbations back to dState
        if self.advect_tracer:
            for i, name in enumerate(self.tracer_names):
                dState.setvar(dc[i], name_var=self.name_var[name])

    def step_adj(self,adState,State,nstep=1,t=0,Xb=None):

        if self.nl > 1:
            # --- Combine surface adjoint (from obs) with per-layer adjoint ---
            # Adjoint of ssh = h_layers.sum(axis=0): broadcast ad_ssh to all layers
            adh_surface = jnp.asarray(self._finite_model_array(adState, self.name_var['SSH']), dtype=self.dtype)
            adu_surface = jnp.asarray(self._finite_model_array(adState, self.name_var['U']),   dtype=self.dtype)
            adv_surface = jnp.asarray(self._finite_model_array(adState, self.name_var['V']),   dtype=self.dtype)

            adh_layers = jnp.asarray(np.nan_to_num(
                            np.asarray(adState.var.get('h_layers',
                            jnp.zeros((self.nl, self.ny, self.nx), dtype=self.dtype)), dtype=float),
                            nan=0.0, posinf=0.0, neginf=0.0), dtype=self.dtype)
            adu_layers = jnp.asarray(np.nan_to_num(
                            np.asarray(adState.var.get('u_layers',
                            jnp.zeros((self.nl, self.ny, self.nx+1), dtype=self.dtype)), dtype=float),
                            nan=0.0, posinf=0.0, neginf=0.0), dtype=self.dtype)
            adv_layers = jnp.asarray(np.nan_to_num(
                            np.asarray(adState.var.get('v_layers',
                            jnp.zeros((self.nl, self.ny+1, self.nx), dtype=self.dtype)), dtype=float),
                            nan=0.0, posinf=0.0, neginf=0.0), dtype=self.dtype)

            # Adjoint of sum: each layer receives the surface SSH adjoint
            adh_layers = adh_layers + adh_surface[None, :, :]
            # Adjoint of selecting layer 0 for surface u/v
            adu_layers = adu_layers.at[0].add(adu_surface)
            adv_layers = adv_layers.at[0].add(adv_surface)

            # Convert to SW convention (1, nl, nx, ny)
            adu = jnp.expand_dims(adu_layers.transpose(0, 2, 1), axis=0)
            adv = jnp.expand_dims(adv_layers.transpose(0, 2, 1), axis=0)
            adh = jnp.expand_dims(adh_layers.transpose(0, 2, 1), axis=0)

            # Forward trajectory from State (per-layer)
            u_layers = np.nan_to_num(np.asarray(State.var['u_layers'], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            v_layers = np.nan_to_num(np.asarray(State.var['v_layers'], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            h_layers = np.nan_to_num(np.asarray(State.var['h_layers'], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            u = jnp.expand_dims(u_layers.astype(self.dtype).transpose(0, 2, 1), axis=0)
            v = jnp.expand_dims(v_layers.astype(self.dtype).transpose(0, 2, 1), axis=0)
            h = jnp.expand_dims(h_layers.astype(self.dtype).transpose(0, 2, 1), axis=0)
        else:
            adh = jnp.expand_dims(jnp.asarray(self._finite_model_array(adState, self.name_var['SSH']), dtype=self.dtype).T, axis=(0,1))
            adu = jnp.expand_dims(jnp.asarray(self._finite_model_array(adState, self.name_var['U']), dtype=self.dtype).T, axis=(0,1))
            adv = jnp.expand_dims(jnp.asarray(self._finite_model_array(adState, self.name_var['V']), dtype=self.dtype).T, axis=(0,1))
            h = jnp.expand_dims(jnp.asarray(self._finite_model_array(State, self.name_var['SSH']), dtype=self.dtype).T, axis=(0,1))
            u = jnp.expand_dims(jnp.asarray(self._finite_model_array(State, self.name_var['U']), dtype=self.dtype).T, axis=(0,1))
            v = jnp.expand_dims(jnp.asarray(self._finite_model_array(State, self.name_var['V']), dtype=self.dtype).T, axis=(0,1))

        # Get parameters (2D forcing)
        Fu = State.params[self.name_var['U']]
        Fv = State.params[self.name_var['V']]
        Fh = State.params[self.name_var['SSH']]
        if 'H' in self.name_params:
            H = State.params['H']
            H = jnp.expand_dims(H.astype(self.dtype).T, axis=0)
        else:
            H = None
        if 'h_wind' in self.name_params:
            h_wind = State.params['h_wind'].astype(self.dtype).T  # (nx, ny)
        else:
            h_wind = None
        if 'wind_strength' in self.name_params:
            wind_strength = State.params['wind_strength'].astype(self.dtype).T  # (nx, ny)
        else:
            wind_strength = None
        
        # Get adjoint parameters (2D — AD traces per-layer conversion automatically)
        adFu = adState.params[self.name_var['U']]
        adFv = adState.params[self.name_var['V']]
        adFh = adState.params[self.name_var['SSH']]
        if 'H' in self.name_params:
            adH = adState.params['H']
            adH = jnp.expand_dims(adH.astype(self.dtype).T, axis=0)
        else:
            adH = None
        if 'h_wind' in self.name_params:
            adh_wind = adState.params['h_wind'].astype(self.dtype).T  # (nx, ny)
        else:
            adh_wind = None
        if 'wind_strength' in self.name_params:
            adwind_strength = adState.params['wind_strength'].astype(self.dtype).T  # (nx, ny)
        else:
            adwind_strength = None
        if 'bc' in self.name_params:
            bc_du = State.params['u_b'].astype(self.dtype)
            bc_dv = State.params['v_b'].astype(self.dtype)
            bc_dh = State.params['h_b'].astype(self.dtype)
            adu_b_acc = adState.params['u_b'].astype(self.dtype)
            adv_b_acc = adState.params['v_b'].astype(self.dtype)
            adh_b_acc = adState.params['h_b'].astype(self.dtype)
        else:
            bc_du = bc_dv = bc_dh = None
            adu_b_acc = adv_b_acc = adh_b_acc = None

        # Tracer setup
        if self.advect_tracer:
            c = jnp.stack(
                [jnp.asarray(self._finite_model_array(State, self.name_var[n], fill_land='nearest'), dtype=self.dtype)
                 for n in self.tracer_names], axis=0)
            Fc = jnp.stack(
                [jnp.asarray(State.params[self.name_var[n]], dtype=self.dtype)
                 for n in self.tracer_names], axis=0)
            adc = jnp.stack(
                [jnp.asarray(self._finite_model_array(adState, self.name_var[n]), dtype=self.dtype)
                 for n in self.tracer_names], axis=0)
            adFc = jnp.stack(
                [jnp.asarray(adState.params[self.name_var[n]], dtype=self.dtype)
                 for n in self.tracer_names], axis=0)
            adc_b_acc = None  # Tracer BCs not presently optimised

        # Build chunk schedule
        chunks = []
        step_done = 0
        while step_done < nstep:
            n_chunk = min(nstep - step_done, self.max_nstep) if self.max_nstep > 0 else (nstep - step_done)
            t_chunk = t + step_done * self.dt
            chunks.append((t_chunk, n_chunk))
            step_done += n_chunk

        # Forward pass: store boundary states at chunk boundaries
        fwd_states = [(u, v, h, c)] if self.advect_tracer else [(u, v, h)]
        if len(chunks) > 1:
            u_fwd, v_fwd, h_fwd = u, v, h
            if self.advect_tracer:
                c_fwd = c
            for chunk_index, (t_chunk, n_chunk) in enumerate(chunks[:-1]):
                u_b, v_b, h_b, c_b, taux, tauy = self._chunk_forcing(
                    Xb, chunk_index, t_chunk, n_chunk,
                )
                if bc_du is not None:
                    u_b = u_b + bc_du
                    v_b = v_b + bc_dv
                    h_b = h_b + bc_dh
                if self.advect_tracer:
                    u_fwd, v_fwd, h_fwd, c_fwd = self.jstep_core_trac_jit(
                        t_chunk, u_fwd, v_fwd, h_fwd, c_fwd, H, Fu, Fv, Fh, Fc,
                        u_b, v_b, h_b, c_b,
                        taux=taux, tauy=tauy, h_wind=h_wind, wind_strength=wind_strength,
                        nstep=n_chunk)
                    fwd_states.append((u_fwd, v_fwd, h_fwd, c_fwd))
                else:
                    u_fwd, v_fwd, h_fwd = self.jstep_jit(
                        t_chunk, u_fwd, v_fwd, h_fwd, H, Fu, Fv, Fh, u_b, v_b, h_b,
                        taux=taux, tauy=tauy, h_wind=h_wind, wind_strength=wind_strength, nstep=n_chunk)
                    fwd_states.append((u_fwd, v_fwd, h_fwd))

        # Reverse adjoint through chunks
        for i in range(len(chunks) - 1, -1, -1):
            t_chunk, n_chunk = chunks[i]
            u_b, v_b, h_b, c_b, taux, tauy = self._chunk_forcing(
                Xb, i, t_chunk, n_chunk,
            )
            if bc_du is not None:
                u_b = u_b + bc_du
                v_b = v_b + bc_dv
                h_b = h_b + bc_dh
            if self.advect_tracer:
                u_i, v_i, h_i, c_i = fwd_states[i]
                (adu, adv, adh, adc,
                 adH, adFu, adFv, adFh, adFc,
                 adh_wind, adwind_strength,
                 adu_b_acc, adv_b_acc, adh_b_acc, adc_b_acc) = self.jstep_adj_trac_jit(
                    t_chunk, adu, adv, adh, adc, adH, adFu, adFv, adFh, adFc,
                    u_i, v_i, h_i, c_i, H, Fu, Fv, Fh, Fc,
                    u_b, v_b, h_b, c_b,
                    taux=taux, tauy=tauy,
                    h_wind=h_wind, adh_wind=adh_wind,
                    wind_strength=wind_strength, adwind_strength=adwind_strength,
                    adu_b=adu_b_acc, adv_b=adv_b_acc, adh_b=adh_b_acc,
                    nstep=n_chunk)
            else:
                u_i, v_i, h_i = fwd_states[i]
                (adu, adv, adh, adH, adFu, adFv, adFh,
                 adh_wind, adwind_strength, adu_b_acc, adv_b_acc, adh_b_acc) = self.jstep_adj_jit(
                    t_chunk, adu, adv, adh, adH, adFu, adFv, adFh,
                    u_i, v_i, h_i, H, Fu, Fv, Fh,
                    u_b, v_b, h_b,
                    taux=taux, tauy=tauy,
                    h_wind=h_wind, adh_wind=adh_wind,
                    wind_strength=wind_strength, adwind_strength=adwind_strength,
                    adu_b=adu_b_acc, adv_b=adv_b_acc, adh_b=adh_b_acc,
                    nstep=n_chunk)

        if self.nl > 1:
            # Convert back: (1, nl, nx, ny) → (nl, ny, nx)
            adh_layers = adh[0].transpose(0, 2, 1)
            adu_layers = adu[0].transpose(0, 2, 1)
            adv_layers = adv[0].transpose(0, 2, 1)
            # Store per-layer adjoints
            adState.var['h_layers'] = adh_layers
            adState.var['u_layers'] = adu_layers
            adState.var['v_layers'] = adv_layers
            # Reset surface adjoints (consumed — distributed to layers above)
            adState.setvar(jnp.zeros((self.ny, self.nx),   dtype=self.dtype), self.name_var['SSH'])
            adState.setvar(jnp.zeros((self.ny, self.nx+1), dtype=self.dtype), self.name_var['U'])
            adState.setvar(jnp.zeros((self.ny+1, self.nx), dtype=self.dtype), self.name_var['V'])
        else:
            adu = adu[0,0].T
            adv = adv[0,0].T
            adh = adh[0,0].T
            adState.setvar(adu,self.name_var['U'])
            adState.setvar(adv,self.name_var['V'])
            adState.setvar(adh,self.name_var['SSH'])

        if 'H' in self.name_params:
            adH = adH[0].T
            adState.params['H'] = adH
        if 'h_wind' in self.name_params:
            adState.params['h_wind'] = adh_wind.T  # back to State convention (ny, nx)
        if 'wind_strength' in self.name_params:
            adState.params['wind_strength'] = adwind_strength.T  # back to State convention (ny, nx)
        if 'bc' in self.name_params:
            adState.params['u_b'] = adu_b_acc
            adState.params['v_b'] = adv_b_acc
            adState.params['h_b'] = adh_b_acc

        # Update adjoint parameters (2D, same for both nl=1 and nl>1)
        adState.params[self.name_var['U']] = adFu 
        adState.params[self.name_var['V']] = adFv 
        adState.params[self.name_var['SSH']] = adFh 

        # Write tracer adjoints back to adState
        if self.advect_tracer:
            for i, name in enumerate(self.tracer_names):
                adState.setvar(adc[i], name_var=self.name_var[name])
            for i, name in enumerate(self.tracer_names):
                adState.params[self.name_var[name]] = adFc[i]

    def adjoint_test_jstep(self, nstep=1, seed=42):
        """
        Low-level adjoint test for jstep_tgl / jstep_adj.

        Checks  <M dx, y> == <dx, M* y>
        where M = jstep_tgl and M* = jstep_adj, operating on the
        differentiated variables (u, v, h, H, Fu, Fv, Fh).

        Boundary conditions (u_b, v_b, h_b) are differentiated when
        'bc' is in self.name_params.
        Wind stress (taux, tauy) is NOT differentiated.
        """
        key = jax.random.PRNGKey(seed)
        dtype = self.dtype
        nx = self.nx
        ny = self.ny

        # --- shapes ---------------------------------------------------------
        # u0/v0/h0 enter jstep in SW-internal convention (transposed + expanded)
        nl = self.nl
        u_shape = (1, nl, nx + 1, ny)
        v_shape = (1, nl, nx, ny + 1)
        h_shape = (1, nl, nx, ny)
        H_shape = (1, nx, ny)
        # Fu/Fv/Fh stay in State convention (ny, nx+1/ny+1/nx) — always 2D
        Fu_shape = (ny, nx + 1)
        Fv_shape = (ny + 1, nx)
        Fh_shape = (ny, nx)
        # boundary conditions (State convention)
        ub_shape = (ny, nx + 1)
        vb_shape = (ny + 1, nx)
        hb_shape = (ny, nx)

        def rand(key, shape):
            key, subkey = jax.random.split(key)
            return key, jax.random.normal(subkey, shape=shape, dtype=dtype) * 1e-4

        # Masks from the SW model (squeeze batch/layer dims for masking)
        mask_u = self.model.masks.u[0, 0]   # (nx+1, ny)
        mask_v = self.model.masks.v[0, 0]   # (nx, ny+1)
        mask_h = self.model.masks.h[0, 0]   # (nx, ny)

        # --- base trajectory ------------------------------------------------
        key, u0 = rand(key, u_shape)
        key, v0 = rand(key, v_shape)
        key, h0 = rand(key, h_shape)
        u0 = u0 * mask_u
        v0 = v0 * mask_v
        h0 = h0 * mask_h

        has_H = 'H' in self.name_params
        if has_H:
            key, H = rand(key, H_shape)
        else:
            H = None

        has_hw = 'h_wind' in self.name_params
        hw_shape = (nx, ny)
        if has_hw:
            key, h_wind = rand(key, hw_shape)
        else:
            h_wind = None

        key, Fu = rand(key, Fu_shape)
        key, Fv = rand(key, Fv_shape)
        key, Fh = rand(key, Fh_shape)

        # boundary conditions
        key, u_b = rand(key, ub_shape)
        key, v_b = rand(key, vb_shape)
        key, h_b = rand(key, hb_shape)

        has_bc = 'bc' in self.name_params

        # --- TLM perturbation -----------------------------------------------
        key, du0 = rand(key, u_shape);  du0 = du0 * mask_u
        key, dv0 = rand(key, v_shape);  dv0 = dv0 * mask_v
        key, dh0 = rand(key, h_shape);  dh0 = dh0 * mask_h
        if has_H:
            key, dH = rand(key, H_shape)
        else:
            dH = None
        if has_hw:
            key, dh_wind = rand(key, hw_shape)
        else:
            dh_wind = None
        if has_bc:
            key, du_b = rand(key, ub_shape)
            key, dv_b = rand(key, vb_shape)
            key, dh_b = rand(key, hb_shape)
        else:
            du_b = dv_b = dh_b = None
        key, dFu = rand(key, Fu_shape)
        key, dFv = rand(key, Fv_shape)
        key, dFh = rand(key, Fh_shape)

        # --- ADJ cotangent (output space: u1, v1, h1) -----------------------
        key, wu = rand(key, u_shape);  wu = wu * mask_u
        key, wv = rand(key, v_shape);  wv = wv * mask_v
        key, wh = rand(key, h_shape);  wh = wh * mask_h

        # --- Run TLM --------------------------------------------------------
        du1, dv1, dh1 = self.jstep_tgl(
            0, du0, dv0, dh0, dH, dFu, dFv, dFh,
            u0, v0, h0, H, Fu, Fv, Fh,
            u_b, v_b, h_b,
            h_wind=h_wind, dh_wind=dh_wind,
            du_b=du_b, dv_b=dv_b, dh_b=dh_b, nstep=nstep)

        # --- Run ADJ --------------------------------------------------------
        # Zero accumulators so output = pure adjoint
        adH_in  = jnp.zeros(H_shape, dtype=dtype) if has_H else None
        adFu_in = jnp.zeros(Fu_shape, dtype=dtype)
        adFv_in = jnp.zeros(Fv_shape, dtype=dtype)
        adFh_in = jnp.zeros(Fh_shape, dtype=dtype)
        adh_wind_in = jnp.zeros(hw_shape, dtype=dtype) if has_hw else None
        adu_b_in = jnp.zeros(ub_shape, dtype=dtype) if has_bc else None
        adv_b_in = jnp.zeros(vb_shape, dtype=dtype) if has_bc else None
        adh_b_in = jnp.zeros(hb_shape, dtype=dtype) if has_bc else None

        (adu0, adv0, adh0, adH_out, adFu_out, adFv_out, adFh_out,
         adh_wind_out, adu_b_out, adv_b_out, adh_b_out) = self.jstep_adj(
            0, wu, wv, wh, adH_in, adFu_in, adFv_in, adFh_in,
            u0, v0, h0, H, Fu, Fv, Fh,
            u_b, v_b, h_b,
            h_wind=h_wind, adh_wind=adh_wind_in,
            adu_b=adu_b_in, adv_b=adv_b_in, adh_b=adh_b_in, nstep=nstep)

        # --- Check NaN ------------------------------------------------------
        has_nan = (
            jnp.any(jnp.isnan(du1)) | jnp.any(jnp.isnan(dv1)) |
            jnp.any(jnp.isnan(dh1)) | jnp.any(jnp.isnan(adu0)) |
            jnp.any(jnp.isnan(adv0)) | jnp.any(jnp.isnan(adh0)))
        if has_nan:
            print(f'  jstep adjoint test (dtype={dtype}, {nstep=}): NaN detected!')
            return float('nan')

        # --- Inner products (f64 accumulation) ------------------------------
        to64 = lambda x: x.astype(jnp.float64)

        # <M dx, y>  (output space)
        ps1 = (jnp.sum(to64(du1) * to64(wu))
             + jnp.sum(to64(dv1) * to64(wv))
             + jnp.sum(to64(dh1) * to64(wh)))

        # <dx, M* y>  (input space: u, v, h, H, Fu, Fv, Fh)
        ps2 = (jnp.sum(to64(du0) * to64(adu0))
             + jnp.sum(to64(dv0) * to64(adv0))
             + jnp.sum(to64(dh0) * to64(adh0))
             + jnp.sum(to64(dFu) * to64(adFu_out))
             + jnp.sum(to64(dFv) * to64(adFv_out))
             + jnp.sum(to64(dFh) * to64(adFh_out)))
        if has_H:
            ps2 += jnp.sum(to64(dH) * to64(adH_out))
        if has_hw:
            ps2 += jnp.sum(to64(dh_wind) * to64(adh_wind_out))
        if has_bc:
            ps2 += (jnp.sum(to64(du_b) * to64(adu_b_out))
                  + jnp.sum(to64(dv_b) * to64(adv_b_out))
                  + jnp.sum(to64(dh_b) * to64(adh_b_out)))

        ratio = float(ps1 / ps2)
        print(f'  jstep adjoint test (dtype={dtype}, {nstep=}): '
              f'<Mdx,y>/<dx,M*y> = {ratio}')
        


class Model_bmit(_ITOpenBoundaryExtensionMixin, M):

    def _load_component_bc_fields(self, component_config, config, State):
        bc_sources = []
        if getattr(component_config, 'bc_file', None) is not None:
            bc_sources.append(type('BC_Config', (), {
                'file': component_config.bc_file,
                'name_lon': getattr(component_config, 'bc_name_lon', 'lon'),
                'name_lat': getattr(component_config, 'bc_name_lat', 'lat'),
                'name_time': getattr(component_config, 'bc_name_time', None),
                'name_var': getattr(component_config, 'bc_name_var', {}),
                'c_grid': getattr(component_config, 'bc_c_grid', False),
                '_config': config,
            })())
        if getattr(component_config, 'bc_files', None) is not None:
            for name, bc_src in component_config.bc_files.items():
                if isinstance(bc_src, str):
                    bc_src = {'file': bc_src}
                name_var = bc_src.get('name_var', name)
                if isinstance(name_var, str):
                    name_var = {name: name_var}
                bc_sources.append(type('BC_Config', (), {
                    'file': bc_src.get('file'),
                    'name_lon': bc_src.get('name_lon', getattr(component_config, 'bc_name_lon', 'lon')),
                    'name_lat': bc_src.get('name_lat', getattr(component_config, 'bc_name_lat', 'lat')),
                    'name_time': bc_src.get('name_time', getattr(component_config, 'bc_name_time', None)),
                    'name_var': name_var,
                    'c_grid': bc_src.get('c_grid', getattr(component_config, 'bc_c_grid', False)),
                    '_config': config,
                })())

        bc_fields = []
        for bc_config in bc_sources:
            try:
                bc_fields.append(_BcFields(bc_config, State))
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to load BM boundary conditions from {bc_config.file}: {e}")
        if len(bc_fields) == 1:
            return bc_fields[0]
        if len(bc_fields) > 1:
            return _MultiBcFields(bc_fields)
        return None

    def _read_component_c(self, component_config, State, title='Rossby phase velocity'):
        if getattr(component_config, 'filec_aux', None) is not None and os.path.exists(component_config.filec_aux):

            ds = xr.open_dataset(component_config.filec_aux)
            name_lon = component_config.name_var_c['lon']
            lon = ds[name_lon]
            if np.sign(lon.data.min()) == -1 and State.lon_unit == '0_360':
                ds = ds.assign_coords({name_lon: ((name_lon, lon.data % 360))})
                ds = ds.sortby(name_lon)
            elif np.sign(lon.data.min()) >= 0 and State.lon_unit == '-180_180':
                ds = ds.assign_coords({name_lon: ((name_lon, (lon.data + 180) % 360 - 180))})
                ds = ds.sortby(name_lon)

            c = grid.interp2d(ds, component_config.name_var_c, State.lon, State.lat)

            c_masked = np.ma.masked_invalid(c)
            c_filled = c_masked.filled(np.nan)
            from scipy import interpolate
            x = np.arange(self.nx)
            y = np.arange(self.ny)
            xx, yy = np.meshgrid(x, y)
            points = np.array([xx[~c_masked.mask], yy[~c_masked.mask]]).T
            values = c_filled[~c_masked.mask]
            c = interpolate.griddata(points, values, (xx, yy), method='nearest')

            if getattr(component_config, 'cmin', None) is not None:
                c[c < component_config.cmin] = component_config.cmin

            if getattr(component_config, 'cmax', None) is not None:
                c[c > component_config.cmax] = component_config.cmax

            if self.config.EXP.flag_plot > 0:
                plt.figure()
                plt.pcolormesh(c)
                plt.colorbar()
                plt.title(title)
                plt.show()

            return c

        return getattr(component_config, 'c0', 2.7) * np.ones((State.ny, State.nx))

    def __init__(self,config,State):

        super().__init__(config,State)

        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

        self.config = config
        self.name_var = config.MOD.name_var
        self.bm_config = Config(config.MOD.bm_model)
        self.it_config = Config(config.MOD.it_model)
        self.bm_config.dtmodel = self.dt
        self.it_config.dtmodel = self.dt
        self._bc_fields = None
        if self.bm_config.super == 'MOD_QGSW':
            self.bm_config.name_var = {
                'U': self.name_var['U_BM'],
                'V': self.name_var['V_BM'],
                'SSH': self.name_var['SSH_BM'],
            }
            self.bm_config.name_init_var = {
                'U': self.name_var['U_BM'],
                'V': self.name_var['V_BM'],
                'SSH': self.name_var['SSH_BM'],
            }
        elif self.bm_config.super == 'MOD_QG1L':
            self.bm_config.name_var = {'SSH': self.name_var['SSH_BM']}
            self.bm_config.name_init_var = {'SSH': self.name_var['SSH_BM']}
        self.it_config.name_var = {
            'U': self.name_var['U_IT'],
            'V': self.name_var['V_IT'],
            'SSH': self.name_var['SSH_IT'],
        }
        self.it_config.name_params = getattr(self.it_config, 'name_params', ['He_mean', 'hbc'])
        self.bm_height_for_He = getattr(config.MOD, 'bm_height_for_He', 'anomaly')
        if self.bm_height_for_He not in ('anomaly', 'full'):
            raise ValueError("MOD_BMIT.bm_height_for_He must be 'anomaly' or 'full'.")
        self.mask_sponge_bc = getattr(config.MOD, 'mask_sponge_bc', True)
        self.bm_config.mask_sponge_bc = self.mask_sponge_bc

        # Feed the existing IT setup code from the selected IT component block.
        # These are internal aliases only; the public MOD_BMIT surface remains
        # componentized through bm_model and it_model.
        for key in [
                'filec_aux', 'name_var_c', 'c0', 'cmin', 'cmax', 'file_H_aux',
                'name_var_H', 'H', 'Ntheta', 'w_waves', 'g',
                'periodic_x', 'periodic_y', 'flag_bc_sponge', 'dist_sponge_bc',
                'sponge_coef', 'use_sponge_on_coast', 'tangential_sponge_factor',
                'bc_it_method', 'bc_it_corner_weight_power',
                'extend_it_open_boundary_sponge']:
            config.MOD[key] = getattr(self.it_config, key)
        config.MOD.mask_sponge_bc = self.mask_sponge_bc
        config.MOD.name_params_it = getattr(self.it_config, 'name_params', [])
        config.MOD.time_scheme_it = getattr(self.it_config, 'time_scheme', 'rk4')

        if self.bm_config.super == 'MOD_QG1L':
            self._bc_fields = self._load_component_bc_fields(self.bm_config, config, State)

        # BM aliases used by the legacy QG1L BM path. QGSW owns its own setup.
        config.MOD.path_mdt = getattr(self.bm_config, 'path_mdt', None)
        config.MOD.name_var_mdt = getattr(self.bm_config, 'name_var_mdt', None)
        config.MOD.time_scheme_bm = getattr(self.bm_config, 'time_scheme', 'Euler')
        config.MOD.Kdiffus = getattr(self.bm_config, 'Kdiffus', 0)

        self.bm_kind = self.bm_config.super
        model_bm = None
        if self.bm_kind == 'MOD_QG1L':
            if getattr(self.bm_config, 'dir_model', None) is None:
                dir_model = os.path.realpath(
                    os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 '..','models','model_qg1l'))
            else:
                dir_model = self.bm_config.dir_model
            qgm = SourceFileLoader("qgm_bmit", f'{dir_model}/jqgm.py').load_module()
            model_bm = qgm.Qgm
        elif self.bm_kind != 'MOD_QGSW':
            raise ValueError(f"Unsupported MOD_BMIT bm_model.super={self.bm_kind!r}")

        if self.it_config.super != 'MOD_CSW1L':
            raise ValueError(f"Unsupported MOD_BMIT it_model.super={self.it_config.super!r}")
        if getattr(self.it_config, 'dir_model', None) is None:
            dir_model = os.path.realpath(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..','models','model_sw1l'))
        else:
            dir_model = self.it_config.dir_model
        swm = SourceFileLoader("swm_bmit", dir_model + "/jswm.py").load_module()
        model_it = swm.CSWm

        # grid
        self.ny = State.ny
        self.nx = State.nx
        self.Xu = self._rho_on_u(State.X)
        self.Yu = self._rho_on_u(State.Y)
        self.Xv = self._rho_on_v(State.X)
        self.Yv = self._rho_on_v(State.Y)

        # Coriolis
        self.f = State.f
        f0 = np.nanmean(self.f)
        self.f[np.isnan(self.f)] = f0

        # Gravity
        self.g = State.g

        # Mask
        self.mask = State.mask

        # Open MDT map if provided
        if config.MOD.path_mdt is not None and os.path.exists(config.MOD.path_mdt):
                      
            ds = xr.open_dataset(config.MOD.path_mdt)
            name_lon = config.MOD.name_var_mdt['lon']
            lon = ds[name_lon]
            # Convert longitude 
            if np.sign(lon.data.min())==-1 and State.lon_unit=='0_360':
                ds = ds.assign_coords({name_lon:((name_lon, lon.data % 360))})
                ds = ds.sortby(name_lon)    
            elif np.sign(lon.data.min())>=0 and State.lon_unit=='-180_180':
                ds = ds.assign_coords({name_lon:((name_lon, (lon.data + 180) % 360 - 180))})
                ds = ds.sortby(name_lon)    

            self.mdt = grid.interp2d(ds,
                                   config.MOD.name_var_mdt,
                                   State.lon,
                                   State.lat)
            self.mdt[np.isnan(self.mdt)] = 0
        
            if config.EXP.flag_plot>0:
                plt.figure()
                plt.pcolormesh(self.mdt)
                plt.colorbar()
                plt.title('MDT')
                plt.show()

        else:
            self.mdt = None

        # Open Rossby Radius if provided
        if config.MOD.filec_aux is not None and os.path.exists(config.MOD.filec_aux):

            ds = xr.open_dataset(config.MOD.filec_aux)
            name_lon = config.MOD.name_var_c['lon']
            lon = ds[name_lon]
            # Convert longitude 
            if np.sign(lon.data.min())==-1 and State.lon_unit=='0_360':
                ds = ds.assign_coords({name_lon:((name_lon, lon.data % 360))})
                ds = ds.sortby(name_lon)    
            elif np.sign(lon.data.min())>=0 and State.lon_unit=='-180_180':
                ds = ds.assign_coords({name_lon:((name_lon, (lon.data + 180) % 360 - 180))})
                ds = ds.sortby(name_lon)    

            self.c = grid.interp2d(ds,
                                   config.MOD.name_var_c,
                                   State.lon,
                                   State.lat)
        
            # Remove NaNs (e.g. over land) by nearest-neighbor interpolation
            # (since c is only used for boundary conditions, this is better than leaving NaNs or filling with a constant.)
            c_masked = np.ma.masked_invalid(self.c)
            c_filled = c_masked.filled(np.nan)
            from scipy import interpolate
            x = np.arange(self.nx)
            y = np.arange(self.ny)
            xx, yy = np.meshgrid(x, y)
            points = np.array([xx[~c_masked.mask], yy[~c_masked.mask]]).T
            values = c_filled[~c_masked.mask]
            self.c = interpolate.griddata(points, values, (xx, yy), method='nearest')
            
            if config.MOD.cmin is not None:
                self.c[self.c<config.MOD.cmin] = config.MOD.cmin
            
            if config.MOD.cmax is not None:
                self.c[self.c>config.MOD.cmax] = config.MOD.cmax
            
            if config.EXP.flag_plot>0:
                plt.figure()
                plt.pcolormesh(self.c)
                plt.colorbar()
                plt.title('Rossby phase velocity')
                plt.show()
                
        else:
            self.c = config.MOD.c0 * np.ones((State.ny,State.nx))
        
        self.f_on_u = (self.f[:, :-1] + self.f[:, 1:]) / 2
        self.f_on_v = (self.f[:-1, :] + self.f[1:, :]) / 2

        # Equivalent depth
        self.Heb = self.c**2 / self.g
        self.c_bm = self.c
        if self.bm_kind == 'MOD_QG1L':
            self.c_bm = self._read_component_c(
                self.bm_config, State, title='BM Rossby phase velocity')

        # Open Bathymetry if provided
        if config.MOD.file_H_aux is not None and os.path.exists(config.MOD.file_H_aux):

            ds = xr.open_dataset(config.MOD.file_H_aux)
            name_lon = config.MOD.name_var_H['lon']
            lon = ds[name_lon]
            # Convert longitude 
            if np.sign(lon.data.min())==-1 and State.lon_unit=='0_360':
                ds = ds.assign_coords({name_lon:((name_lon, lon.data % 360))})
                ds = ds.sortby(name_lon)    
            elif np.sign(lon.data.min())>=0 and State.lon_unit=='-180_180':
                ds = ds.assign_coords({name_lon:((name_lon, (lon.data + 180) % 360 - 180))})
                ds = ds.sortby(name_lon)    

            self.H = grid.interp2d(ds,
                                   config.MOD.name_var_H,
                                   State.lon,
                                   State.lat)

            # Remove NaNs (e.g. over land) by nearest-neighbor interpolation
            H_masked = np.ma.masked_invalid(self.H)
            H_filled = H_masked.filled(np.nan)
            from scipy import interpolate
            x = np.arange(self.nx)
            y = np.arange(self.ny)
            xx, yy = np.meshgrid(x, y)
            points = np.array([xx[~H_masked.mask], yy[~H_masked.mask]]).T
            values = H_filled[~H_masked.mask]
            self.H = interpolate.griddata(points, values, (xx, yy), method='nearest')
            
            if config.EXP.flag_plot>0:
                plt.figure()
                plt.pcolormesh(self.H)
                plt.colorbar()
                plt.title('Bathymetry')
                plt.show()

        else:
            self.H = config.MOD.H 
        
        # Boundary angles
        _ntheta = config.MOD.Ntheta
        if _ntheta == -1:
            # Auto: boundary tangential Nyquist criterion.
            # A wave entering at angle theta has along-boundary wavenumber k_t = k*sin(theta).
            # At grazing incidence k_t_max = k_max = omega_max/c_min.  With 2*Ntheta+1 angular
            # samples covering [-k_max, k_max], the Nyquist condition on a boundary of length L
            # gives:  Ntheta >= k_max * L / (2*pi) = L / lambda_min
            # where lambda_min = 2*pi*c_min / omega_max and L = max boundary length.
            _omegas = np.asarray(config.MOD.w_waves)
            _He_bdy = np.concatenate([
                self.Heb[0,:], self.Heb[-1,:], self.Heb[:,0], self.Heb[:,-1]])
            _c_min = np.sqrt(self.g * np.nanmin(_He_bdy[_He_bdy > 0]))
            _lambda_min = 2*pi * _c_min / np.max(np.abs(_omegas))
            _L_bdy = max(
                float(np.asarray(State.X).max() - np.asarray(State.X).min()),
                float(np.asarray(State.Y).max() - np.asarray(State.Y).min()))
            _ntheta = int(np.ceil(_L_bdy / _lambda_min))
            print(f'  Auto Ntheta (boundary tangential Nyquist): '
                    f'lambda_min={_lambda_min/1e3:.1f} km, '
                    f'L_bdy={_L_bdy/1e3:.1f} km -> Ntheta={_ntheta}')
        if _ntheta > 0:
            theta_p = np.arange(0, pi/2 + pi/2/_ntheta, pi/2/_ntheta)
            self.bc_theta = np.append(theta_p - pi/2, theta_p[1:])
        else:
            self.bc_theta = np.array([0])
        
        # External boundary conditions
        self.bc = {}
        for _name_var_mod in self.name_var:
            self.bc[_name_var_mod] = {}
        self.init_from_bc_bm = getattr(self.bm_config, 'init_from_bc', False)
        
        # Tide frequencies
        self.omegas = np.asarray(config.MOD.w_waves)
        
        # CFL
        if config.MOD.cfl is not None:
            grid_spacing = min(np.nanmean(State.DX), np.nanmean(State.DY)) 
            dt = config.MOD.cfl * grid_spacing / np.nanmax(self.c)
            divisors = [i for i in range(1, 3600 + 1) if 3600 % i == 0]  # Find all divisors of one hour in seconds
            lower_divisors = [d for d in divisors if d <= dt]
            self.dt = max(lower_divisors)  # Get closest
            print('CFL condition for max(c)=', np.nanmax(self.c), 'm/s and grid spacing ~', grid_spacing, 'm: dt<=', dt, 's')
            print('Model time-step', self.dt)
            # Time parameters
            if self.dt>0:
                self.nt = 1 + int((config.EXP.final_date - config.EXP.init_date).total_seconds()//self.dt)
                self.timestamps = [] 
                t = config.EXP.init_date
                while t<=config.EXP.final_date:
                    self.timestamps.append(t)
                    t += timedelta(seconds=self.dt)
                self.timestamps = np.asarray(self.timestamps)
            else:
                self.nt = 1
                self.timestamps = np.array([config.EXP.init_date])
            self.T = np.arange(self.nt) * self.dt

        # Initialize model state
        self.name_var = config.MOD.name_var
        self.var_to_save = [self.name_var['SSH']] # ssh
        if (config.GRID.super == 'GRID_FROM_FILE') and (config.MOD.name_init_var is not None):
            dsin = grid.open_grid_restart(config)
            for name in self.name_var:
                if name in config.MOD.name_init_var:
                    var_init = dsin[config.MOD.name_init_var[name]]
                    if len(var_init.shape)==3:
                        var_init = var_init[0,:,:]
                    if config.GRID.subsampling is not None:
                        var_init = var_init[::config.GRID.subsampling,::config.GRID.subsampling]
                    dsin.close()
                    del dsin
                    State.var[self.name_var[name]] = var_init.values
                else:
                    if name=='U_IT':
                        State.var[self.name_var[name]] = np.zeros((State.ny,State.nx-1))
                    elif name=='V_IT':
                        State.var[self.name_var[name]] = np.zeros((State.ny-1,State.nx))
                    elif name in ['SSH_IT', 'SSH_BM', 'SSH']:
                        State.var[self.name_var[name]] = np.zeros((State.ny,State.nx))
        else:
            for name in self.name_var:  
                if name=='U_IT':
                    State.var[self.name_var[name]] = np.zeros((State.ny,State.nx-1))
                elif name=='V_IT':
                    State.var[self.name_var[name]] = np.zeros((State.ny-1,State.nx))
                elif name in ['SSH_IT', 'SSH_BM', 'SSH', 'U_BM', 'V_BM']:
                    State.var[self.name_var[name]] = np.zeros((State.ny,State.nx))

        # BM Model Parameters (Flux)
        State.params[self.name_var['SSH_BM']] = np.zeros((State.ny,State.nx))

        # IT Model Parameters (sponge, He and alpha coefficient)
        self.name_params_it = config.MOD.name_params_it
        if 'He_mean' in self.name_params_it:
            State.params['He_mean'] = np.zeros((self.ny, self.nx))
        if 'alpha' in self.name_params_it:
            State.params['alpha'] = np.zeros((self.ny, self.nx))
        if 'alpha_He' in self.name_params_it:
            State.params['alpha_He'] = np.zeros((self.ny, self.nx))
        if 'alpha_Uu' in self.name_params_it:
            State.params['alpha_Uu'] = np.zeros((self.ny, self.nx))
        if 'alpha_Up' in self.name_params_it:
            State.params['alpha_Up'] = np.zeros((self.ny, self.nx))
        if 'hbc' in self.name_params_it:
            self.shapehbcx = [len(self.omegas), # tide frequencies
                              2, # North/South
                              2, # cos/sin
                              len(self.bc_theta), # Angles
                              State.nx # NX
                              ]
            self.shapehbcy = [len(self.omegas), # tide frequencies
                              2, # North/South
                              2, # cos/sin
                              len(self.bc_theta), # Angles
                              State.ny # NY
                              ]
            State.params['hbcx'] = np.zeros((self.shapehbcx))
            State.params['hbcy'] = np.zeros((self.shapehbcy))
        

        # ---- Background IT params (from a previous experiment) ----
        # If a netcdf file is provided via `config.MOD.path_background_params`,
        # load `He_mean` anomaly background and physical alpha references.
        # Alpha controls are dimensionless logit increments:
        #     alpha_total = sigmoid(logit(alpha_ref) + alpha_control)
        # with neutral alpha_ref=0.5 by default. The dynamics below still uses
        # the centered coefficient alpha_total - 0.5, preserving the former
        # zero-control neutral behaviour while bounding physical alpha in (0, 1).
        self.He_mean_bg  = jnp.zeros((self.ny, self.nx))
        self.alpha_eps = float(getattr(config.MOD, 'alpha_eps', 1e-6))
        self.alpha_background_physical = getattr(config.MOD, 'alpha_background_physical', None)
        self.alpha_He_ref = 0.5 * jnp.ones((self.ny, self.nx))
        self.alpha_Uu_ref = 0.5 * jnp.ones((self.ny, self.nx))
        self.alpha_Up_ref = 0.5 * jnp.ones((self.ny, self.nx))
        # Python-bool flags so JIT branches can use them as static.
        self.has_alpha_Uu_ref = False
        self.has_alpha_Up_ref = False

        path_bg = getattr(config.MOD, 'path_background_params', None)
        if path_bg is not None and os.path.exists(path_bg):
            name_var_bg = getattr(config.MOD, 'name_var_background_params', None) or {}
            ds_bg = xr.open_dataset(path_bg)
            name_lon = name_var_bg.get('lon', 'lon')
            name_lat = name_var_bg.get('lat', 'lat')
            # Optional longitude convention conversion
            if name_lon in ds_bg.coords or name_lon in ds_bg.variables:
                lon_arr = ds_bg[name_lon]
                if np.sign(lon_arr.data.min()) == -1 and State.lon_unit == '0_360':
                    ds_bg = ds_bg.assign_coords({name_lon: ((name_lon, lon_arr.data % 360))})
                    ds_bg = ds_bg.sortby(name_lon)
                elif np.sign(lon_arr.data.min()) >= 0 and State.lon_unit == '-180_180':
                    ds_bg = ds_bg.assign_coords({name_lon: ((name_lon, (lon_arr.data + 180) % 360 - 180))})
                    ds_bg = ds_bg.sortby(name_lon)

            def _read_bg(key, default_name, fill_value=0.):
                vname = name_var_bg.get(key, default_name)
                if vname not in ds_bg.variables:
                    return None
                arr = grid.interp2d(
                    ds_bg,
                    {'lon': name_lon, 'lat': name_lat, key: vname},
                    State.lon, State.lat)
                arr = np.asarray(arr, dtype=float)
                arr[np.isnan(arr)] = fill_value
                return jnp.asarray(arr)

            def _read_alpha_ref(key, default_name):
                vname = name_var_bg.get(key, default_name)
                if vname not in ds_bg.variables:
                    return None
                is_physical = self.alpha_background_physical
                if is_physical is None:
                    is_physical = (
                        f'{vname}_control' in ds_bg.variables
                        or f'{key}_control' in ds_bg.variables
                    )
                arr = _read_bg(key, default_name, fill_value=0.5 if is_physical else 0.)
                arr_np = np.asarray(arr, dtype=float)
                if not is_physical:
                    arr_np = arr_np + 0.5
                if np.nanmin(arr_np) < 0. or np.nanmax(arr_np) > 1.:
                    kind = 'physical alpha' if is_physical else 'legacy centered alpha anomaly'
                    raise ValueError(
                        f'Background {key} interpreted as {kind} gives physical range '
                        f'[{np.nanmin(arr_np)}, {np.nanmax(arr_np)}], outside [0, 1].'
                    )
                return jnp.asarray(np.clip(arr_np, self.alpha_eps, 1. - self.alpha_eps))

            He_mean_bg  = _read_bg('He_mean',  'He_mean')
            alpha_He_ref = _read_alpha_ref('alpha_He', 'alpha_He')
            alpha_Uu_ref = _read_alpha_ref('alpha_Uu', 'alpha_Uu')
            alpha_Up_ref = _read_alpha_ref('alpha_Up', 'alpha_Up')

            if He_mean_bg  is not None: self.He_mean_bg  = He_mean_bg
            if alpha_He_ref is not None: self.alpha_He_ref = alpha_He_ref
            if alpha_Uu_ref is not None: self.alpha_Uu_ref = alpha_Uu_ref
            if alpha_Up_ref is not None: self.alpha_Up_ref = alpha_Up_ref

            self.has_alpha_Uu_ref = bool(np.any(np.abs(np.asarray(self.alpha_Uu_ref) - 0.5) > 1e-12))
            self.has_alpha_Up_ref = bool(np.any(np.abs(np.asarray(self.alpha_Up_ref) - 0.5) > 1e-12))

            ds_bg.close()
            print('[Model_bmit] background IT params loaded from', path_bg,
                  '- alpha params are now dimensionless logit controls around physical references.')

            if config.EXP.flag_plot > 0:
                _, axes = plt.subplots(1, 4, figsize=(22, 4))
                for ax, fld, ttl in zip(
                        axes,
                        [self.He_mean_bg, self.alpha_He_ref, self.alpha_Uu_ref, self.alpha_Up_ref],
                        ['He_mean_bg', 'alpha_He_ref', 'alpha_Uu_ref', 'alpha_Up_ref']):
                    im = ax.pcolormesh(np.asarray(fld))
                    plt.colorbar(im, ax=ax)
                    ax.set_title(ttl)
                plt.suptitle('Background IT params / alpha references')
                plt.show()

        # Sponge layer
        self.sponge_width = config.MOD.dist_sponge_bc
        self.sponge_coef = config.MOD.sponge_coef
        self.flag_bc_sponge = config.MOD.flag_bc_sponge and self.sponge_width is not None and self.sponge_width > 0
        if self.flag_bc_sponge:
            self.Xh = State.X
            self.Yh = State.Y
            self.Xu = 0.5 * (State.X[:, :-1] + State.X[:, 1:])
            self.Yu = 0.5 * (State.Y[:, :-1] + State.Y[:, 1:])
            self.Xv = 0.5 * (State.X[:-1, :] + State.X[1:, :])
            self.Yv = 0.5 * (State.Y[:-1, :] + State.Y[1:, :])

            # Sponge profile via grid.compute_sponge_components
            lon_h = State.lon
            lat_h = State.lat
            lon_u = 0.5 * (lon_h[:, :-1] + lon_h[:, 1:])
            lat_u = 0.5 * (lat_h[:, :-1] + lat_h[:, 1:])
            lon_v = 0.5 * (lon_h[:-1, :] + lon_h[1:, :])
            lat_v = 0.5 * (lat_h[:-1, :] + lat_h[1:, :])

            if config.MOD.use_sponge_on_coast:
                mask_h = State.mask.copy()
                mask_u = mask_h[:, :-1] & mask_h[:, 1:]
                mask_v = mask_h[:-1, :] & mask_h[1:, :]
            else:
                mask_h = np.zeros_like(lon_h, dtype=bool)
                mask_u = np.zeros_like(lon_u, dtype=bool)
                mask_v = np.zeros_like(lon_v, dtype=bool)

            alpha = config.MOD.tangential_sponge_factor

            # h: isotropic sponge
            wc_h, wNS_h, wWE_h = grid.compute_sponge_components(lon_h, lat_h, mask_h, config.MOD.dist_sponge_bc)
            if config.MOD.periodic_y:
                wNS_h[:] = 0.0
            if config.MOD.periodic_x:
                wWE_h[:] = 0.0
            self.sponge_h = 1.0 - (1.0 - wc_h) * (1.0 - wNS_h) * (1.0 - wWE_h)

            # u: normal at W/E (full), tangential at N/S (reduced)
            wc_u, wNS_u, wWE_u = grid.compute_sponge_components(lon_u, lat_u, mask_u, config.MOD.dist_sponge_bc)
            if config.MOD.periodic_y:
                wNS_u[:] = 0.0
            if config.MOD.periodic_x:
                wWE_u[:] = 0.0
            self.sponge_u = 1.0 - (1.0 - wc_u) * (1.0 - alpha * wNS_u) * (1.0 - wWE_u)

            # v: normal at N/S (full), tangential at W/E (reduced)
            wc_v, wNS_v, wWE_v = grid.compute_sponge_components(lon_v, lat_v, mask_v, config.MOD.dist_sponge_bc)
            if config.MOD.periodic_y:
                wNS_v[:] = 0.0
            if config.MOD.periodic_x:
                wWE_v[:] = 0.0
            self.sponge_v = 1.0 - (1.0 - wc_v) * (1.0 - wNS_v) * (1.0 - alpha * wWE_v)

            if config.EXP.flag_plot > 0:
                _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
                im1 = ax1.pcolormesh(self.sponge_h)
                plt.colorbar(im1, ax=ax1)
                ax1.set_title('Sponge SSH')
                im2 = ax2.pcolormesh(self.sponge_u)
                plt.colorbar(im2, ax=ax2)
                ax2.set_title('Sponge U')
                im3 = ax3.pcolormesh(self.sponge_v)
                plt.colorbar(im3, ax=ax3)
                ax3.set_title('Sponge V')
                plt.show()

            if not config.MOD.periodic_y:
                self.sponge_on_h_S = (np.abs(self.Yh-self.Yh[0,:][None,:])<=config.MOD.dist_sponge_bc*1e3) 
                self.sponge_on_u_S = (np.abs(self.Yu-self.Yu[0,:][None,:])<=config.MOD.dist_sponge_bc*1e3)
                self.sponge_on_v_S = (np.abs(self.Yv-self.Yv[0,:][None,:])<=config.MOD.dist_sponge_bc*1e3)
                self.sponge_on_h_N = (np.abs(self.Yh-self.Yh[-1,:][None,:])<=config.MOD.dist_sponge_bc*1e3) 
                self.sponge_on_u_N = (np.abs(self.Yu-self.Yu[-1,:][None,:])<=config.MOD.dist_sponge_bc*1e3) 
                self.sponge_on_v_N = (np.abs(self.Yv-self.Yv[-1,:][None,:])<=config.MOD.dist_sponge_bc*1e3) 
            else:
                self.sponge_on_h_S = np.zeros_like(self.Yh, dtype=bool)
                self.sponge_on_u_S = np.zeros_like(self.Yu, dtype=bool)
                self.sponge_on_v_S = np.zeros_like(self.Yv, dtype=bool)
                self.sponge_on_h_N = np.zeros_like(self.Yh, dtype=bool)
                self.sponge_on_u_N = np.zeros_like(self.Yu, dtype=bool)
                self.sponge_on_v_N = np.zeros_like(self.Yv, dtype=bool)
            if not config.MOD.periodic_x:
                self.sponge_on_h_W = (np.abs(self.Xh-self.Xh[:,0][:,None])<=config.MOD.dist_sponge_bc*1e3) 
                self.sponge_on_u_W = (np.abs(self.Xu-self.Xu[:,0][:,None])<=config.MOD.dist_sponge_bc*1e3)
                self.sponge_on_v_W = (np.abs(self.Xv-self.Xv[:,0][:,None])<=config.MOD.dist_sponge_bc*1e3)
                self.sponge_on_h_E = (np.abs(self.Xh-self.Xh[:,-1][:,None])<=config.MOD.dist_sponge_bc*1e3) 
                self.sponge_on_u_E = (np.abs(self.Xu-self.Xu[:,-1][:,None])<=config.MOD.dist_sponge_bc*1e3)
                self.sponge_on_v_E = (np.abs(self.Xv-self.Xv[:,-1][:,None])<=config.MOD.dist_sponge_bc*1e3)
            else:
                self.sponge_on_h_W = np.zeros_like(self.Yh, dtype=bool)
                self.sponge_on_u_W = np.zeros_like(self.Yu, dtype=bool)
                self.sponge_on_v_W = np.zeros_like(self.Yv, dtype=bool)
                self.sponge_on_h_E = np.zeros_like(self.Yh, dtype=bool)
                self.sponge_on_u_E = np.zeros_like(self.Yu, dtype=bool)
                self.sponge_on_v_E = np.zeros_like(self.Yv, dtype=bool)
            
            self.weight_sponge_u = np.array(self.sponge_on_u_S, dtype=float) + np.array(self.sponge_on_u_N, dtype=float) + np.array(self.sponge_on_u_W, dtype=float) + np.array(self.sponge_on_u_E, dtype=float)
            self.weight_sponge_v = np.array(self.sponge_on_v_S, dtype=float) + np.array(self.sponge_on_v_N, dtype=float) + np.array(self.sponge_on_v_W, dtype=float) + np.array(self.sponge_on_v_E, dtype=float)
            self.weight_sponge_h = np.array(self.sponge_on_h_S, dtype=float) + np.array(self.sponge_on_h_N, dtype=float) + np.array(self.sponge_on_h_W, dtype=float) + np.array(self.sponge_on_h_E, dtype=float)
            self.weight_sponge_u[self.weight_sponge_u==0] = 1
            self.weight_sponge_v[self.weight_sponge_v==0] = 1
            self.weight_sponge_h[self.weight_sponge_h==0] = 1

            # Keep sponge exclusion separate from State.mask so model and basis
            # still include sponge cells.
            if config.MOD.mask_sponge_bc:
                sponge_mask = (self.sponge_on_h_S | self.sponge_on_h_N |
                               self.sponge_on_h_W | self.sponge_on_h_E)
                if getattr(State, 'sponge_mask', None) is None:
                    State.sponge_mask = sponge_mask.copy()
                else:
                    State.sponge_mask = np.asarray(State.sponge_mask, dtype=bool) | sponge_mask
        
        self._init_it_open_boundary_extension(config.MOD)
        self.H_it_open_boundary = self._extend_it_open_boundary_static_field(self.H)

        # Coupling terms from vertical modes
        self.flag_coupling_from_bm = config.MOD.flag_coupling_from_bm
        if self.flag_coupling_from_bm:
            if config.MOD.path_interaction_terms is not None and os.path.exists(config.MOD.path_interaction_terms):
                self._read_interaction_terms(config, State)
            else:
                self._compute_coupling_coefficients(config.MOD.path_vertical_modes)

        # Model initialization
        if self.bm_kind == 'MOD_QGSW':
            config_bm = Config({
                'EXP': config.EXP,
                'GRID': config.GRID,
                'MOD': self.bm_config,
                'INV': config.INV,
            })
            self.model_bm = Model_qgsw(config_bm, State)
            if getattr(self.model_bm, 'nl', 1) != 1:
                raise NotImplementedError(
                    'MOD_BMIT with a QGSW Balanced Motion component currently '
                    'supports nl=1 only; multi-layer BM needs per-layer BM '
                    'state variables in the coupled model.')
        else:
            self.model_bm = model_bm(
                dx=State.DX,
                dy=State.DY,
                dt=self.dt,
                SSH=State.getvar(name_var=self.name_var['SSH_BM']),
                c=self.c_bm,
                time_scheme=config.MOD.time_scheme_bm,
                g=config.MOD.g,
                f=self.f,
                Kdiffus=config.MOD.Kdiffus,
                mdt=self.mdt
                )
            # QG1L BM exposes diagnosed velocity fields on the h-grid.
            ssh_bm = State.getvar(name_var=self.name_var['SSH_BM'])
            ssh_full = ssh_bm + self.mdt if self.mdt is not None else ssh_bm
            u_bm, v_bm = self.model_bm.h2uv(ssh_full)
            State.setvar([u_bm, v_bm], [self.name_var['U_BM'], self.name_var['V_BM']])
            State.params[self.name_var['SSH_BM']] = np.zeros((State.ny, State.nx))
            State.params[self.name_var['U_BM']] = np.zeros_like(State.var[self.name_var['U_BM']])
            State.params[self.name_var['V_BM']] = np.zeros_like(State.var[self.name_var['V_BM']])
        
        self.model_it = model_it(
            X=State.X,
            Y=State.Y,
            dt=self.dt,
            omegas=self.omegas,
            bc_theta=self.bc_theta,
            f=self.f,
            Heb=self.Heb,
            periodic_x=config.MOD.periodic_x,
            periodic_y=config.MOD.periodic_y,
            time_scheme=config.MOD.time_scheme_it,
            )
    
        # Set sponge attributes on IT model for compute_IT_2D
        if self.flag_bc_sponge:
            self.model_it.flag_sponge_bc = True
            self.model_it.sponge_coef = self.sponge_coef
            self.model_it.sponge_u = self.sponge_u
            self.model_it.sponge_v = self.sponge_v
            self.model_it.sponge_h = self.sponge_h
            self.model_it.sponge_on_h_S = self.sponge_on_h_S
            self.model_it.sponge_on_h_N = self.sponge_on_h_N
            self.model_it.sponge_on_h_W = self.sponge_on_h_W
            self.model_it.sponge_on_h_E = self.sponge_on_h_E
            self.model_it.sponge_on_u_S = self.sponge_on_u_S
            self.model_it.sponge_on_u_N = self.sponge_on_u_N
            self.model_it.sponge_on_u_W = self.sponge_on_u_W
            self.model_it.sponge_on_u_E = self.sponge_on_u_E
            self.model_it.sponge_on_v_S = self.sponge_on_v_S
            self.model_it.sponge_on_v_N = self.sponge_on_v_N
            self.model_it.sponge_on_v_W = self.sponge_on_v_W
            self.model_it.sponge_on_v_E = self.sponge_on_v_E
            self.model_it.weight_sponge_u = self.weight_sponge_u
            self.model_it.weight_sponge_v = self.weight_sponge_v
            self.model_it.weight_sponge_h = self.weight_sponge_h

        # IT wave-phase method for sponge BC
        self.model_it.bc_it_method = getattr(config.MOD, 'bc_it_method', 'plane_wave_bdy')
        self.model_it.bc_it_corner_weight_power = getattr(config.MOD, 'bc_it_corner_weight_power', self.model_it.bc_it_corner_weight_power)
        self.max_nstep = getattr(config.MOD, 'max_nstep', 240)

        # Compile jax-related functions
        self._jstep_jit = jit(self._jstep, static_argnames=['nstep'])
        self._jstep_tgl_jit = jit(self._jstep_tgl, static_argnames=['nstep'])
        self._jstep_adj_jit = jit(self._jstep_adj, static_argnames=['nstep'])

        # Functions related to IT time_scheme
        self.time_scheme_it = config.MOD.time_scheme_it
        if self.time_scheme_it == 'Euler':
            self.model_it_step_nstep = self.model_it._step_euler_nstep
        elif self.time_scheme_it == 'rk3':
            self.model_it_step_nstep = self.model_it._step_rk3_nstep
        elif self.time_scheme_it == 'rk4':
            self.model_it_step_nstep = self.model_it._step_rk4_nstep
        else:
            raise ValueError(f"Unsupported time_scheme_it: {self.time_scheme_it}")
        
        if config.INV is not None and config.INV.super=='INV_4DVAR' and config.INV.compute_test:
            print('BM/IT Tangent test:')
            tangent_test(self,State,nstep=10)
            print('BM/IT Adjoint test:')
            adjoint_test(self,State,nstep=10)
    
    def init(self, State, t0=0):

        if self.bm_kind == 'MOD_QGSW':
            self.model_bm.init(State, t0=t0)
            has_u_bc = hasattr(self.model_bm, 'bc') and t0 in self.model_bm.bc.get('U', {})
            has_v_bc = hasattr(self.model_bm, 'bc') and t0 in self.model_bm.bc.get('V', {})
            if (not getattr(self.model_bm, '_is_qg_class', False)) and (not has_u_bc or not has_v_bc):
                h_bm = State.getvar(name_var=self.name_var['SSH_BM'])
                u_bm, v_bm = self.model_bm.ssh2uv(h_bm)
                values = []
                names = []
                if not has_u_bc:
                    values.append(u_bm)
                    names.append(self.name_var['U_BM'])
                if not has_v_bc:
                    values.append(v_bm)
                    names.append(self.name_var['V_BM'])
                State.setvar(values, names)

        if self.bm_kind == 'MOD_QG1L':
            if type(self.init_from_bc_bm)==dict:
                for name in self.init_from_bc_bm:
                    bmit_name = {'U': 'U_BM', 'V': 'V_BM', 'SSH': 'SSH_BM'}.get(name, name)
                    if self.init_from_bc_bm[name] and t0 in self.bc[bmit_name]:
                        State.setvar(self.bc[bmit_name][t0], self.name_var[bmit_name])
            elif self.init_from_bc_bm:
                for name in ('U_BM', 'V_BM', 'SSH_BM'):
                    if t0 in self.bc[name]:
                         State.setvar(self.bc[name][t0], self.name_var[name])

        if self.bm_kind == 'MOD_QG1L':
            h_bm = State.getvar(name_var=self.name_var['SSH_BM'])
            u_bm, v_bm = self._bm_velocity_on_h(
                State.getvar(name_var=self.name_var['U_BM']),
                State.getvar(name_var=self.name_var['V_BM']),
                h_bm)
            State.setvar([u_bm, v_bm], [self.name_var['U_BM'], self.name_var['V_BM']])

        State.var[self.name_var['SSH']] = State.var[self.name_var['SSH_BM']] + State.var[self.name_var['SSH_IT']]
        self._set_land_nan(State)

    def save_output(self,State,present_date,name_var=None,t=None):

        name_var_to_save = [self.name_var['U_BM'],
                            self.name_var['V_BM'],
                            self.name_var['SSH_BM'],
                            self.name_var['SSH_IT'],
                            self.name_var['SSH'],
                            self.name_var['U_IT']+'_interp',
                            self.name_var['V_IT']+'_interp',
                            self.name_var['U_BM']+'_interp',
                            self.name_var['V_BM']+'_interp',
                            'He',
                            'He_mean']
        bm_name_params = getattr(self.model_bm, 'name_params', [])
        if self.bm_kind == 'MOD_QGSW' and 'H' in bm_name_params:
            State.var['H'] = self.model_bm._H_control_to_total_state(State.params['H'])
            State.var['H_control'] = +State.params['H']
            name_var_to_save += ['H', 'H_control']
        if self.bm_kind == 'MOD_QGSW' and 'h_wind' in bm_name_params:
            State.var['h_wind'] = +State.params['h_wind']
            name_var_to_save += ['h_wind']
        if self.bm_kind == 'MOD_QGSW' and 'wind_strength' in bm_name_params:
            State.var['wind_strength'] = +State.params['wind_strength']
            name_var_to_save += ['wind_strength']
        if 'alpha' in self.name_params_it:
            name_var_to_save += ['alpha', 'alpha_control']
        else:
            name_var_to_save += ['alpha_He']
            if 'alpha_He' in self.name_params_it:
                name_var_to_save += ['alpha_He_control']

        mdt = self._bm_mdt_h()
        if mdt is not None:
            State.var[self.name_var['SSH_BM']+'_full'] = State.getvar(name_var=self.name_var['SSH_BM']) + mdt
            State.var[self.name_var['SSH']+'_full'] = State.getvar(name_var=self.name_var['SSH']) + mdt
            name_var_to_save += [self.name_var['SSH_BM']+'_full', self.name_var['SSH']+'_full']

        u = +State.getvar(name_var=self.name_var['U_IT'])
        v = +State.getvar(name_var=self.name_var['V_IT'])
        u_to_save = np.zeros((State.ny,State.nx))
        v_to_save = np.zeros((State.ny,State.nx))
        u_to_save[:,1:-1] = (u[:,1:] + u[:,:-1]) * .5
        v_to_save[1:-1,:] = (v[1:,:] + v[:-1,:]) * .5
        State.var[self.name_var['U_IT']+'_interp'] = u_to_save
        State.var[self.name_var['V_IT']+'_interp'] = v_to_save

        u_bm_h, v_bm_h = self._bm_velocity_on_h(
            State.getvar(name_var=self.name_var['U_BM']),
            State.getvar(name_var=self.name_var['V_BM']),
            State.getvar(name_var=self.name_var['SSH_BM']))
        State.var[self.name_var['U_BM']+'_interp'] = np.asarray(u_bm_h)
        State.var[self.name_var['V_BM']+'_interp'] = np.asarray(v_bm_h)

        if 'He_mean' in self.name_params_it:
            He_mean = +State.params['He_mean']
        else:
            He_mean = jnp.zeros((self.ny,self.nx))
        if 'alpha_He' in self.name_params_it:
            alpha_He_control = +State.params['alpha_He']
        elif 'alpha' in self.name_params_it:
            alpha_He_control = +State.params['alpha']
        else:
            alpha_He_control = None
        h_for_He = self._bm_height_for_He(+State.getvar(name_var=self.name_var['SSH_BM']))
        He2d = self._compute_He_from_bm(He_mean, alpha_He_control, h_for_He)
        alpha_He = self._alpha_control_to_physical_state(alpha_He_control, self.alpha_He_ref)

        State.var['He'] = He2d + self.Heb
        State.var['He_mean'] = He_mean
        State.var['alpha_He'] = alpha_He
        if 'alpha' in self.name_params_it:
            State.var['alpha'] = alpha_He
            State.var['alpha_control'] = State.params['alpha']
        elif 'alpha_He' in self.name_params_it:
            State.var['alpha_He_control'] = State.params['alpha_He']

        if 'alpha_Uu' in self.name_params_it:
            State.var['alpha_Uu'] = self._alpha_control_to_physical_state(
                State.params['alpha_Uu'], self.alpha_Uu_ref)
            State.var['alpha_Uu_control'] = State.params['alpha_Uu']
            name_var_to_save += ['alpha_Uu', 'alpha_Uu_control']
        if 'alpha_Up' in self.name_params_it:
            State.var['alpha_Up'] = self._alpha_control_to_physical_state(
                State.params['alpha_Up'], self.alpha_Up_ref)
            State.var['alpha_Up_control'] = State.params['alpha_Up']
            name_var_to_save += ['alpha_Up', 'alpha_Up_control']

        State.save_output(present_date,
                          name_var=name_var_to_save)

    def set_bc(self,time_bc,var_bc=None,t_bc=None,**kwargs):

        time_keys = t_bc if t_bc is not None else time_bc

        # A QGSW BM component owns its own boundary source and storage.
        if self.bm_kind == 'MOD_QGSW' and var_bc is None:
            self.model_bm.set_bc(time_bc, t_bc=time_keys)
            return

        # QG1L BM boundary conditions are loaded from bm_model and stored by BMIT.
        if var_bc is None and self._bc_fields is not None:
            var_bc = self._bc_fields.interp(time_bc)
        
        if var_bc is None:
            return

        for bm_role, bmit_role in {'U': 'U_BM', 'V': 'V_BM', 'SSH': 'SSH_BM'}.items():
            if bm_role in var_bc and bmit_role not in var_bc:
                var_bc[bmit_role] = var_bc[bm_role]

        if self.bm_kind == 'MOD_QGSW':
            bm_var_bc = {}
            if 'U_BM' in var_bc:
                bm_var_bc['U'] = var_bc['U_BM']
            if 'V_BM' in var_bc:
                bm_var_bc['V'] = var_bc['V_BM']
            if 'SSH_BM' in var_bc:
                bm_var_bc['SSH'] = var_bc['SSH_BM']
            if bm_var_bc:
                self.model_bm.set_bc(time_bc, bm_var_bc, t_bc=time_keys)

        for _name_var_bc in var_bc:
            for _name_var_mod in self.name_var:
                if _name_var_bc==_name_var_mod:
                    for i,t in enumerate(time_keys):
                        var_bc_t = +var_bc[_name_var_bc][i]
                        # Remove nan
                        var_bc_t[np.isnan(var_bc_t)] = 0.
                        # Fill bc dictionnary
                        self.bc[_name_var_mod][t] = var_bc_t
                elif _name_var_bc==f'{_name_var_mod}_params':
                    for i,t in enumerate(time_keys):
                        var_bc[_name_var_bc][i][np.isnan(var_bc[_name_var_bc][i])] = 0.
        
        # For step_jax
        self.bc_time_jax = jnp.array(list(self.bc['SSH_BM'].keys()))
        self.bc_time = np.array(list(self.bc['SSH_BM'].keys()))
        self.bc_values = {t: jnp.array(self.bc['SSH_BM'][t]) for t in self.bc['SSH_BM']}
        for name in var_bc:
            self.bc_values[name] = jnp.array(list(self.bc['SSH_BM'].values()))
            
    def _apply_bc(self,t0):
        
        ssh_bc = jnp.zeros((self.ny,self.nx,))
        
        if 'SSH_BM' not in self.bc:
            return ssh_bc
        elif len(self.bc['SSH_BM'].keys())==0:
             return ssh_bc
        elif t0 not in self.bc_time:
            # Find closest time
            idx_closest = np.argmin(np.abs(self.bc_time-t0))
            t0 = self.bc_time[idx_closest]

        ssh_bc = self.bc_values[t0]
        
        return ssh_bc

    def _rho_on_u(self,rho):
        
        return (rho[:,1:] + rho[:,:-1])/2 
    
    def _rho_on_v(self,rho):
        
        return (rho[1:,:] + rho[:-1,:])/2 
    
    def _compute_coupling_coefficients(self, path_vertical_modes):

        # Open dataset with vertical mode structures
        ds = xr.open_dataset(path_vertical_modes)
        phi1  = ds.phi.sel(mode=1)      # (s_rho, y_rho)
        phi1 = phi1/phi1[-1]
        phi1p = ds.dphidz.sel(mode=1)   # (s_w,  y_rho)  → given derivative!
        c1    = ds.c.sel(mode=1)        # (y_rho)
        N2    = ds.N2                   # (y_rho, s_w)
        z_r   = ds.z_r                  # (s_rho, y_rho)
        z_w = ds.z_w
        dz_r  = ds.dz                   # (s_rho, y_rho)
        s_w = ds.s_w

        #########################################################################
        # Compute U11 coupling term for mode 1
        #########################################################################
        integrand = phi1/phi1[-1] * (phi1**2) * dz_r  # multiply by layer thickness
        U11 = (1/self.H) * integrand.sum(dim="s_rho")   # integral in z
        self.U11 = jnp.array(U11.values) 
        self.U11 = jnp.where(self.U11<0, 0, self.U11)
        plt.figure()
        plt.pcolormesh(self.U11)
        plt.colorbar()
        plt.title('U11')
        plt.show()

        #########################################################################
        # Compute U11p coupling term for mode 1
        #########################################################################
        φ = -(c1**2 / N2)* phi1p          # (s_w, y_rho)
        phi1_w = phi1.interp(s_rho=s_w)   # now on (s_w, y_rho)
        phi1_w[0] = phi1_w[1]
        phi1_w[-1] = phi1_w[-2]

        integrand = (
            phi1_w/phi1_w[-1]
            * (N2 / c1**2)
            * φ**2
        )   # dims: (s_w, y_rho)
        # thickness between w-levels: same size as s_w except top/bottom.
        dz_w = z_w.diff("s_w")                   # (s_w_minus1, y_rho)
        # pad to same length as s_w (xarray aligns automatically)
        dz_w = dz_w.reindex(s_w=s_w, fill_value=np.nan)
        U11p = (1/self.H) * (integrand * dz_w).sum(dim="s_w")   # integral in z
        self.U11p = jnp.array(U11p.values)
        self.U11p = jnp.where(self.U11p<0, 0, self.U11p)
        plt.figure()
        plt.pcolormesh(self.U11p)
        plt.colorbar()
        plt.title('U11p')
        plt.show()

        #########################################################################
        # Compute dHe coupling term for mode 1
        #########################################################################
        # forward and backward slopes
        slope_f = (phi1.shift(s_rho=-1) - phi1) / (z_r.shift(s_rho=-1) - z_r)
        slope_b = (phi1 - phi1.shift(s_rho=1)) / (z_r - z_r.shift(s_rho=1))

        # central 2nd derivative
        phi1pp = 2 * (slope_f - slope_b) / (z_r.shift(s_rho=-1) - z_r.shift(s_rho=1))

        # fill top/bottom NaNs (simple, safe)
        phi1pp = phi1pp.fillna(0)

        # vertical interpolation from s_w → s_rho
        phi1p_rho = phi1p.interp(s_w=ds.s_rho)

        N2_rho = N2.interp(s_w=ds.s_rho)

        c1_rho = c1.broadcast_like(phi1p_rho)

        A = - (c1_rho**2 / N2_rho) * phi1p_rho
        A2 = A**2

        integrand = phi1pp * A2
        dHe = (1/self.H) * (integrand * dz_r).sum("s_rho")
        self.dHe = jnp.array(dHe.values) 
        self.dHe = jnp.where(self.dHe<0, 0, self.dHe)
        plt.figure()
        plt.pcolormesh(self.dHe)
        plt.colorbar()
        plt.title('dHe')
        plt.show()

    def _read_interaction_terms(self, config, State):
        """Read interaction terms (U11, U11p, dHe) from a netcdf file
        instead of computing them from vertical modes."""

        ds = xr.open_dataset(config.MOD.path_interaction_terms)
        name_lon = config.MOD.name_var_interaction_terms['lon']
        lon = ds[name_lon]
        # Convert longitude
        if np.sign(lon.data.min()) == -1 and State.lon_unit == '0_360':
            ds = ds.assign_coords({name_lon: ((name_lon, lon.data % 360))})
            ds = ds.sortby(name_lon)
        elif np.sign(lon.data.min()) >= 0 and State.lon_unit == '-180_180':
            ds = ds.assign_coords({name_lon: ((name_lon, (lon.data + 180) % 360 - 180))})
            ds = ds.sortby(name_lon)

        name_vars = config.MOD.name_var_interaction_terms

        def _read_field(var_key, ds, name_vars, State, divisor):
            """Read and interpolate a field if variable name is defined and exists in the dataset."""
            if name_vars.get(var_key) and name_vars[var_key] in ds:
                return jnp.array(grid.interp2d(
                    ds, {'lon': name_vars['lon'], 'lat': name_vars['lat'], 'var': name_vars[var_key]},
                    State.lon, State.lat)) / divisor
            else:
                print(f'Warning: {var_key} not found or not defined. Setting to zero.')
                return jnp.zeros((State.ny, State.nx))

        # Read U11
        self.U11 = _read_field('U11_u', ds, name_vars, State, self.H)
        self.U11 = jnp.where(self.U11<0, 0, self.U11)  # Ensure non-negativity
        self.U11 = jnp.where(self.mask, 0, self.U11)  # Mask land points
        self.U11 = jnp.where(jnp.isnan(self.U11), 0, self.U11)  # Ensure no NaNs
        
        # Read U11p
        self.U11p = _read_field('U11_p', ds, name_vars, State, self.H)
        self.U11p = jnp.where(self.U11p<0, 0, self.U11p)  # Ensure non-negativity
        self.U11p = jnp.where(self.mask, 0, self.U11p)  # Mask land points
        self.U11p = jnp.where(jnp.isnan(self.U11p), 0, self.U11p)  # Ensure no NaNs
        
        # Read dc2 (correction to c^2) and convert to dHe = dc2 / g
        self.dHe = _read_field('dc2', ds, name_vars, State, self.H * self.g)
        self.dHe = jnp.where(self.dHe<0, 0, self.dHe)  # Ensure non-negativity
        self.dHe = jnp.where(self.mask, 0, self.dHe)  # Mask land points
        self.dHe = jnp.where(jnp.isnan(self.dHe), 0, self.dHe)  # Ensure no NaNs
        
        if config.EXP.flag_plot > 0:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for ax, data, title in zip(axes,
                                       [self.U11, self.U11p, self.dHe],
                                       ['U11', 'U11p', 'dHe']):
                im = ax.pcolormesh(data)
                plt.colorbar(im, ax=ax)
                ax.set_title(title)
            plt.suptitle('Interaction terms (from file)')
            plt.show()

    def _alpha_control_to_physical(self, alpha_control, alpha_ref):

        alpha_ref = jnp.clip(jnp.asarray(alpha_ref), self.alpha_eps, 1. - self.alpha_eps)
        logit_ref = jnp.log(alpha_ref) - jnp.log1p(-alpha_ref)
        if alpha_control is None:
            alpha_control = 0.
        return jax.nn.sigmoid(logit_ref + jnp.asarray(alpha_control))

    def _alpha_control_to_centered(self, alpha_control, alpha_ref):

        return self._alpha_control_to_physical(alpha_control, alpha_ref) - 0.5

    def _alpha_control_to_physical_state(self, alpha_control, alpha_ref):

        return np.asarray(self._alpha_control_to_physical(alpha_control, alpha_ref))

    def _compute_He_from_bm(self, He, alpha_He, h_bm):
        
        """ Compute equivalent depth with coupling term.

        He is interpreted as an anomaly around `self.He_mean_bg`.
        alpha_He is a dimensionless logit-control around the physical
        reference `self.alpha_He_ref`; the dynamics uses alpha - 0.5.
        --------------
        Inputs:
        He        : IT equivalent depth anomaly control parameters (without units)
        alpha_He  : IT He coupling anomaly coefficient (without units)
        h_bm      : BM sea surface height (m) 
        --------------
        Outputs:
        He_tot    : Total equivalent depth with coupling term (m)
        """

        # Add background to He_mean anomaly
        if He is not None:
            He_eff = He + self.He_mean_bg
        else:
            He_eff = self.He_mean_bg

        if self.flag_coupling_from_bm and h_bm is not None:
            alpha_He_eff = self._alpha_control_to_centered(alpha_He, self.alpha_He_ref)
            dHe_bm = self.dHe * h_bm
            if self.flag_bc_sponge:
                dHe_bm = dHe_bm * (1 - self.sponge_h)
            He2d = He_eff + (.5 + alpha_He_eff) * dHe_bm
        else:
            He2d = He_eff

        He2d = jnp.where(self.mask, 0., He2d)
        
        return He2d

    def _compute_it_open_boundary_He_total(self, He, alpha_He, h_bm):

        Heb = self._extend_it_open_boundary_field(self.Heb)
        if He is not None:
            He_eff = He + self.He_mean_bg
        else:
            He_eff = self.He_mean_bg
        He_eff = self._extend_it_open_boundary_field(He_eff)

        if self.flag_coupling_from_bm and h_bm is not None:
            alpha_He_eff = self._alpha_control_to_centered(alpha_He, self.alpha_He_ref)
            alpha_He_eff = self._extend_it_open_boundary_field(alpha_He_eff)
            dHe_bm = self.dHe * h_bm
            if self.flag_bc_sponge:
                dHe_bm = dHe_bm * (1 - self.sponge_h)
            He2d = He_eff + (.5 + alpha_He_eff) * dHe_bm
        else:
            He2d = He_eff

        He2d = jnp.where(self.mask, 0., He2d)
        return Heb + He2d

    def _compute_advective_terms_from_bm(self, alpha_Uu, alpha_Up, u_bm, v_bm):

        """ Compute advective terms with coupling term.

        alpha_Uu and alpha_Up are dimensionless logit-controls around the
        physical references `self.alpha_Uu_ref` and `self.alpha_Up_ref`; the
        dynamics uses alpha - 0.5.
        --------------
        Inputs:
        alpha_Uu  : IT advective anomaly control parameter for U11 (without units)
        alpha_Up  : IT advective anomaly control parameter for U11p (without units)
        u_bm      : BM zonal velocity (m/s)
        v_bm      : BM meridional velocity (m/s)
        --------------
        Outputs:
        u11, v11  : advective coupling terms for u and v (m/s)
        u11p, v11p: advective coupling terms for u and v (m/s)
        """

        if self.flag_coupling_from_bm and u_bm is not None and v_bm is not None:
            # alpha_Uu physical value is bounded in (0, 1); aUu is centered.
            if alpha_Uu is not None or self.has_alpha_Uu_ref:
                aUu = self._alpha_control_to_centered(alpha_Uu, self.alpha_Uu_ref)
                u11 = ((.5 - aUu) + (.5 + aUu) * self.U11) * u_bm 
                v11 = ((.5 - aUu) + (.5 + aUu) * self.U11) * v_bm 
                if self.flag_bc_sponge:
                    u11 = u11 * (1 - self.sponge_h)
                    v11 = v11 * (1 - self.sponge_h)
            else:
                u11 = None
                v11 = None
            # alpha_Up physical value is bounded in (0, 1); aUp is centered.
            if alpha_Up is not None or self.has_alpha_Up_ref:
                aUp = self._alpha_control_to_centered(alpha_Up, self.alpha_Up_ref)
                u11p = ((.5 - aUp) + (.5 + aUp) * self.U11p) * u_bm 
                v11p = ((.5 - aUp) + (.5 + aUp) * self.U11p) * v_bm 
                if self.flag_bc_sponge:
                    u11p = u11p * (1 - self.sponge_h)
                    v11p = v11p * (1 - self.sponge_h)
            else:
                u11p = None
                v11p = None
        else:
            u11 = None
            v11 = None
            u11p = None
            v11p = None
        
        return u11, v11, u11p, v11p

    def _bm_mdt_h(self):
        if self.bm_kind == 'MOD_QGSW' and getattr(self.model_bm, 'mdt', None) is not None:
            return self.model_bm.mdt[0, 0].T
        if getattr(self, 'mdt', None) is not None:
            return self.mdt
        return None

    def _bm_mduv(self):
        if self.bm_kind == 'MOD_QGSW' and getattr(self.model_bm, 'mdu', None) is not None:
            return self.model_bm.mdu[0, 0].T, self.model_bm.mdv[0, 0].T
        return None, None

    def _bm_height_for_He(self, h_bm):
        if self.bm_height_for_He == 'full':
            mdt = self._bm_mdt_h()
            if mdt is not None:
                return h_bm + mdt
        return h_bm

    def _bm_velocity_on_h(self, u_bm, v_bm, h_bm):
        if self.bm_kind == 'MOD_QGSW':
            u_full = u_bm
            v_full = v_bm
            mdu, mdv = self._bm_mduv()
            if mdu is not None:
                u_full = u_full + mdu
                v_full = v_full + mdv
            u_h = 0.5 * (u_full[:, :-1] + u_full[:, 1:])
            v_h = 0.5 * (v_full[:-1, :] + v_full[1:, :])
            return u_h, v_h

        mdt = self._bm_mdt_h()
        ssh_full = h_bm + mdt if mdt is not None else h_bm
        return self.model_bm.h2uv(ssh_full)

    def _bm_step_qgsw(self, t, u_bm, v_bm, h_bm,
                      Fu_bm, Fv_bm, Fh_bm, H_bm,
                      u_bm_bc, v_bm_bc, h_bm_bc,
                      taux_bm=None, tauy_bm=None, nstep=1):
        dtype = self.model_bm.dtype
        u0 = jnp.expand_dims(jnp.asarray(u_bm, dtype=dtype).T, axis=(0, 1))
        v0 = jnp.expand_dims(jnp.asarray(v_bm, dtype=dtype).T, axis=(0, 1))
        h0 = jnp.expand_dims(jnp.asarray(h_bm, dtype=dtype).T, axis=(0, 1))
        H = None if H_bm is None else jnp.expand_dims(jnp.asarray(H_bm, dtype=dtype).T, axis=0)
        u1, v1, h1 = self.model_bm.jstep_jit(
            t, u0, v0, h0, H, Fu_bm, Fv_bm, Fh_bm,
            u_bm_bc, v_bm_bc, h_bm_bc,
            taux=taux_bm, tauy=tauy_bm,
            h_wind=None, wind_strength=None, nstep=nstep)
        return u1[0, 0].T, v1[0, 0].T, h1[0, 0].T

    def _jstep(self, t,
            u_bm, v_bm, h_bm, u_it, v_it, h_it, h_tot,
            Fu_bm, Fv_bm, Fh_bm, H_bm, He_mean, alpha_He, alpha_Uu, alpha_Up,
            h_SN, h_WE, u_bm_bc, v_bm_bc, h_bm_bc,
            taux_bm=None, tauy_bm=None, nstep=1):

        """Multi-step forward integration of the coupled BM/IT model."""

        h_for_He = self._bm_height_for_He(h_bm)
        He2d = self._compute_He_from_bm(He_mean, alpha_He, h_for_He)
        He_tot = self._compute_it_open_boundary_He_total(He_mean, alpha_He, h_for_He)

        u_bm_h, v_bm_h = self._bm_velocity_on_h(u_bm, v_bm, h_bm)
        u11, v11, u11p, v11p = self._compute_advective_terms_from_bm(
            alpha_Uu, alpha_Up, u_bm_h, v_bm_h)

        # BM nstep forward
        if self.bm_kind == 'MOD_QGSW':
            u_bm, v_bm, h_bm = self._bm_step_qgsw(
                t, u_bm, v_bm, h_bm,
                Fu_bm, Fv_bm, Fh_bm, H_bm,
                u_bm_bc, v_bm_bc, h_bm_bc,
                taux_bm=taux_bm, tauy_bm=tauy_bm, nstep=nstep)
        else:
            h_bm = self.model_bm.step(h_bm, h_bm_bc, nstep=nstep)
            if Fh_bm is not None:
                h_bm = h_bm + nstep * self.dt/(3600*24) * Fh_bm
            u_bm, v_bm = self._bm_velocity_on_h(u_bm, v_bm, h_bm)

        # IT nstep forward (scan + checkpoint inside swm)
        u_it, v_it, h_it = self.model_it_step_nstep(
            u_it, v_it, h_it, He2d,
            u11u=u11, v11u=v11, u11p=u11p, v11p=v11p,
            nstep=nstep, t=t, He_total=He_tot, h_SN=h_SN, h_WE=h_WE)

        h_tot = h_bm + h_it

        return u_bm, v_bm, h_bm, u_it, v_it, h_it, h_tot

    def _jstep_tgl(self, t,
                du_bm, dv_bm, dh_bm, du_it, dv_it, dh_it, dh_tot,
                dFu_bm, dFv_bm, dFh_bm, dH_bm, dHe_mean, dalpha_He, dalpha_Uu, dalpha_Up,
                dh_SN, dh_WE,
                u_bm, v_bm, h_bm, u_it, v_it, h_it, h_tot,
                Fu_bm, Fv_bm, Fh_bm, H_bm, He_mean, alpha_He, alpha_Uu, alpha_Up,
                h_SN, h_WE, u_bm_bc, v_bm_bc, h_bm_bc,
                taux_bm=None, tauy_bm=None, nstep=1):
        """Multi-step tangent linear of the coupled BM/IT model using JVP."""

        def wrapped_jstep(x):
            u_bm, v_bm, h_bm, u_it, v_it, h_it, h_tot, \
                Fu_bm, Fv_bm, Fh_bm, H_bm, He_mean, alpha_He, alpha_Uu, alpha_Up, h_SN, h_WE = x
            return self._jstep(
                t, u_bm, v_bm, h_bm, u_it, v_it, h_it, h_tot,
                Fu_bm, Fv_bm, Fh_bm, H_bm, He_mean, alpha_He, alpha_Uu, alpha_Up,
                h_SN, h_WE, u_bm_bc, v_bm_bc, h_bm_bc,
                taux_bm=taux_bm, tauy_bm=tauy_bm, nstep=nstep)

        primals = ((u_bm, v_bm, h_bm, u_it, v_it, h_it, h_tot,
                    Fu_bm, Fv_bm, Fh_bm, H_bm, He_mean, alpha_He, alpha_Uu, alpha_Up, h_SN, h_WE),)
        tangents = ((du_bm, dv_bm, dh_bm, du_it, dv_it, dh_it, dh_tot,
                     dFu_bm, dFv_bm, dFh_bm, dH_bm, dHe_mean, dalpha_He, dalpha_Uu, dalpha_Up, dh_SN, dh_WE),)

        _, dy = jax.jvp(wrapped_jstep, primals, tangents)
        return dy

    def _jstep_adj(self, t,
                adu_bm, adv_bm, adh_bm, adu_it, adv_it, adh_it, adh_tot,
                adFu_bm, adFv_bm, adFh_bm, adH_bm, adHe_mean, adalpha_He, adalpha_Uu, adalpha_Up,
                adh_SN, adh_WE,
                u_bm, v_bm, h_bm, u_it, v_it, h_it, h_tot,
                Fu_bm, Fv_bm, Fh_bm, H_bm, He_mean, alpha_He, alpha_Uu, alpha_Up,
                h_SN, h_WE, u_bm_bc, v_bm_bc, h_bm_bc,
                taux_bm=None, tauy_bm=None, nstep=1):
        """Multi-step adjoint of the coupled BM/IT model using VJP."""

        def wrapped_jstep(x):
            u_bm, v_bm, h_bm, u_it, v_it, h_it, h_tot, \
                Fu_bm, Fv_bm, Fh_bm, H_bm, He_mean, alpha_He, alpha_Uu, alpha_Up, h_SN, h_WE = x
            return self._jstep(
                t, u_bm, v_bm, h_bm, u_it, v_it, h_it, h_tot,
                Fu_bm, Fv_bm, Fh_bm, H_bm, He_mean, alpha_He, alpha_Uu, alpha_Up,
                h_SN, h_WE, u_bm_bc, v_bm_bc, h_bm_bc,
                taux_bm=taux_bm, tauy_bm=tauy_bm, nstep=nstep)

        primals = ((u_bm, v_bm, h_bm, u_it, v_it, h_it, h_tot,
                    Fu_bm, Fv_bm, Fh_bm, H_bm, He_mean, alpha_He, alpha_Uu, alpha_Up, h_SN, h_WE),)
        cotangents = (adu_bm, adv_bm, adh_bm, adu_it, adv_it, adh_it, adh_tot)

        _, vjp_fn = jax.vjp(wrapped_jstep, *primals)
        adjoints = vjp_fn(cotangents)

        adu_bm, adv_bm, adh_bm, adu_it, adv_it, adh_it, adh_tot, \
            _adFu_bm, _adFv_bm, _adFh_bm, _adH_bm, _adHe_mean, _adalpha_He, \
            _adalpha_Uu, _adalpha_Up, _adhSN, _adhWE = adjoints[0]

        adFu_bm += _adFu_bm
        adFv_bm += _adFv_bm
        adFh_bm += _adFh_bm
        if adH_bm is not None:
            adH_bm += _adH_bm
        if adHe_mean is not None:
            adHe_mean += _adHe_mean
        if adalpha_He is not None:
            adalpha_He += _adalpha_He
        if adalpha_Uu is not None:
            adalpha_Uu += _adalpha_Uu
        if adalpha_Up is not None:
            adalpha_Up += _adalpha_Up
        if adh_SN is not None:
            adh_SN += _adhSN
        if adh_WE is not None:
            adh_WE += _adhWE

        return (adu_bm, adv_bm, adh_bm, adu_it, adv_it, adh_it, adh_tot,
                adFu_bm, adFv_bm, adFh_bm, adH_bm, adHe_mean, adalpha_He,
                adalpha_Uu, adalpha_Up, adh_SN, adh_WE)

    def _bm_bc_for_chunk(self, t_chunk, n_chunk):
        if self.bm_kind == 'MOD_QGSW':
            return self.model_bm._apply_bc(t_chunk, int(t_chunk + n_chunk * self.dt))
        h_b = self._apply_bc(t_chunk)
        u_b = jnp.zeros((self.ny, self.nx))
        v_b = jnp.zeros((self.ny, self.nx))
        return u_b, v_b, h_b

    def _bm_wind_for_chunk(self, t_chunk):
        if self.bm_kind == 'MOD_QGSW':
            return self.model_bm._get_wind_stress(t_chunk)
        return None, None

    def _bm_controls(self, State):
        u_name = self.name_var['U_BM']
        v_name = self.name_var['V_BM']
        h_name = self.name_var['SSH_BM']
        Fu = State.params.get(u_name, np.zeros_like(State.var[u_name]))
        Fv = State.params.get(v_name, np.zeros_like(State.var[v_name]))
        Fh = State.params.get(h_name, np.zeros_like(State.var[h_name]))
        return Fu, Fv, Fh

    def _bm_H_control(self, State):
        if self.bm_kind == 'MOD_QGSW' and 'H' in getattr(self.model_bm, 'name_params', []):
            return State.params['H']
        return None

    def _it_controls(self, State):
        if 'He_mean' in self.name_params_it:
            He_mean = +State.params['He_mean']
        else:
            He_mean = jnp.zeros((self.ny,self.nx))
        if 'alpha_He' in self.name_params_it:
            alpha_He = +State.params['alpha_He']
        elif 'alpha' in self.name_params_it:
            alpha_He = +State.params['alpha']
        else:
            alpha_He = None
        if 'alpha_Uu' in self.name_params_it:
            alpha_Uu = +State.params['alpha_Uu']
        elif 'alpha' in self.name_params_it:
            alpha_Uu = +State.params['alpha']
        else:
            alpha_Uu = None
        if 'alpha_Up' in self.name_params_it:
            alpha_Up = +State.params['alpha_Up']
        elif 'alpha' in self.name_params_it:
            alpha_Up = +State.params['alpha']
        else:
            alpha_Up = None
        if 'hbc' in self.name_params_it:
            h_SN = +State.params['hbcx']
            h_WE = +State.params['hbcy']
        else:
            h_SN = None
            h_WE = None
        return He_mean, alpha_He, alpha_Uu, alpha_Up, h_SN, h_WE

    def step(self,State,nstep=1,t=0):

        u_bm = +State.getvar(name_var=self.name_var['U_BM'])
        v_bm = +State.getvar(name_var=self.name_var['V_BM'])
        h_bm = +State.getvar(name_var=self.name_var['SSH_BM'])
        u_it = +State.getvar(name_var=self.name_var['U_IT'])
        v_it = +State.getvar(name_var=self.name_var['V_IT'])
        h_it = +State.getvar(name_var=self.name_var['SSH_IT'])
        h_tot = +State.getvar(name_var=self.name_var['SSH'])

        Fu_bm, Fv_bm, Fh_bm = self._bm_controls(State)
        H_bm = self._bm_H_control(State)
        He_mean, alpha_He, alpha_Uu, alpha_Up, h_SN, h_WE = self._it_controls(State)

        step_done = 0
        while step_done < nstep:
            n_chunk = min(nstep - step_done, self.max_nstep) if self.max_nstep > 0 else (nstep - step_done)
            t_chunk = t + step_done * self.dt
            u_bm_bc, v_bm_bc, h_bm_bc = self._bm_bc_for_chunk(t_chunk, n_chunk)
            taux_bm, tauy_bm = self._bm_wind_for_chunk(t_chunk)
            u_bm, v_bm, h_bm, u_it, v_it, h_it, h_tot = self._jstep_jit(
                t_chunk,
                u_bm, v_bm, h_bm, u_it, v_it, h_it, h_tot,
                Fu_bm, Fv_bm, Fh_bm, H_bm, He_mean, alpha_He, alpha_Uu, alpha_Up,
                h_SN, h_WE, u_bm_bc, v_bm_bc, h_bm_bc,
                taux_bm=taux_bm, tauy_bm=tauy_bm, nstep=n_chunk)
            step_done += n_chunk

        State.setvar([u_bm, v_bm, h_bm, u_it, v_it, h_it, h_tot],[
            self.name_var['U_BM'],
            self.name_var['V_BM'],
            self.name_var['SSH_BM'],
            self.name_var['U_IT'],
            self.name_var['V_IT'],
            self.name_var['SSH_IT'],
            self.name_var['SSH']]
            )

    def step_tgl(self,dState,State,nstep=1,t=0):

        du_bm = +dState.getvar(name_var=self.name_var['U_BM'])
        dv_bm = +dState.getvar(name_var=self.name_var['V_BM'])
        dh_bm = +dState.getvar(name_var=self.name_var['SSH_BM'])
        du_it = +dState.getvar(name_var=self.name_var['U_IT'])
        dv_it = +dState.getvar(name_var=self.name_var['V_IT'])
        dh_it = +dState.getvar(name_var=self.name_var['SSH_IT'])
        dh_tot = +dState.getvar(name_var=self.name_var['SSH'])
        u_bm = +State.getvar(name_var=self.name_var['U_BM'])
        v_bm = +State.getvar(name_var=self.name_var['V_BM'])
        h_bm = +State.getvar(name_var=self.name_var['SSH_BM'])
        u_it = +State.getvar(name_var=self.name_var['U_IT'])
        v_it = +State.getvar(name_var=self.name_var['V_IT'])
        h_it = +State.getvar(name_var=self.name_var['SSH_IT'])
        h_tot = +State.getvar(name_var=self.name_var['SSH'])

        Fu_bm, Fv_bm, Fh_bm = self._bm_controls(State)
        dFu_bm, dFv_bm, dFh_bm = self._bm_controls(dState)
        H_bm = self._bm_H_control(State)
        dH_bm = self._bm_H_control(dState)
        He_mean, alpha_He, alpha_Uu, alpha_Up, h_SN, h_WE = self._it_controls(State)
        dHe_mean, dalpha_He, dalpha_Uu, dalpha_Up, dh_SN, dh_WE = self._it_controls(dState)

        step_done = 0
        while step_done < nstep:
            n_chunk = min(nstep - step_done, self.max_nstep) if self.max_nstep > 0 else (nstep - step_done)
            t_chunk = t + step_done * self.dt
            u_bm_bc, v_bm_bc, h_bm_bc = self._bm_bc_for_chunk(t_chunk, n_chunk)
            taux_bm, tauy_bm = self._bm_wind_for_chunk(t_chunk)
            du_bm, dv_bm, dh_bm, du_it, dv_it, dh_it, dh_tot = self._jstep_tgl_jit(
                t_chunk,
                du_bm, dv_bm, dh_bm, du_it, dv_it, dh_it, dh_tot,
                dFu_bm, dFv_bm, dFh_bm, dH_bm, dHe_mean, dalpha_He, dalpha_Uu, dalpha_Up, dh_SN, dh_WE,
                u_bm, v_bm, h_bm, u_it, v_it, h_it, h_tot,
                Fu_bm, Fv_bm, Fh_bm, H_bm, He_mean, alpha_He, alpha_Uu, alpha_Up, h_SN, h_WE,
                u_bm_bc, v_bm_bc, h_bm_bc,
                taux_bm=taux_bm, tauy_bm=tauy_bm, nstep=n_chunk)
            step_done += n_chunk
            if step_done < nstep:
                u_bm, v_bm, h_bm, u_it, v_it, h_it, h_tot = self._jstep_jit(
                    t_chunk,
                    u_bm, v_bm, h_bm, u_it, v_it, h_it, h_tot,
                    Fu_bm, Fv_bm, Fh_bm, H_bm, He_mean, alpha_He, alpha_Uu, alpha_Up,
                    h_SN, h_WE, u_bm_bc, v_bm_bc, h_bm_bc,
                    taux_bm=taux_bm, tauy_bm=tauy_bm, nstep=n_chunk)

        dState.setvar([du_bm, dv_bm, dh_bm, du_it, dv_it, dh_it, dh_tot],[
            self.name_var['U_BM'],
            self.name_var['V_BM'],
            self.name_var['SSH_BM'],
            self.name_var['U_IT'],
            self.name_var['V_IT'],
            self.name_var['SSH_IT'],
            self.name_var['SSH']]
            )

    def step_adj(self, adState, State, nstep=1, t=0):

        adu_bm = +adState.getvar(name_var=self.name_var['U_BM'])
        adv_bm = +adState.getvar(name_var=self.name_var['V_BM'])
        adh_bm = +adState.getvar(name_var=self.name_var['SSH_BM'])
        adu_it = +adState.getvar(name_var=self.name_var['U_IT'])
        adv_it = +adState.getvar(name_var=self.name_var['V_IT'])
        adh_it = +adState.getvar(name_var=self.name_var['SSH_IT'])
        adh_tot = +adState.getvar(name_var=self.name_var['SSH'])
        u_bm = +State.getvar(name_var=self.name_var['U_BM'])
        v_bm = +State.getvar(name_var=self.name_var['V_BM'])
        h_bm = +State.getvar(name_var=self.name_var['SSH_BM'])
        u_it = +State.getvar(name_var=self.name_var['U_IT'])
        v_it = +State.getvar(name_var=self.name_var['V_IT'])
        h_it = +State.getvar(name_var=self.name_var['SSH_IT'])
        h_tot = +State.getvar(name_var=self.name_var['SSH'])

        Fu_bm, Fv_bm, Fh_bm = self._bm_controls(State)
        adFu_bm, adFv_bm, adFh_bm = self._bm_controls(adState)
        H_bm = self._bm_H_control(State)
        adH_bm = self._bm_H_control(adState)
        He_mean, alpha_He, alpha_Uu, alpha_Up, h_SN, h_WE = self._it_controls(State)
        adHe_mean, adalpha_He, adalpha_Uu, adalpha_Up, adh_SN, adh_WE = self._it_controls(adState)

        chunks = []
        step_done = 0
        while step_done < nstep:
            n_chunk = min(nstep - step_done, self.max_nstep) if self.max_nstep > 0 else (nstep - step_done)
            t_chunk = t + step_done * self.dt
            chunks.append((t_chunk, n_chunk))
            step_done += n_chunk

        fwd_states = [(u_bm, v_bm, h_bm, u_it, v_it, h_it, h_tot)]
        u_fwd, v_fwd, h_fwd, uit_fwd, vit_fwd, hit_fwd, htot_fwd = fwd_states[0]
        for t_chunk, n_chunk in chunks[:-1]:
            u_bm_bc, v_bm_bc, h_bm_bc = self._bm_bc_for_chunk(t_chunk, n_chunk)
            taux_bm, tauy_bm = self._bm_wind_for_chunk(t_chunk)
            u_fwd, v_fwd, h_fwd, uit_fwd, vit_fwd, hit_fwd, htot_fwd = self._jstep_jit(
                t_chunk,
                u_fwd, v_fwd, h_fwd, uit_fwd, vit_fwd, hit_fwd, htot_fwd,
                Fu_bm, Fv_bm, Fh_bm, H_bm, He_mean, alpha_He, alpha_Uu, alpha_Up,
                h_SN, h_WE, u_bm_bc, v_bm_bc, h_bm_bc,
                taux_bm=taux_bm, tauy_bm=tauy_bm, nstep=n_chunk)
            fwd_states.append((u_fwd, v_fwd, h_fwd, uit_fwd, vit_fwd, hit_fwd, htot_fwd))

        for i in range(len(chunks) - 1, -1, -1):
            t_chunk, n_chunk = chunks[i]
            u_i, v_i, h_i, uit_i, vit_i, hit_i, htot_i = fwd_states[i]
            u_bm_bc, v_bm_bc, h_bm_bc = self._bm_bc_for_chunk(t_chunk, n_chunk)
            taux_bm, tauy_bm = self._bm_wind_for_chunk(t_chunk)
            (adu_bm, adv_bm, adh_bm, adu_it, adv_it, adh_it, adh_tot,
             adFu_bm, adFv_bm, adFh_bm, adH_bm, adHe_mean, adalpha_He,
             adalpha_Uu, adalpha_Up, adh_SN, adh_WE) = self._jstep_adj_jit(
                t_chunk,
                adu_bm, adv_bm, adh_bm, adu_it, adv_it, adh_it, adh_tot,
                adFu_bm, adFv_bm, adFh_bm, adH_bm, adHe_mean, adalpha_He, adalpha_Uu, adalpha_Up,
                adh_SN, adh_WE,
                u_i, v_i, h_i, uit_i, vit_i, hit_i, htot_i,
                Fu_bm, Fv_bm, Fh_bm, H_bm, He_mean, alpha_He, alpha_Uu, alpha_Up,
                h_SN, h_WE, u_bm_bc, v_bm_bc, h_bm_bc,
                taux_bm=taux_bm, tauy_bm=tauy_bm, nstep=n_chunk)

        adState.setvar([adu_bm, adv_bm, adh_bm, adu_it, adv_it, adh_it, adh_tot],[
            self.name_var['U_BM'],
            self.name_var['V_BM'],
            self.name_var['SSH_BM'],
            self.name_var['U_IT'],
            self.name_var['V_IT'],
            self.name_var['SSH_IT'],
            self.name_var['SSH']]
            )

        adState.params[self.name_var['U_BM']] = adFu_bm
        adState.params[self.name_var['V_BM']] = adFv_bm
        adState.params[self.name_var['SSH_BM']] = adFh_bm
        if adH_bm is not None:
            adState.params['H'] = adH_bm
        if 'He_mean' in self.name_params_it:
            adState.params['He_mean'] = adHe_mean
        adalpha = 0.
        if 'alpha_He' in self.name_params_it:
            adState.params['alpha_He'] = adalpha_He
        elif 'alpha' in self.name_params_it:
            adalpha += adalpha_He
        if 'alpha_Uu' in self.name_params_it:
            adState.params['alpha_Uu'] = adalpha_Uu
        elif 'alpha' in self.name_params_it:
            adalpha += adalpha_Uu
        if 'alpha_Up' in self.name_params_it:
            adState.params['alpha_Up'] = adalpha_Up
        elif 'alpha' in self.name_params_it:
            adalpha += adalpha_Up
        if 'alpha' in self.name_params_it:
            adState.params['alpha'] = adalpha
        if 'hbc' in self.name_params_it:
            adState.params['hbcx'] = adh_SN
            adState.params['hbcy'] = adh_WE


###############################################################################
#                             Multi-models                                    #
###############################################################################      

class Model_multi:

    def __init__(self,config,State):

        # Initialize models
        self.Models = []
        _config = config.copy()

        for _MOD in config.MOD:
            _config.MOD = config.MOD[_MOD]
            self.Models.append(Model(_config,State))
            print()

        # Time parameters
        dts = [_Model.dt for _Model in self.Models]
        self.dt = int(np.max(dts)) # We take the longer timestep 
        if False:
            for _MOD,_Model,dt in zip(config.MOD,self.Models, dts):
                mult = int(np.ceil(self.dt / dt))
                adj_dt =  int(self.dt//mult)
                if adj_dt != dt:
                    print(f'Adjusting dt {dt} for {_MOD} -> {adj_dt} to be a divisor of dt {self.dt}')
                _Model.dt = adj_dt
        self.nt = 1 + int((config.EXP.final_date - config.EXP.init_date).total_seconds()//self.dt)
        self.T = np.arange(self.nt) * self.dt
        self.timestamps = [] 
        t = config.EXP.init_date
        while t<=config.EXP.final_date:
            self.timestamps.append(t)
            t += timedelta(seconds=self.dt)

        # Model variables: for each variable ('SSH', 'SST', 'Chl' etc...), 
        # we initialize a new variable for the sum of the different contributions
        self.name_var = {}
        self.name_var_tot = {}
        _name_var_tmp = []
        self.var_to_save = []
        for M in self.Models:
            self.var_to_save = np.concatenate((self.var_to_save, M.var_to_save))
            for name in M.name_var:
                if name not in _name_var_tmp:
                    _name_var_tmp.append(name)
                else:
                    # At least two component for the same variable, so we initialize a global variable
                    new_name = f'{name}_tot'
                    self.name_var[name] = new_name
                    self.name_var_tot[name] = new_name
                    # Initialize new State variable
                    if new_name in State.var:
                        State.var[new_name] += State.var[M.name_var[name]]
                    else:
                        State.var[new_name] = State.var[M.name_var[name]].copy()
                    if M.name_var[name] in M.var_to_save and new_name not in self.var_to_save:
                        self.var_to_save = np.append(self.var_to_save,new_name)
        self.var_to_save = list(self.var_to_save)

        for M in self.Models:
            for name in M.name_var:
                if name not in self.name_var_tot:
                    self.name_var[name] = M.name_var[name]

        # Tests tgl & adj
        if config.INV is not None and config.INV.super=='INV_4DVAR' and config.INV.compute_test:
            #for M in self.Models:
                #print('Tangent test:')
                #tangent_test(M,State,nstep=10)
                #print('Adjoint test:')
                #adjoint_test(M,State,nstep=10)
            print('MultiModel Tangent test:')
            tangent_test(self,State,nstep=1)
            print('MultiModel Adjoint test:')
            adjoint_test(self,State,nstep=1)

    def init(self,State,t0=0):

        # Intialization
        var_tot_tmp = {}
        for name in self.name_var:
            var_tot_tmp[name] = np.zeros_like(State.var[self.name_var[name]]) 

        for M in self.Models:
            M.init(State, t0=t0)
            for name in self.name_var:
                if name in M.name_var:
                    var_tot_tmp[name] += State.var[M.name_var[name]]

        # Update state
        for name in self.name_var:
            State.var[self.name_var[name]] = var_tot_tmp[name]
        for M in self.Models:
            M._set_land_nan(State)

    def set_bc(self,time_bc,var_bc=None,**kwargs):

        for M in self.Models:
            M.set_bc(time_bc, var_bc, **kwargs)

    def save_output(self,State,present_date,name_var=None,t=None):
        # Component models may add diagnosed variables in their save methods.
        # Collect those fields in memory so all components and totals are
        # committed in one archive transaction for this timestamp.
        collected = {}
        previous_collector = getattr(
            State, '_output_var_collector', None)
        State._output_var_collector = collected
        try:
            for M in self.Models:
                M.save_output(State, present_date, t=t)
        finally:
            State._output_var_collector = previous_collector

        # Totals are maintained by Model_multi.step() on the original state
        # and were historically written by this final save call.
        for name in self.var_to_save:
            collected[name] = State.var[name]

        if previous_collector is not None:
            previous_collector.update(collected)
            return

        output_state = State.copy()
        output_state._output_var_collector = None
        output_state.var.update(collected)
        output_state.save_output(
            present_date, name_var=list(collected.keys()))

    def prepare_scan_boundary_conditions(self, times, nstep):
        """Prepare one scan-forcing pytree entry per component model.

        Models without device-side forcing preparation keep a static ``None``
        leaf. Models such as QG and QGSW receive their own effective number
        of time steps, matching the conversion performed by ``step``.
        """
        forcing = []
        for M in self.Models:
            _nstep = nstep * self.dt // M.dt
            prepare = getattr(M, 'prepare_scan_boundary_conditions', None)
            forcing.append(
                prepare(times, _nstep) if prepare is not None else None
            )
        return tuple(forcing)

    @staticmethod
    def _component_boundary(boundary_conditions, model_index):
        if boundary_conditions is None:
            return None
        return boundary_conditions[model_index]

    def step(self,State,nstep=1,t=None,Xb=None):

        # Intialization
        var_tot_tmp = {}
        for name in self.name_var_tot:
            var_tot_tmp[name] = jnp.zeros_like(State.var[self.name_var[name]]) 
        
        # Loop over models
        for model_index, M in enumerate(self.Models):
            _nstep = nstep*self.dt//M.dt
            component_boundary = self._component_boundary(Xb, model_index)
            # Forward propagation
            if component_boundary is None:
                M.step(State,nstep=_nstep,t=t)
            else:
                M.step(
                    State,
                    nstep=_nstep,
                    t=t,
                    Xb=component_boundary,
                )
            # Add to total variables
            for name in self.name_var:
                if name in M.name_var and (name in self.name_var_tot):
                    var_tot_tmp[name] += +State.var[M.name_var[name]]
                    
        # Update state
        for name in self.name_var_tot:
            State.var[self.name_var_tot[name]] = var_tot_tmp[name]

    def step_tgl(self,dState,State,nstep=1,t=None,Xb=None):

        # Intialization
        var_tot_tmp = {}
        for name in self.name_var_tot:
            var_tot_tmp[name] = np.zeros_like(State.var[self.name_var[name]]) 

        # Loop over models
        for model_index, M in enumerate(self.Models):
            _nstep = nstep*self.dt//M.dt
            component_boundary = self._component_boundary(Xb, model_index)
            # Tangent propagation
            if component_boundary is None:
                M.step_tgl(dState,State,nstep=_nstep,t=t)
            else:
                M.step_tgl(
                    dState,
                    State,
                    nstep=_nstep,
                    t=t,
                    Xb=component_boundary,
                )
            # Add to total variables
            for name in self.name_var:
                if name in M.name_var and name in var_tot_tmp:
                    var_tot_tmp[name] += dState.var[M.name_var[name]]
        
        # Update state
        for name in self.name_var_tot:
            dState.var[self.name_var_tot[name]] = var_tot_tmp[name]

    def step_adj(self,adState,State,nstep=1,t=None,Xb=None):

        # Intialization
        var_tot_tmp = {}
        for name in self.name_var_tot:
            var_tot_tmp[name] = adState.var[self.name_var_tot[name]]
        
        # Loop over models
        for model_index, M in enumerate(self.Models):
            _nstep = nstep*self.dt//M.dt
            component_boundary = self._component_boundary(Xb, model_index)
            # Add to local variable
            for name in self.name_var:
                if name in M.name_var and name in self.name_var_tot:
                    adState.var[M.name_var[name]] += var_tot_tmp[name]  
            # Adjoint propagation
            if component_boundary is None:
                M.step_adj(adState,State,nstep=_nstep,t=t)
            else:
                M.step_adj(
                    adState,
                    State,
                    nstep=_nstep,
                    t=t,
                    Xb=component_boundary,
                )
        
        for name in self.name_var_tot:
            adState.var[self.name_var_tot[name]] *= 0 
                

###############################################################################
#                       Tangent and Adjoint tests                             #
###############################################################################     

def tangent_test(M,State,t0=0,nstep=1, ampl=1):

    # Boundary conditions
    var_bc = {}

    for name in M.name_var:
        var_bc[name] = {0:ampl*np.random.random(State.var[M.name_var[name]].shape).astype('float64'),
                        1:ampl*np.random.random(State.var[M.name_var[name]].shape).astype('float64')}
    M.set_bc([t0,t0+nstep*M.dt],var_bc)

    State0 = State.random(ampl=ampl)
    dState = State.random(ampl=ampl)
    State0_tmp = State0.copy()
    
    M.step(t=t0,State=State0_tmp,nstep=nstep)
    X2 = State0_tmp.getvar(vect=True) 
    
    for p in range(10):
        
        lambd = 10**(-p)
        
        State1 = dState.copy()
        State1.scalar(lambd)
        State1.Sum(State0)

        M.step(t=t0,State=State1,nstep=nstep)
        X1 = State1.getvar(vect=True)
        
        dState1 = dState.copy()
        dState1.scalar(lambd)
        M.step_tgl(t=t0,dState=dState1,State=State0,nstep=nstep)

        dX = dState1.getvar(vect=True)
        
        mask = np.isnan(X1+X2+dX)
        
        ps = np.linalg.norm(X1[~mask]-X2[~mask]-dX[~mask])/np.linalg.norm(dX[~mask])
    
        print('%.E' % lambd,'%.E' % ps)
        
def adjoint_test(M, State, t0=0, nstep=1, ampl=1):

    model_dtype = getattr(M, 'dtype', np.float64)
    np_dtype = np.float32 if model_dtype == jnp.float32 else np.float64

    # ------------------------------------------------------------------
    # Build per-shape ocean masks so that land points stay at zero.
    # Non-zero land values create large gradients at land-ocean boundaries
    # that amplify float32 roundoff in WENO/padding operations, breaking
    # the inner-product identity even though the operators are exact duals.
    # ------------------------------------------------------------------
    masks_by_shape = {}
    if State.mask is not None:
        ny, nx = State.mask.shape
        mask_h = np.asarray(State.mask, dtype=bool)
        masks_by_shape[(ny, nx)] = mask_h.astype(np_dtype)
        # U-grid mask: (ny, nx+1)
        mask_u = np.zeros((ny, nx + 1), dtype=np_dtype)
        mask_u[:, 1:nx] = (mask_h[:, :-1] & mask_h[:, 1:]).astype(np_dtype)
        masks_by_shape[(ny, nx + 1)] = mask_u
        # V-grid mask: (ny+1, nx)
        mask_v = np.zeros((ny + 1, nx), dtype=np_dtype)
        mask_v[1:ny, :] = (mask_h[:-1, :] & mask_h[1:, :]).astype(np_dtype)
        masks_by_shape[(ny + 1, nx)] = mask_v

    # ------------------------------------------------------------------
    # Use DIFFERENT seeds for trajectory / perturbation / adjoint vector.
    # State.random() always resets np.random.seed(0), so calling it three
    # times produces three IDENTICAL states.  With identical dState and
    # adState the inner-product test degenerates and f32 rounding errors
    # no longer cancel, giving a spurious ~0.1 % deviation from 1.0.
    # ------------------------------------------------------------------
    def _apply_mask(arr):
        m = masks_by_shape.get(arr.shape)
        return arr * m if m is not None else arr

    def make_rand_state(seed):
        np.random.seed(seed)
        s = State.copy(free=True)
        for name in s.var:
            s.var[name] = _apply_mask(
                (ampl * np.random.random(s.var[name].shape)).astype(np_dtype))
        for name in s.params:
            s.params[name] = _apply_mask(
                (ampl * np.random.random(s.params[name].shape)).astype(np_dtype))
        return s

    # Boundary conditions
    np.random.seed(99)
    var_bc = {}
    for name in M.name_var:
        var_bc[name] = {
            0: _apply_mask(ampl * np.random.random(State.var[M.name_var[name]].shape).astype(np_dtype)),
            1: _apply_mask(ampl * np.random.random(State.var[M.name_var[name]].shape).astype(np_dtype)),
        }
    M.set_bc([t0, t0 + nstep * M.dt], var_bc)

    State0  = make_rand_state(seed=10)   # trajectory
    dState  = make_rand_state(seed=20)   # TLM perturbation
    adState = make_rand_state(seed=30)   # ADJ input

    # Snapshot before TLM / ADJ
    dX0  = np.concatenate((dState.getvar(vect=True), dState.getparams(vect=True)))
    adX0 = np.concatenate((adState.getvar(vect=True), adState.getparams(vect=True)))

    # Run TLM
    M.step_tgl(t=t0, dState=dState, State=State0, nstep=nstep)
    dX1  = np.concatenate((dState.getvar(vect=True), dState.getparams(vect=True)))

    # Run ADJ
    M.step_adj(t=t0, adState=adState, State=State0, nstep=nstep)
    adX1 = np.concatenate((adState.getvar(vect=True), adState.getparams(vect=True)))

    mask = np.isnan(adX0 + dX0 + adX1 + dX1)

    # Compute inner products in f64 for accurate accumulation
    ps1 = np.inner(dX1[~mask], adX0[~mask])
    ps2 = np.inner(dX0[~mask], adX1[~mask])

    print(f'  adjoint_test ({np_dtype.__name__}):  <Mdx,y> = {ps1} <dx,M*y> = {ps2} \n<Mdx,y>/<dx,M*y> = {ps1/ps2}')

    
    

    
    
    
    
