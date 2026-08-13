#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Florian Le Guillou on June 2026.

Provides shared numerical, interpolation, and filtering utilities.
"""
import numpy as np 
import xarray as xr
import scipy.linalg as spl
from scipy import interpolate, spatial
import scipy



def gaspari_cohn(r,c=1):
    """
    NAME 
        bfn_gaspari_cohn

    DESCRIPTION 
        Gaspari-Cohn function. Inspired from E.Cosmes.
        
        Args: 
            r : array of value whose the Gaspari-Cohn function will be applied
            c : Distance above which the return values are zeros


        Returns:  smoothed values 
            
    """ 
    if type(r) in [float,int,np.float32,np.float64]:
        ra = np.array([r])
    else:
        ra = r
    if c<=0:
        return np.zeros_like(ra)
    else:
        ra = 2*np.abs(ra)/c
        gp = np.zeros_like(ra)
        i= np.where(ra<=1.)[0]
        gp[i]=-0.25*ra[i]**5+0.5*ra[i]**4+0.625*ra[i]**3-5./3.*ra[i]**2+1.
        i =np.where((ra>1.)*(ra<=2.))[0]
        gp[i] = 1./12.*ra[i]**5-0.5*ra[i]**4+0.625*ra[i]**3+5./3.*ra[i]**2-5.*ra[i]+4.-2./3./ra[i]
        if type(r) is float:
            gp = gp[0]
    return gp


def lonlat2dxdy(lon,lat):
    dlon = np.gradient(lon)
    dlat = np.gradient(lat)
    dx = np.sqrt((dlon[1]*111000*np.cos(np.deg2rad(lat)))**2
                 + (dlat[1]*111000)**2)
    dy = np.sqrt((dlon[0]*111000*np.cos(np.deg2rad(lat)))**2
                 + (dlat[0]*111000)**2)
    dx[0,:] = dx[1,:]
    dx[-1,: ]= dx[-2,:]
    dx[:,0] = dx[:,1]
    dx[:,-1] = dx[:,-2]
    dy[0,:] = dy[1,:]
    dy[-1,:] = dy[-2,:]
    dy[:,0] = dy[:,1]
    dy[:,-1] = dy[:,-2]

    return dx,dy


def dxdy2xy(dx,dy,x0=0,y0=0):
    ny,nx = dx.shape
    X = np.zeros((ny,nx))
    Y = np.zeros((ny,nx))
    for i in range(ny):
        for j in range(nx):
            X[i,j] = x0 + np.sum(dx[i,:j])
            Y[i,j] = y0 + np.sum(dy[:i,j])
    return X,Y


def geo2cart(coords):
    """
    NAME
        geo2cart

    DESCRIPTION
        Transform coordinates from geodetic to cartesian

        Args:
            coords : a set of lan/lon coordinates (e.g. a tuple or
             an array of tuples)


        Returns: a set of cartesian coordinates (x,y,z)

    """

    A = 6378.137
    E2 = 6.69437999014e-3

    coords = np.asarray(coords).astype(float)

    if coords.ndim == 1:
        coords = np.array([coords])

    lat_rad = np.radians(coords[:, 1])
    lon_rad = np.radians(coords[:, 0])

    r_n = A / (np.sqrt(1 - E2 * (np.sin(lat_rad) ** 2)))
    x = r_n * np.cos(lat_rad) * np.cos(lon_rad)
    y = r_n * np.cos(lat_rad) * np.sin(lon_rad)
    z = r_n * (1 - E2) * np.sin(lat_rad)

    return np.column_stack((x, y, z))


def interp2d(ds,name_vars,lon_out,lat_out):

    ds = ds.squeeze()

    ds = ds.assign_coords(
                 {name_vars['lon']:(ds[name_vars['lon']]),
                  name_vars['lat']:ds[name_vars['lat']]})

    if ds[name_vars['var']].shape[0]!=ds[name_vars['lat']].shape[0]:
        ds[name_vars['var']] = ds[name_vars['var']].transpose()

    lon_full = np.asarray(ds[name_vars['lon']].values)
    lat_full = np.asarray(ds[name_vars['lat']].values)
    var_full = np.asarray(ds[name_vars['var']].values)

    def _nearest_finite(lon_values, lat_values, values):
        finite = (np.isfinite(lon_values.ravel())
                  & np.isfinite(lat_values.ravel())
                  & np.isfinite(values.ravel()))
        if not np.any(finite):
            raise ValueError(f"No finite values in {name_vars['var']!r}")
        points = np.column_stack((lon_values.ravel()[finite],
                                  lat_values.ravel()[finite]))
        tree = spatial.cKDTree(points)
        query = np.column_stack((np.asarray(lon_out).ravel(),
                                 np.asarray(lat_out).ravel()))
        _, nearest = tree.query(query, k=1)
        return values.ravel()[finite][nearest].reshape(np.asarray(lat_out).shape)

    if lon_full.ndim == 2:
        dlon = np.nanmax(np.abs(np.diff(lon_full, axis=1)))
        dlat = np.nanmax(np.abs(np.diff(lat_full, axis=0)))
    else:
        dlon = np.nanmax(np.abs(np.diff(lon_full)))
        dlat = np.nanmax(np.abs(np.diff(lat_full)))

    if len(ds[name_vars['lon']].shape)==2:
        ds = ds.where((ds[name_vars['lon']]<=lon_out.max()+dlon) &\
                      (ds[name_vars['lon']]>=lon_out.min()-dlon) &\
                      (ds[name_vars['lat']]<=lat_out.max()+dlat) &\
                      (ds[name_vars['lat']]>=lat_out.min()-dlat),drop=True)

        lon_sel = ds[name_vars['lon']].values
        lat_sel = ds[name_vars['lat']].values

    else:

        ds = ds.where((ds[name_vars['lon']]<=lon_out.max()+dlon) &\
                      (ds[name_vars['lon']]>=lon_out.min()-dlon) &\
                      (ds[name_vars['lat']]<=lat_out.max()+dlat) &\
                      (ds[name_vars['lat']]>=lat_out.min()-dlat),drop=True)

        lon_sel,lat_sel = np.meshgrid(
            ds[name_vars['lon']].values,
            ds[name_vars['lat']].values)

    var_sel = ds[name_vars['var']].values

    finite_selected = (np.isfinite(lon_sel.ravel())
                       & np.isfinite(lat_sel.ravel())
                       & np.isfinite(var_sel.ravel()))
    if np.count_nonzero(finite_selected) < 3:
        return _nearest_finite(lon_full, lat_full, var_full)

    try:
        var_out = interpolate.griddata(
            (lon_sel.ravel()[finite_selected], lat_sel.ravel()[finite_selected]),
            var_sel.ravel()[finite_selected],
            (lon_out.ravel(), lat_out.ravel()),
        ).reshape(lat_out.shape)
    except (ValueError, RuntimeError):
        return _nearest_finite(lon_full, lat_full, var_full)

    invalid = ~np.isfinite(var_out)
    if np.any(invalid):
        nearest = _nearest_finite(lon_full, lat_full, var_full)
        var_out[invalid] = nearest[invalid]

    return var_out


def compute_weight_map(lon2d,lat2d,mask,dist_scale,bc=True,slope=10):

    coords = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    coords_cart = geo2cart(coords)
    ground_pixel_tree = spatial.cKDTree(coords_cart)
    n_probe = min(100, len(coords_cart))
    dd, _ = ground_pixel_tree.query(coords_cart[:n_probe], k=2)
    dist_threshold = dd[:, 1].min()
    mask = mask.copy()
    if bc:
        mask[0,:] = True
        mask[-1,:] = True
        mask[:,0] = True
        mask[:,-1] = True

    lon_bc = lon2d[mask]
    lat_bc = lat2d[mask]
    coords_bc = np.column_stack((lon_bc, lat_bc))
    bc_tree = spatial.cKDTree(geo2cart(coords_bc))

    dist_to_bc, _ = bc_tree.query(coords_cart, k=1)
    dist_to_bc = np.maximum(dist_to_bc - 0.5 * dist_threshold, 0.0)

    x = np.clip(dist_to_bc / dist_scale, 0.0, 1.0)
    bc_weight = 1.0 - x**3 * (10.0 - 15.0 * x + 6.0 * x**2)

    bc_weight = bc_weight.reshape(lon2d.shape)

    return bc_weight


def compute_sponge_components(lon2d, lat2d, mask, dist_scale):
    """Compute separate sponge weight maps for coast, N/S edges, and W/E edges."""

    w_coast = compute_weight_map(lon2d, lat2d, mask, dist_scale, bc=False)

    mask_NS = np.zeros_like(mask)
    mask_NS[0, :] = True;  mask_NS[-1, :] = True
    w_NS = compute_weight_map(lon2d, lat2d, mask_NS, dist_scale, bc=False)

    mask_WE = np.zeros_like(mask)
    mask_WE[:, 0] = True;  mask_WE[:, -1] = True
    w_WE = compute_weight_map(lon2d, lat2d, mask_WE, dist_scale, bc=False)

    return w_coast, w_NS, w_WE


def smooth_weight_map(lon2d, lat2d, mask, dist_scale):
    """Isotropic sponge weight with smooth corner blending."""

    w_coast, w_NS, w_WE = compute_sponge_components(lon2d, lat2d, mask, dist_scale)
    return 1.0 - (1.0 - w_coast) * (1.0 - w_NS) * (1.0 - w_WE)


def detrendn(da, axes=None):
    
    """
    Detrend by subtracting out the least-square plane or least-square cubic fit
    depending on the number of axis.
    Parameters
    ----------
    da : `dask.array`
        The data to be detrended
    Returns
    -------
    da : `numpy.array`
        The detrended input data
    """
    
    if axes is None:
        axes = range(len(da.shape))
        
    N = [da.shape[n] for n in axes]
    M = []
    for n in range(da.ndim):
        if n not in axes:
            M.append(da.shape[n])
            
    if len(N) == 2:
        G = np.ones((N[0]*N[1],3))
        for i in range(N[0]):
            G[N[1]*i:N[1]*i+N[1], 1] = i+1
            G[N[1]*i:N[1]*i+N[1], 2] = np.arange(1, N[1]+1)
        if type(da) == xr.DataArray:
            d_obs = np.reshape(da.copy().values, (N[0]*N[1],1))
        else:
            d_obs = np.reshape(da.copy(), (N[0]*N[1],1))
    elif len(N) == 3:
        if type(da) == xr.DataArray:
            if da.ndim > 3:
                raise NotImplementedError("Cubic detrend is not implemented "
                                         "for 4-dimensional `xarray.DataArray`."
                                         " We suggest converting it to "
                                         "`dask.array`.")
            else:
                d_obs = np.reshape(da.copy().values, (N[0]*N[1]*N[2],1))
        else:
            d_obs = np.reshape(da.copy(), (N[0]*N[1]*N[2],1))

        G = np.ones((N[0]*N[1]*N[2],4))
        G[:,3] = np.tile(np.arange(1,N[2]+1), N[0]*N[1])
        ys = np.zeros(N[1]*N[2])
        for i in range(N[1]):
            ys[N[2]*i:N[2]*i+N[2]] = i+1
        G[:,2] = np.tile(ys, N[0])
        for i in range(N[0]):
            G[len(ys)*i:len(ys)*i+len(ys),1] = i+1
    else:
        raise NotImplementedError("Detrending over more than 4 axes is "
                                 "not implemented.")

    m_est = np.dot(np.dot(spl.inv(np.dot(G.T, G)), G.T), d_obs)
    d_est = np.dot(G, m_est)

    lin_trend = np.reshape(d_est, da.shape)

    return da - lin_trend


def read_auxdata(file_aux,name_var,lon_unit):

    # Read database
    ds = xr.open_dataset(file_aux)
    
    if np.sign(ds[name_var['lon']].data.min())==-1 and lon_unit=='0_360':
        ds = ds.assign_coords({name_var['lon']:((name_var['lon'], ds[name_var['lon']].data % 360))})
    elif (np.sign(ds[name_var['lon']].data.min())==1 or ds[name_var['lon']].data.max()>180) and lon_unit=='-180_180':
        ds = ds.assign_coords({name_var['lon']:((name_var['lon'], (ds[name_var['lon']].data + 180) % 360 - 180))})
    ds = ds.sortby(ds[name_var['lon']])    
    
    lon = ds[name_var['lon']].values
    lat = ds[name_var['lat']].values
    if 'data' in name_var:
        data = ds[name_var['data']].values.squeeze()
    elif 'mdt' in name_var:
        data = ds[name_var['mdt']].values.squeeze()
    elif 'var' in name_var:
        data = ds[name_var['var']].values.squeeze()
    
    if len(np.shape(data))==3: 
        data = np.mean(data,0)
    
    if data.shape[1]==lon.size:
        data = data.transpose()
    
    if len(lon.shape)==1:
        finterp = scipy.interpolate.RegularGridInterpolator((lon,lat),data,bounds_error=False,fill_value=None)
    else:
        finterp = scipy.interpolate.LinearNDInterpolator(list(zip(lon.ravel(),lat.ravel())),data.ravel())

    return finterp


def ssh2uv(ssh, State=None, lon=None, lat=None, xac=None, g=9.81):

    if lon is not None and lat is not None:
        if len(lon.shape) == 1:
            lon, lat = np.meshgrid(lon, lat)
        f = 4 * np.pi / 86164 * np.sin(lat * np.pi / 180)
        dx, dy = lonlat2dxdy(lon, lat)
    else:
        f = State.f
        dx = State.DX
        dy = State.DY

    ssh_shapelen = len(ssh.shape)
    if ssh_shapelen == 2:
        _dx = dx
        _dy = dy
    elif ssh_shapelen == 3:
        ssh = np.moveaxis(ssh, 0, -1)
        f = f[:, :, np.newaxis]
        _dx = dx[:, :, np.newaxis]
        _dy = dy[:, :, np.newaxis]

    u = np.zeros_like(ssh) * np.nan
    v = np.zeros_like(ssh) * np.nan
    u[1:-1,1:] = -g / f[1:-1,1:] * \
            (ssh[2:,:-1] + ssh[2:,1:] - ssh[:-2,1:] - ssh[:-2,:-1]) / (4 * _dy[1:-1,1:])
    v[1:,1:-1] = +g / f[1:,1:-1] * \
        (ssh[1:,2:] + ssh[:-1,2:] - ssh[:-1,:-2] - ssh[1:,:-2]) / (4 * _dx[1:,1:-1])

    if xac is not None:
        u = _masked_edge(u, xac)
        v = _masked_edge(v, xac)

    if ssh_shapelen == 3:
        u = np.moveaxis(u, -1, 0)
        v = np.moveaxis(v, -1, 0)

    return u, v


def ssh2rv(ssh, State=None, lon=None, lat=None, xac=None, g=9.81, norm=False):

    if lon is not None and lat is not None:
        if len(lon.shape) == 1:
            lon, lat = np.meshgrid(lon, lat)
        f = 4 * np.pi / 86164 * np.sin(lat * np.pi / 180)
        dx, dy = lonlat2dxdy(lon, lat)
    else:
        f = State.f
        dx = State.DX
        dy = State.DY

    ssh_shapelen = len(ssh.shape)
    if ssh_shapelen == 2:
        _dx = dx[1:-1,1:-1]
        _dy = dy[1:-1,1:-1]
    elif ssh_shapelen == 3:
        ssh = np.moveaxis(ssh, 0, -1)
        f = f[:, :, np.newaxis]
        _dx = dx[1:-1,1:-1, np.newaxis]
        _dy = dy[1:-1,1:-1, np.newaxis]

    rv = np.zeros_like(ssh) * np.nan
    rv[1:-1,1:-1] = g / f[1:-1,1:-1] * \
        ((ssh[2:,1:-1] + ssh[:-2,1:-1] - 2 * ssh[1:-1,1:-1]) / _dy**2 \
        + (ssh[1:-1,2:] + ssh[1:-1,:-2] - 2 * ssh[1:-1,1:-1]) / _dx**2)
    if norm:
        rv /= f

    if xac is not None:
        rv = _masked_edge(rv, xac)

    if ssh_shapelen == 3:
        rv = np.moveaxis(rv, -1, 0)

    return rv


def uv2rv(UV, State=None, lon=None, lat=None, xac=None, norm=False):
    try:
        dx, dy = lonlat2dxdy(lon, lat)
    except Exception:
        dx = State.DX
        dy = State.DY

    u = UV[0]
    v = UV[1]

    uv_shapelen = len(u.shape)
    if uv_shapelen == 2:
        _dx = dx
        _dy = dy
    elif uv_shapelen == 3:
        u = np.moveaxis(u, 0, -1)
        v = np.moveaxis(v, 0, -1)
        _dx = dx[:, :, np.newaxis]
        _dy = dy[:, :, np.newaxis]

    mask = np.isnan(u)
    rv = np.gradient(v, axis=1) / _dx - np.gradient(u, axis=0) / _dy

    if norm:
        f = 4 * np.pi / 86164 * np.sin(lat * np.pi / 180)
        rv /= f

    if xac is not None:
        rv = _masked_edge(rv, xac)

    rv[mask] = np.nan

    if uv_shapelen == 3:
        rv = np.moveaxis(rv, -1, 0)

    return rv


def _masked_edge(var, xac):

    """Mask the edges of the swath gap."""

    if np.any(xac > 0):
        ind_gap = (xac == np.nanmin(xac[xac > 0]))
        if ind_gap.size == var.size:
            if ind_gap.shape != var.shape:
                ind_gap = ind_gap.transpose()
            var[ind_gap] = np.nan
        elif ind_gap.size == var.shape[1]:
            var[:, ind_gap] = np.nan
    if np.any(xac < 0):
        ind_gap = (xac == np.nanmax(xac[xac < 0]))
        if ind_gap.size == var.size:
            if ind_gap.shape != var.shape:
                ind_gap = ind_gap.transpose()
            var[ind_gap] = np.nan
        elif ind_gap.size == var.shape[1]:
            var[:, ind_gap] = np.nan

    return var

