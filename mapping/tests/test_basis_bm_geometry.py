import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from src.basis import _Basis_bm, _bounded_wavelet_frequencies


def test_bounded_wavelet_frequencies_respect_requested_interval():
    ff = _bounded_wavelet_frequencies(200., 600., 3.5, 1.5)

    np.testing.assert_allclose(
        1. / ff,
        [583.0903790087463, 408.1632653061224,
         285.71428571428567, 200.],
    )
    assert np.all(1. / ff <= 600.)
    assert np.all(1. / ff >= 200. - 1.e-10)


def test_bounded_wavelet_frequencies_reject_too_narrow_interval():
    with pytest.raises(ValueError, match='At least two wavelength bands'):
        _bounded_wavelet_frequencies(500., 600., 3.5, 1.5)


def test_bm_discards_centres_without_grid_support():
    basis_config = SimpleNamespace(
        flux=False,
        facns=1.,
        facnlt=2.,
        npsp=3.5,
        facpsp=1.5,
        lmin=200.,
        lmax=600.,
        tdecmin=2.5,
        tdecmax=40.,
        factdec=0.5,
        sloptdec=-1.28,
        Qmax=1.e-3,
        facQ=1.,
        slopQ=-5.,
        lmeso=300.,
        tmeso=20.,
        name_mod_var='ssh',
        norm_time=True,
        path_background=None,
        var_background=None,
        c_grid_var=None,
        compute_velocities=False,
        name_mod_u='u',
        name_mod_v='v',
        file_depth=None,
        name_var_depth={'lon': '', 'lat': '', 'var': ''},
        depth1=0.,
        depth2=30.,
    )
    config = SimpleNamespace(
        BASIS=basis_config,
        EXP=SimpleNamespace(
            init_date=datetime.datetime(2020, 1, 1),
            tmp_DA_path='/tmp',
        ),
    )
    lon, lat = np.meshgrid(
        np.linspace(-1.5, 1.5, 4), np.linspace(44., 46., 3))
    state = SimpleNamespace(
        ny=3,
        nx=4,
        lon=lon,
        lat=lat,
        lon_min=lon.min(),
        lon_max=lon.max(),
        lat_min=lat.min(),
        lat_max=lat.max(),
        mask=None,
        f=np.full((3, 4), 1.e-4),
        DX=np.full((3, 4), 8.e4),
        DY=np.full((3, 4), 8.e4),
    )

    basis = _Basis_bm(config, state)
    basis.set_basis(np.arange(3.), return_q=True)

    for spatial_operator in basis.Gx:
        assert np.all(np.diff(spatial_operator.indptr) > 0)
