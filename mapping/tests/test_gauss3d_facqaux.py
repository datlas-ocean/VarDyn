"""Fourier-scale selection for Gaussian-basis prior variance corrections."""
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import xarray as xr

from src import config_default as defaults
from src.basis import Basis_gauss3d, _gauss3d_facq_at_scale
from src.exp import Config
from src.state import State


NAMES = dict(lon='longitude', lat='latitude', var='factor', wavenumber='k')


def spectral_field():
    return xr.DataArray(
        np.broadcast_to(np.array([.36, .64, 1.])[:, None, None], (3, 2, 2)),
        dims=('k', 'latitude', 'longitude'),
        coords={'k': [0., .005, .01], 'latitude': [30., 50.],
                'longitude': [90., 110.]}, name='factor')


class GaussianAuxiliaryTests(unittest.TestCase):
    def test_exact_fourier_scale(self):
        selected = _gauss3d_facq_at_scale(spectral_field(), NAMES, 100.)
        self.assertNotIn('k', selected.dims)
        np.testing.assert_allclose(selected, .64)

    def test_interpolate_in_wavenumber_including_zero(self):
        field = spectral_field().isel(k=slice(None, None, -1))
        # sigma_D=200 km -> lambda=400 km -> k=.0025 cycles/km.
        np.testing.assert_allclose(_gauss3d_facq_at_scale(field, NAMES, 200.), .5)

    def test_endpoint_and_single_frequency(self):
        with self.assertLogs(level='WARNING'):
            np.testing.assert_allclose(
                _gauss3d_facq_at_scale(spectral_field(), NAMES, 10.), 1.)
        one = spectral_field().isel(k=[1])
        np.testing.assert_allclose(_gauss3d_facq_at_scale(one, NAMES, 100.), .64)

    def test_spatial_map_unchanged(self):
        field = spectral_field().isel(k=0, drop=True)
        self.assertIs(_gauss3d_facq_at_scale(field, NAMES, 100.), field)

    def test_unmapped_spectrum_is_not_silently_averaged(self):
        with self.assertRaisesRegex(ValueError, 'wavenumber'):
            _gauss3d_facq_at_scale(spectral_field(), dict(NAMES, wavenumber=None), 100.)

    def test_invalid_frequencies(self):
        for frequencies in ([0., .005, .005], [0., .005, np.inf], [-.01, 0., .01]):
            with self.subTest(frequencies=frequencies):
                with self.assertRaisesRegex(ValueError, 'wavenumbers'):
                    _gauss3d_facq_at_scale(
                        spectral_field().assign_coords(k=frequencies), NAMES, 100.)

    def test_netcdf_correction_reaches_basis_standard_deviations(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'factor.nc'
            spectral_field().to_netcdf(path)
            cfg = Config()
            cfg.EXP = Config(defaults.EXP | dict(
                path_save=directory, tmp_DA_path=directory, flag_plot=0,
                init_date=datetime(2020, 1, 1), final_date=datetime(2020, 1, 2)))
            cfg.GRID = Config(defaults.GRID_GEO | dict(
                super='GRID_GEO', lon_min=100., lon_max=101., lat_min=40.,
                lat_max=41., dlon=.2, dlat=.2, name_init_mask=None))
            cfg.MOD = Config(defaults.MOD_QG1L | dict(super='MOD_QG1L'))
            cfg.BASIS = Config(defaults.BASIS_GAUSS3D | dict(
                super='BASIS_GAUSS3D', sigma_D=100., sigma_T=1.,
                file_facQaux=None, name_var_facQaux=NAMES))
            cfg.INV = None
            state = State(cfg, verbose=False)
            basis = Basis_gauss3d(cfg, state)
            times = np.array([0., 1.])
            _, baseline = basis.set_basis(times, return_q=True)
            basis.file_facQaux = path
            _, corrected = basis.set_basis(times, return_q=True)
            np.testing.assert_allclose(corrected / baseline, .8)
            basis.sigma_D = 200.
            basis.file_facQaux = None
            _, baseline = basis.set_basis(times, return_q=True)
            basis.file_facQaux = path
            _, corrected = basis.set_basis(times, return_q=True)
            np.testing.assert_allclose(corrected / baseline, np.sqrt(.5))


if __name__ == '__main__':
    unittest.main()
