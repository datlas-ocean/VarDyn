import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip('zarr')
pytest.importorskip('pyinterp')
pytest.importorskip('jax')

from src.state import State


def _batch(times, offset=0.0):
    times = pd.DatetimeIndex(times)
    values = np.arange(len(times) * 6, dtype=np.float32).reshape(len(times), 2, 3)
    values += offset
    return xr.Dataset(
        {'sla': (('time', 'y', 'x'), values)},
        coords={
            'time': times,
            'lon': (('y', 'x'), np.broadcast_to(
                np.arange(3, dtype=np.float32), (2, 3))),
            'lat': (('y', 'x'), np.broadcast_to(
                np.arange(2, dtype=np.float32)[:, None], (2, 3))),
        },
    )


def test_zarr_batch_appends_and_overwrites_restart_prefix(tmp_path):
    archive = tmp_path / 'trajectory.zarr'
    times = pd.date_range('2025-01-01', periods=6, freq='6h')

    State._save_zarr_records(
        _batch(times[:4]), str(archive),
        window_start=times[0], window_end=times[-1],
        zarr_time_chunk=4, zarr_spatial_chunk=2)
    State._save_zarr_records(
        _batch(times[4:]), str(archive),
        window_start=times[0], window_end=times[-1],
        zarr_time_chunk=4, zarr_spatial_chunk=2)

    # Restarting the final trajectory updates an existing contiguous batch
    # without duplicating timestamps.
    State._save_zarr_records(
        _batch(times[:4], offset=100.0), str(archive),
        window_start=times[0], window_end=times[-1],
        zarr_time_chunk=4, zarr_spatial_chunk=2)

    with xr.open_zarr(archive, consolidated=False) as result:
        assert result.sizes['time'] == 6
        np.testing.assert_array_equal(result.time.values, times.values)
        np.testing.assert_allclose(
            result.sla.isel(time=slice(0, 4)).values,
            _batch(times[:4], offset=100.0).sla.values)
        np.testing.assert_allclose(
            result.sla.isel(time=slice(4, None)).values,
            _batch(times[4:]).sla.values)
