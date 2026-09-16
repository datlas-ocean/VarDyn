from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator


pytest.importorskip("astropy")
pytest.importorskip("cartopy")

from src.run_assimilation import (  # noqa: E402
    _CompactLinearInterpolator,
    _compact_delaunay_projection,
    _compact_regular_projection,
    _merge_date_arrays,
)


def test_compact_interpolators_match_scipy():
    lat = np.array([-1.0, 0.0, 1.0])
    lon = np.array([10.0, 12.0, 14.0, 16.0])
    target_lat, target_lon = np.meshgrid(
        np.linspace(-2, 2, 9), np.linspace(9, 17, 17), indexing="ij")
    targets = np.column_stack((target_lat.ravel(), target_lon.ravel()))
    values = lat[:, None] * 3.0 + lon[None, :] * 2.0

    output, source, coefficients = _compact_regular_projection(
        lat, lon, targets)
    compact = _CompactLinearInterpolator(output, source, coefficients)
    actual = np.full(len(targets), np.nan)
    actual[output] = compact(values)
    expected = RegularGridInterpolator(
        (lat, lon), values, bounds_error=False, fill_value=np.nan)(targets)
    np.testing.assert_allclose(
        actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
    assert len(output) < len(targets)

    rng = np.random.default_rng(3)
    points = rng.random((80, 2))
    targets = rng.random((300, 2))
    values = points[:, 0] - 2.0 * points[:, 1]
    reference = LinearNDInterpolator(points, values)
    output, source, coefficients = _compact_delaunay_projection(
        reference.tri, targets)
    compact = _CompactLinearInterpolator(output, source, coefficients)
    actual = np.full(len(targets), np.nan)
    actual[output] = compact(values)
    np.testing.assert_allclose(
        actual, reference(targets), rtol=1e-12, atol=1e-12,
        equal_nan=True)


class _IdentityProjection:
    def __call__(self, values):
        return np.asarray(values).reshape(-1)


class _TileState:
    ny = 2
    nx = 2

    def __init__(self, values=None, fail=False):
        self.values = values
        self.fail = fail

    def load_output(self, date):
        if self.fail:
            raise OSError("missing tile")
        return xr.Dataset({"sla": (("y", "x"), self.values)})


def test_date_merge_uses_compact_support_and_renormalizes_missing_tile():
    target = SimpleNamespace(ny=2, nx=3, mask=None)
    first_indices = np.array([0, 1, 3, 4], dtype=np.int32)
    second_indices = np.array([1, 2, 4, 5], dtype=np.int32)
    runtime = [
        (first_indices, np.array([1, 0.5, 1, 0.5]),
         _IdentityProjection()),
        (second_indices, np.array([0.5, 1, 0.5, 1]),
         _IdentityProjection()),
    ]
    no_coverage = np.zeros((2, 3), dtype=bool)

    merged = _merge_date_arrays(
        None,
        target,
        [_TileState(np.ones((2, 2))), _TileState(np.full((2, 2), 3.0))],
        ["sla"],
        runtime,
        no_coverage,
        np.float32,
    )["sla"]
    np.testing.assert_allclose(merged, [[1, 2, 3], [1, 2, 3]])

    partial = _merge_date_arrays(
        None,
        target,
        [_TileState(np.ones((2, 2))), _TileState(fail=True)],
        ["sla"],
        runtime,
        no_coverage,
        np.float32,
    )["sla"]
    np.testing.assert_allclose(partial[:, :2], 1.0)
    assert np.isnan(partial[:, 2]).all()
