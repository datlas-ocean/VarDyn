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

    def __init__(self, values=None, fail=False, variables=None):
        self.values = values
        self.fail = fail
        self.variables = variables

    def load_output(self, date):
        if self.fail:
            raise OSError("missing tile")
        if self.variables is not None:
            return xr.Dataset({
                name: (("y", "x"), values)
                for name, values in self.variables.items()
            })
        return xr.Dataset({"sla": (("y", "x"), self.values)})


def test_date_merge_excludes_land_from_support_and_normalization():
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

    land = _TileState(fail=True)
    land.mask = np.ones((2, 2), dtype=bool)
    partial = _merge_date_arrays(
        None,
        target,
        [_TileState(np.ones((2, 2))), land],
        ["sla"],
        runtime,
        no_coverage,
        np.float32,
    )["sla"]
    np.testing.assert_allclose(partial[:, :2], 1.0)
    assert np.isnan(partial[:, 2]).all()


def test_missing_ocean_output_is_fatal():
    target = SimpleNamespace(ny=2, nx=2, mask=None)
    with pytest.raises(RuntimeError, match="Required ocean tile"):
        _merge_date_arrays("2024-07-27", target, [_TileState(fail=True)],
                           ["sla"], [(None, np.ones((2, 2)), None)],
                           np.zeros((2, 2), dtype=bool), np.float32)


def test_missing_variable_is_nan_on_that_tile_footprint():
    target = SimpleNamespace(ny=2, nx=3, mask=None)
    first_indices = np.array([0, 1, 3, 4], dtype=np.int32)
    second_indices = np.array([1, 2, 4, 5], dtype=np.int32)
    runtime = [
        (first_indices, np.array([1, 0.5, 1, 0.5]),
         _IdentityProjection()),
        (second_indices, np.array([0.5, 1, 0.5, 1]),
         _IdentityProjection()),
    ]
    merged = _merge_date_arrays(
        None,
        target,
        [
            _TileState(variables={
                "sla": np.ones((2, 2)),
            }),
            _TileState(variables={
                "sla": np.full((2, 2), 3.0),
                "diagnostic": np.full((2, 2), 4.0),
            }),
        ],
        ["sla", "diagnostic"],
        runtime,
        np.zeros((2, 3), dtype=bool),
        np.float32,
    )

    np.testing.assert_allclose(merged["sla"], [[1, 2, 3], [1, 2, 3]])
    assert np.isnan(merged["diagnostic"][:, :2]).all()
    np.testing.assert_allclose(merged["diagnostic"][:, 2], 4.0)


def test_parallel_merge_aborts_on_first_reported_failure(monkeypatch):
    from src import run_assimilation as module

    class Process:
        def __init__(self, **kwargs):
            self.alive = False
            self.terminated = False
        def start(self):
            self.alive = True
        def is_alive(self):
            return self.alive
        def terminate(self):
            self.terminated = True
            self.alive = False
        def join(self, timeout):
            pass

    class Context:
        def __init__(self):
            self.processes = []
            self.reads = 0
        def Queue(self):
            return self
        def get(self, timeout):
            self.reads += 1
            assert self.reads == 1, "must not wait for the other worker"
            return {"kind": "result", "worker": 0, "error": "missing ocean output"}
        def Process(self, **kwargs):
            process = Process(**kwargs)
            self.processes.append(process)
            return process

    context = Context()
    monkeypatch.setattr(module.mp, "get_context", lambda method: context)
    with pytest.raises(RuntimeError, match="missing ocean output"):
        module.parallel_merge([1, 2], None, [None, None], ["sla"],
                              None, None, None, None, num_workers=2)
    assert len(context.processes) == 2
    assert all(p.terminated for p in context.processes)
