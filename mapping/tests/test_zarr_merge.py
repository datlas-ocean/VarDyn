"""Exercise real region writes, interruptions, and legacy source chunk layouts."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.zarr_merge import write_merged_zarr


def _dataset():
    values = np.arange(70 * 6 * 5, dtype=np.float32).reshape(70, 6, 5)
    values[:, 0, 0] = np.nan
    return xr.Dataset(
        {'sla': (('time', 'y', 'x'), values)},
        coords={'time': pd.date_range('2024-01-01', periods=70),
                'lon': (('y', 'x'), np.arange(30).reshape(6, 5)),
                'lat': (('y', 'x'), np.ones((6, 5)))})


def test_resume_rewrites_partial_lot_and_preserves_completed_lots(tmp_path, monkeypatch):
    source = tmp_path / 'source.zarr'
    expected = _dataset()
    expected.chunk({'time': 3, 'y': 3, 'x': 5}).to_zarr(source, zarr_format=2)
    source_files = {str(p): p.stat().st_mtime_ns for p in source.rglob('*') if p.is_file()}
    output = tmp_path / 'final.zarr'
    previous = expected.isel(time=slice(0, 1))
    previous.to_zarr(output, zarr_format=2)
    original = xr.Dataset.to_zarr

    def interrupted(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if kwargs.get('region', {}).get('time') == slice(32, 64):
            raise RuntimeError('simulated interruption after a partial lot')
        return result

    with xr.open_zarr(source, consolidated=False) as dataset:
        with monkeypatch.context() as patch:
            patch.setattr(xr.Dataset, 'to_zarr', interrupted)
            with pytest.raises(RuntimeError, match='simulated interruption'):
                write_merged_zarr(dataset, output, [source],
                                  zarr_time_chunk=4, zarr_spatial_chunk=2)
        work = tmp_path / 'final.zarr.merge-work'
        with xr.open_zarr(output) as preserved:
            xr.testing.assert_equal(preserved, previous)
        chunk = work / 'archive.zarr' / 'sla' / '0.0.0'
        first_mtime = chunk.stat().st_mtime_ns
        # A different output encoding must never reuse these completed lots.
        with pytest.raises(RuntimeError, match='settings changed'):
            write_merged_zarr(dataset, output, [source], zarr_time_chunk=1)
        with pytest.raises(RuntimeError, match='settings changed'):
            write_merged_zarr(dataset, output, [source], zarr_time_chunk=4,
                              zarr_spatial_chunk=2,
                              merge_plan={'blend': 'different overlaps'})
        # Observe the completed chunk just before publication removes work.
        from src import zarr_merge
        replace = zarr_merge.os.replace

        def check_replace(src, dst):
            if str(src) == str(work / 'archive.zarr'):
                assert chunk.stat().st_mtime_ns == first_mtime
            return replace(src, dst)

        monkeypatch.setattr(zarr_merge.os, 'replace', check_replace)
        write_merged_zarr(dataset, output, [source],
                          zarr_time_chunk=4, zarr_spatial_chunk=2)
    with xr.open_zarr(output, consolidated=True) as result:
        xr.testing.assert_equal(result.drop_attrs(), expected)
        assert result.sla.encoding['chunks'] == (4, 2, 2)
    assert not work.exists()
    assert source_files == {str(p): p.stat().st_mtime_ns for p in source.rglob('*') if p.is_file()}


def test_temporal_merge_reads_existing_windows_with_different_chunks(tmp_path):
    from src.run_assimilation import merge_time_windows_outputs

    expected = _dataset()
    times = pd.DatetimeIndex(expected.time.values)
    middles = [times[20], times[50]]
    for begin, end, middle, chunks, offset in [
            (0, 41, middles[0], 3, 0), (30, 70, middles[1], 5, 100)]:
        root = tmp_path / f'subwindow_{str(middle)[:10]}'
        root.mkdir()
        window = expected.isel(time=slice(begin, end)).copy(deep=True)
        window['sla'] += offset
        window.chunk({'time': chunks, 'y': 3, 'x': 5}).to_zarr(
            root / 'test.zarr', zarr_format=2)
    sources = {str(p): p.stat().st_mtime_ns
               for p in tmp_path.glob('subwindow_*/*/**/*') if p.is_file()}
    config = SimpleNamespace(EXP=SimpleNamespace(
        path_save=str(tmp_path), name_exp_save='test',
        saveoutput_time_step=pd.Timedelta(days=1)))
    merge_time_windows_outputs(
        config, [times[0], times[30]], middles, [times[40], times[-1]],
        10, zarr_output=True, zarr_time_chunk=4, zarr_spatial_chunk=2)
    expected['sla'][30:41] += xr.DataArray(
        (50 * (1 - np.cos(np.pi * np.arange(11) / 10))).astype(np.float32),
        dims='time', coords={'time': times[30:41]})
    expected['sla'][41:] += 100
    with xr.open_zarr(tmp_path / 'test.zarr') as result:
        xr.testing.assert_allclose(result.drop_attrs(), expected)
    assert sources == {str(p): p.stat().st_mtime_ns
                       for p in tmp_path.glob('subwindow_*/*/**/*') if p.is_file()}


def test_resume_after_publication_before_cleanup(tmp_path, monkeypatch):
    from src import zarr_merge

    expected = _dataset()
    output = tmp_path / 'final.zarr'
    work = tmp_path / 'final.zarr.merge-work'
    rmtree = zarr_merge.shutil.rmtree

    def interrupted(path, *args, **kwargs):
        if path == work:
            raise RuntimeError('simulated kill after publication')
        return rmtree(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(zarr_merge.shutil, 'rmtree', interrupted)
        with pytest.raises(RuntimeError, match='after publication'):
            write_merged_zarr(expected, output, [], zarr_time_chunk=4)
    chunk = output / 'sla' / '0.0.0'
    mtime = chunk.stat().st_mtime_ns
    write_merged_zarr(expected, output, [], zarr_time_chunk=4)
    assert chunk.stat().st_mtime_ns == mtime
    assert not work.exists()
    with xr.open_zarr(output) as result:
        xr.testing.assert_equal(result.drop_attrs(), expected)
