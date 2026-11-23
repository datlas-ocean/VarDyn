"""Bounded, restartable publication of a final temporal Zarr product."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time

import pandas as pd
import xarray as xr


def _source_signature(paths):
    """Track source metadata without scanning millions of source chunks."""
    signature = []
    for path in paths:
        root = Path(path).resolve()
        entries = [root, *sorted(root.glob('*/.zarray')),
                   *sorted(root.glob('*/.zattrs')), root / '.zattrs',
                   root / '.zmetadata']
        signature.append([
            (str(entry), entry.stat().st_size, entry.stat().st_mtime_ns)
            for entry in entries if entry.exists()])
    return signature


def _save_manifest(path, payload):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_merged_zarr(dataset, output_path, source_paths, *,
                      zarr_time_chunk=4, zarr_spatial_chunk=256,
                      zarr_compression_level=3, merge_plan=None):
    """Write disjoint aligned time regions; preserve completed lots on failure.

    Source windows must remain immutable during a merge. The manifest binds a
    restart to their metadata, the output schema, timestamps and encoding.
    Existing source chunk sizes are independent of the final archive's chunks.
    """
    from .state import _set_zarr_time_encoding, _zarr_encoding

    output = Path(output_path)
    work = Path(f'{output}.merge-work')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the lock inode outside work: removing it would race another opener.
    with Path(f'{output}.merge.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another merge owns {output}') from exc
        work.mkdir(parents=True, exist_ok=True)
        return _write_locked(
            dataset, output, work, source_paths, zarr_time_chunk,
            zarr_spatial_chunk, zarr_compression_level,
            _set_zarr_time_encoding, _zarr_encoding, merge_plan)


def _write_locked(dataset, output, work, source_paths, time_chunk,
                  spatial_chunk, compression_level, set_time, encoding_for,
                  merge_plan):
    archive = work / 'archive.zarr'
    manifest_path = work / 'progress.json'
    backup = work / 'previous.zarr'
    # Restore the previous product first, even if changed settings subsequently
    # prevent reuse of this checkpoint.
    if backup.exists() and not output.exists():
        os.replace(backup, output)
    encoding = encoding_for(dataset, time_chunk, spatial_chunk, compression_level)
    times = pd.DatetimeIndex(dataset.time.values)
    if not len(times):
        raise ValueError('Cannot merge an empty time axis')
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError('Final merge timestamps must be unique and ordered')
    sources = _source_signature(source_paths)
    identity = {
        'version': 1, 'sources': sources, 'merge_plan': merge_plan,
        'times': times.asi8.tolist(),
        'variables': {name: [list(var.dims), list(var.shape), str(var.dtype)]
                      for name, var in dataset.variables.items()},
        'encoding': [int(time_chunk), int(spatial_chunk), int(compression_level)],
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    total = len(times)
    bytes_per_time = sum(
        var.nbytes // total for var in dataset.variables.values()
        if 'time' in var.dims)
    # About 128 MiB of uncompressed output per lot, at least one whole chunk.
    batch_size = max(int(time_chunk), min(32, max(
        1, (128 * 1024**2) // max(1, bytes_per_time)))
        // int(time_chunk) * int(time_chunk))
    manifest = {'identity': digest, 'next_time': 0, 'batch_size': batch_size}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['identity'] != digest:
            raise RuntimeError(
                f'Merge inputs or settings changed. Preserve or remove {work} '
                'before starting a different merge; source windows are untouched.')
        batch_size = manifest['batch_size']
        if not archive.exists():
            # A kill after publication but before checkpoint cleanup must not
            # cause completed data to be written again.
            if output.exists():
                with xr.open_zarr(output, consolidated=True) as published:
                    if (published.attrs.get('_vardyn_merge_identity') == digest
                            and pd.DatetimeIndex(published.time.values).equals(times)):
                        shutil.rmtree(work)
                        print(f'Final Zarr archive already published: {output}', flush=True)
                        return
            raise RuntimeError(f'Checkpoint archive is missing: {archive}')
    else:
        if archive.exists():
            shutil.rmtree(archive)
        template = dataset.copy()
        import dask.array as da
        # Build a small metadata-only graph, independent of all source reads.
        for name, var in template.variables.items():
            if 'time' in var.dims and name not in template.indexes:
                template[name].data = da.empty(
                    var.shape, chunks=encoding[name]['chunks'], dtype=var.dtype)
        for var in template.variables.values():
            var.encoding = {}
        set_time(template)
        # Only initialize metadata; no delayed field writes are computed here.
        template.to_zarr(archive, mode='w', encoding=encoding,
                         compute=False, consolidated=False, zarr_format=2,
                         safe_chunks=False)
        static_names = [name for name, var in template.variables.items()
                        if 'time' not in var.dims]
        if static_names:
            static = template[static_names].drop_vars('time', errors='ignore')
            static = static.chunk({dim: min(size, int(spatial_chunk))
                                   for dim, size in static.sizes.items()})
            static.to_zarr(archive, mode='r+', consolidated=False,
                           safe_chunks=True)
        _save_manifest(manifest_path, manifest)

    start = manifest['next_time']
    if not 0 <= start <= total or (start != total and start % int(time_chunk)):
        raise RuntimeError(f'Invalid merge checkpoint in {manifest_path}')
    print(f'Final Zarr merge: resume at {start}/{total} timestamps; '
          f'lots={batch_size}, chunks={time_chunk}x{spatial_chunk}x{spatial_chunk}',
          flush=True)
    for begin in range(start, total, batch_size):
        end = min(total, begin + batch_size)
        lot = dataset.isel(time=slice(begin, end)).drop_vars([
            name for name, var in dataset.variables.items()
            if 'time' not in var.dims])
        for var in lot.variables.values():
            var.encoding = {}
        set_time(lot)
        # Every Dask task owns complete target chunks, including at overlap
        # boundaries. Only the final archive edge may have a shorter chunk.
        lot = lot.chunk({dim: int(time_chunk) if dim == 'time'
                         else min(size, int(spatial_chunk))
                         for dim, size in lot.sizes.items()})
        print(f'Final Zarr lot {begin}:{end}/{total}: '
              f'{times[begin]} → {times[end - 1]}', flush=True)
        started = time.monotonic()
        stopped = threading.Event()

        def heartbeat():
            while not stopped.wait(30):
                print(f'Final Zarr lot {begin}:{end}: writing, '
                      f'{time.monotonic() - started:.0f}s elapsed', flush=True)

        reporter = threading.Thread(target=heartbeat, daemon=True)
        reporter.start()
        try:
            lot.to_zarr(archive, mode='r+', region={'time': slice(begin, end)},
                        consolidated=False, safe_chunks=True)
        finally:
            stopped.set()
            reporter.join()
        # The marker advances only after all tasks succeeded. A partial lot is
        # overwritten in full on restart, never trusted because files exist.
        manifest['next_time'] = end
        _save_manifest(manifest_path, manifest)
        elapsed = max(time.monotonic() - started, 1e-9)
        print(f'Final Zarr progress: {end}/{total} ({100 * end / total:.1f}%), '
              f'lot {elapsed:.1f}s, {lot.nbytes / elapsed / 1024**2:.1f} MiB/s '
              'uncompressed', flush=True)

    if _source_signature(source_paths) != sources:
        raise RuntimeError('Source window metadata changed during the merge')
    import zarr
    zarr.open_group(str(archive), mode='a').attrs['_vardyn_merge_identity'] = digest
    zarr.consolidate_metadata(str(archive))
    with xr.open_zarr(archive, consolidated=True) as candidate:
        if not pd.DatetimeIndex(candidate.time.values).equals(times):
            raise RuntimeError('Final Zarr timestamps differ from expected dates')
        for name, var in dataset.variables.items():
            if candidate[name].dims != var.dims or candidate[name].shape != var.shape:
                raise RuntimeError(f'Final Zarr schema mismatch: {name}')
    if output.exists():
        if backup.exists():
            shutil.rmtree(backup)
        os.replace(output, backup)
    try:
        os.replace(archive, output)
    except BaseException:
        if backup.exists() and not output.exists():
            os.replace(backup, output)
        raise
    if backup.exists():
        shutil.rmtree(backup)
    shutil.rmtree(work)
    print(f'Published final Zarr archive: {output}', flush=True)
