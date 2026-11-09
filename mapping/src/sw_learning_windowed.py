"""Sequential, resumable E-steps by window and one shared annual M-step."""
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import numpy as np
from .sw_learning import LearningOptions, parameter_groups, spatial_edges, maximize_window_statistics
from .sw_learning_moments import window_moments, combine_moments


def partition_windows(start, end, days=30, overlap=10):
    if days < 1 or not 0 <= overlap < days or end <= start or (end-start).total_seconds() % 86400:
        raise ValueError('Windows require increasing whole-day dates and positive duration')
    result = []
    left = start
    while left < end:
        right = min(left+timedelta(days=days), end)
        result.append(dict(id=len(result), start=left.isoformat(), end=right.isoformat(),
                           days=(right-left).days,
                           owned_start=0 if not result else (previous_end-left).days+1,
                           initial_offset_days=0 if not result else days-overlap))
        if right == end:
            break
        previous_end = right
        left += timedelta(days=days-overlap)
    return result


def atomic_npz(path, **values):
    path = Path(path)
    with path.with_suffix('.tmp').open('wb') as stream:
        np.savez_compressed(stream, **values)
    path.with_suffix('.tmp').replace(path)


def learn_windows(windows, records, initial_statistics, run_window, output, options=None,
                  provenance='', resume=False):
    """run_window(window, statistics_file, warm_start, folder, seed, preceding_posterior) -> posterior.npz.

    Workers are released after each window. Only O(natom) summed moments and
    hyperparameters remain resident in the coordinator. No monthly parameter
    estimates are averaged and no cross-window lag is fabricated.
    """
    options = options or LearningOptions()
    options.validate()
    if not windows:
        raise ValueError('No windows')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(json.dumps(dict(windows=windows, options=asdict(options),
        provenance=provenance), sort_keys=True).encode()+np.asarray(records).tobytes()
        +b''.join(np.asarray(initial_statistics[k]).tobytes() for k in ('std','tdec','mean'))).hexdigest()
    manifest = output/'windowed_manifest.json'
    if manifest.exists():
        if not resume:
            raise FileExistsError('Use --resume or a new output directory')
        if json.loads(manifest.read_text())['signature'] != signature:
            raise ValueError('Windowed checkpoint configuration/layout mismatch')
    else:
        manifest.write_text(json.dumps(dict(signature=signature, windows=windows, options=asdict(options),
            observations='All observations in every window, including overlap; local likelihoods are not an independent annual likelihood',
            moment_ownership='Each date contributes moments from the first window covering it; leading overlap excluded only from M-step',
            initial_state='First: GLORYS filtered. Subsequent: preceding MAP at overlap start; held fixed in local differentiation',
            boundary_controls='Local copies in overlap; only uniquely owned core moments enter M-step; no cross-core lag'), indent=2))
    std, tdec, mean = [np.array(initial_statistics[k], copy=True) for k in ('std','tdec','mean')]
    anchor = np.log(np.stack((std,tdec)))
    history = []
    checkpoint = output/'windowed_latest.npz'
    if checkpoint.exists():
        with np.load(checkpoint) as saved:
            if str(saved['signature']) != signature:
                raise ValueError('Windowed checkpoint mismatch')
            std,tdec,mean,anchor = [saved[k].copy() for k in ('std','tdec','mean','anchor')]
            history = json.loads(str(saved['history']))
    groups = parameter_groups(records); edges = spatial_edges(records)
    def e_step(label, iteration, statistics, previous):
        folder = output/label; folder.mkdir(exist_ok=True)
        input_file = folder/'statistics.npz'
        if input_file.exists():
            with np.load(input_file) as saved:
                if str(saved['signature']) != signature or any(not np.array_equal(saved[k],v) for k,v in statistics.items()):
                    raise ValueError('E-step statistics mismatch')
        else:
            atomic_npz(input_file, **statistics, signature=signature, records=records)
        total = None; diagnostics = []; paths = []
        for window in windows:
            wid = window['id']; work = folder/f'window_{wid:03d}'; work.mkdir(exist_ok=True)
            done = work/'complete.json'
            token = f'{signature}:{label}:{wid}'
            if done.exists():
                saved = json.loads(done.read_text())
                if saved['token'] != token:
                    raise ValueError('Window completion mismatch')
                path = Path(saved['posterior'])
            else:
                warm = output/previous/f'window_{wid:03d}'/'posterior.npz' if previous else None
                print(f'{label}: window {wid+1}/{len(windows)} {window["start"]}', flush=True)
                path = Path(run_window(window, input_file, warm, work, options.seed+iteration*len(windows)+wid, paths[-1] if paths else None)).resolve()
            with np.load(path) as result:
                if not np.array_equal(result['records'], records):
                    raise ValueError('Window spatial basis mismatch')
                if any(not np.array_equal(result[k],v) for k,v in statistics.items()):
                    raise ValueError('Window posterior uses different hyperparameters')
                a,p,c = (result[k] for k in ('analysis','variance','lag_covariance'))
                if a.shape != (window['days']+1,len(records)):
                    raise ValueError('Window posterior time layout mismatch')
                core = window['owned_start']
                moments = window_moments(a[core:],p[core:],c[core:])
                diagnostic = json.loads(str(result['diagnostics']))
            total = moments if total is None else combine_moments((total,moments))
            diagnostics.append(diagnostic); paths.append(str(path))
            if not done.exists():
                tmp = done.with_suffix('.tmp')
                tmp.write_text(json.dumps(dict(token=token,posterior=str(path)),indent=2)); tmp.replace(done)
        return total, diagnostics, paths
    for iteration in range(len(history),options.outer_iterations):
        if history and history[-1]['phase']=='local' and history[-1]['parameter_change']<options.tolerance:
            break
        statistics = dict(std=std,tdec=tdec,mean=mean)
        moments, diagnostics, _ = e_step(f'iteration_{iteration:03d}', iteration, statistics,
                                        f'iteration_{iteration-1:03d}' if iteration else None)
        pooled = iteration < options.pooled_iterations
        ns,nt,nm,objective = maximize_window_statistics(moments,std,tdec,
            groups if pooled else np.arange(len(records)), np.empty((0,2),int) if pooled else edges,anchor,options)
        change = max(float(np.max(np.abs(np.log(ns/std)))),float(np.max(np.abs(np.log(nt/tdec)))),
                     float(np.max(np.abs(nm-mean)/std)))
        std,tdec,mean = ns,nt,nm
        if pooled: anchor=np.log(np.stack((std,tdec)))
        history.append(dict(iteration=iteration,phase='pooled' if pooled else 'local',parameter_change=change,
                            expected_objective=objective,windows=diagnostics))
        atomic_npz(checkpoint,std=std,tdec=tdec,mean=mean,anchor=anchor,signature=signature,history=json.dumps(history))
        (output/'history.json').write_text(json.dumps(history,indent=2))
    # Recompute posteriors at the final annual parameters, independently by window.
    _, diagnostics, paths = e_step('final',options.outer_iterations+1,dict(std=std,tdec=tdec,mean=mean),
                                   f'iteration_{len(history)-1:03d}' if history else None)
    atomic_npz(output/'annual_statistics.npz',std=std,tdec=tdec,mean=mean,records=records,signature=signature)
    (output/'result.json').write_text(json.dumps(dict(iterations=len(history),windows=windows,
        converged=bool(history and history[-1]['phase']=='local' and history[-1]['parameter_change']<options.tolerance),
        final_posteriors=paths,final_diagnostics=diagnostics,
        approximation='Sequential overlapping-window MAP/Gauss-Newton moments; shared stationary AR1 parameters; fixed analyzed initial states; core-only M-step surrogate'),indent=2))
    return dict(std=std,tdec=tdec,mean=mean,history=history,posteriors=paths)
