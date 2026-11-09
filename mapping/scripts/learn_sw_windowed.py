#!/usr/bin/env python3
"""Shared annual SW forcing statistics from sequential overlapping windows."""
import argparse
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
os.environ.setdefault('HDF5_USE_FILE_LOCKING','FALSE')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('MPLBACKEND','Agg')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'mapping'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--window-days',type=int,default=30)
    parser.add_argument('--overlap-days',type=int,default=10)
    parser.add_argument('--days',type=int,help='Shorten the total experiment for validation')
    parser.add_argument('--bounds',type=float,nargs=4)
    parser.add_argument('--wavelengths',type=float,nargs='+')
    parser.add_argument('--outer-iterations',type=int,default=8)
    parser.add_argument('--pooled-iterations',type=int,default=2)
    parser.add_argument('--map-maxiter',type=int,default=3000)
    parser.add_argument('--map-maxcor',type=int,default=20)
    parser.add_argument('--cg-maxiter',type=int,default=1500)
    parser.add_argument('--cg-rtol',type=float,default=1.e-4)
    parser.add_argument('--cg-preconditioner',choices=['none','spectral','spectral-host'],default='spectral')
    parser.add_argument('--cg-spectral-rank',type=int,default=256)
    parser.add_argument('--probes',type=int,default=16)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    args = parser.parse_args()
    import numpy as np
    import xarray as xr
    from src import exp
    from learning_local.sw_learning import LearningOptions
    from learning_local.sw_learning_windowed import learn_windows, partition_windows
    cfg = exp.Exp(str(args.config.resolve()))
    end = cfg.EXP.init_date+timedelta(days=args.days) if args.days is not None else cfg.EXP.final_date
    windows = partition_windows(cfg.EXP.init_date,end,args.window_days,args.overlap_days)
    options = LearningOptions(**{key:getattr(args,key) for key in ('outer_iterations','pooled_iterations',
        'map_maxiter','map_maxcor','cg_maxiter','cg_rtol','cg_preconditioner','cg_spectral_rank','probes','seed')})
    options.validate()
    output = args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    base = [sys.executable,'-u',str(ROOT/'mapping/scripts/learn_sw_daily.py'),'--config',str(args.config.resolve())]
    if args.bounds: base += ['--bounds',*map(str,args.bounds)]
    if args.wavelengths: base += ['--wavelengths',*map(str,args.wavelengths)]
    def launch(command,log):
        with Path(log).open('a') as stream:
            result = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Window worker failed (exit {result.returncode}); inspect {log}. Use --resume after correction.')
    bootstrap = output/'bootstrap'
    launch(base+['--output',str(bootstrap),'--start-date',windows[0]['start'],
                 '--days',str(windows[0]['days']),'--prepare-only'],output/'bootstrap.log')
    with np.load(bootstrap/'basis_metadata.npz') as saved:
        records=saved['records'].copy()
        statistics={key:saved[key].copy() for key in ('std','tdec','mean')}
    sources = [ROOT/'mapping/learning_local/sw_learning_windowed.py',ROOT/'mapping/learning_local/sw_learning_moments.py',
               Path(__file__), ROOT/'mapping/scripts/learn_sw_daily.py']
    provenance = hashlib.sha256(json.dumps(json.loads((bootstrap/'provenance.json').read_text()),sort_keys=True).encode()
        +(bootstrap/'resolved_config.json').read_bytes()+b''.join(p.read_bytes() for p in sources)).hexdigest()
    preparation = dict(windows=windows,spatial_atoms=len(records),
        maximum_controls=max(w['days']+1 for w in windows)*len(records),
        unique_observation_dates=sum(w['days']+1-w['owned_start'] for w in windows),
        options=asdict(options),provenance=provenance)
    plan = output/'window_plan.json'
    if plan.exists() and json.loads(plan.read_text()) != preparation:
        raise ValueError('Existing window plan differs; use a new output directory')
    plan.write_text(json.dumps(preparation,indent=2))
    print(json.dumps(preparation,indent=2),flush=True)
    if args.prepare_only: return
    def run_window(window,statistics_file,warm,work,seed,preceding):
        command=base+['--output',str(work),'--start-date',window['start'],'--days',str(window['days']),
            '--e-step','--statistics',str(statistics_file),'--seed',str(seed)]
        for key in ('map_maxiter','map_maxcor','cg_maxiter','cg_rtol','cg_preconditioner','cg_spectral_rank','probes'):
            command += ['--'+key.replace('_','-'),str(getattr(options,key))]
        if warm is not None: command += ['--warm-start',str(warm)]
        if preceding is not None:
            command += ['--initial-trajectory',str(preceding),'--initial-day',str(window['initial_offset_days'])]
        launch(command,work/'worker.log')
        return work/'posterior.npz'
    result=learn_windows(windows,records,statistics,run_window,output,options,provenance,args.resume)
    data={key:(('atom',),result[key]) for key in ('std','tdec','mean')}
    for i,name in enumerate(('component','wavelength_km','radius_km','center_x','center_y','basin','support','direct_support')):
        data[name]=(('atom',),records[:,i])
    xr.Dataset(data,coords={'atom':np.arange(len(records))},attrs=dict(schema='sw-windowed-learning-v1',
        window_days=args.window_days,overlap_days=args.overlap_days,
        potential_units='m2/s/day',height_units='m interface/day',tdec_units='day',
        approximation='Sequential windows; fixed preceding MAP initial state; unique-core moments; shared annual parameters'
        )).to_netcdf(output/'annual_statistics.nc')


if __name__=='__main__': main()
