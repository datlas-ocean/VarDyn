#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Florian Le Guillou on June 2026.

Runs inversion drivers and assimilation orchestration.
"""
from .config import USE_FLOAT64
from dataclasses import dataclass
from typing import Callable
import sys, os
import time
import numpy as np
import pandas as pd
import os
import matplotlib.pylab as plt
from datetime import datetime, timedelta
import scipy.optimize as opt
import gc
import xarray as xr

import glob
from contextlib import ExitStack, nullcontext
from importlib.machinery import SourceFileLoader

import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import jit
from jax.lax import scan
jax.config.update("jax_enable_x64", USE_FLOAT64)

from . import tools as grid



class ConvergenceReached(Exception):
    pass

class CrazyGradient(Exception):
    pass

def _block_until_ready(obj):
    """Wait for pending JAX work before reporting wall-clock timings."""
    return jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x,
        obj,
    )


@dataclass
class MinimizationResult:
    """Result and diagnostics from one minimizer adapter."""

    control: object
    cost_history: list[float]
    gradient_norm_history: list[float]
    gradient_mean_history: list[float]
    iteration_seconds: list[float]
    cumulative_seconds: list[float]
    evaluation_count_history: list[int]
    function_evaluations: int
    iterations_completed: int
    converged: bool
    status: str
    optimizer_init_seconds: float = 0.0
    minimization_seconds: float = 0.0
    line_search_evaluations: list[int] | None = None
    step_size_history: list[float] | None = None


def _criterion_reached(
    *,
    cost_history,
    gradient_norm_history,
    relative_gradient_tolerance,
    relative_cost_tolerance,
    patience,
    minimum_iterations,
):
    iterations_completed = len(cost_history) - 1
    if iterations_completed < minimum_iterations:
        return False
    if len(cost_history) - 1 < patience:
        return False

    initial_gradient_norm = max(float(gradient_norm_history[0]), 1e-30)
    criteria = []
    if relative_gradient_tolerance is not None:
        recent_gradients = gradient_norm_history[-patience:]
        criteria.append(
            all(
                float(norm) / initial_gradient_norm
                <= relative_gradient_tolerance
                for norm in recent_gradients
            )
        )
    if relative_cost_tolerance is not None:
        recent_pairs = zip(
            cost_history[-patience - 1 : -1],
            cost_history[-patience:],
        )
        criteria.append(
            all(
                abs(float(previous) - float(current))
                / max(abs(float(previous)), abs(float(current)), 1.0)
                <= relative_cost_tolerance
                for previous, current in recent_pairs
            )
        )
    return any(criteria)


def minimize_optax_decoupled(
    evaluate: Callable,
    initial_control,
    *,
    maxiter,
    history_size=10,
    relative_gradient_tolerance=None,
    relative_cost_tolerance=None,
    convergence_patience=1,
    minimum_iterations=1,
    gradient_max_norm=None,
    iteration_callback=None,
    initial_value=None,
    initial_gradient=None,
):
    """Run L-BFGS on device with a scalar Python Armijo line search.

    ``evaluate`` must accept and return JAX device arrays. The control,
    gradients, L-BFGS history, directions, and trial points remain on device.
    Python receives only scalar diagnostics used to accept a step.
    """
    try:
        import optax
    except ImportError as exc:
        raise ImportError(
            "minimizer='optax-decoupled' requires the 'optax' package"
        ) from exc

    if maxiter < 1:
        raise ValueError("maxiter must be >= 1")
    if history_size < 1:
        raise ValueError("history_size must be >= 1")
    if convergence_patience < 1:
        raise ValueError("convergence_patience must be >= 1")
    if minimum_iterations < 1:
        raise ValueError("minimum_iterations must be >= 1")

    params = jnp.asarray(initial_control)
    if initial_value is None or initial_gradient is None:
        initial_value, initial_gradient = evaluate(params)
        _block_until_ready((initial_value, initial_gradient))

    direction_transform = optax.scale_by_lbfgs(memory_size=history_size)

    @jax.jit
    def compute_direction(params, gradient, direction_state):
        preconditioned, direction_state = direction_transform.update(
            gradient,
            direction_state,
            params,
        )
        direction = -preconditioned
        directional_derivative = jnp.vdot(gradient, direction).real
        gradient_norm = jnp.linalg.norm(gradient)
        steepest = -gradient / jnp.maximum(gradient_norm, 1.0)
        direction = jnp.where(
            directional_derivative < 0.0,
            direction,
            steepest,
        )
        directional_derivative = jnp.vdot(gradient, direction).real
        return direction, direction_state, directional_derivative

    @jax.jit
    def trial_point(params, direction, step_size):
        return params + step_size * direction

    optimizer_init_start = time.perf_counter()
    direction_state = direction_transform.init(params)
    warm_direction, _, warm_derivative = compute_direction(
        params,
        initial_gradient,
        direction_state,
    )
    warm_trial = trial_point(
        params,
        warm_direction,
        jnp.asarray(1.0, dtype=params.dtype),
    )
    _block_until_ready((warm_derivative, warm_trial))
    optimizer_init_seconds = time.perf_counter() - optimizer_init_start

    value = initial_value
    gradient = initial_gradient
    initial_cost = float(np.asarray(jax.device_get(value)))
    initial_gradient_norm = float(
        np.asarray(jax.device_get(jnp.linalg.norm(gradient)))
    )
    initial_gradient_mean = float(
        np.asarray(jax.device_get(jnp.mean(jnp.abs(gradient))))
    )
    cost_history = [initial_cost]
    gradient_norm_history = [initial_gradient_norm]
    gradient_mean_history = [initial_gradient_mean]
    iteration_seconds = []
    cumulative_seconds = [0.0]
    evaluation_count_history = [1]
    line_search_evaluations = []
    step_size_history = []
    evaluations = 1
    converged = False
    status = "maximum iterations reached"
    minimization_start = time.perf_counter()

    armijo_c1 = 1e-4
    contraction = 0.5
    max_linesearch_steps = 20
    minimum_step = 2.0**-20

    for iteration in range(1, maxiter + 1):
        iteration_start = time.perf_counter()
        direction, proposed_state, directional_derivative = compute_direction(
            params,
            gradient,
            direction_state,
        )
        _block_until_ready(
            (direction, proposed_state, directional_derivative)
        )
        current_cost = float(np.asarray(jax.device_get(value)))
        derivative_host = float(
            np.asarray(jax.device_get(directional_derivative))
        )

        step_size = 1.0
        accepted = None
        best_trial = None
        trials = 0
        for _ in range(max_linesearch_steps):
            candidate = trial_point(
                params,
                direction,
                jnp.asarray(step_size, dtype=params.dtype),
            )
            candidate_value, candidate_gradient = evaluate(candidate)
            _block_until_ready(
                (candidate, candidate_value, candidate_gradient)
            )
            candidate_cost = float(
                np.asarray(jax.device_get(candidate_value))
            )
            candidate_gradient_max = float(
                np.asarray(
                    jax.device_get(jnp.max(jnp.abs(candidate_gradient)))
                )
            )
            trials += 1
            evaluations += 1

            stable_gradient = (
                gradient_max_norm is None
                or candidate_gradient_max <= gradient_max_norm
            )
            if np.isfinite(candidate_cost) and stable_gradient:
                if best_trial is None or candidate_cost < best_trial[0]:
                    best_trial = (
                        candidate_cost,
                        candidate,
                        candidate_value,
                        candidate_gradient,
                        step_size,
                    )
                armijo_bound = (
                    current_cost
                    + armijo_c1 * step_size * derivative_host
                )
                if candidate_cost <= armijo_bound:
                    accepted = (
                        candidate,
                        candidate_value,
                        candidate_gradient,
                        step_size,
                    )
                    break

            step_size *= contraction
            if step_size < minimum_step:
                break

        if accepted is None and best_trial is not None:
            if best_trial[0] < current_cost:
                accepted = best_trial[1:]
        if accepted is None:
            status = "line search failed"
            break

        params, value, gradient, accepted_step = accepted
        direction_state = proposed_state
        _block_until_ready((params, value, gradient, direction_state))

        cost = float(np.asarray(jax.device_get(value)))
        gradient_norm = float(
            np.asarray(jax.device_get(jnp.linalg.norm(gradient)))
        )
        gradient_mean = float(
            np.asarray(jax.device_get(jnp.mean(jnp.abs(gradient))))
        )
        iteration_seconds.append(time.perf_counter() - iteration_start)
        cost_history.append(cost)
        gradient_norm_history.append(gradient_norm)
        gradient_mean_history.append(gradient_mean)
        cumulative_seconds.append(time.perf_counter() - minimization_start)
        evaluation_count_history.append(evaluations)
        line_search_evaluations.append(trials)
        step_size_history.append(float(accepted_step))

        if iteration_callback is not None:
            iteration_callback(
                iteration,
                params,
                cost,
                gradient,
                {
                    "gradient_norm": gradient_norm,
                    "gradient_mean": gradient_mean,
                    "relative_gradient_norm": (
                        gradient_norm / max(initial_gradient_norm, 1e-30)
                    ),
                    "step_size": float(accepted_step),
                    "line_search_evaluations": trials,
                    "iteration_seconds": iteration_seconds[-1],
                },
            )

        if _criterion_reached(
            cost_history=cost_history,
            gradient_norm_history=gradient_norm_history,
            relative_gradient_tolerance=relative_gradient_tolerance,
            relative_cost_tolerance=relative_cost_tolerance,
            patience=convergence_patience,
            minimum_iterations=minimum_iterations,
        ):
            converged = True
            status = "convergence criterion reached"
            break

    return MinimizationResult(
        control=params,
        cost_history=cost_history,
        gradient_norm_history=gradient_norm_history,
        gradient_mean_history=gradient_mean_history,
        iteration_seconds=iteration_seconds,
        cumulative_seconds=cumulative_seconds,
        evaluation_count_history=evaluation_count_history,
        function_evaluations=evaluations,
        iterations_completed=len(iteration_seconds),
        converged=converged,
        status=status,
        optimizer_init_seconds=optimizer_init_seconds,
        minimization_seconds=time.perf_counter() - minimization_start,
        line_search_evaluations=line_search_evaluations,
        step_size_history=step_size_history,
    )

def Inv(config, State=None, Model=None, dict_obs=None, Obsop=None, Basis=None, *args, **kwargs):

    """
    NAME
        Inv

    DESCRIPTION
        Main function calling subfunctions for specific Inversion algorithms
    """
    
    if config.INV is None:
        return Inv_forward(config, State=State, Model=Model)
    
    print(config.INV)
    
    if config.INV.super=='INV_4DVAR':
        return Inv_4Dvar(config, State=State, Model=Model, dict_obs=dict_obs, Obsop=Obsop, Basis=Basis)
    
    else:
        sys.exit(config.INV.super + ' not implemented yet')
        
def Inv_forward(config,State,Model):
    
    """
    NAME
        Inv_forward

    DESCRIPTION
        Run a model forward integration  
    
    """

    if 'JAX' in config.MOD.super:
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    
    present_date = config.EXP.init_date
    nstep = int(config.EXP.saveoutput_time_step.total_seconds()//Model.dt)

    # Set boundary conditions at output times (if configured)
    time_bc = np.array([np.datetime64(time) for time in Model.timestamps[::nstep]])
    t_bc = np.array([t for t in Model.T[::nstep]])
    Model.set_bc(time_bc, t_bc=t_bc)

    t = 0
    Model.init(State,t)
    Model.save_output(State,present_date,name_var=Model.var_to_save,t=t)
    State.plot(title='Start of forward integration')

    while present_date + timedelta(seconds=nstep*Model.dt) <= config.EXP.final_date :
        
        # Propagation
        Model.step(State,nstep,t=t)

        # Time increment
        present_date += timedelta(seconds=nstep*Model.dt)
        t += nstep*Model.dt

        # Save
        if config.EXP.saveoutputs:
            Model.save_output(State,present_date,name_var=Model.var_to_save,t=t)    
    
    State.plot(title='End of forward integration')
        
    return
       
def Inv_4Dvar(config=None,State=None,Model=None,dict_obs=None,Obsop=None,Basis=None,verbose=True,gpu_device=None) :

    
    '''
    Run a 4Dvar analysis
    '''

    #if 'JAX' in config.MOD.super:
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    if gpu_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)

    minimizer_name = getattr(config.INV, 'minimizer', 'scipy')
    supported_minimizers = ('scipy', 'optax-decoupled')
    if minimizer_name not in supported_minimizers:
        raise ValueError(
            f"Unknown 4DVar minimizer {minimizer_name!r}; expected one of "
            f"{supported_minimizers}"
        )
    if minimizer_name == 'optax-decoupled':
        if not getattr(config.INV, 'device_resident_state', False):
            raise ValueError(
                "minimizer='optax-decoupled' requires "
                "device_resident_state=True"
            )
        if not getattr(config.INV, 'jit_cost_and_grad', False):
            raise ValueError(
                "minimizer='optax-decoupled' requires "
                "jit_cost_and_grad=True"
            )
        if getattr(config.INV, 'cost_and_grad_schedule', None) != 'scan':
            raise ValueError(
                "minimizer='optax-decoupled' requires "
                "cost_and_grad_schedule='scan'"
            )
        
    
    # Module initializations
    if Model is None:
        # initialize Model operator 
        from . import mod
        Model = mod.Model(config, State, verbose=verbose)
    if dict_obs is None:
        # initialize Obs
        from . import obs
        dict_obs = obs.Obs(config, State)
    if Obsop is None:
        # initialize Obsop
        from . import obsop
        Obsop = obsop.Obsop(config, State, dict_obs, Model, verbose=verbose)
    if Basis is None:
        # initialize Basis
        from . import basis
        Basis = basis.Basis(config, State, verbose=verbose)

    # Process observations
    print('process observation operators')
    Obsop.process_obs()
    
    # Compute checkpoints when the cost function will be evaluated 
    nstep_check = int(config.INV.timestep_checkpoint.total_seconds()//Model.dt)
    checkpoints = [0]
    time_checkpoints = [np.datetime64(Model.timestamps[0])]
    t_checkpoints = [Model.T[0]]
    check = 0
    for i,t in enumerate(Model.timestamps[:-1]):
        if i>0 and (Obsop.is_obs(t) or check==nstep_check):
            checkpoints.append(i)
            time_checkpoints.append(np.datetime64(t))
            t_checkpoints.append(Model.T[i])
            if check==nstep_check:
                check = 0
        check += 1 
    checkpoints.append(len(Model.timestamps)-1) # last timestep
    time_checkpoints.append(np.datetime64(Model.timestamps[-1]))
    t_checkpoints.append(Model.T[-1])
    checkpoints = np.asarray(checkpoints)
    time_checkpoints = np.asarray(time_checkpoints)
    print(f'--> {checkpoints.size} checkpoints to evaluate the cost function')

    # Boundary conditions at checkpoints
    Model.set_bc(time_checkpoints, t_bc=np.asarray(t_checkpoints))
    
    # Observations operator 
    if config.INV.anomaly_from_bc and Model._bc_fields is not None:  # Remove boundary fields if anomaly mode is chosen
        var_bc = Model._bc_fields.interp(np.array([np.datetime64(date) for date in Obsop.date_obs]))
    else:
        var_bc = None
    
    # Initial model state
    Model.init(State)
    State.plot(title='Init State')
    State.plot(title='Init params', params=True)

    # Set Reduced Basis
    if Basis is not None:
        time_basis = np.arange(0,Model.T[-1]+nstep_check*Model.dt,nstep_check*Model.dt)/24/3600 # Time (in days) for which the basis components will be compute (at each timestep_checkpoint)
        Xb, Q = Basis.set_basis(time_basis, return_q=True, State=State) # Q is the standard deviation. To get the variance, use Q^2
    else:
        sys.exit('4Dvar only work with reduced basis!!')

    # Initialization, I/O and plotting are complete. Keep every state field and
    # control parameter resident on the selected accelerator from here on.
    if getattr(config.INV, 'device_resident_state', False):
        State.to_device()

    # Covariance matrix
    if config.INV.sigma_B is not None:     
        print('Warning: sigma_B is prescribed --> ignore Q of the reduced basis')
        # Least squares
        B = Cov(config.INV.sigma_B)
        R = Cov(config.INV.sigma_R)
    else:
        B = Cov(Q)
        R = Cov(config.INV.sigma_R)
    
    # Read Background vector 
    if config.INV.path_background is not None: 
        # Read previous minimum 
        print('Read background basis:',config.INV.path_background)
        ds = xr.open_dataset(config.INV.path_background)
        Xb[:len(ds.res.values)] = ds.res.values   
        ds.close()

    # Variational object initialization
    var = Variational(
        config=config, M=Model, H=Obsop, State=State, B=B, R=R, Basis=Basis, Xb=Xb, checkpoints=checkpoints, freq_it_plot=config.INV.freq_it_plot, print_time=config.INV.print_time)
    
    # Initial Control vector 
    if config.INV.path_init_4Dvar is None:
        Xopt = np.zeros((Xb.size,))
    else:
        # Read previous minimum 
        print('Read previous minimum:',config.INV.path_init_4Dvar)
        ds = xr.open_dataset(config.INV.path_init_4Dvar)
        Xopt = np.zeros(var.Xb.shape, dtype=np.float64)
        Xopt[:ds.res.size] = ds.res.values
        ds.close()
        if config.INV.prec:
            with var.cost_precision():
                Xopt = B.invsqr(Xopt - var.Xb)
    
    # Path where to save the control vector at each 4Dvar iteration 
    # (carefull, depending on the number of control variables, these files may use large disk space)
    if config.INV.path_save_control_vectors is not None:
        path_save_control_vectors = config.INV.path_save_control_vectors
    else:
        path_save_control_vectors = config.EXP.tmp_DA_path
    if not os.path.exists(path_save_control_vectors):
        os.makedirs(path_save_control_vectors)

    # Restart mode
    maxiter = config.INV.maxiter
    if config.INV.restart_4Dvar:
        tmp_files = sorted(glob.glob(os.path.join(path_save_control_vectors,'X_it*.nc')))
        if len(tmp_files)>0:
            print('Restart at:',tmp_files[-1])
            try:
                ds = xr.open_dataset(tmp_files[-1])
            except:
                if len(tmp_files)>1:
                    ds = xr.open_dataset(tmp_files[-2])
            try:
                Xopt = ds.res.values
                maxiter = max(config.INV.maxiter - len(tmp_files), 0)
                ds.close()
            except:
                Xopt = +Xopt

    # The historical SciPy adapter owns a host vector. The Optax adapter
    # converts this initial/restart value once, then keeps every vector on GPU.
    Xopt = np.asarray(jax.device_get(Xopt), dtype=np.float64)
    skip_minimization = (
        (config.INV.restart_4Dvar or config.INV.path_init_4Dvar is not None)
        and maxiter == 0
    )

    if minimizer_name == 'optax-decoupled' and not skip_minimization:
        print('\n*** Minimization (Optax L-BFGS, decoupled line search) ***\n')

        iterations_path = os.path.join(
            path_save_control_vectors,
            'iterations.txt',
        )
        with open(iterations_path, 'w') as stream:
            stream.write('Minimization\n')

        def optax_callback(iteration, control, cost, gradient, diagnostics):
            text = (
                f"J={cost:.6E}, "
                f"G={diagnostics['gradient_mean']:.6E}, "
                f"relative_G={diagnostics['relative_gradient_norm']:.6E}, "
                f"step={diagnostics['step_size']:.6E}, "
                f"evaluations={diagnostics['line_search_evaluations']}, "
                f"time={diagnostics['iteration_seconds']:.3f}s"
            )
            if verbose:
                print(f"* iteration {iteration} {text}")
            with open(iterations_path, 'a') as stream:
                stream.write(f"iteration {iteration}, {text}\n")
            if config.INV.save_minimization:
                control_host = np.asarray(jax.device_get(control))
                dataset = xr.Dataset({'res': (('x',), control_host)})
                dataset.to_netcdf(os.path.join(
                    path_save_control_vectors,
                    'X_it.nc',
                ))
                dataset.close()

        patience = getattr(config.INV, 'convergence_nit', None)
        if patience is None:
            patience = 1
        minimization_result = minimize_optax_decoupled(
            var.cost_and_grad,
            jnp.asarray(Xopt, dtype=var.cost_dtype),
            maxiter=maxiter,
            history_size=getattr(config.INV, 'lbfgs_history_size', 10),
            relative_gradient_tolerance=getattr(
                config.INV,
                'relative_gradient_tolerance',
                None,
            ),
            relative_cost_tolerance=getattr(config.INV, 'ftol', None),
            convergence_patience=patience,
            minimum_iterations=getattr(
                config.INV,
                'minimum_iterations',
                1,
            ),
            gradient_max_norm=getattr(
                config.INV,
                'gradient_max_norm',
                None,
            ),
            iteration_callback=optax_callback,
        )
        Xres = minimization_result.control
        print(f"\nMinimization status: {minimization_result.status}")
        print(f"\nFinal cost function value: {minimization_result.cost_history[-1]}")
        print(f"\nNumber of iterations: {minimization_result.iterations_completed}")

        if config.INV.save_minimization:
            dataset = xr.Dataset({
                'cost': (('i',), np.asarray(minimization_result.cost_history)),
                'grad': (('i',), np.asarray(
                    minimization_result.gradient_mean_history
                )),
                'grad_norm': (('i',), np.asarray(
                    minimization_result.gradient_norm_history
                )),
            })
            dataset.to_netcdf(os.path.join(
                path_save_control_vectors,
                'minimization_trajectory.nc',
            ))
            dataset.close()

    elif not skip_minimization:
        print('\n*** Minimization ***\n')
        ###################
        # Minimization    #
        ###################

        # Main function
        raw_fun = var.cost_and_grad
        first_cost_eval = True
        cost_eval = 0
        transfer_guard = getattr(config.INV, 'transfer_guard', None)
        profile_dir = getattr(config.INV, 'profile_dir', None)
        profile_cost_eval = getattr(config.INV, 'profile_cost_eval', 2)

        def fun(XX):
            nonlocal first_cost_eval, cost_eval

            cost_eval += 1
            is_first = first_cost_eval
            first_cost_eval = False
            should_profile = profile_dir is not None and cost_eval == profile_cost_eval
            XX_device = jax.device_put(XX)

            with ExitStack() as stack:
                # The first call may still load captured constants while XLA
                # builds the executable. Guard steady-state evaluations, where
                # any transfer indicates a real regression in the hot path.
                if transfer_guard is not None and not is_first:
                    stack.enter_context(jax.transfer_guard(transfer_guard))
                if should_profile:
                    os.makedirs(profile_dir, exist_ok=True)
                    stack.enter_context(jax.profiler.trace(
                        profile_dir,
                        create_perfetto_trace=True,
                    ))

                time0 = time.time()
                J, G = raw_fun(XX_device)
                if is_first or should_profile:
                    _block_until_ready((J, G))

            if is_first:
                print("[cost] first evaluation compile+run time: {:.2f} seconds".format(time.time() - time0))
            if should_profile:
                print(f"[cost] profiled evaluation {cost_eval} in {profile_dir}")

            # Historical SciPy Minimizer boundary: convert the shared device-native
            # evaluation to the host arrays required by scipy.optimize.
            J_host, G_host = jax.device_get((J, G))
            return np.float64(J_host), np.asarray(G_host, dtype=np.float64)

        # Callback function called at every minimization iterations
        def callback(XX):
            if config.INV.save_minimization:
                ds = xr.Dataset({'res':(('x',),XX)})
                ds.to_netcdf(os.path.join(path_save_control_vectors,'X_it.nc'))
                ds.close()
                
        # Minimization options
        options = {}
        if verbose:
            options['disp'] = True
        else:
            options['disp'] = False
        options['maxiter'] = maxiter

        if config.INV.ftol is not None:
            options['ftol'] = config.INV.ftol

        if config.INV.gtol is not None:
            _, g0 = fun(Xopt*0.)
            projg0 = np.max(np.abs(g0))
            options['gtol'] = config.INV.gtol*projg0
        
        # Sustained convergence: override scipy stopping criteria
        convergence_nit = getattr(config.INV, 'convergence_nit', None)
        ftol_threshold = options.get('ftol', None)
        gtol_threshold = options.get('gtol', None)
        if convergence_nit is not None:
            options['ftol'] = 0
            options['gtol'] = 0

        gradient_max_norm = getattr(config.INV, 'gradient_max_norm', None)
        
        # Run minimization 
        from decimal import Decimal
        import time
        class Wrapper:
            def __init__(self):
                J0, G0 = fun(Xopt)
                self.cache = {
                    'cost':J0,
                    'grad':G0
                }
                self.J_list = []
                self.G_list = []
                self.time = time.time()
                self.it = 1
                if 'gtol' in options:
                    self.gtol = options['gtol']
                else:
                    self.gtol = None
                self.filename_out = os.path.join(path_save_control_vectors, 'iterations.txt')
                with open(self.filename_out, "w") as f:
                    f.write("Minimization\n")  # Header
                self.convergence_count = 0
                self.last_x = None
                # For crazy gradient recovery
                self.best_cost = float(J0)
                self.best_x = np.array(Xopt).copy()
                self.best_grad_norm = float(np.max(np.abs(G0)))

            def __call__(self, x, *args):
                cost, grad = fun(x)
                ftol = (cost - self.cache['cost']) / max(cost, self.cache['cost'], 1)
                mean_grad = np.mean(np.abs(grad))
                mean_grad_previous = np.mean(np.abs(self.cache['grad']))
                gtol = (mean_grad - mean_grad_previous) / max(mean_grad, mean_grad_previous, 1)
                self.cache['cost'] = cost
                self.cache['grad'] = grad
                time0 = time.time()
                text = "computed in %.2E second:" % (time0 - self.time) + ', x=%.2E' % Decimal(float(x.mean()))  + ', J=%.2E' % Decimal(float(cost)) + ', G=%.2E' % Decimal(float(mean_grad))  + ', ftol=%.2E' % Decimal(abs(float(ftol)))  + ', gtol=%.2E' % Decimal(abs(float(gtol))) 
                print(f"* iteration {self.it}", text)
                with open(self.filename_out, "a") as f:
                    f.write(f"iteration {self.it}, {text}\n")
                self.time = time0
                self.it += 1

                self.J_list.append(float(cost))
                self.G_list.append(float(mean_grad))

                # Track best state
                grad_norm = np.max(np.abs(grad))
                if gradient_max_norm is not None and np.isfinite(cost) and np.isfinite(grad_norm) and grad_norm < gradient_max_norm:
                    if float(cost) < self.best_cost:
                        self.best_cost = float(cost)
                        self.best_x = np.array(x).copy()
                        self.best_grad_norm = grad_norm

                # Check for crazy gradient
                if gradient_max_norm is not None and ((not np.isfinite(cost)) or (not np.isfinite(grad_norm)) or grad_norm > gradient_max_norm):
                    print(f"\nCrazy gradient detected (cost={cost}, grad_norm={grad_norm}). Will restart from best state.")
                    raise CrazyGradient()

                # Check sustained convergence
                if convergence_nit is not None:
                    converged = False
                    if ftol_threshold is not None and abs(float(ftol)) <= ftol_threshold:
                        converged = True
                    if gtol_threshold is not None and np.max(np.abs(grad)) <= gtol_threshold:
                        converged = True
                    if converged:
                        self.convergence_count += 1
                    else:
                        self.convergence_count = 0
                    self.last_x = np.array(x).copy()
                    if self.convergence_count >= convergence_nit:
                        print(f'\nConvergence criteria met for {convergence_nit} consecutive iterations. Stopping.')
                        raise ConvergenceReached()

                return cost

            def jac(self, x, *args):
                return self.cache['grad']
        
        wrapper = Wrapper()
        max_retries = getattr(config.INV, 'max_retries', 3)
        retry = 0
        while True:
            try:
                res = opt.minimize(wrapper, Xopt,
                                method=config.INV.opt_method,
                                jac=wrapper.jac,
                                options=options,
                                callback=callback)
                print ('\nIs the minimization successful? {}'.format(res.success))
                print ('\nFinal cost function value: {}'.format(res.fun))
                print ('\nNumber of iterations: {}'.format(res.nit))
                Xres = res.x
                break
            except ConvergenceReached:
                print(f'\nNumber of iterations: {wrapper.it - 1}')
                Xres = wrapper.last_x
                break
            except CrazyGradient:
                retry += 1
                if retry > max_retries:
                    print(f"Maximum retries ({max_retries}) reached. Stopping minimization.")
                    Xres = wrapper.best_x
                    break
                print(f"Restarting minimization from best state (retry {retry}/{max_retries})...")
                Xopt = wrapper.best_x
                wrapper = Wrapper()
        
        # Save minimization trajectory
        if config.INV.save_minimization:
            ds = xr.Dataset({'cost':(('i'),np.array(wrapper.J_list)),
                             'grad':(('i'),np.array(wrapper.G_list))
                             })
            ds.to_netcdf(os.path.join(path_save_control_vectors,'minimization_trajectory.nc'))
            ds.close()
    else:
        print('You ask for restart_4Dvar and maxiter==0, so we save directly the trajectory')
        Xres = +Xopt
        
    ########################
    #    Saving trajectory #
    ########################
    print('\n*** Saving trajectory ***\n')
    
    with var.cost_precision():
        if config.INV.prec:
            Xa = var.Xb + B.sqr(Xres)
        else:
            Xa = var.Xb + Xres
    Xa_model = jnp.asarray(
        Xa,
        dtype=jnp.float64 if USE_FLOAT64 else jnp.float32,
    )
        
    # Save minimum for next experiments
    ds = xr.Dataset({'res':(('x',), np.asarray(jax.device_get(Xa)))})
    ds.to_netcdf(os.path.join(path_save_control_vectors, 'Xres.nc'))
    ds.close()

    # Init
    State0 = State.copy()
    Model.init(State0)
    date = config.EXP.init_date
    Model.save_output(State0, date, name_var=Model.var_to_save, t=0) 
    
    nstep = min(nstep_check, int(config.EXP.saveoutput_time_step.total_seconds()//Model.dt))
    # Forward propagation
    while date<config.EXP.final_date:
        
        # current time in secondes
        t = (date - config.EXP.init_date).total_seconds()
        
        # Reduced basis
        if t%int(config.INV.timestep_checkpoint.total_seconds())==0:
            Basis.operg(t/3600/24, Xa_model, State=State0.params)

        # Forward propagation
        Model.step(t=t, State=State0, nstep=nstep)
        date += timedelta(seconds=nstep*Model.dt)

        # Save output
        if (((date - config.EXP.init_date).total_seconds()
            /config.EXP.saveoutput_time_step.total_seconds())%1 == 0)\
            & (date>=config.EXP.init_date) & (date<=config.EXP.final_date) :
            Model.save_output(State0, date, name_var=Model.var_to_save, t=t) 
        
    del State, State0, Xa, dict_obs, B, R, Model, Basis, var, Xopt, Xres, checkpoints, time_checkpoints, t_checkpoints
    gc.collect()
    print()


# =============================================================================
# Merged from tools_4Dvar.py
# =============================================================================



class Cov :
    # case of a simple diagonal covariance matrix
    def __init__(self,sigma=None):
        
        if sigma is None:
            sigma = 1
            
        self.sigma = sigma
        
    def inv(self,X):
        return 1/self.sigma**2 * X    
    
    def sqr(self,X):
        return self.sigma * X
    
    def invsqr(self,X):
        return 1/self.sigma * X
    
    
class Variational:
    
    def __init__(self, 
                 config=None, M=None, H=None, State=None, R=None,B=None, Basis=None, Xb=None, checkpoints=None, nstep=None, freq_it_plot=1, print_time=False):
        
        # Objects
        self.M = M # model
        self.H = H # observational operator
        self.State = State # state variables
    
        # Covariance matrixes
        self.B = B
        self.R = R

        self.jit_cost_and_grad = bool(
            getattr(config.INV, 'jit_cost_and_grad', False)
        )
        self.cost_and_grad_schedule = getattr(
            config.INV,
            'cost_and_grad_schedule',
            'python',
        )
        if self.cost_and_grad_schedule not in ('python', 'scan'):
            raise ValueError(
                "cost_and_grad_schedule must be 'python' or 'scan', got "
                f"{self.cost_and_grad_schedule!r}"
            )
        self.cost_and_grad_scan_unroll = int(
            getattr(config.INV, 'cost_and_grad_scan_unroll', 1)
        )
        if self.cost_and_grad_scan_unroll < 1:
            raise ValueError('cost_and_grad_scan_unroll must be >= 1')
        requested_cost_float64 = bool(
            getattr(config.INV, 'cost_float64', True)
        )
        if self.jit_cost_and_grad and not USE_FLOAT64:
            # Tracing the whole mixed-precision QG graph under an x64 context
            # promotes Python scalar constants inside model kernels. Keep the
            # compiled algebra consistently float32; the SciPy adapter still
            # returns host float64 values for its line search.
            self.cost_dtype = jnp.float32
            if requested_cost_float64:
                print(
                    '[cost] jit_cost_and_grad uses float32 internally because '
                    'USE_FLOAT64=False; SciPy outputs remain float64.'
                )
        else:
            self.cost_dtype = (
                jnp.float64
                if requested_cost_float64
                else (jnp.float64 if USE_FLOAT64 else jnp.float32)
            )
        self.model_control_dtype = jnp.float64 if USE_FLOAT64 else jnp.float32

        with self.cost_precision():
            if self.B is not None:
                self.B.sigma = jnp.asarray(self.B.sigma, dtype=self.cost_dtype)
            if self.R is not None:
                self.R.sigma = jnp.asarray(self.R.sigma, dtype=self.cost_dtype)
        
        # Background state
        with self.cost_precision():
            self.Xb = jnp.asarray(Xb, dtype=self.cost_dtype)
        
        # Temporary path where to save model trajectories
        self.tmp_DA_path = config.EXP.tmp_DA_path

        # checkpoint 
        self.checkpoints = checkpoints
        
        # preconditioning
        self.prec = config.INV.prec
        
        # Wavelet reduced basis
        self.dtbasis = int(config.INV.timestep_checkpoint.total_seconds()//M.dt)
        self.basis = Basis 
        
        # Save cost function and its gradient at each iteration 
        self.save_minimization = config.INV.save_minimization
        if self.save_minimization:
            self.J = []
            self.dJ = [] # For incremental 4Dvar only
            self.G = []
        
        # For incremental 4Dvar only
        self.X0 = self.Xb*0

        self.freq_it_plot = freq_it_plot
        self.it_plot = 0
        self.print_time = print_time
        self.plot_state_during_minimization = getattr(
            config.INV,
            'plot_state_during_minimization',
            False,
        )

        self._scan_has_boundary_conditions = False
        if self.jit_cost_and_grad and self.cost_and_grad_schedule == 'scan':
            self._prepare_scan_schedule()

        if self.jit_cost_and_grad:
            if not getattr(config.INV, 'device_resident_state', False):
                raise ValueError(
                    'jit_cost_and_grad requires device_resident_state=True'
                )
            if self.print_time:
                raise ValueError(
                    'jit_cost_and_grad is incompatible with print_time=True; '
                    'use the outer cost timing/profiler instead'
                )
            if self.plot_state_during_minimization:
                raise ValueError(
                    'jit_cost_and_grad is incompatible with state plotting '
                    'inside the compiled evaluation'
                )
            implementation = (
                self._cost_and_grad_scan_impl
                if self.cost_and_grad_schedule == 'scan'
                else self._cost_and_grad_impl
            )
            self._compiled_cost_and_grad = jax.jit(implementation)
            print(
                '[cost] complete cost-and-gradient schedule: JIT enabled '
                f'({self.cost_and_grad_schedule})'
            )
        else:
            self._compiled_cost_and_grad = None
        
        # Grad test
        if config.INV.compute_test:
            print('Gradient test:')
            np.random.seed(0)
            if self.prec:
                X = (np.random.random(self.basis.nbasis)-0.5)
            else:
                X = self.B.sqr(np.random.random(self.basis.nbasis)-0.5) + self.Xb
            
            def cost(X):
                return self.cost_and_grad(X)[0]
            def grad(X):
                return self.cost_and_grad(X)[1]
            grad_test(cost,grad,X)

    def cost_precision(self):
        """Scope float64 to cost/control algebra, never to model tracing."""
        if self.cost_dtype == jnp.float64 and not USE_FLOAT64:
            return jax.experimental.enable_x64()
        return nullcontext()

    def _prepare_scan_schedule(self):
        """Validate and materialize fixed-shape inputs for checkpoint scans."""
        if not hasattr(self.H, 'scan_misfit') or not hasattr(self.H, 'scan_adj'):
            raise TypeError(
                'cost_and_grad_schedule=scan requires scan_misfit and '
                'scan_adj observation operators'
            )

        interval_lengths = np.diff(self.checkpoints)
        if interval_lengths.size == 0:
            raise ValueError('lax.scan requires at least one checkpoint interval')
        if not np.all(interval_lengths == interval_lengths[0]):
            raise ValueError(
                'cost_and_grad_schedule=scan currently requires uniform '
                f'checkpoint intervals, got {np.unique(interval_lengths)}'
            )

        self._scan_nstep = int(interval_lengths[0])
        scan_times = np.asarray([
            self.M.T[index] for index in self.checkpoints[:-1]
        ])
        self._scan_times = jnp.asarray(scan_times)
        self._scan_final_time = jnp.asarray(
            self.M.T[self.checkpoints[-1]]
        )
        self._scan_basis_active = jnp.asarray(
            self.checkpoints[:-1] % self.dtbasis == 0
        )

        if hasattr(self.M, 'prepare_scan_boundary_conditions'):
            self._scan_boundary_conditions = (
                self.M.prepare_scan_boundary_conditions(
                    scan_times,
                    self._scan_nstep,
                )
            )
            self._scan_has_boundary_conditions = True
        else:
            self._scan_boundary_conditions = jnp.zeros(
                (scan_times.size, 0),
                dtype=self.model_control_dtype,
            )

        # Force validation while shapes and concrete observation arrays are
        # still available, before the first (potentially expensive) trace.
        self.H.scan_misfit_size()
        print(
            '[cost] lax.scan checkpoint schedule: '
            f'{scan_times.size} uniform intervals, nstep={self._scan_nstep}, '
            f'unroll={self.cost_and_grad_scan_unroll}'
        )

        
    def cost(self,X0):
                
        # Initial state
        State = self.State.copy()
        #State.plot(title='State variables at the start of cost function evaluation')
        # Background cost function evaluation 
        if self.B is not None:
            if self.prec :
                X  = self.B.sqr(X0) + self.Xb
                Jb = X0.dot(X0) # cost of background term
            else:
                X  = X0 + self.Xb
                Jb = np.dot(X0,self.B.inv(X0)) # cost of background term
        else:
            X  = X0 - self.Xb
            Jb = 0
    
        # Observational cost function evaluation
        Jo = 0.

        time_misfit = 0
        l = 0
        time_model = 0
        j = 0
        time_basis = 0
        k = 0
        
        for i in range(len(self.checkpoints)-1):
            
            timestamp = self.M.timestamps[self.checkpoints[i]]
            t = self.M.T[self.checkpoints[i]]
            nstep = self.checkpoints[i+1] - self.checkpoints[i]
            
            # 1. Misfit
            if self.H.is_obs(timestamp):
                start = time.time()
                misfit = self.H.misfit(timestamp,State) # d=Hx-xobs   
                end = time.time()
                time_misfit += end - start
                l += 1
                # Accumulate Jo in float64: model misfit may be float32 (mixed precision)
                _m = np.asarray(misfit, dtype=np.float64)
                Jo += _m.dot(np.asarray(self.R.inv(misfit), dtype=np.float64))
            
            # 2. Reduced basis
            if self.checkpoints[i]%self.dtbasis==0:
                start = time.time()
                self.basis.operg(t/3600/24, X, State=State.params)
                end = time.time()
                time_basis += end - start
                k += 1
            
            State.save(os.path.join(self.tmp_DA_path,
                        'model_state_' + str(self.checkpoints[i]) + '.nc'))

            # 3. Run forward model
            start = time.time()
            self.M.step(t=t,State=State,nstep=nstep)
            end = time.time()
            time_model += end - start
            j += 1

            if i==int(len(self.checkpoints)/2):
                State.plot(title='State variables at the middle of cost function evaluation')

        timestamp = self.M.timestamps[self.checkpoints[-1]]
        if self.H.is_obs(timestamp):
            start = time.time()
            misfit = self.H.misfit(timestamp,State) # d=Hx-xobsx
            time_misfit += end - start
            l += 1
            _m = np.asarray(misfit, dtype=np.float64)
            Jo += _m.dot(np.asarray(self.R.inv(misfit), dtype=np.float64))
        
        print('misfit', l, time_misfit/l)
        print('basis', k, time_basis/k)
        print('model', j, time_model/j)
        # Cost function (float64 for L-BFGS-B line-search stability)
        J = np.float64(0.5 * (Jo + Jb))
        
        if self.save_minimization:
            self.J.append(J)

        return J
    
    def grad(self,X0): 
                
        X = +X0 
        
        
        if self.B is not None:
            if self.prec :
                X  = self.B.sqr(X0) + self.Xb
                gb = X0      # gradient of background term
            else:
                X  = X0 + self.Xb
                gb = self.B.inv(X0) # gradient of background term
        else:
            X  = X0 + self.Xb
            gb = 0
            
        # Current trajectory
        State = self.State.copy()
        
        # Ajoint initialization   
        adState = self.State.copy(free=True)
        adX = X*0

        # Last timestamp
        timestamp = self.M.timestamps[self.checkpoints[-1]]
        if self.H.is_obs(timestamp):
            self.H.adj(timestamp,adState,self.R)

        # Time loop
        for i in reversed(range(0,len(self.checkpoints)-1)):
            
            nstep = self.checkpoints[i+1] - self.checkpoints[i]
            timestamp = self.M.timestamps[self.checkpoints[i]]
            t = self.M.T[self.checkpoints[i]]
            
            # Read model state
            State.load(os.path.join(self.tmp_DA_path,
                       'model_state_' + str(self.checkpoints[i]) + '.nc'))
            
            # 3. Run adjoint model 
            self.M.step_adj(t=t, adState=adState, State=State, nstep=nstep) # i+1 --> i
            
            # 2. Reduced basis
            if self.checkpoints[i]%self.dtbasis==0:
                adX += self.basis.operg_transpose(t=t/3600/24,adState=adState.params)
            
            # 1. Misfit 
            if self.H.is_obs(timestamp):
                self.H.adj(timestamp,adState,self.R)

        if self.prec :
            adX = np.transpose(self.B.sqr(adX)) 
        
        g = adX + gb  # total gradient

        #adState.plot(title='adjoint variables at the end of gradient function evaluation')
        #State.plot(title='adjoint parameters at the end of gradient function evaluation',params=True)
        
        # Cast to float64 for L-BFGS-B line-search stability
        g = np.asarray(g, dtype=np.float64)

        if self.save_minimization:
            self.G.append(np.max(np.abs(g)))

        return g  

    def cost_and_grad(self, X0):
        """Evaluate through either the compiled or legacy adapter."""
        if self._compiled_cost_and_grad is not None:
            X0 = jnp.asarray(X0, dtype=self.cost_dtype)
            J, G = self._compiled_cost_and_grad(X0)
        else:
            J, G = self._cost_and_grad_impl(X0)

        self.it_plot += 1
        return J, G

    def _state_from_scan_trees(self, var, params):
        """Build a trace-time State facade around device pytree leaves."""
        state = self.State.copy()
        state.var = dict(var)
        state.params = dict(params)
        return state

    def _cost_and_grad_scan_impl(self, X0):
        """Cost and explicit adjoint with rolled checkpoint loops.

        The historical implementation below deliberately remains available as
        ``cost_and_grad_schedule='python'``. This variant carries plain pytrees
        through two ``lax.scan`` primitives so the monthly schedule is not
        duplicated in StableHLO.
        """
        with self.cost_precision():
            X0 = jnp.asarray(X0, dtype=self.cost_dtype)
            if self.B is not None:
                if self.prec:
                    X = self.B.sqr(X0) + self.Xb
                    Jb = jnp.vdot(X0, X0)
                else:
                    X = X0 + self.Xb
                    Jb = jnp.vdot(X0, self.B.inv(X0))
            else:
                X = X0 - self.Xb
                Jb = jnp.asarray(0., dtype=self.cost_dtype)
            Jo0 = jnp.asarray(0., dtype=self.cost_dtype)

        X_model = jnp.asarray(X, dtype=self.model_control_dtype)
        initial_state = self.State.copy()
        initial_var = dict(initial_state.var)
        initial_params = dict(initial_state.params)

        def project_basis(t, params):
            projected = dict(params)
            self.basis.operg(t / (3600 * 24), X_model, State=projected)
            return projected

        def forward_body(carry, inputs):
            state_var, state_params, Jo = carry
            t, basis_active, boundary = inputs
            state = self._state_from_scan_trees(state_var, state_params)

            misfit = self.H.scan_misfit(t, state.var)
            with self.cost_precision():
                weighted = jnp.asarray(misfit, dtype=self.cost_dtype)
                Jo = Jo + jnp.vdot(
                    weighted,
                    jnp.asarray(self.R.inv(weighted), dtype=self.cost_dtype),
                )

            state.params = lax.cond(
                basis_active,
                lambda params: project_basis(t, params),
                lambda params: params,
                state.params,
            )
            trajectory_var = dict(state.var)
            trajectory_params = dict(state.params)

            if self._scan_has_boundary_conditions:
                self.M.step(
                    t=t,
                    State=state,
                    nstep=self._scan_nstep,
                    Xb=boundary,
                )
            else:
                self.M.step(
                    t=t,
                    State=state,
                    nstep=self._scan_nstep,
                )

            trajectory = (
                trajectory_var,
                trajectory_params,
                misfit,
            )
            return (dict(state.var), dict(state.params), Jo), trajectory

        scan_inputs = (
            self._scan_times,
            self._scan_basis_active,
            self._scan_boundary_conditions,
        )
        (final_var, final_params, Jo), trajectory = lax.scan(
            forward_body,
            (initial_var, initial_params, Jo0),
            scan_inputs,
            unroll=self.cost_and_grad_scan_unroll,
        )
        trajectory_var, trajectory_params, trajectory_misfit = trajectory

        final_misfit = self.H.scan_misfit(
            self._scan_final_time,
            final_var,
        )
        with self.cost_precision():
            final_weighted = jnp.asarray(
                final_misfit,
                dtype=self.cost_dtype,
            )
            Jo = Jo + jnp.vdot(
                final_weighted,
                jnp.asarray(
                    self.R.inv(final_weighted),
                    dtype=self.cost_dtype,
                ),
            )
            J = jnp.asarray(0.5 * (Jo + Jb), dtype=self.cost_dtype)

        if self.B is not None:
            gb = X0 if self.prec else self.B.inv(X0)
        else:
            gb = 0

        adjoint_state = self.State.copy(free=True)
        adjoint_var = self.H.scan_adj(
            self._scan_final_time,
            dict(adjoint_state.var),
            final_var,
            final_misfit,
        )
        adjoint_params = dict(adjoint_state.params)
        adX0 = jnp.zeros_like(X_model)

        def transpose_basis(t, operands):
            ad_params, ad_control = operands
            projected_params = dict(ad_params)
            increment = self.basis.operg_transpose(
                t=t / (3600 * 24),
                adState=projected_params,
            )
            return projected_params, ad_control + increment

        def backward_body(carry, inputs):
            ad_var, ad_params, ad_control = carry
            t, basis_active, boundary, state_var, state_params, misfit = inputs
            state = self._state_from_scan_trees(state_var, state_params)
            adjoint = self._state_from_scan_trees(ad_var, ad_params)

            if self._scan_has_boundary_conditions:
                self.M.step_adj(
                    t=t,
                    adState=adjoint,
                    State=state,
                    nstep=self._scan_nstep,
                    Xb=boundary,
                )
            else:
                self.M.step_adj(
                    t=t,
                    adState=adjoint,
                    State=state,
                    nstep=self._scan_nstep,
                )

            ad_params, ad_control = lax.cond(
                basis_active,
                lambda operands: transpose_basis(t, operands),
                lambda operands: operands,
                (dict(adjoint.params), ad_control),
            )
            ad_var = self.H.scan_adj(
                t,
                dict(adjoint.var),
                state.var,
                misfit,
            )
            return (ad_var, ad_params, ad_control), None

        reverse_inputs = (
            self._scan_times,
            self._scan_basis_active,
            self._scan_boundary_conditions,
            trajectory_var,
            trajectory_params,
            trajectory_misfit,
        )
        (adjoint_var, adjoint_params, adX), _ = lax.scan(
            backward_body,
            (adjoint_var, adjoint_params, adX0),
            reverse_inputs,
            reverse=True,
            unroll=self.cost_and_grad_scan_unroll,
        )

        with self.cost_precision():
            adX = jnp.asarray(adX, dtype=self.cost_dtype)
            if self.prec:
                adX = jnp.transpose(self.B.sqr(adX))
            G = jnp.asarray(adX + gb, dtype=self.cost_dtype)

        return J, G

    def _cost_and_grad_impl(self, X0):
         
        ########################################
        # COST FUNCTION
        ########################################

        with self.cost_precision():
            X0 = jnp.asarray(X0, dtype=self.cost_dtype)

        # Initial state
        State = self.State.copy()

        # Background cost function
        with self.cost_precision():
            if self.B is not None:
                if self.prec :
                    X  = self.B.sqr(X0) + self.Xb
                    Jb = jnp.vdot(X0, X0) # cost of background term
                else:
                    X  = X0 + self.Xb
                    Jb = jnp.vdot(X0,self.B.inv(X0)) # cost of background term
            else:
                X  = X0 - self.Xb
                Jb = jnp.asarray(0., dtype=self.cost_dtype)
        X_model = jnp.asarray(X, dtype=self.model_control_dtype)
        
        cost_misfit = []
        cost_basis = []
        cost_model = []
    
        # Observational cost function evaluation
        State_dict = {}
        misfit_dict = {}
        with self.cost_precision():
            Jo = jnp.asarray(0., dtype=self.cost_dtype)

        # Under jit_cost_and_grad this heterogeneous checkpoint schedule runs
        # only while tracing. XLA then executes one device program per cost
        # evaluation; each QG interval already uses lax.scan internally.
        for i in range(len(self.checkpoints)-1):
            
            t = self.M.T[self.checkpoints[i]]
            nstep = self.checkpoints[i+1] - self.checkpoints[i]
            
            # 1. Misfit
            if self.H.is_obs_time(t):
                if self.print_time:
                    time0 = time.time()
                misfit = self.H.misfit(t,State) # d=Hx-xobs   
                if self.print_time:
                    _block_until_ready(misfit)
                misfit_dict[t] = misfit
                # Accumulate Jo in float64 (model misfit may be float32)
                with self.cost_precision():
                    _m = jnp.asarray(misfit, dtype=self.cost_dtype)
                    Jo = Jo + jnp.vdot(
                        _m,
                        jnp.asarray(self.R.inv(_m), dtype=self.cost_dtype),
                    )
                if self.print_time:
                    cost_misfit.append(time.time()-time0)
            
            # 2. Reduced basis
            if self.checkpoints[i]%self.dtbasis==0:
                if self.print_time:
                    time0 = time.time()
                self.basis.operg(t/3600/24, X_model, State=State.params)
                if self.print_time:
                    _block_until_ready(State.params)
                    cost_basis.append(time.time()-time0)
            
            State_dict[t] = State.copy()

            # 3. Run forward model
            if self.print_time:
                    time0 = time.time()
            self.M.step(t=t,State=State,nstep=nstep)
            if self.print_time:
                    _block_until_ready(State.var)
                    cost_model.append(time.time()-time0)

            if self.plot_state_during_minimization and i==int(len(self.checkpoints)/2):
                if self.it_plot % self.freq_it_plot == 0:
                    State.plot(title='State variables at the middle of cost function evaluation', name_save=f'state_cost_it{self.it_plot}')
                    State.plot(title='State params at the middle of cost function evaluation', params=True, name_save=f'state_params_cost_it{self.it_plot}')

        t = self.M.T[-1]
        State_dict[t] = State.copy()
        if self.H.is_obs_time(t):
            misfit = self.H.misfit(t,State) # d=Hx-xobsx
            if self.print_time:
                _block_until_ready(misfit)
            misfit_dict[t] = misfit
            with self.cost_precision():
                _m = jnp.asarray(misfit, dtype=self.cost_dtype)
                Jo = Jo + jnp.vdot(
                    _m,
                    jnp.asarray(self.R.inv(_m), dtype=self.cost_dtype),
                )
        
        # Cost function (float64 for L-BFGS-B line-search stability)
        with self.cost_precision():
            J = jnp.asarray(0.5 * (Jo + Jb), dtype=self.cost_dtype)


        ########################################
        # GRAD FUNCTION
        ########################################

        # Gradient of the background term
        if self.B is not None:
            if self.prec :
                gb = X0      # gradient of background term
            else:
                gb = self.B.inv(X0) # gradient of background term
        else:
            gb = 0

        # Ajoint initialization   
        adState = self.State.copy(free=True)
        adX = jnp.zeros_like(X_model)

        # Last timestamp
        t = self.M.T[self.checkpoints[-1]]
        if self.H.is_obs_time(t):
            self.H.adj(t, adState, State_dict[t], misfit_dict[t])
            if self.print_time:
                _block_until_ready((adState.var, adState.params))
    
        grad_misfit = []
        grad_basis = []
        grad_model = []

        # Time loop
        # This reverse schedule is likewise trace-time construction, not a
        # Python loop in steady-state minimizer iterations.
        for i in reversed(range(0,len(self.checkpoints)-1)):

            nstep = self.checkpoints[i+1] - self.checkpoints[i]
            t = self.M.T[self.checkpoints[i]]

            # 3. Run adjoint model 
            if self.print_time:
                time0 = time.time()
            self.M.step_adj(t=t, adState=adState, State=State_dict[t], nstep=nstep) # i+1 --> i
            if self.print_time:
                _block_until_ready((adState.var, adState.params))
                grad_model.append(time.time()-time0)

            # 2. Reduced basis
            if self.checkpoints[i]%self.dtbasis==0:
                if self.print_time:
                    time0 = time.time()
                _adX = self.basis.operg_transpose(t=t/3600/24,adState=adState.params)
                adX += _adX
                if self.print_time:
                    _block_until_ready(adX)
                    grad_basis.append(time.time()-time0)
            
            # 1. Misfit 
            if self.H.is_obs_time(t):
                if self.print_time:
                    time0 = time.time()
                self.H.adj(t,adState,State_dict[t],misfit_dict[t])
                if self.print_time:
                    _block_until_ready((adState.var, adState.params))
                    grad_misfit.append(time.time()-time0)

            if self.plot_state_during_minimization and i==int(len(self.checkpoints)/2):
                if self.it_plot % self.freq_it_plot == 0:
                    adState.plot(title='Adjoint State variables at the middle of cost function evaluation', name_save=f'adjoint_state_grad_it{self.it_plot}')
            
    
        if self.print_time:
            def _time_stats(name, values):
                values = np.asarray(values, dtype=float)
                if values.size == 0:
                    return f"{name}: n=0"
                return (
                    f"{name}: mean={values.mean():.2e}, "
                    f"total={values.sum():.2e}, n={values.size}"
                )

            print("[cost] computation time [seconds]: " + ", ".join([
                _time_stats('misfit', cost_misfit),
                _time_stats('basis', cost_basis),
                _time_stats('model', cost_model),
            ]))
            print("[grad] computation time [seconds]: " + ", ".join([
                _time_stats('misfit', grad_misfit),
                _time_stats('basis', grad_basis),
                _time_stats('model', grad_model),
            ]))

        with self.cost_precision():
            adX = jnp.asarray(adX, dtype=self.cost_dtype)
            if self.prec :
                adX = jnp.transpose(self.B.sqr(adX))

            G = jnp.asarray(adX + gb, dtype=self.cost_dtype)

        return J, G  
   
def grad_test(J, G, X):
    np.random.seed(0)
    h = np.random.random(X.size)
    h /= np.linalg.norm(h)
    JX = J(X)
    GX = G(X)
    Gh = h.dot(np.where(np.isnan(GX),0,GX))
    for p in range(10):
        lambd = 10**(-p)
        test = np.abs(1. - (J(X+lambd*h) - JX)/(lambd*Gh))
        print(f'{lambd:.1E} , {test:.2E}')

def plot_grad_test(L) :
    '''
    plots the result of a gradient test, L is a list containing
    the test results
    '''
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    ax.plot(L[0],L[1],'o','red')
    ax.plot(L[0],L[1],'orange')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('gradient test')
    ax.set_xlabel('order')
    ax.invert_xaxis()
    plt.show()
