#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Florian Le Guillou on June 2026.

Runs inversion drivers and assimilation orchestration.
"""
from .config import USE_FLOAT64
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
        Xopt = var.Xb*0
        Xopt[:ds.res.size] = ds.res.values
        ds.close()
        if config.INV.prec:
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

    if not ((config.INV.restart_4Dvar or config.INV.path_init_4Dvar is not None) and maxiter==0):
        print('\n*** Minimization ***\n')
        ###################
        # Minimization    #
        ###################

        # Main function
        raw_fun = var.cost_and_grad
        first_cost_eval = True

        def fun(XX):
            nonlocal first_cost_eval

            if not first_cost_eval:
                return raw_fun(XX)

            first_cost_eval = False
            time0 = time.time()
            J, G = raw_fun(XX)
            _block_until_ready((J, G))
            print("[cost] first evaluation compile+run time: {:.2f} seconds".format(time.time() - time0))
            return J, G

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
    
    if config.INV.prec:
        Xa = var.Xb + B.sqr(Xres)
    else:
        Xa = var.Xb + Xres
        
    # Save minimum for next experiments
    ds = xr.Dataset({'res':(('x',), Xa)})
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
            Basis.operg(t/3600/24, Xa, State=State0.params)

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
        
        # Background state
        self.Xb = Xb
        
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
         
        ########################################
        # COST FUNCTION
        ########################################

        # Initial state
        State = self.State.copy()

        # Background cost function
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
        
        cost_misfit = []
        cost_basis = []
        cost_model = []
    
        # Observational cost function evaluation
        State_dict = {}
        misfit_dict = {}
        Jo = 0.

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
                _m = np.asarray(misfit, dtype=np.float64)
                Jo += _m.dot(np.asarray(self.R.inv(misfit), dtype=np.float64))
                if self.print_time:
                    cost_misfit.append(time.time()-time0)
            
            # 2. Reduced basis
            if self.checkpoints[i]%self.dtbasis==0:
                if self.print_time:
                    time0 = time.time()
                self.basis.operg(t/3600/24, X, State=State.params)
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

            if i==int(len(self.checkpoints)/2):
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
            _m = np.asarray(misfit, dtype=np.float64)
            Jo += _m.dot(np.asarray(self.R.inv(misfit), dtype=np.float64))
        
        # Cost function (float64 for L-BFGS-B line-search stability)
        J = np.float64(0.5 * (Jo + Jb))


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
        adX = X*0

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

            if i==int(len(self.checkpoints)/2):
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

        self.it_plot += 1

        if self.prec :
            adX = np.transpose(self.B.sqr(adX)) 

        G = adX + gb  # total gradient

        # Cast to float64 for L-BFGS-B line-search stability
        G = np.asarray(G, dtype=np.float64)

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


        
        
        
        
        
        
        
        