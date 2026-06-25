"""
Created by Florian Le Guillou on June 2026.

Implements the JSWM shallow-water model with boundary handling.
"""

import sys 
sys.path.insert(0, '../../src') # add src to path to import modules
from src.config import USE_FLOAT64

import numpy as np 
import jax.numpy as jnp 
from jax import jit
from jax import jvp,vjp
import jax
from jax import checkpoint as jax_checkpoint
from jax.lax import scan, dynamic_index_in_dim
from functools import partial

import matplotlib.pylab as plt

jax.config.update("jax_enable_x64", USE_FLOAT64)
    
class CSWm: 
    
    ###########################################################################
    #                             Initialization                              #
    ###########################################################################
    
    def __init__(self,X=None,Y=None,dt=None,time_scheme='rk4',g=9.81,f=1e-4,Heb=0.7,periodic_x=False, periodic_y=False, omegas=None, bc_theta=None, **arr_kwargs):
        
        self.X = X
        self.Y = Y
        self.Xu = self.rho_on_u(X)
        self.Yu = self.rho_on_u(Y)
        self.Xv = self.rho_on_v(X)
        self.Yv = self.rho_on_v(Y)
        self.dt = dt
        self.g = g
        if hasattr(f, "__len__") and f.shape==self.X.shape:
            self.f = f
        else: 
            self.f = f * jnp.ones_like(self.X)
        self.f_on_u = self.rho_on_u(self.f)
        self.f_on_v = self.rho_on_v(self.f)

        self.DX = self.X[:,1:]-self.X[:,:-1]
        self.DY = self.Y[1:,:]-self.Y[:-1,:]
        self.DXu = self.Xu[:,1:]-self.Xu[:,:-1]
        self.DYv = self.Yv[1:,:]-self.Yv[:-1,:]
        
        if hasattr(Heb, "__len__") and Heb.shape==self.X.shape:
            self.Heb = Heb
        else: 
            self.Heb = Heb * jnp.ones_like(self.X)
        
        self.ny,self.nx = self.X.shape
                
        self.nu = self.Xu.size
        self.nv = self.Xv.size
        self.nh = self.X.size
        self.nstates = self.nu + self.nv + self.nh
        self.nHe = self.nh
        self.nBc = 2*(self.ny + self.nx)
        self.nparams = self.nHe + self.nBc
        
        self.sliceu = slice(0,
                            self.nu)
        self.slicev = slice(self.nu,
                            self.nu+self.nv)
        self.sliceh = slice(self.nu+self.nv,
                            self.nu+self.nv+self.nh)
        self.sliceHe = slice(self.nu+self.nv+self.nh,
                             self.nu+self.nv+self.nh+self.nHe)
        self.sliceBc = slice(self.nu+self.nv+self.nh+self.nHe,
                             self.nu+self.nv+self.nh+self.nHe+self.nBc)
                             
        
        self.shapeu = self.Xu.shape
        self.shapev = self.Xv.shape
        self.shapeh = self.X.shape
        self.shapeHe = self.X.shape
        
        # Periodic Boundary Conditions
        self.periodic_x = periodic_x
        self.periodic_y = periodic_y

        # Wave parameters for IT boundary conditions
        self.omegas = np.asarray(omegas) if omegas is not None else np.array([])
        self.bc_theta = np.asarray(bc_theta) if bc_theta is not None else np.array([0])

        # Warn if Ntheta is below the boundary tangential Nyquist minimum.
        # Criterion: Ntheta >= L_bdy / lambda_min, where L_bdy = max boundary length and
        # lambda_min = 2*pi*c_min/omega_max.  This ensures all entering wave directions
        # (up to grazing incidence) are represented without aliasing along the boundary.
        if self.omegas.size > 0:
            _He_bdy = np.concatenate([
                np.asarray(self.Heb[0,:]), np.asarray(self.Heb[-1,:]),
                np.asarray(self.Heb[:,0]), np.asarray(self.Heb[:,-1])])
            _He_pos = _He_bdy[_He_bdy > 0]
            if _He_pos.size > 0:
                _c_min = np.sqrt(self.g * np.nanmin(_He_pos))
                _lambda_min = 2*np.pi * _c_min / np.max(np.abs(self.omegas))
                _L_bdy = max(
                    float(np.asarray(self.X).max() - np.asarray(self.X).min()),
                    float(np.asarray(self.Y).max() - np.asarray(self.Y).min()))
                _ntheta_min = int(np.ceil(_L_bdy / _lambda_min))
                _ntheta_actual = (len(self.bc_theta) - 1) // 2
                if _ntheta_actual < _ntheta_min:
                    import warnings
                    warnings.warn(
                        f'CSWm: Ntheta={_ntheta_actual} is below the boundary Nyquist minimum '
                        f'Ntheta_min={_ntheta_min} '
                        f'(lambda_min={_lambda_min/1e3:.1f} km, L_bdy={_L_bdy/1e3:.1f} km). '
                        f'Aliasing in along-boundary wave angle. '
                        f'Consider increasing Ntheta or setting Ntheta=-1 for auto.',
                        stacklevel=2)

        # Sponge BC (attributes set externally by wrapper when used)
        self.flag_sponge_bc = False
        self.sponge_coef = 0.

        # IT phase method for sponge boundary conditions
        # 'plane_wave'     : local k(x,y) * coords  (original, phase inconsistent with varying He)
        # 'plane_wave_bdy' : k from boundary row/col -> true 1D plane wave (recommended)
        # 'wkb'            : cumulative-path-integral phase + He^{-1/4} amplitude correction
        self.bc_it_method = arr_kwargs.get('bc_it_method', 'plane_wave_bdy')
        self.bc_it_corner_weight_power = float(arr_kwargs.get('bc_it_corner_weight_power', 1.0))
        
        # JAX compiling — always-needed utilities
        self.u_on_v_jit = jit(self.u_on_v)
        self.v_on_u_jit = jit(self.v_on_u)
        self.rhs_u_jit = jit(self.rhs_u)
        self.rhs_v_jit = jit(self.rhs_v)
        self.rhs_h_jit = jit(self.rhs_h)

        # Only JIT the active time scheme to avoid unnecessary compile overhead.
        # Leapfrog and the unused scheme are still available as plain methods;
        # call jit() explicitly if needed.
        self.step_leapfrog_jit = jit(self.step_leapfrog)
        if time_scheme == 'rk4':
            self.step_rk4_jit   = jit(self.step_rk4)
            self.step_rk4_tgl_jit = jit(self.step_rk4_tgl)
            self.step_rk4_adj_jit = jit(self.step_rk4_adj)
            self.step_rk4_nstep = jit(self._step_rk4_nstep, static_argnames=['nstep'])
            self.step          = self.step_rk4_jit
            self.step_tgl      = self.step_rk4_tgl_jit
            self.step_adj      = self.step_rk4_adj_jit
            self.step_nstep    = self.step_rk4_nstep
        elif time_scheme == 'rk3':
            self.step_rk3_jit   = jit(self.step_rk3)
            self.step_rk3_tgl_jit = jit(self.step_rk3_tgl)
            self.step_rk3_adj_jit = jit(self.step_rk3_adj)
            self.step_rk3_nstep = jit(self._step_rk3_nstep, static_argnames=['nstep'])
            self.step          = self.step_rk3_jit
            self.step_tgl      = self.step_rk3_tgl_jit
            self.step_adj      = self.step_rk3_adj_jit
            self.step_nstep    = self.step_rk3_nstep
        elif time_scheme == 'Euler':
            self.step_euler_jit   = jit(self.step_euler)
            self.step_euler_tgl_jit = jit(self.step_euler_tgl)
            self.step_euler_adj_jit = jit(self.step_euler_adj)
            self.step_euler_nstep = jit(self._step_euler_nstep, static_argnames=['nstep'])
            self.step          = self.step_euler_jit
            self.step_tgl      = self.step_euler_tgl_jit
            self.step_adj      = self.step_euler_adj_jit
            self.step_nstep    = self.step_euler_nstep
        else:
            raise ValueError(f"Unsupported time_scheme: {time_scheme}")

    ###########################################################################
    #                           Spatial scheme                                #
    ###########################################################################
    
    def rho_on_u(self,rho):
        
        return (rho[:,1:] + rho[:,:-1])/2 
    
    def rho_on_v(self,rho):
        
        return (rho[1:,:] + rho[:-1,:])/2 
    
    def u_on_rho(self,u):
        
        um = 0.5 * (u[:,1:] + u[:,:-1]) #(ny,nx-2)
        um = jnp.pad(um, ((0,0),(1,1)), mode='edge')  # (ny,nx)
        
        return um
    
    def v_on_rho(self,v):
        
        vm = 0.5 * (v[1:,:] + v[:-1,:]) #(ny-2,nx)
        vm = jnp.pad(vm, ((1,1),(0,0)), mode='edge')  # (ny,nx)
        
        return vm
    
    def u_on_v(self,u):
        
        um = 0.25 * (u[1:,:-1] + u[1:,1:] + u[:-1,:-1] + u[:-1,1:])
        
        return um

    
    def v_on_u(self,v):
        
        vm = 0.25 * (v[:-1,1:] + v[:-1,:-1] + v[1:,1:] + v[1:,:-1])
        
        return vm
    
    def adv(self, up, vp, um, vm, q0):

        """
            3rd-order upwind scheme
        """

        dx = self.Xu[1:-1,1:] - self.Xu[1:-1,:-1] # shape (ny-2, nx-2)
        dy = self.Yv[1:,1:-1] - self.Yv[:-1,1:-1] # shape (ny-2, nx-2)

        _adv =  + up[2:-2,2:-2] * 1 / (6 * dx[1:-1,1:-1]) * (2 * q0[2:-2, 3:-1] + 3 * q0[2:-2, 2:-2] - 6 * q0[2:-2, 1:-3] + q0[2:-2, :-4]) \
                - um[2:-2,2:-2] * 1 / (6 * dx[1:-1,1:-1]) * (q0[2:-2, 4:] - 6 * q0[2:-2, 3:-1] + 3 * q0[2:-2, 2:-2] + 2 * q0[2:-2, 1:-3]) \
                + vp[2:-2,2:-2] * 1 / (6 * dy[1:-1,1:-1]) * (2 * q0[3:-1, 2:-2] + 3 * q0[2:-2, 2:-2] - 6 * q0[1:-3, 2:-2] + q0[:-4, 2:-2]) \
                - vm[2:-2,2:-2] * 1 / (6 * dy[1:-1,1:-1]) * (q0[4:, 2:-2] - 6 * q0[3:-1, 2:-2] + 3 * q0[2:-2, 2:-2] + 2 * q0[1:-3, 2:-2])
    

        return _adv 
    

    ###########################################################################
    #                          Right hand sides                               #
    ###########################################################################

    def rhs_u(self,u,v,h, u11u=None, v11u=None, u11z=None, v11z=None):
        
        rhs_u = jnp.zeros_like(u)
        
        # --- Pressure gradient + Coriolis ---
        rhs_u = rhs_u.at[1:-1,:].set(
            self.f_on_u[1:-1,:] * self.v_on_u(v) -\
            self.g * (h[1:-1,1:] - h[1:-1,:-1]) / self.DX[1:-1,:]
            )

        # -----------------------------------------
        # Mean-flow advection (u11u, v11u)
        # -----------------------------------------
        if u11u is not None and v11u is not None:
            # Shape u = (ny, nx-1)
            u_on_T = self.u_on_rho(u) # (ny, nx)
            
            # --- split velocities into positive and negative parts ---
            up = jnp.where(u11u < 0, 0, u11u) # (ny, nx)
            um = jnp.where(u11u > 0, 0, u11u)
            vp = jnp.where(v11u < 0, 0, v11u)
            vm = jnp.where(v11u > 0, 0, v11u)

            # --- advection term on T points ---
            adv_term_on_T = self.adv(up, vp, um, vm, u_on_T) # shape (ny-4, nx-4)

            # --- interpolate advection term back to u points ---
            adv_term_on_u = self.rho_on_u(adv_term_on_T) # shape (ny-4, nx-5)

            # --- add advection term ---
            rhs_u = rhs_u.at[2:-2,2:-2].set(rhs_u[2:-2,2:-2] - adv_term_on_u) 
        
        # -----------------------------------------
        # Mean-flow vertical shear (u11z, v11z)
        # -----------------------------------------
        if u11z is not None and v11z is not None:
            div = (u[1:-1,1:] - u[1:-1,:-1]) / self.DXu[1:-1,:] + \
                  (v[1:,1:-1] - v[:-1,1:-1]) / self.DYv[:,1:-1]
            shear_term_on_rho = u11z[1:-1,1:-1] * div # shape (ny-2, nx-2)
            shear_term_on_u = self.rho_on_u(shear_term_on_rho) # shape (ny-2, nx-3)
            rhs_u = rhs_u.at[1:-1,1:-1].set(rhs_u[1:-1,1:-1] + shear_term_on_u)
            
        return rhs_u

    def rhs_v(self,u,v,h, u11u=None, v11u=None, u11z=None, v11z=None):
        
        rhs_v = jnp.zeros_like(v)
        
        # --- Pressure gradient + Coriolis ---
        rhs_v = rhs_v.at[:,1:-1].set(
            -self.f_on_v[:,1:-1] * self.u_on_v(u) -\
            self.g * (h[1:,1:-1] - h[:-1,1:-1]) / self.DY[:,1:-1]
            )

        # -----------------------------------------
        # Mean-flow advection (u11u, v11u)
        # -----------------------------------------
        if u11u is not None and v11u is not None:

            # Shape v = (ny-1, nx)
            v_on_T = self.v_on_rho(v) # (ny, nx)
            
            # --- split velocities into positive and negative parts ---
            up = jnp.where(u11u < 0, 0, u11u) # (ny, nx) 
            um = jnp.where(u11u > 0, 0, u11u)
            vp = jnp.where(v11u < 0, 0, v11u)
            vm = jnp.where(v11u > 0, 0, v11u)

            # --- advection term on T points ---
            adv_term_on_T = self.adv(up, vp, um, vm, v_on_T) # shape (ny-4, nx-4)

            # --- interpolate advection term back to u points ---
            adv_term_on_v = self.rho_on_v(adv_term_on_T) # shape (ny-5, nx-4)

            # --- add advection term ---
            rhs_v = rhs_v.at[2:-2,2:-2].set(rhs_v[2:-2,2:-2] - adv_term_on_v) 
        
        # -----------------------------------------
        # Mean-flow vertical shear (u11z, v11z)
        # -----------------------------------------
        if u11z is not None and v11z is not None:
            div = (u[1:-1,1:] - u[1:-1,:-1]) / self.DXu[1:-1,:] + \
                (v[1:,1:-1] - v[:-1,1:-1]) / self.DYv[:,1:-1]
            shear_term_on_T = v11z[1:-1,1:-1] * div # shape (ny-2, nx-2)
            shear_term_on_v = self.rho_on_v(shear_term_on_T) # shape (ny-3, nx-2)
            rhs_v = rhs_v.at[1:-1,1:-1].set(rhs_v[1:-1,1:-1] + shear_term_on_v) 
            
        return rhs_v
    
    def rhs_h(self,u,v,h, He, u11p=None, v11p=None):

        rhs_h = jnp.zeros_like(h)

        # --- Continuity equation ---
        rhs_h = rhs_h.at[1:-1,1:-1].set(- He[1:-1,1:-1] * (\
                (u[1:-1,1:] - u[1:-1,:-1]) / self.DXu[1:-1,:] + \
                (v[1:,1:-1] - v[:-1,1:-1]) / self.DYv[:,1:-1]))

        # -----------------------------------------
        # Mean-flow divergence (u11p, v11p)
        # -----------------------------------------
        if u11p is not None and v11p is not None:

            # --- interpolate mean flow to h points ---
            u11p_on_h = u11p
            v11p_on_h = v11p

            # --- split velocities into positive and negative parts ---
            up = jnp.where(u11p_on_h < 0, 0, u11p_on_h)
            um = jnp.where(u11p_on_h > 0, 0, u11p_on_h)
            vp = jnp.where(v11p_on_h < 0, 0, v11p_on_h)
            vm = jnp.where(v11p_on_h > 0, 0, v11p_on_h)

            # --- advection term on T points ---
            adv_term_on_T = self.adv(up, vp, um, vm, h) # shape (ny-4, nx-4)

            # --- add advection term ---
            rhs_h = rhs_h.at[2:-2,2:-2].set(rhs_h[2:-2,2:-2] - adv_term_on_T)
          
        return rhs_h
    
    
    ###########################################################################
    #                         Boundary Handling                               #
    ###########################################################################

    def boundary_conditions(self,u,v,h,u0,v0,h0,w1ext):
        return u,v,h
    
    def periodic_boundary_conditions(self, u, v, h):
        """
        Periodic boundary conditions for staggered C-grid with ghost cells.

        h : (ny+2, nx+2)  cell centers (ghosts in x and y)
        u : (ny+2, nx+1)  x-faces     (ghosts in y only)
        v : (ny+1, nx+2)  y-faces     (ghosts in x only)
        """

        # --------------------
        # Periodic in x
        # --------------------
        if self.periodic_x:
            # h (cell centers)
            h = h.at[:, 0].set(h[:, -2])
            h = h.at[:, -1].set(h[:, 1])

            # v (y-faces) — has x-ghosts
            v = v.at[:, 0].set(v[:, -2])
            v = v.at[:, -1].set(v[:, 1])

            # u has NO x-ghosts → nothing to do

        # --------------------
        # Periodic in y
        # --------------------
        if self.periodic_y:
            # h (cell centers)
            h = h.at[0, :].set(h[-2, :])
            h = h.at[-1, :].set(h[1, :])

            # u (x-faces) — has y-ghosts
            u = u.at[0, :].set(u[-2, :])
            u = u.at[-1, :].set(u[1, :])

            # v has NO y-ghosts → nothing to do

        return u, v, h
        
    ###########################################################################
    #                            One time step                                #
    ###########################################################################

    def asselin_filter(self, q_nm1, q_n, q_np1, nu=0.1):
        """
        Asselin time filter to remove the leapfrog computational mode.
        """
        return q_n + nu * (q_nm1 - 2.0*q_n + q_np1)
            
    def step_euler(self,u0, v0, h0, He=None, w1ext=None, u11u=None, v11u=None, u11p=None, v11p=None, dc2=None):
        
        #######################
        #   Init local state  #
        #######################
        u1 = +u0
        v1 = +v0
        h1 = +h0
        He = self.Heb if He is None else He + self.Heb

        #######################
        # Boundary conditions #
        #######################
        u1,v1,h1 = self.periodic_boundary_conditions(u1,v1,h1)
        
        #######################
        #  Right hand sides   #
        #######################
        ku = self.rhs_u(u1,v1,h1)
        kv = self.rhs_v(u1,v1,h1)
        kh = self.rhs_h(u1,v1,h1,He)
        
        #######################
        #  Time propagation   #
        #######################
        u = u1 + self.dt*ku 
        v = v1 + self.dt*kv
        h = h1 + self.dt*kh
        
        #######################
        # Boundary conditions #
        #######################
        u,v,h = self.boundary_conditions(u,v,h,u0,v0,h0,w1ext)
        
        return u, v, h
    
    def step_rk4(self, u0, v0, h0, He=None, w1ext=None, u11u=None, v11u=None, u11z=None, v11z=None, u11p=None, v11p=None):
        
        #######################
        #   Init local state  #
        #######################
        u1 = +u0
        v1 = +v0
        h1 = +h0
        He = self.Heb if He is None else He + self.Heb

        #######################
        # Boundary conditions #
        #######################
        u1,v1,h1 = self.periodic_boundary_conditions(u1,v1,h1)
        
        #######################
        #  Right hand sides   #
        #######################
        # k1  (use plain methods — inner jit is a no-op under the outer jit)
        ku1 = self.rhs_u(u1,v1,h1, u11u, v11u, u11z, v11z)*self.dt
        kv1 = self.rhs_v(u1,v1,h1, u11u, v11u, u11z, v11z)*self.dt
        kh1 = self.rhs_h(u1,v1,h1,He, u11p, v11p)*self.dt
        # k2
        ku2 = self.rhs_u(u1+0.5*ku1,v1+0.5*kv1,h1+0.5*kh1, u11u, v11u, u11z, v11z)*self.dt
        kv2 = self.rhs_v(u1+0.5*ku1,v1+0.5*kv1,h1+0.5*kh1, u11u, v11u, u11z, v11z)*self.dt
        kh2 = self.rhs_h(u1+0.5*ku1,v1+0.5*kv1,h1+0.5*kh1,He, u11p, v11p)*self.dt
        # k3
        ku3 = self.rhs_u(u1+0.5*ku2,v1+0.5*kv2,h1+0.5*kh2, u11u, v11u, u11z, v11z)*self.dt
        kv3 = self.rhs_v(u1+0.5*ku2,v1+0.5*kv2,h1+0.5*kh2, u11u, v11u, u11z, v11z)*self.dt
        kh3 = self.rhs_h(u1+0.5*ku2,v1+0.5*kv2,h1+0.5*kh2,He, u11p, v11p)*self.dt
        # k4
        ku4 = self.rhs_u(u1+ku3,v1+kv3,h1+kh3, u11u, v11u, u11z, v11z)*self.dt
        kv4 = self.rhs_v(u1+ku3,v1+kv3,h1+kh3, u11u, v11u, u11z, v11z)*self.dt
        kh4 = self.rhs_h(u1+ku3,v1+kv3,h1+kh3,He, u11p, v11p)*self.dt
        
        #######################
        #   Time propagation  #
        #######################
        u = u1 + 1/6*(ku1+2*ku2+2*ku3+ku4)
        v = v1 + 1/6*(kv1+2*kv2+2*kv3+kv4)
        h = h1 + 1/6*(kh1+2*kh2+2*kh3+kh4)
        
        #######################
        # Boundary conditions #
        #######################
        u,v,h = self.boundary_conditions(u,v,h,u0,v0,h0,w1ext)
        
        return u, v, h


    def step_rk3(self, u0, v0, h0, He=None, w1ext=None, u11u=None, v11u=None, u11z=None, v11z=None, u11p=None, v11p=None):
        
        #######################
        #   Init local state  #
        #######################
        u = +u0
        v = +v0
        h = +h0
        He = self.Heb if He is None else He + self.Heb

        #######################
        # Boundary conditions #
        #######################
        u, v, h = self.periodic_boundary_conditions(u, v, h)

        def _rhs(u_, v_, h_):
            du = self.rhs_u(u_, v_, h_, u11u, v11u, u11z, v11z)
            dv = self.rhs_v(u_, v_, h_, u11u, v11u, u11z, v11z)
            dh = self.rhs_h(u_, v_, h_, He, u11p, v11p)
            return du, dv, dh

        #######################
        #  RK3-SSP stages     #
        #######################
        du0, dv0, dh0 = _rhs(u, v, h)
        u = u + self.dt * du0
        v = v + self.dt * dv0
        h = h + self.dt * dh0

        du1, dv1, dh1 = _rhs(u, v, h)
        u = u + (self.dt / 4.0) * (du1 - 3.0 * du0)
        v = v + (self.dt / 4.0) * (dv1 - 3.0 * dv0)
        h = h + (self.dt / 4.0) * (dh1 - 3.0 * dh0)

        du2, dv2, dh2 = _rhs(u, v, h)
        u = u + (self.dt / 12.0) * (8.0 * du2 - du1 - du0)
        v = v + (self.dt / 12.0) * (8.0 * dv2 - dv1 - dv0)
        h = h + (self.dt / 12.0) * (8.0 * dh2 - dh1 - dh0)

        #######################
        # Boundary conditions #
        #######################
        u, v, h = self.boundary_conditions(u, v, h, u0, v0, h0, w1ext)
        
        return u, v, h

    def step_leapfrog(self, u_nm1, v_nm1, h_nm1,
                        u_n,   v_n,   h_n,
                        He=None, w1ext=None):

        #######################
        #   Init local state  #
        #######################
        u1 = +u_n
        v1 = +v_n
        h1 = +h_n
        He = self.Heb if He is None else He + self.Heb

        #######################
        # Boundary conditions #
        #######################
        u1,v1,h1 = self.periodic_boundary_conditions(u1,v1,h1)

        #######################
        #  Right hand sides   #
        #######################
        ku = self.rhs_u(u1, v1, h1)
        kv = self.rhs_v(u1, v1, h1)
        kh = self.rhs_h(u1, v1, h1, He)

        #######################
        #  Time propagation   #
        #######################
        u_np1 = u_nm1 + 2.0 * self.dt * ku
        v_np1 = v_nm1 + 2.0 * self.dt * kv
        h_np1 = h_nm1 + 2.0 * self.dt * kh

        #######################
        # Boundary conditions #
        #######################
        u_np1, v_np1, h_np1 = self.boundary_conditions(
            u_np1, v_np1, h_np1,
            u_n, v_n, h_n,
            w1ext
        )

        return u_np1, v_np1, h_np1

    def _wave_phases(self, w, He, He_on_u, He_on_v):
        """
        Compute per-boundary phase and amplitude arrays for ALL angles in
        self.bc_theta at once, for a given frequency w and equivalent depth He.

        Instead of iterating over theta in Python (which unrolls under jit/scan),
        we vectorise over theta using standard broadcasting: theta has shape
        (n_theta,) and grid arrays have shape (..., ny, nx).  This reduces the
        XLA graph to O(1) nodes regardless of n_theta, dramatically cutting both
        compile time and the number of dispatched kernels at runtime.

        Returns
        -------
        For each of the 12 combinations (direction S/N/W/E × grid h/u/v):

          phase  : array (n_theta, ny_g, nx_g)   -- phase kxy(theta)
          amp    : array (n_theta, ny_g, nx_g)   -- amplitude correction (WKB) or 1.0
          kx_g   : array (n_theta, ny_g, nx_g)   -- k_x component
          ky_g   : array (n_theta, ny_g, nx_g)   -- k_y component

        All returned as the compound dict:
          out[direction][grid] = (phase, amp, kx, ky)

        Method is controlled by self.bc_it_method (plane_wave / plane_wave_bdy / wkb).
        """
        thetas = jnp.asarray(self.bc_theta)   # (n_theta,)
        sin_t  = jnp.sin(thetas)               # (n_theta,)
        cos_t  = jnp.cos(thetas)               # (n_theta,)
        # add trailing spatial dims for broadcasting: (n_theta, 1, 1)
        sin_t3 = sin_t[:, None, None]
        cos_t3 = cos_t[:, None, None]

        out = {d: {} for d in ('S', 'N', 'W', 'E')}

        grids = [
            ('h', He,      self.f,       self.X,  self.Y),
            ('u', He_on_u, self.f_on_u,  self.Xu, self.Yu),
            ('v', He_on_v, self.f_on_v,  self.Xv, self.Yv),
        ]

        for gname, He_g, f_g, X_g, Y_g in grids:
            ny_g, nx_g = He_g.shape
            # wavenumber on this grid: (ny_g, nx_g)
            k_g = jnp.sqrt((w**2 - f_g**2) / (self.g * He_g))

            # k components for each (theta, y, x): (n_theta, ny_g, nx_g)
            # Direction conventions:
            #   S : (kx, ky) = ( sin θ,  cos θ) * k
            #   N : (kx, ky) = ( sin θ, -cos θ) * k
            #   W : (kx, ky) = ( cos θ,  sin θ) * k
            #   E : (kx, ky) = (-cos θ,  sin θ) * k
            kx_S =  sin_t3 * k_g;  ky_S =  cos_t3 * k_g
            kx_N =  sin_t3 * k_g;  ky_N = -cos_t3 * k_g
            kx_W =  cos_t3 * k_g;  ky_W =  sin_t3 * k_g
            kx_E = -cos_t3 * k_g;  ky_E =  sin_t3 * k_g

            # default amplitude: 1.0 (no correction)
            amp_ones = jnp.ones((len(thetas), ny_g, nx_g))

            # Local boundary coordinates.  Tangential phase is a path
            # integral along the boundary; using k(s) * s would add an artificial
            # s * dk/ds term and creates stripes perpendicular to the boundary.
            y_S = Y_g - Y_g[0:1, :]
            y_N = Y_g[-1:, :] - Y_g
            x_W = X_g - X_g[:, 0:1]
            x_E = X_g[:, -1:] - X_g

            def _cumtrapz_last(coord, kval):
                ds = coord[..., 1:] - coord[..., :-1]
                dphi = 0.5 * (kval[..., :-1] + kval[..., 1:]) * ds
                return jnp.concatenate([jnp.zeros_like(kval[..., :1]),
                                        jnp.cumsum(dphi, axis=-1)], axis=-1)

            def _cumtrapz_first(coord, kval):
                return jnp.swapaxes(_cumtrapz_last(jnp.swapaxes(coord, -1, -2),
                                                   jnp.swapaxes(kval, -1, -2)), -1, -2)

            if self.bc_it_method == 'plane_wave':
                phi_x = _cumtrapz_last(X_g, k_g)
                phi_y = _cumtrapz_first(Y_g, k_g)
                phi_S = sin_t3 * phi_x + cos_t3 * k_g * y_S
                phi_N = sin_t3 * phi_x + cos_t3 * k_g * y_N
                phi_W = cos_t3 * k_g * x_W + sin_t3 * phi_y
                phi_E = cos_t3 * k_g * x_E + sin_t3 * phi_y
                amp_S = amp_N = amp_W = amp_E = amp_ones

            elif self.bc_it_method == 'plane_wave_bdy':
                # k evaluated at the boundary edge, broadcast inward
                k_S_bdy = jnp.sqrt((w**2 - f_g[0 ,:]**2) / (self.g * He_g[0 ,:]))   # (nx_g,)
                k_N_bdy = jnp.sqrt((w**2 - f_g[-1,:]**2) / (self.g * He_g[-1,:]))
                k_W_bdy = jnp.sqrt((w**2 - f_g[:,0 ]**2) / (self.g * He_g[:,0 ]))   # (ny_g,)
                k_E_bdy = jnp.sqrt((w**2 - f_g[:,-1]**2) / (self.g * He_g[:,-1]))
                phi_S_t = _cumtrapz_last(X_g[0, :], k_S_bdy)[None, :]
                phi_N_t = _cumtrapz_last(X_g[-1, :], k_N_bdy)[None, :]
                phi_W_t = _cumtrapz_last(Y_g[:, 0], k_W_bdy)[:, None]
                phi_E_t = _cumtrapz_last(Y_g[:, -1], k_E_bdy)[:, None]
                phi_S = sin_t3 * phi_S_t + cos_t3 * k_S_bdy[None, :] * y_S
                phi_N = sin_t3 * phi_N_t + cos_t3 * k_N_bdy[None, :] * y_N
                phi_W = cos_t3 * k_W_bdy[:, None] * x_W + sin_t3 * phi_W_t
                phi_E = cos_t3 * k_E_bdy[:, None] * x_E + sin_t3 * phi_E_t
                amp_S = amp_N = amp_W = amp_E = amp_ones

            else:  # 'wkb'
                DX_g = X_g[:,1:] - X_g[:,:-1]   # (ny_g, nx_g-1)
                DY_g = Y_g[1:,:] - Y_g[:-1,:]   # (ny_g-1, nx_g)

                # normal / tangential wavenumber components (n_theta, ny_g, nx_g)
                abs_cos = jnp.abs(cos_t3);  abs_sin = jnp.abs(sin_t3)

                # S / N  (normal = y)
                ky_n  = abs_cos * k_g;  kx_t = abs_sin * k_g
                dPhi_y  = 0.5*(ky_n[:,:-1,:] + ky_n[:,1:,:]) * DY_g   # (n_t, ny_g-1, nx_g)
                dPhi_x  = 0.5*(kx_t[:,:,:-1] + kx_t[:,:,1:]) * DX_g  # (n_t, ny_g, nx_g-1)
                phi_yS  = jnp.concatenate([jnp.zeros((len(thetas),1,nx_g)),
                                            jnp.cumsum(dPhi_y, axis=1)], axis=1)
                phi_yN  = jnp.concatenate([jnp.flip(jnp.cumsum(
                                            jnp.flip(dPhi_y, axis=1), axis=1), axis=1),
                                            jnp.zeros((len(thetas),1,nx_g))], axis=1)
                phi_xSN = jnp.concatenate([jnp.zeros((len(thetas),ny_g,1)),
                                            jnp.cumsum(dPhi_x, axis=2)], axis=2)
                phi_S   = phi_yS  + phi_xSN
                phi_N   = phi_yN  + phi_xSN

                # W / E  (normal = x)
                kx_n  = abs_cos * k_g;  ky_t = abs_sin * k_g
                dPhi_xWE = 0.5*(kx_n[:,:,:-1] + kx_n[:,:,1:]) * DX_g
                dPhi_yWE = 0.5*(ky_t[:,:-1,:] + ky_t[:,1:,:]) * DY_g
                phi_xW  = jnp.concatenate([jnp.zeros((len(thetas),ny_g,1)),
                                            jnp.cumsum(dPhi_xWE, axis=2)], axis=2)
                phi_xE  = jnp.concatenate([jnp.flip(jnp.cumsum(
                                            jnp.flip(dPhi_xWE, axis=2), axis=2), axis=2),
                                            jnp.zeros((len(thetas),ny_g,1))], axis=2)
                phi_yWE = jnp.concatenate([jnp.zeros((len(thetas),1,nx_g)),
                                            jnp.cumsum(dPhi_yWE, axis=1)], axis=1)
                phi_W   = phi_xW + phi_yWE
                phi_E   = phi_xE + phi_yWE

                # WKB amplitude: He^{-1/4} normalised at boundary (n_theta broadcast)
                amp_S = jnp.broadcast_to((He_g[0:1,:] / He_g)**0.25, (len(thetas), ny_g, nx_g))
                amp_N = jnp.broadcast_to((He_g[-1:,:] / He_g)**0.25, (len(thetas), ny_g, nx_g))
                amp_W = jnp.broadcast_to((He_g[:,0:1] / He_g)**0.25, (len(thetas), ny_g, nx_g))
                amp_E = jnp.broadcast_to((He_g[:,-1:] / He_g)**0.25, (len(thetas), ny_g, nx_g))

            out['S'][gname] = (phi_S, amp_S, kx_S, ky_S)
            out['N'][gname] = (phi_N, amp_N, kx_N, ky_N)
            out['W'][gname] = (phi_W, amp_W, kx_W, ky_W)
            out['E'][gname] = (phi_E, amp_E, kx_E, ky_E)

        return out  # out[dir][grid] = (phase, amp, kx, ky)  all (n_theta, ny_g, nx_g)

    def _smootherstep(self, x):
        return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)

    def _edge_weight(self, dist, mask):
        mask = jnp.asarray(mask, dtype=bool)
        dist = jnp.asarray(dist)
        width = jnp.max(jnp.where(mask, dist, 0.0))
        width = jnp.maximum(width, 1e-12)
        r = jnp.clip(dist / width, 0.0, 1.0)
        weight = 1.0 - self._smootherstep(r)
        if self.bc_it_corner_weight_power != 1.0:
            weight = weight ** self.bc_it_corner_weight_power
        return jnp.where(mask, weight, 0.0)

    def _it_boundary_weights(self, grid):
        """Smooth partition of unity for S/N/W/E sponge fields on one grid."""
        if grid == 'h':
            X, Y = self.X, self.Y
            names = ('sponge_on_h_S', 'sponge_on_h_N', 'sponge_on_h_W', 'sponge_on_h_E')
        elif grid == 'u':
            X, Y = self.Xu, self.Yu
            names = ('sponge_on_u_S', 'sponge_on_u_N', 'sponge_on_u_W', 'sponge_on_u_E')
        else:
            X, Y = self.Xv, self.Yv
            names = ('sponge_on_v_S', 'sponge_on_v_N', 'sponge_on_v_W', 'sponge_on_v_E')

        if not all(hasattr(self, name) for name in names):
            weight = 0.25 * jnp.ones_like(X)
            return weight, weight, weight, weight

        X = jnp.asarray(X)
        Y = jnp.asarray(Y)
        mask_S = getattr(self, names[0])
        mask_N = getattr(self, names[1])
        mask_W = getattr(self, names[2])
        mask_E = getattr(self, names[3])

        w_S = self._edge_weight(jnp.abs(Y - Y[0:1, :]), mask_S)
        w_N = self._edge_weight(jnp.abs(Y[-1:, :] - Y), mask_N)
        w_W = self._edge_weight(jnp.abs(X - X[:, 0:1]), mask_W)
        w_E = self._edge_weight(jnp.abs(X[:, -1:] - X), mask_E)

        weight_sum = w_S + w_N + w_W + w_E
        active = weight_sum > 0.0
        weight_sum = jnp.where(active, weight_sum, 1.0)
        return tuple(jnp.where(active, w / weight_sum, 0.0)
                     for w in (w_S, w_N, w_W, w_E))

    def compute_IT_2D(self, t, He, h_SN, h_WE, flag_tangent=True):
        """
        Compute 2D IT wave fields for sponge boundary conditions.

        Each S/N/W/E wave field is blended with a smooth partition of unity.
        This avoids the hard corner jumps produced by boolean sponge-mask
        averaging while leaving the HBC amplitudes untouched.

        Parameters
        ----------
        t     : float   -- time in seconds
        He    : (ny,nx) -- total equivalent depth (Heb + anomaly)
        h_SN  : (n_omega, 2, 2, n_theta, nx) -- SSH amplitudes S/N borders [border,cos/sin,theta,x]
        h_WE  : (n_omega, 2, 2, n_theta, ny) -- SSH amplitudes W/E borders [border,cos/sin,theta,y]
        flag_tangent : bool -- compute tangential velocity components

        Returns
        -------
        u_it, v_it, h_it : 2D arrays
        """
        He_on_u = (He[:,1:] + He[:,:-1]) / 2
        He_on_v = (He[1:,:] + He[:-1,:]) / 2

        wh_S, wh_N, wh_W, wh_E = self._it_boundary_weights('h')
        wu_S, wu_N, wu_W, wu_E = self._it_boundary_weights('u')
        wv_S, wv_N, wv_W, wv_E = self._it_boundary_weights('v')

        # Accumulate contributions from all omega/theta (as sums of 2D arrays)
        u_S = jnp.zeros((self.ny,   self.nx-1))
        v_S = jnp.zeros((self.ny-1, self.nx))
        h_S = jnp.zeros((self.ny,   self.nx))
        u_N = jnp.zeros_like(u_S);  v_N = jnp.zeros_like(v_S);  h_N = jnp.zeros_like(h_S)
        u_W = jnp.zeros_like(u_S);  v_W = jnp.zeros_like(v_S);  h_W = jnp.zeros_like(h_S)
        u_E = jnp.zeros_like(u_S);  v_E = jnp.zeros_like(v_S);  h_E = jnp.zeros_like(h_S)

        for j, w in enumerate(self.omegas):
            # --- compute phase/amp/kx/ky for all thetas at once (no theta loop) ---
            phases = self._wave_phases(w, He, He_on_u, He_on_v)
            # phases[dir][grid] = (phi, amp, kx, ky)  each (n_theta, ny_g, nx_g)

            w2f2_v = w**2 - self.f_on_v**2   # used for v formula   (ny-1, nx)
            w2f2_u = w**2 - self.f_on_u**2   # used for u formula   (ny, nx-1)

            # Helpers (hc, hs already broadcast to (n_theta, ny_g, nx_g) by the caller):
            def _h_field(phi, amp, hc, hs):
                c = jnp.cos(w*t - phi)   # (n_theta, ny_g, nx_g)
                s = jnp.sin(w*t - phi)
                return jnp.sum(amp * (hc*c + hs*s), axis=0)

            def _vel_field(phi, amp, kp, km, fp, hc, hs, w2f2):
                c = jnp.cos(w*t - phi)
                s = jnp.sin(w*t - phi)
                return jnp.sum(
                    amp * (self.g / w2f2) * (
                        hc * (w*kp*c + fp*km*s) +
                        hs * (w*kp*s - fp*km*c)
                    ), axis=0)

            # ------- South -------
            # h_SN[j, border, cs, :, :] shape: (n_theta, nx)
            hc_nx = h_SN[j,0,0,:]   # (n_theta, nx)
            hs_nx = h_SN[j,0,1,:]
            # broadcast to (n_theta, 1, nx) for h/v grids (ny or ny-1, nx)
            hc_xb = hc_nx[:, None, :]
            hs_xb = hs_nx[:, None, :]
            # interpolate to nx-1 for u-grid, then broadcast to (n_theta, 1, nx-1)
            hc_ub = ((hc_nx[:, :-1] + hc_nx[:, 1:]) * 0.5)[:, None, :]
            hs_ub = ((hs_nx[:, :-1] + hs_nx[:, 1:]) * 0.5)[:, None, :]

            phi_h, amp_h, kx_h, ky_h = phases['S']['h']
            phi_v, amp_v, kx_v, ky_v = phases['S']['v']
            h_S = h_S + _h_field(phi_h, amp_h, hc_xb, hs_xb)
            v_S = v_S + _vel_field(
                    phi_v, amp_v, ky_v, kx_v, self.f_on_v, hc_xb, hs_xb, w2f2_v)
            if flag_tangent:
                phi_u, amp_u, kx_u, ky_u = phases['S']['u']
                u_S = u_S + _vel_field(
                        phi_u, amp_u, kx_u, ky_u, self.f_on_u, hc_ub, hs_ub, w2f2_u)

            # ------- North -------
            hc_nx = h_SN[j,1,0,:]
            hs_nx = h_SN[j,1,1,:]
            hc_xb = hc_nx[:, None, :]
            hs_xb = hs_nx[:, None, :]
            hc_ub = ((hc_nx[:, :-1] + hc_nx[:, 1:]) * 0.5)[:, None, :]
            hs_ub = ((hs_nx[:, :-1] + hs_nx[:, 1:]) * 0.5)[:, None, :]

            phi_h, amp_h, kx_h, ky_h = phases['N']['h']
            phi_v, amp_v, kx_v, ky_v = phases['N']['v']
            h_N = h_N + _h_field(phi_h, amp_h, hc_xb, hs_xb)
            v_N = v_N + _vel_field(
                    phi_v, amp_v, ky_v, kx_v, self.f_on_v, hc_xb, hs_xb, w2f2_v)
            if flag_tangent:
                phi_u, amp_u, kx_u, ky_u = phases['N']['u']
                u_N = u_N + _vel_field(
                        phi_u, amp_u, kx_u, ky_u, self.f_on_u, hc_ub, hs_ub, w2f2_u)

            # ------- West -------
            # h_WE[j, border, cs, :, :] shape: (n_theta, ny)
            hc_ny = h_WE[j,0,0,:]   # (n_theta, ny)
            hs_ny = h_WE[j,0,1,:]
            # broadcast to (n_theta, ny, 1) for h/u grids (ny, nx or nx-1)
            hc_yb = hc_ny[:, :, None]
            hs_yb = hs_ny[:, :, None]
            # interpolate to ny-1 for v-grid, then broadcast to (n_theta, ny-1, 1)
            hc_vb = ((hc_ny[:, :-1] + hc_ny[:, 1:]) * 0.5)[:, :, None]
            hs_vb = ((hs_ny[:, :-1] + hs_ny[:, 1:]) * 0.5)[:, :, None]

            phi_h, amp_h, kx_h, ky_h = phases['W']['h']
            phi_u, amp_u, kx_u, ky_u = phases['W']['u']
            h_W = h_W + _h_field(phi_h, amp_h, hc_yb, hs_yb)
            u_W = u_W + _vel_field(
                    phi_u, amp_u, kx_u, ky_u, self.f_on_u, hc_yb, hs_yb, w2f2_u)
            if flag_tangent:
                phi_v, amp_v, kx_v, ky_v = phases['W']['v']
                v_W = v_W + _vel_field(
                        phi_v, amp_v, ky_v, kx_v, self.f_on_v, hc_vb, hs_vb, w2f2_v)

            # ------- East -------
            hc_ny = h_WE[j,1,0,:]
            hs_ny = h_WE[j,1,1,:]
            hc_yb = hc_ny[:, :, None]
            hs_yb = hs_ny[:, :, None]
            hc_vb = ((hc_ny[:, :-1] + hc_ny[:, 1:]) * 0.5)[:, :, None]
            hs_vb = ((hs_ny[:, :-1] + hs_ny[:, 1:]) * 0.5)[:, :, None]

            phi_h, amp_h, kx_h, ky_h = phases['E']['h']
            phi_u, amp_u, kx_u, ky_u = phases['E']['u']
            h_E = h_E + _h_field(phi_h, amp_h, hc_yb, hs_yb)
            u_E = u_E + _vel_field(
                    phi_u, amp_u, kx_u, ky_u, self.f_on_u, hc_yb, hs_yb, w2f2_u)
            if flag_tangent:
                phi_v, amp_v, kx_v, ky_v = phases['E']['v']
                v_E = v_E + _vel_field(
                        phi_v, amp_v, ky_v, kx_v, self.f_on_v, hc_vb, hs_vb, w2f2_v)

        u_it = wu_S * u_S + wu_N * u_N + wu_W * u_W + wu_E * u_E
        v_it = wv_S * v_S + wv_N * v_N + wv_W * v_W + wv_E * v_E
        h_it = wh_S * h_S + wh_N * h_N + wh_W * h_W + wh_E * h_E

        return u_it, v_it, h_it

    def _step_euler_nstep(self, u0, v0, h0, He=None, w1ext=None,
                          u11u=None, v11u=None, u11p=None, v11p=None, dc2=None,
                          nstep=1, t=0., He_total=None, h_SN=None, h_WE=None):
        """Multi-step Euler with lax.scan and checkpointing."""

        def body(carry, _):
            u, v, h, tc = carry
            u1, v1, h1 = self.step_euler(u, v, h, He, w1ext, u11u, v11u, u11p, v11p, dc2)
            if self.flag_sponge_bc:
                _u_b, _v_b, _h_b = self.compute_IT_2D(tc, He_total, h_SN, h_WE)
                u1 = u1 + self.sponge_coef * self.sponge_u * (_u_b - u)
                v1 = v1 + self.sponge_coef * self.sponge_v * (_v_b - v)
                h1 = h1 + self.sponge_coef * self.sponge_h * (_h_b - h)
            return (u1, v1, h1, tc + self.dt), None

        body = jax_checkpoint(body)
        (u, v, h, _), _ = scan(body, (u0, v0, h0, t), None, length=nstep)
        return u, v, h

    def _step_rk4_nstep(self, u0, v0, h0, He=None, w1ext=None,
                         u11u=None, v11u=None, u11z=None, v11z=None, u11p=None, v11p=None,
                         nstep=1, t=0., He_total=None, h_SN=None, h_WE=None):
        """Multi-step RK4 with lax.scan and checkpointing."""

        def body(carry, _):
            u, v, h, tc = carry
            u1, v1, h1 = self.step_rk4(u, v, h, He, w1ext, u11u, v11u, u11z, v11z, u11p, v11p)
            if self.flag_sponge_bc:
                _u_b, _v_b, _h_b = self.compute_IT_2D(tc, He_total, h_SN, h_WE)
                u1 = u1 + self.sponge_coef * self.sponge_u * (_u_b - u)
                v1 = v1 + self.sponge_coef * self.sponge_v * (_v_b - v)
                h1 = h1 + self.sponge_coef * self.sponge_h * (_h_b - h)
            return (u1, v1, h1, tc + self.dt), None

        body = jax_checkpoint(body)
        (u, v, h, _), _ = scan(body, (u0, v0, h0, t), None, length=nstep)
        return u, v, h
      

    def _step_rk3_nstep(self, u0, v0, h0, He=None, w1ext=None,
                         u11u=None, v11u=None, u11z=None, v11z=None, u11p=None, v11p=None,
                         nstep=1, t=0., He_total=None, h_SN=None, h_WE=None):
        """Multi-step RK3-SSP with lax.scan and checkpointing."""

        def body(carry, _):
            u, v, h, tc = carry
            u1, v1, h1 = self.step_rk3(u, v, h, He, w1ext, u11u, v11u, u11z, v11z, u11p, v11p)
            if self.flag_sponge_bc:
                _u_b, _v_b, _h_b = self.compute_IT_2D(tc, He_total, h_SN, h_WE)
                u1 = u1 + self.sponge_coef * self.sponge_u * (_u_b - u)
                v1 = v1 + self.sponge_coef * self.sponge_v * (_v_b - v)
                h1 = h1 + self.sponge_coef * self.sponge_h * (_h_b - h)
            return (u1, v1, h1, tc + self.dt), None

        body = jax_checkpoint(body)
        (u, v, h, _), _ = scan(body, (u0, v0, h0, t), None, length=nstep)
        return u, v, h

    def step_euler_tgl(self,
                       du0, dv0, dh0, u0, v0, h0, 
                       dHe=None, dw1ext=None, du11=None, dv11=None, du11p=None, dv11p=None, ddc2=None,
                       He=None, w1ext=None, u11u=None, v11u=None, u11p=None, v11p=None, dc2=None):
        
        def wrapped_step(x):
            u0, v0, h0, He, w1ext, u11u, v11u, u11p, v11p, dc2 = x
            return self.step_euler(u0, v0, h0, He, w1ext, u11u, v11u, u11p, v11p, dc2)

        primals = ((u0, v0, h0, He, w1ext, u11u, v11u, u11p, v11p, dc2),)
        tangents = ((du0, dv0, dh0, dHe, dw1ext, du11, dv11, du11p, dv11p, ddc2),)

        _, dy = jax.jvp(wrapped_step, primals, tangents)

        return dy  # returns (du, dv, dh)
     
    def step_rk4_tgl(self,
                     du0, dv0, dh0, u0, v0, h0, 
                     dHe=None, dw1ext=None, du11u=None, dv11u=None, du11z=None, dv11z=None, du11p=None, dv11p=None, ddc2=None,
                     He=None, w1ext=None, u11u=None, v11u=None, u11z=None, v11z=None, u11p=None, v11p=None, dc2=None):
        
        def wrapped_step(x):
            u0, v0, h0, He, w1ext, u11u, v11u, u11z, v11z, u11p, v11p, dc2 = x
            return self.step_rk4(u0, v0, h0, He, w1ext, u11u, v11u, u11z, v11z, u11p, v11p, dc2)

        primals = ((u0, v0, h0, He, w1ext, u11u, v11u, u11z, v11z, u11p, v11p, dc2),)
        tangents = ((du0, dv0, dh0, dHe, dw1ext, du11u, dv11u, du11z, dv11z, du11p, dv11p, ddc2),)
        _, dy = jax.jvp(wrapped_step, primals, tangents)

        return dy  # returns (du, dv, dh)
      

    def step_rk3_tgl(self,
                     du0, dv0, dh0, u0, v0, h0, 
                     dHe=None, dw1ext=None, du11u=None, dv11u=None, du11z=None, dv11z=None, du11p=None, dv11p=None, ddc2=None,
                     He=None, w1ext=None, u11u=None, v11u=None, u11z=None, v11z=None, u11p=None, v11p=None, dc2=None):
        
        def wrapped_step(x):
            u0, v0, h0, He, w1ext, u11u, v11u, u11z, v11z, u11p, v11p, dc2 = x
            return self.step_rk3(u0, v0, h0, He, w1ext, u11u, v11u, u11z, v11z, u11p, v11p)

        primals = ((u0, v0, h0, He, w1ext, u11u, v11u, u11z, v11z, u11p, v11p, dc2),)
        tangents = ((du0, dv0, dh0, dHe, dw1ext, du11u, dv11u, du11z, dv11z, du11p, dv11p, ddc2),)
        _, dy = jax.jvp(wrapped_step, primals, tangents)

        return dy  # returns (du, dv, dh)

    def step_euler_adj(self,
                       adu0, adv0, adh0, u0, v0, h0,
                       He=None, w1ext=None, u11u=None, v11u=None, u11z=None, v11z=None, u11p=None, v11p=None, dc2=None):
        
        def wrapped_step(x):
            u0, v0, h0, He, w1ext, u11u, v11u, u11z, v11z, u11p, v11p, dc2 = x
            return self.step_euler(u0, v0, h0, He, w1ext, u11u, v11u, u11z, v11z, u11p, v11p, dc2)
        
        primals = ((u0, v0, h0, He, w1ext, u11u, v11u, u11z, v11z, u11p, v11p, dc2),)
        cotangents = (adu0, adv0, adh0)  

        _, vjp_fn = jax.vjp(wrapped_step, *primals)
        adjoints = vjp_fn(cotangents)

        return adjoints  # returns (adj_u0, adj_v0, adj_h0, adj_He, adj_w1ext, adj_u11, adj_v11, adj_u11p, adj_v11p, adj_dc2)

    def step_rk4_adj(self,
                     adu0, adv0, adh0, u0, v0, h0,
                    He=None, w1ext=None, u11u=None, v11u=None, u11z=None, v11z=None, u11p=None, v11p=None, dc2=None):
        
        def wrapped_step(x):
            u0, v0, h0, He, w1ext, u11u, v11u, u11z, v11z, u11p, v11p, dc2 = x
            return self.step_rk4(u0, v0, h0, He, w1ext, u11u, v11u, u11z, v11z, u11p, v11p, dc2)
        
        primals = ((u0, v0, h0, He, w1ext, u11u, v11u, u11z, v11z, u11p, v11p, dc2),)
        cotangents = (adu0, adv0, adh0)  

        _, vjp_fn = jax.vjp(wrapped_step, *primals)
        adjoints = vjp_fn(cotangents)

        return adjoints  # returns (adj_u0, adj_v0, adj_h0, adj_He, adj_w1ext, adj_u11, adj_v11, adj_u11z, adj_v11z, adj_u11p, adj_v11p, adj_dc2)


    def step_rk3_adj(self,
                     adu0, adv0, adh0, u0, v0, h0,
                    He=None, w1ext=None, u11u=None, v11u=None, u11z=None, v11z=None, u11p=None, v11p=None, dc2=None):
        
        def wrapped_step(x):
            u0, v0, h0, He, w1ext, u11u, v11u, u11z, v11z, u11p, v11p, dc2 = x
            return self.step_rk3(u0, v0, h0, He, w1ext, u11u, v11u, u11z, v11z, u11p, v11p)
        
        primals = ((u0, v0, h0, He, w1ext, u11u, v11u, u11z, v11z, u11p, v11p, dc2),)
        cotangents = (adu0, adv0, adh0)  

        _, vjp_fn = jax.vjp(wrapped_step, *primals)
        adjoints = vjp_fn(cotangents)

        return adjoints  # returns (adj_u0, adj_v0, adj_h0, adj_He, adj_w1ext, adj_u11, adj_v11, adj_u11z, adj_v11z, adj_u11p, adj_v11p, adj_dc2)
   

if __name__ == "__main__":
    
    import numpy 
    
    x = numpy.arange(0,1e6,10e3)
    y = numpy.arange(0,1e6,10e3)
    ny,nx = y.size,x.size
    X,Y = numpy.meshgrid(x,y)
    dt = 900
    
    swm = Swm(X=X,Y=Y,dt=dt)

    N = swm.nstates + swm.nparams
    
    X0 = numpy.zeros((N,))
    
    X0[swm.sliceHe] = 0.7
    X0[swm.sliceBc][:swm.nx] = 0.02
    
    for i in range(100):
        X0 = swm.step_rk4(X0)
    
    X0 = numpy.random.random((N,))
    dX0 = numpy.random.random((N,))
    adX0 = numpy.random.random((N,))
    
    print('Tangent test:')
    X2 = swm.step_rk4_jit(X0)
    for p in range(10):
        
        lambd = 10**(-p)
        
        X1 = swm.step_rk4_jit(X0+lambd*dX0)
        
        dX1 = swm.step_rk4_tgl_jit(dX0=lambd*dX0,X0=X0)
        
        ps = numpy.linalg.norm(X1-X2-dX1)/jnp.linalg.norm(dX1)

        print('%.E' % lambd,'%.E' % ps)
    
    print('\nAdjoint test:')
    dX1 = swm.step_rk4_tgl_jit(dX0=dX0,X0=X0)
    adX1= swm.step_rk4_adj_jit(adX0,X0)
    
    ps1 = numpy.inner(dX1,adX0)
    ps2 = numpy.inner(dX0,adX1)
    
    print(ps1/ps2)
    
   