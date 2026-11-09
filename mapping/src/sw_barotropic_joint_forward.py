"""Continuous SW trajectory with an additive static-model barotropic SSH.

The barotropic component cannot feed back on the SW state. Its zero-diffusion
MOD_DIFF evolution is therefore integrated separately at the same six-hour
checkpoints, then added in physical metres at each daily SSH observation.
"""
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np


def make_daily_residual(model, state, basis, observations, day_times, selected=None, return_states=False):
    times = np.asarray(day_times, float)
    if (len(times) != basis.ntime or not np.allclose(np.diff(times), 86400.)
            or not np.isclose(times[0]/86400., basis.start)):
        raise ValueError('Forward dates must match daily control nodes exactly')
    nstep = int(round(86400./model.dt))
    if not np.isclose(nstep*model.dt, 86400.):
        raise ValueError('Model timestep must divide one day for the learning scan')
    if model.advect_tracer or set(model.name_params)-{'c', 'H', 'g_prime', 'h_wind', 'bottom_drag_coef'}:
        raise ValueError('Unsupported joint SW parameters')
    observations.scan_misfit_size()
    boundary = tuple(jnp.stack(values) for values in zip(*[model._apply_bc(t, t+model.dt) for t in times]))
    if model.taux_wind is not None:
        wx, wy = jnp.asarray(model.taux_wind), jnp.asarray(model.tauy_wind)
        taper_x, taper_y = model.wind_taper_u, model.wind_taper_v
        wind_dt = model.wind_dt
    else:
        def constant_wind(name, shape, axis):
            value = jnp.asarray(getattr(model.model, name, 0.), dtype=model.dtype)
            result = jnp.zeros(shape, dtype=model.dtype)
            if value.ndim == 0:
                return result+value
            return result.at[:, 1:-1].set(value.T) if axis == 0 else result.at[1:-1].set(value.T)
        wx = constant_wind('taux', basis.shapes[0], 0)[None]
        wy = constant_wind('tauy', basis.shapes[1], 1)[None]
        taper_x = taper_y = 1.
        wind_dt = 86400.
    selected = np.ones(len(times), bool) if selected is None else np.asarray(selected, bool)
    if selected.shape != times.shape or not selected.any():
        raise ValueError('At least one selected observation day is required')
    selected_indices = np.flatnonzero(selected)
    template = state.copy()
    template.to_device()
    initial = dict(template.var)
    params = dict(template.params)
    ssh_scale = jnp.asarray(model.ssh_observation_scale)
    def facade(fields, forcing, static_parameters):
        s = template.copy()
        s.var, s.params = dict(fields), dict(params)
        s.params.update(static_parameters)
        for name, field in zip(basis.name_mod_var, forcing):
            s.params[name] = field
        return s
    def trajectory(coefficients, static_parameters, barotropic_ssh):
        if barotropic_ssh.shape != (len(times), state.ny, state.nx):
            raise ValueError('Barotropic daily SSH shape differs from the model grid')
        def observation_fields(fields, correction):
            observed = dict(fields)
            observed[model.name_var['SSH']] = (
                fields[model.name_var['SSH']] + correction / ssh_scale)
            return observed
        def day(fields, inputs):
            t, bleft, bright, correction = inputs
            fleft = basis.project(t/86400., coefficients.ravel())
            fright = basis.project(t/86400.+1, coefficients.ravel())
            residual = observations.scan_misfit(
                t, observation_fields(fields, correction))
            def step(fields, k):
                fraction = (k+.5)/nstep
                forcing = tuple((1-fraction)*a+fraction*b for a, b in zip(fleft, fright))
                bc = tuple((1-fraction)*a+fraction*b for a, b in zip(bleft, bright))
                instant = t+k*model.dt
                wind_index = jnp.clip(jnp.floor(instant/wind_dt).astype('int32'), 0, wx.shape[0]-1)
                external = tuple(a[None] for a in (*bc, jnp.zeros((0, model.ny, model.nx)),
                                  taper_x*wx[wind_index], taper_y*wy[wind_index]))
                s = facade(fields, forcing, static_parameters)
                model.step(s, t=instant, nstep=1, Xb=external)
                return dict(s.var), None
            final, _ = lax.scan(jax.checkpoint(step), fields, jnp.arange(nstep))
            return final, (residual, fields) if return_states else residual
        final, residuals = lax.scan(jax.checkpoint(day), initial,
            (jnp.asarray(times[:-1]),
             tuple(b[:-1] for b in boundary), tuple(b[1:] for b in boundary),
             barotropic_ssh[:-1]))
        if return_states:
            _, states = residuals
            result = {name: jnp.concatenate((values, final[name][None]), axis=0)
                      for name, values in states.items()}
            result['sla_barotrop'] = barotropic_ssh
            result['SSH_tot'] = (result[model.name_var['SSH']] * ssh_scale
                                 + barotropic_ssh)
            return result
        last = observations.scan_misfit(
            times[-1], observation_fields(final, barotropic_ssh[-1]))
        return jnp.concatenate((residuals, last[None]), axis=0)
    # Selection occurs before covariance sampling so withheld rows are excluded
    # both from MAP and posterior precision, not merely from reported metrics.
    if return_states:
        return trajectory
    return lambda coefficients, static_parameters, barotropic_ssh: trajectory(
        coefficients, static_parameters, barotropic_ssh)[selected_indices].ravel()
