"""Daily SSH from the zero-diffusion UpperDyn barotropic flux component."""

import numpy as np
import jax
import jax.numpy as jnp


def cumulative_flux_weights(basis, days, checkpoint_hours=6):
    """Integrate GAUSS3D flux with the native MOD_DIFF checkpoint convention.

    INV projects the basis at each checkpoint and MOD_DIFF adds
    ``checkpoint_hours / 24`` times that flux. The first row is the zero
    initial barotropic state; subsequent rows are its daily values.
    """
    if not basis.flux or basis.time_spinup is not None:
        raise ValueError('Expected an unspun GAUSS3D flux basis')
    steps_per_day = 24 // checkpoint_hours
    if steps_per_day * checkpoint_hours != 24 or days < 1:
        raise ValueError('Checkpoints must divide a day')
    checkpoints = np.arange(days * steps_per_day, dtype=np.float64) / steps_per_day
    distance = (checkpoints[:, None] - np.asarray(basis.ENST)[None, :]) / basis.sigma_T
    weights = np.where(
        np.abs(distance) < 1.,
        -np.pi * np.sin(.5 * np.pi * distance) * np.cos(.5 * np.pi * distance),
        0.,
    )
    if basis.normalize_fact:
        weights /= basis.norm_fact
    increments = weights.reshape(days, steps_per_day, -1).sum(axis=1) / steps_per_day
    return np.concatenate((np.zeros((1, increments.shape[1])),
                           np.cumsum(increments, axis=0)), axis=0).astype('float32')


class BarotropicComponent:
    """Whitened GAUSS3D controls mapped to physical-metre daily SSH."""

    def __init__(self, basis, standard_deviation, days, checkpoint_hours=6):
        self.basis = basis
        self.standard_deviation = jnp.asarray(standard_deviation, dtype=jnp.float32)
        self.weights = jnp.asarray(cumulative_flux_weights(basis, days, checkpoint_hours))
        self.size = int(basis.nbasis)
        self.ntime = days + 1
        if self.standard_deviation.shape != (self.size,) or not bool(
                jnp.all(jnp.isfinite(self.standard_deviation) & (self.standard_deviation > 0))):
            raise ValueError('Invalid barotropic prior standard deviation')

    def states(self, whitened):
        controls = (jnp.asarray(whitened).ravel() * self.standard_deviation).reshape(
            self.basis.ENST.size, self.basis.Nx)
        spatial_amplitudes = self.weights @ controls
        fields = jax.lax.map(lambda row: self.basis.Gauss_xy @ row,
                             spatial_amplitudes)
        return fields.reshape((self.ntime, *self.basis.shape_phys))
