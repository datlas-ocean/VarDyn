"""Local empirical statistics and compact BMaux-like temporal controls.

The existing daily SW forward discretization is retained. Compact temporal
windows are evaluated on its daily nodes before the continuous annual scan.
"""
import numpy as np
import jax
import jax.numpy as jnp


def summarize_local_statistics(local, records):
    """Display summaries; these never replace local parameters in the basis."""
    from .sw_learning_empirical import scale_groups
    keys, groups = scale_groups(records)
    report = []
    for i, (component, wavelength, basin) in enumerate(keys):
        ids = groups == i
        tau = local['tdec'][ids]
        report.append(dict(component=int(component), wavelength_km=float(wavelength), basin=int(basin),
            atoms=int(ids.sum()), mean=float(np.mean(local['mean'][ids])),
            raw_std=float(np.sqrt(np.mean(local['raw_std'][ids]**2))),
            std=float(np.sqrt(np.mean(local['std'][ids]**2))),
            std_floor=float(np.sqrt(np.mean(local['std_floor'][ids]**2))),
            floor_active=bool(np.any(local['floor_active'][ids])),
            floor_active_atoms=int(np.sum(local['floor_active'][ids])),
            std_prior_retained_atoms=int(np.sum(local['std_prior_retained'][ids])),
            tdec_days=float(np.nanmedian(tau)) if np.isfinite(tau).any() else None,
            width_days_median=float(np.median(local['window_width_days'][ids])),
            spacing_days_median=float(np.median(local['spacing_days'][ids])),
            status='summary_of_local_window_acf_fits',
            summary_definition='STD: RMS over atoms; tdec: median of local 1/e times'))
    return report

def window_matrix(times, width, facnlt=2):
    """Actual normalized compact windows, including all boundary overlaps."""
    times = np.asarray(times, float)
    spacing = width/facnlt
    first = int(np.floor(times[0]/spacing))-facnlt+1
    last = int(np.ceil(times[-1]/spacing))+facnlt-1
    centers = np.arange(first, last+1)*spacing
    d = (times[:, None]-centers)/width
    w = np.where(np.abs(d) < 1, np.cos(np.pi/2*d)**2, 0.)
    return w/w.sum(axis=1, keepdims=True)


def window_acf_table(times, max_lag, facnlt=2, min_spacing_days=1., candidates=256,
                     return_uncentered=False):
    """ACF of the exact discrete window bank used by MAP2.

Expected covariance is W W^T. Temporal centering matches the MAP1 statistic;
the reported e-folding time refers to the uncentered stationary-mean prior.
"""
    widths = np.geomspace(facnlt*min_spacing_days, 240., candidates)
    curves = []; effective = []; uncentered_curves = []
    for width in widths:
        w = window_matrix(times, width, facnlt)
        cov = w@w.T
        uncentered = np.array([np.diag(cov, k).mean() for k in range(len(times))])
        uncentered /= uncentered[0]
        uncentered_curves.append(uncentered[:max_lag+1])
        below = np.flatnonzero(uncentered <= np.exp(-1))
        if len(below):
            k = below[0]
            effective.append(k-1+(uncentered[k-1]-np.exp(-1))/(uncentered[k-1]-uncentered[k]))
        else:
            effective.append(np.nan)
        centered = cov-cov.mean(0)[None, :]-cov.mean(1)[:, None]+cov.mean()
        curve = np.array([np.diag(centered, k).mean() for k in range(max_lag+1)])
        curves.append(curve/curve[0])
    result = (widths, np.asarray(curves), np.asarray(effective))
    return (*result, np.asarray(uncentered_curves)) if return_uncentered else result


def local_statistics(analysis, reference_std, reference_tdec, max_lag=30,
                     floor_fraction=.05, chunk_size=1024, facnlt=2, min_spacing_days=1.):
    """Per-spatial-atom variance and fit to the actual temporal-window ACF.

No spatial pooling: local sample means remain separate. The floor is anchored
to the MAP1 prior, rather than repeatedly shrinking with later priors.
"""
    a = np.asarray(analysis)
    if a.ndim != 2 or a.shape[0] < 4 or not np.isfinite(a).all():
        raise ValueError('Need finite daily coefficients and at least four dates')
    nt, na = a.shape
    ref = np.broadcast_to(reference_std, (na,))
    ref_tau = np.broadcast_to(reference_tdec, (na,))
    if (not np.isfinite(ref).all() or np.any(ref <= 0)
            or not np.isfinite(ref_tau).all() or np.any(ref_tau <= 0)
            or not 0 < floor_fraction <= 1 or max_lag < 2):
        raise ValueError('Invalid local statistics settings')
    nlags = min(max_lag, nt//3)
    nlags = max(2, nlags)
    mean = a.mean(axis=0, dtype=np.float64)
    raw = np.zeros(na); tau = np.empty(na); error = np.full(na, np.nan)
    acf = np.zeros((nlags+1, na)); used = np.zeros(na, dtype=np.int32)
    status = np.empty(na, dtype='U40')
    for start in range(0, na, chunk_size):
        end = min(start+chunk_size, na)
        x = a[:, start:end].astype(np.float64)-mean[start:end]
        cov = np.array([np.mean(x[:nt-lag]*x[lag:], axis=0) for lag in range(nlags+1)])
        raw[start:end] = np.sqrt(np.maximum(cov[0], 0))
        acf[:, start:end] = np.divide(cov, cov[:1], out=np.zeros_like(cov), where=cov[:1] > 0)
    widths, curves, effective, uncentered_curves = window_acf_table(
        np.arange(nt), nlags, facnlt, min_spacing_days, return_uncentered=True)
    chosen = np.zeros(na, dtype=int)
    for start in range(0, na, chunk_size):
        end = min(start+chunk_size, na)
        observed = acf[1:, start:end]
        valid = np.cumprod(observed > 0, axis=0).astype(bool)
        weights = valid*(nt-np.arange(1,nlags+1))[:, None]
        weights = weights/np.maximum(weights.sum(axis=0), 1)
        loss = (curves[:, 1:]**2@weights - 2*curves[:, 1:]@(weights*observed)
                + np.sum(weights*observed**2,axis=0)[None, :])
        choice = np.argmin(loss,axis=0)
        chosen[start:end] = choice
        used[start:end] = valid.sum(axis=0)
        error[start:end] = np.sqrt(np.maximum(loss[choice,np.arange(end-start)],0))
    status[:] = 'fitted_window_acf'
    status[chosen == 0] = 'lower_width_bound'
    status[chosen == len(widths)-1] = 'upper_width_bound'
    unresolved = used < 2
    # No measured time scale in these cases: retain a model closest to the
    # reference correlation time and explicitly flag it, rather than fit noise.
    zero = raw <= np.finfo(float).eps*np.maximum(1.,np.abs(mean))
    for i in np.flatnonzero(unresolved | zero):
        target = np.exp(-np.arange(1,nlags+1)/ref_tau[i])
        chosen[i] = np.argmin(np.mean((uncentered_curves[:,1:]-target)**2,axis=1))
    status[unresolved] = 'unresolved_reference_scale'
    status[zero] = 'zero_variance_reference_scale'
    error[unresolved | zero] = np.nan
    tau[:] = effective[chosen]
    # On very short pilots, 1/e may lie beyond the sampled interval. Width is
    # still defined; report NaN rather than invent a measurable decorrelation.
    floor = floor_fraction*ref
    regularized_std = np.maximum(raw, floor)
    regularized_std[zero] = ref[zero]
    return dict(mean=mean, raw_std=raw, std=regularized_std,
                std_floor=floor, floor_active=raw < floor, tdec=tau,
                std_prior_retained=zero,
                window_width_days=widths[chosen], spacing_days=widths[chosen]/facnlt,
                acf=acf, fitted_acf=curves[chosen].T, used_lags=used,
                fit_rmse=error, fit_status=status, temporal_facnlt=facnlt,
                kernel='compact_cosine_squared_partition_unity',
                tdec_definition='1/e crossing of uncentered window-bank ACF; fit uses centered ACF',
                width_candidates_days=widths)


class ReducedTemporalBasis:
    """Ragged temporal centers per spatial atom, spacing = width / facnlt.

Window weights partition unity, so constant means and zero initial forcing
are represented exactly. Independent amplitudes are normalized to reproduce
the prescribed daily-node marginal variance averaged over the experiment.
"""
    def __init__(self, times, std, tdec, mean, facnlt=2, min_spacing_days=1., *, window_width):
        self.times = np.asarray(times, float)
        self.std, self.tdec, self.mean = [np.asarray(v, float) for v in (std,tdec,mean)]
        if (self.times.ndim != 1 or len(self.times) < 2 or not np.isfinite(self.times).all()
                or np.any(np.diff(self.times) <= 0) or self.std.ndim != 1
                or self.std.shape != self.tdec.shape or self.std.shape != self.mean.shape
                or not all(np.isfinite(v).all() for v in (self.std,self.mean))
                or np.any(self.std <= 0)
                or facnlt < 2 or int(facnlt) != facnlt or min_spacing_days <= 0):
            raise ValueError('Invalid reduced temporal layout')
        self.facnlt = int(facnlt)
        self.natom = len(self.std)
        self.width = np.asarray(window_width, float)
        if (self.width.shape != self.std.shape or not np.isfinite(self.width).all()
                or np.any(self.width < self.facnlt*min_spacing_days)):
            raise ValueError('Window widths must come from the matching ACF fit')
        self.spacing = self.width/self.facnlt
        self.first = np.floor(self.times[0]/self.spacing).astype('int32')-self.facnlt+1
        self.last = np.ceil(self.times[-1]/self.spacing).astype('int32')+self.facnlt-1
        self.counts = self.last-self.first+1
        self.offsets = np.r_[0, np.cumsum(self.counts)].astype('int32')
        self.nbasis = int(self.offsets[-1])
        # Memory O(natom*facnlt), no global dense/sparse temporal matrix.
        energy = np.zeros(self.natom)
        neighbors = np.arange(-self.facnlt+1, self.facnlt+1)
        for t in self.times:
            center = np.floor(t/self.spacing).astype(int)[:, None]+neighbors
            distance = (t-center*self.spacing[:, None])/self.width[:, None]
            w = np.where(np.abs(distance) < 1, np.cos(np.pi/2*distance)**2, 0.)
            w /= w.sum(axis=1, keepdims=True)
            energy += np.sum(w*w, axis=1)
        self.gain = np.sqrt(len(self.times)/energy)
        self.control_std = np.repeat(self.std*self.gain, self.counts).astype('float32')
        self.control_mean = np.repeat(self.mean, self.counts).astype('float32')

    def coefficients(self, t, control):
        spacing = jnp.asarray(self.spacing, dtype=control.dtype)
        center = (jnp.floor(t/spacing).astype('int32')[:, None]
                  + jnp.arange(-self.facnlt+1, self.facnlt+1))
        local = center-jnp.asarray(self.first)[:, None]
        valid = (local >= 0) & (local < jnp.asarray(self.counts)[:, None])
        idx = jnp.asarray(self.offsets[:-1])[:, None]+jnp.clip(local, 0, jnp.asarray(self.counts)[:, None]-1)
        distance = (t-center*spacing[:, None])/(spacing[:, None]*self.facnlt)
        weight = jnp.where(valid & (jnp.abs(distance) < 1), jnp.cos(jnp.pi/2*distance)**2, 0.)
        weight /= jnp.sum(weight, axis=1, keepdims=True)
        return jnp.sum(weight*control.ravel()[idx], axis=1)

    def expand(self, control):
        # Keep gather indices at O(natom*facnlt), rather than materializing
        # an annual (ntime,natom,overlap) index/weight tensor in XLA constants.
        return jax.lax.map(jax.checkpoint(lambda t: self.coefficients(t, control)),
                           jnp.asarray(self.times, dtype=control.dtype))

    def metadata(self):
        return dict(times=self.times, std=self.std, tdec=self.tdec, mean=self.mean,
                    spacing_days=self.spacing, width_days=self.width, first_center_index=self.first,
                    counts=self.counts, offsets=self.offsets, gain=self.gain, facnlt=self.facnlt,
                    controls=self.nbasis, dense_controls=len(self.times)*self.natom,
                    temporal_kind='compact_cosine_squared_partition_unity')
