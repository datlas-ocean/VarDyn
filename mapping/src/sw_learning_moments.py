"""Sufficient moments for independent AR1 windows with a shared annual mean."""
import numpy as np
import jax.numpy as jnp


KEYS = ('start', 'start2', 'left', 'right', 'left2', 'right2', 'cross')


def window_moments(analysis, variance, lag):
    a, p, c = (np.asarray(x, dtype=np.float64) for x in (analysis, variance, lag))
    if a.ndim != 2 or p.shape != a.shape or c.shape != (len(a)-1, a.shape[1]) or len(a) < 1:
        raise ValueError('Invalid window posterior shapes')
    if not all(np.isfinite(x).all() for x in (a, p, c)) or np.any(p < 0):
        raise ValueError('Invalid posterior moments')
    reference = a.mean(axis=0)
    a = a-reference
    return dict(reference=reference, windows=1, nodes=len(a), transitions=len(a)-1,
                start=a[0].copy(), start2=a[0]**2+p[0],
                left=a[:-1].sum(0), right=a[1:].sum(0),
                left2=(a[:-1]**2+p[:-1]).sum(0), right2=(a[1:]**2+p[1:]).sum(0),
                cross=(a[:-1]*a[1:]+c).sum(0))


def combine_moments(windows):
    result = None
    for item in windows:
        if result is None:
            result = {key: np.array(value, copy=True) for key, value in item.items()}
        else:
            if result['start'].shape != item['start'].shape:
                raise ValueError('Incompatible spatial layouts')
            shift = item['reference']-result['reference']
            rebased = dict(item)
            for key,count in (('start','windows'),('left','transitions'),('right','transitions')):
                rebased[key+'2'] = item[key+'2']+2*shift*item[key]+item[count]*shift**2
                rebased[key] = item[key]+item[count]*shift
            rebased['cross'] = item['cross']+shift*(item['left']+item['right'])+item['transitions']*shift**2
            for key in (*KEYS,'windows','nodes','transitions'):
                result[key] += rebased[key]
    if result is None:
        raise ValueError('No windows')
    return result


def common_mean(m, rho):
    return (m['start']+(m['right']-rho*m['left'])/(1+rho))/(m['windows']+m['transitions']*(1-rho)/(1+rho))


def expected_ar1(moments, log_std, log_tau):
    """Sum of expected stationary AR1 negative log likelihoods, up to constants."""
    m = {key: jnp.asarray(value) for key, value in moments.items()}
    rho = jnp.exp(-jnp.exp(-log_tau))
    den = -jnp.expm1(-2*jnp.exp(-log_tau))
    mean = common_mean(m, rho)
    start = m['start2']-2*mean*m['start']+m['windows']*mean**2
    innovations = (m['right2']+rho**2*m['left2']-2*rho*m['cross']
                   -2*mean*(1-rho)*(m['right']-rho*m['left'])
                   +m['transitions']*(1-rho)**2*mean**2)
    return .5*jnp.sum(2*m['nodes']*log_std+m['transitions']*jnp.log(den)
                      +(start+innovations/den)*jnp.exp(-2*log_std))
