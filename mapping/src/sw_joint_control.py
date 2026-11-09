"""Whitened forcing/physics controls with a fixed-physics MAP2 mode."""
import numpy as np
import jax.numpy as jnp
from .basis import DailyAR1Covariance
from .sw_learning_empirical import scale_groups


class JointControl:
    def __init__(self,daily,temporal=None,physics=None,fixed=None):
        self.daily,self.temporal=daily,temporal
        self.physics=physics or {}
        self.fixed={k:jnp.asarray(v,dtype=jnp.float32) for k,v in (fixed or {}).items()}
        if self.physics and self.fixed:raise ValueError('Choose estimated or fixed physics')
        self.nforcing=daily.nbasis if temporal is None else temporal.nbasis
        self.covariance=DailyAR1Covariance(daily.std,daily.tdec,daily.ntime) if temporal is None else None
        self.slices={};offset=self.nforcing
        for name,(basis,std) in self.physics.items():
            self.slices[name]=slice(offset,offset+basis.nbasis);offset+=basis.nbasis
        self.size=offset

    def initial(self):
        z=np.zeros(self.size,dtype='float32')
        if self.temporal is None:
            center=np.tile(self.daily.mean,self.daily.ntime).astype('float32')
            z[:self.nforcing]=np.asarray(self.covariance.inverse_sqrt(-jnp.asarray(center)))
        else:
            z[:self.nforcing]=-self.temporal.control_mean/self.temporal.control_std
        return z

    def decode(self,z):
        z=z.ravel();q=z[:self.nforcing]
        if self.temporal is None:
            daily=self.covariance.sqrt(q).reshape(self.daily.ntime,self.daily.natom)+jnp.asarray(self.daily.mean)
        else:
            daily=self.temporal.expand(jnp.asarray(self.temporal.control_mean)+jnp.asarray(self.temporal.control_std)*q)
        params=dict(self.fixed)
        for name,(basis,std) in self.physics.items():
            params[name]=basis.operg(0.,jnp.asarray(std)*z[self.slices[name]])
        return daily,params

    def groups(self,strategy):
        if strategy=='prior':return None
        if strategy=='component_scale':_,atoms=scale_groups(self.daily.records)
        elif strategy=='local_atom':atoms=np.arange(self.daily.natom)
        else:raise ValueError('Unknown preconditioner')
        groups=np.tile(atoms,self.daily.ntime) if self.temporal is None else np.repeat(atoms,self.temporal.counts)
        next_group=int(groups.max())+1
        extra=[]
        for name,(basis,std) in self.physics.items():
            extra.append(np.full(basis.nbasis,next_group,dtype='int32'));next_group+=1
        return np.concatenate([groups,*extra]).astype('int32')
