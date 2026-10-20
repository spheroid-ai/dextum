"""MLP with fixed camera/goal scaling; motor and privileged-state normalization remain adaptive."""
import copy
import jax
import jax.numpy as j
import numpy as np
from brax.io import model
from brax.training.agents.ppo import networks
from brax.training.acme import running_statistics


def fixed_scaling(size):
    mean=j.zeros(size);std=j.ones(size)
    if size==34:
        std=std.at[20:23].set(.1)
        mean=mean.at[23:25].set(.5);std=std.at[23:25].set(.5)
        mean=mean.at[25:27].set(6.);std=std.at[25:27].set(6.)
        return 14,mean,std
    if size==89:return 81,mean,std
    raise ValueError('Unknown observation contract')


def preprocess(x,stats):
    if x.shape[-1]==102:
        # Share motor scaling across lags; quality/goal scales stay physical.
        blocks=x.reshape(x.shape[:-1]+(3,34));_,mean,std=fixed_scaling(34)
        return j.concatenate(((blocks[...,:14]-stats.mean[:14])/stats.std[:14],
            (blocks[...,14:]-mean[14:])/std[14:]),axis=-1).reshape(x.shape)
    start,mean,std=fixed_scaling(x.shape[-1])
    return j.concatenate(((x[...,:start]-stats.mean[:start])/stats.std[:start],
        (x[...,start:]-mean[start:])/std[start:]),axis=-1)


def network(*args,**kwargs):
    kwargs['preprocess_observations_fn']=preprocess
    return networks.make_ppo_networks(*args,**kwargs)


def make_network(history=1):
    if history not in (1,3):raise ValueError('Registered history lengths are1 and3')
    return network(observation_size={'state':34*history,'privileged_state':89},action_size=7,
        policy_hidden_layer_sizes=(512,256,128),value_hidden_layer_sizes=(512,256,128),
        policy_obs_key='state',value_obs_key='privileged_state')


def migrate(params):
    """Affine change of first-layer coordinates preserves actor and critic functions."""
    new=copy.deepcopy(params);stats= params[0]
    for key,index,size in [('state',1,34),('privileged_state',2,89)]:
        start,mean,std=fixed_scaling(size);first=new[index]['params']['hidden_0'];old=first['kernel']
        first['bias']=first['bias']+((mean[start:]-stats.mean[key][start:])/stats.std[key][start:])@old[start:]
        first['kernel']=old.at[start:].set(old[start:]*(std[start:]/stats.std[key][start:])[:,None])
    # Restore enough exploration for new goals, preserving every mean action.
    last=new[1]['params']['hidden_3'];last['kernel']=last['kernel'].at[:,7:].set(0.)
    last['bias']=last['bias'].at[7:].set(j.log(j.expm1(.12-.001)))
    return new


def migrate_history(params):
    """Preserve the whole distribution/critic while adding unused past frames."""
    new=copy.deepcopy(params);first=new[1]['params']['hidden_0'];old=first['kernel']
    if old.shape[0]!=34:raise ValueError('Only migrate a single-frame checkpoint')
    first['kernel']=j.concatenate((old,j.zeros((68,old.shape[1]),dtype=old.dtype)))
    stats=new[0]
    fields={name:{**getattr(stats,name),'state':j.tile(getattr(stats,name)['state'],3)}
            for name in ('mean','std','summed_variance')}
    return (stats.replace(**fields),*new[1:])


class OrientationPolicy:
    def __init__(self,path):
        self.params=model.load_params(path)
        size=self.params[1]['params']['hidden_0']['kernel'].shape[0]
        if size not in (34,102):raise ValueError('Unknown orientation actor input contract')
        self.history=size//34
        self.apply=jax.jit(networks.make_inference_fn(make_network(self.history))(self.params,deterministic=True))
        self.key=jax.random.PRNGKey(14)

    def __call__(self,obs):
        obs=np.asarray(obs,np.float32)
        if obs.shape[-1]!=34*self.history or not np.isfinite(obs).all():raise ValueError('Invalid orientation observation')
        out=np.asarray(self.apply({'state':j.array(obs)},self.key)[0])
        if not np.isfinite(out).all():raise ValueError('Nonfinite action')
        return out
