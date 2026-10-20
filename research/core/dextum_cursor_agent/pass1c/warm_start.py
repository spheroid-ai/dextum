"""Preserve Ideal actor channel meanings at the observable Kit B interface.

The legacy feature prefix is estimated from the packet only. CAD transforms
are fixed reference-pose calibration, not runtime simulator state. Missing
thumb-base touch and barometric shear are zeros, not privileged replacements.
"""
import copy
import numpy as np
import jax
import jax.numpy as j
from dextum import touch_mjx


def calibrate(sensors, model):
    import mujoco
    d=mujoco.MjData(model);mujoco.mj_forward(model,d)
    frames=d.xmat[sensors.frames].reshape(-1,3,3)
    rotations=d.site_xmat[sensors.sites].reshape(-1,3,3)
    sensors.legacy_rotation=np.einsum('nji,njk->nik',frames[sensors.pad],rotations)
    sensors.legacy_position=np.einsum('nji,nj->ni',frames[sensors.pad],d.site_xpos[sensors.sites]-d.xpos[sensors.frames[sensors.pad]])


def prefix(sensors, packet):
    optical=packet[:60].reshape(20,3)
    normal=j.concatenate((optical[:,0],packet[100:120]))
    cell_force=j.concatenate((optical[:,j.array([1,2,0])],j.stack((j.zeros(20),j.zeros(20),normal[20:]),-1)))
    force=j.asarray(sensors.membership).T@j.einsum('nij,nj->ni',j.asarray(sensors.legacy_rotation),cell_force)
    fn=j.asarray(sensors.membership).T@normal
    centroid=(j.asarray(sensors.membership).T@(normal[:,None]*j.asarray(sensors.legacy_position)))/j.maximum(fn[:,None],1e-6)
    centroid=j.where((fn>.3)[:,None],centroid,0)
    slip=j.concatenate((packet[155:160]>.5,j.zeros(6,bool)))
    ages=j.max(j.where(j.asarray(sensors.membership)>0,packet[160:200,None],0),axis=0)
    reading=touch_mjx.Reading(force,fn,j.zeros(11),centroid,j.zeros((11,3)),j.zeros((11,3)),slip,j.zeros(11),fn>.3)
    return touch_mjx.actor_features(reading,ages*20)


def expand(params, old_frame, new_frame, fresh_normalizer):
    """Each history frame retains its old prefix; appended weights start zero."""
    normalizer,policy,value=params
    policy=copy.deepcopy(policy)
    layers=policy['params']
    first=next(k for k in layers if 'kernel' in layers[k])
    old=layers[first]['kernel']
    if old.shape[0]!=3*old_frame:raise ValueError('Unexpected Ideal actor input width')
    expanded=j.zeros((3*new_frame,old.shape[1]),old.dtype)
    for frame in range(3):expanded=expanded.at[frame*new_frame:frame*new_frame+old_frame].set(old[frame*old_frame:(frame+1)*old_frame])
    layers[first]['kernel']=expanded
    # Actor preprocessing ignores running statistics; retain privileged critic
    # statistics exactly, initialize only newly sized actor bookkeeping leaves.
    normalizer=normalizer.replace(mean={**normalizer.mean,'state':fresh_normalizer.mean['state']},
        std={**normalizer.std,'state':fresh_normalizer.std['state']},
        summed_variance={**normalizer.summed_variance,'state':fresh_normalizer.summed_variance['state']})
    return normalizer,policy,value


def check(checkpoint, output):
    """Actor parity on identical old channels with appended channels zero."""
    import json
    from pathlib import Path
    from brax.io import model
    from brax.training.acme import running_statistics, specs
    from brax.training.agents.ppo import networks
    from experiments.train_e3 import make_network as ideal_network
    from pass1c.train import make_network
    from pass1c.environment import frame_size, CRITIC_SIZE
    from envs.blind_rotation import frame_size as ideal_frame
    old=model.load_params(checkpoint)
    fresh=running_statistics.init_state({'state':specs.Array((3*frame_size('c_touch'),),j.dtype('float32')),
                                         'privileged_state':specs.Array((CRITIC_SIZE,),j.dtype('float32'))})
    new=expand(old,ideal_frame('c_touch'),frame_size('c_touch'),fresh)
    obs=jax.random.uniform(jax.random.PRNGKey(72),(16,3,ideal_frame('c_touch')),minval=-2,maxval=2)
    padded=j.pad(obs,((0,0),(0,0),(0,frame_size('c_touch')-ideal_frame('c_touch'))))
    old_fn=networks.make_inference_fn(ideal_network('c_touch'))(old,deterministic=True)
    new_fn=networks.make_inference_fn(make_network('c_touch'))(new,deterministic=True)
    key=jax.random.PRNGKey(0)
    a=old_fn({'state':obs.reshape(16,-1),'privileged_state':j.zeros((16,CRITIC_SIZE))},key)[0]
    b=new_fn({'state':padded.reshape(16,-1),'privileged_state':j.zeros((16,CRITIC_SIZE))},key)[0]
    error=float(j.max(j.abs(a-b)))
    result=dict(state='PASS' if error<1e-5 else 'FAIL',maximum_action_error=error,
                old_width=obs.shape[-1]*3,new_width=padded.shape[-1]*3,inputs=16,
                criterion='Same old feature values; appended zero fields; action maximum error <1e-5')
    Path(output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if result['state']!='PASS':raise RuntimeError('Warm-start actor parity failed')


if __name__=='__main__':
    import sys
    check(sys.argv[1],sys.argv[2])
