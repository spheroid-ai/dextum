"""Remote diagnostic only: isolate reset numerics from rollout differences."""
import functools
import json
import os
from pathlib import Path
import sys
if not os.environ.get('CONTAINER_ID'):
    raise RuntimeError('Cloud only')
import jax
import jax.numpy as jnp
import numpy as np
import vendor.aero_official.vision_env
from envs.blind_rotation import BlindRotation, WARMUP, EPISODE
from experiments.evaluate_e3 import reset_one, step_one
from mujoco_playground._src.manipulation.aero_hand.rotate_z import domain_randomize
from pass0.faithful import wrap_for_training, MODEL_KEY


def main():
    out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=False)
    jax.config.update('jax_default_matmul_precision','highest')
    env=BlindRotation('c_touch'); n=16
    dr=jax.random.split(jax.random.PRNGKey(7),n)
    rng=jax.random.split(jax.random.PRNGKey(2026),n)
    keys=jax.vmap(jax.random.split)(rng)[...,1]
    model,axes=domain_randomize(env.mjx_model,dr)
    wrapped=wrap_for_training(env,episode_length=EPISODE,randomization_fn=functools.partial(domain_randomize,rng=dr))
    initial=jax.jit(wrapped.reset)(rng)
    zeros=jnp.zeros((n,env.action_size))
    direct_reset=jax.jit(jax.vmap(functools.partial(reset_one,env),in_axes=(axes,0,None)))(model,keys,jnp.zeros(2))
    initial_error={}
    for name in ('qpos','qvel','qacc','qacc_warmstart','ctrl'):
        initial_error[name]=float(np.max(np.abs(np.asarray(getattr(initial.data,name))-np.asarray(getattr(direct_reset.data,name)))))
    (out/'initial.json').write_text(json.dumps(initial_error,indent=2)); print(initial_error,flush=True)
    step_v=jax.vmap(functools.partial(step_one,env),in_axes=(axes,0,0))
    common=direct_reset.replace(data=initial.data)
    @jax.jit
    def direct(m,state):
        def body(s,_):
            s=step_v(m,s,zeros)
            return s,s.data.qpos
        return jax.lax.scan(body,state,None,length=WARMUP)[1]
    @jax.jit
    def train(state):
        def body(s,_):
            s=wrapped.step(s,zeros)
            return s,s.data.qpos
        return jax.lax.scan(body,state,None,length=WARMUP)[1]
    a=np.asarray(train(initial)); b=np.asarray(direct(model,common))
    err=np.abs(a-b)
    r=dict(state='DIAGNOSTIC_ONLY',same_initial_data=True,initial_error=initial_error,
           max_qpos_error=float(err.max()),max_by_step=err.max((1,2)).tolist(),
           note='Does not replace or relax the registered D5 qualification')
    (out/'result.json').write_text(json.dumps(r,indent=2)); print(json.dumps(r),flush=True)


if __name__=='__main__': main()
