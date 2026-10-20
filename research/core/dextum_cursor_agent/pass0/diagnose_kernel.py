"""Cloud diagnostic: compare each transition from identical current states."""
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
from envs.blind_rotation import BlindRotation,WARMUP,EPISODE
from experiments.evaluate_e3 import step_one
from mujoco_playground._src.manipulation.aero_hand.rotate_z import domain_randomize
from pass0.faithful import wrap_for_training,MODEL_KEY


def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
    jax.config.update('jax_default_matmul_precision','highest')
    env=BlindRotation('c_touch');n=16
    rng=jax.random.split(jax.random.PRNGKey(2026),n)
    dr=jax.random.split(jax.random.PRNGKey(7),n)
    model,axes=domain_randomize(env.mjx_model,dr)
    fixed=wrap_for_training(env,episode_length=EPISODE,randomization_fn=functools.partial(domain_randomize,rng=dr))
    initial=jax.jit(fixed.reset)(rng)
    step_v=jax.vmap(functools.partial(step_one,env),in_axes=(axes,0,0))
    zero=jnp.zeros((n,env.action_size))
    @jax.jit
    def probe(state):
        def body(s,_):
            info={k:v for k,v in s.info.items() if k!=MODEL_KEY}
            raw=step_v(s.info[MODEL_KEY],s.replace(info=info),zero)
            nxt=fixed.step(s,zero)
            error=jnp.max(jnp.abs(raw.data.qpos-nxt.data.qpos),axis=-1)
            # Native terminal state versus auto-reset state is intentionally
            # different; report those separately rather than compare them.
            comparable=(nxt.done<.5)&(raw.done<.5)
            return nxt,(error,comparable)
        return jax.lax.scan(body,state,None,length=WARMUP)[1]
    error,mask=map(np.asarray,probe(initial))
    r=dict(state='DIAGNOSTIC_ONLY',max_comparable_qpos_error=float(error[mask].max()),
           compared_transitions=int(mask.sum()),terminal_transitions=int((~mask).sum()),
           max_by_step=np.max(np.where(mask,error,0),axis=1).tolist(),
           scope='Each transition receives identical current data/model; no replacement of D5 result')
    (out/'result.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)


if __name__=='__main__':main()
