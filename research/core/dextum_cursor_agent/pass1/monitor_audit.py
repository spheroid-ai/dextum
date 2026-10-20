"""Compare the actual Brax joint reset/unroll evaluator with materialized reset."""
import argparse
import functools
import hashlib
import importlib
import json
import os
from pathlib import Path
if not os.environ.get('CONTAINER_ID'):
    raise RuntimeError('Vast.ai only')
import jax
import numpy as np
import vendor.aero_official.vision_env
from brax.envs.wrappers import training
from brax.io import model
from brax.training import acting
from brax.training.agents.ppo import networks
from envs.blind_rotation import BlindRotation, EPISODE
from pass0.train_e3 import make_network
from pass0.review_gate import exact_mcnemar
from mujoco_playground._src.manipulation.aero_hand.rotate_z import domain_randomize


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variant',choices=['baseline','barrier'],required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    physics=importlib.import_module('pass0.faithful' if a.variant=='baseline' else 'pass1.faithful')
    jax.config.update('jax_default_matmul_precision','highest')
    params=model.load_params(a.checkpoint)
    factory=functools.partial(networks.make_inference_fn(make_network('c_touch')),deterministic=True)
    dr=jax.random.split(jax.random.PRNGKey(7),200)
    wrapped=physics.wrap_for_training(BlindRotation('c_touch'),episode_length=EPISODE,
                randomization_fn=functools.partial(domain_randomize,rng=dr))
    evaluator=acting.Evaluator(wrapped,factory,200,EPISODE,1,jax.random.PRNGKey(0))
    key=jax.random.PRNGKey(2026)
    print('Actual Brax joint reset/rollout',a.variant,flush=True)
    joint=evaluator._generate_eval_unroll(params,key)
    joint_metrics=jax.tree.map(np.asarray,joint.info['eval_metrics'].episode_metrics)
    split_env=training.EvalWrapper(wrapped)
    initial=jax.jit(split_env.reset)(jax.random.split(key,200))

    @jax.jit
    def split(state,policy_params,rollout_key):
        return acting.generate_unroll(split_env,state,factory(policy_params),rollout_key,EPISODE)[0]

    print('Same evaluator with reset materialized before rollout',flush=True)
    separate=split(initial,params,key)
    split_metrics=jax.tree.map(np.asarray,separate.info['eval_metrics'].episode_metrics)
    x=joint_metrics['e3/success']>0;y=split_metrics['e3/success']>0
    b=int(np.sum(x&~y));c=int(np.sum(y&~x))
    result=dict(state='MEASURED',variant=a.variant,n=200,reset_seed=2026,dr_seed=7,
        checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        wrapper_sha256=hashlib.sha256(Path(physics.__file__).read_bytes()).hexdigest(),
        joint_success=int(x.sum()),materialized_success=int(y.sum()),joint_only=b,materialized_only=c,
        agreement=float(np.mean(x==y)),exact_mcnemar_p=exact_mcnemar(b,c),
        aggregate_gap=float(abs(x.mean()-y.mean())),finite=bool(all(np.isfinite(v).all() for d in (joint_metrics,split_metrics) for v in d.values())),
        shared_bank_sha256={'reset':hashlib.sha256(np.asarray(jax.random.split(key,200)).tobytes()).hexdigest(),
                            'dr':hashlib.sha256(np.asarray(dr).tobytes()).hexdigest()},
        interpretation='Same actual Brax EvalWrapper and generate_unroll; compare joint compilation versus materialized reset. No policy updates.')
    np.savez(a.output/'metrics.npz',**{'joint_'+k.replace('/','_'):v for k,v in joint_metrics.items()},
             **{'materialized_'+k.replace('/','_'):v for k,v in split_metrics.items()})
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
