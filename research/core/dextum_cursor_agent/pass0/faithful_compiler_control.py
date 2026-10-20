"""Remote direct-evaluator compiler control, independent of the D5 gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
if not os.environ.get('CONTAINER_ID'):
    raise RuntimeError('Vast.ai only')

import jax
import jax.numpy as jnp
import numpy as np
import vendor.aero_official.vision_env  # noqa: F401
from envs.blind_rotation import BlindRotation
from experiments.evaluate_e3 import load_policy, make_rollout, summarise
from mujoco_playground._src.manipulation.aero_hand.rotate_z import domain_randomize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=['default', 'no_constant_folding'], required=True)
    a = parser.parse_args()
    flags = {'default': '', 'no_constant_folding': '--xla_disable_hlo_passes=constant_folding'}
    assert os.environ.get('XLA_FLAGS', '') == flags[a.variant]
    a.output.mkdir(parents=True, exist_ok=False)
    jax.config.update('jax_default_matmul_precision', 'highest')
    env = BlindRotation('c_touch')
    policy, steps = load_policy(Path('runs/e3p1-s1'), 'c_touch', steps=108134400)
    rng = jax.random.split(jax.random.PRNGKey(2026), 200)
    keys = jax.vmap(jax.random.split)(rng)[..., 1]
    dr = jax.random.split(jax.random.PRNGKey(7), 200)
    model, axes = domain_randomize(env.mjx_model, dr)
    bank = dict(reset_keys=np.asarray(rng), direct_reset_keys=np.asarray(keys), dr_keys=np.asarray(dr))
    np.savez(a.output/'start_bank.npz', **bank)
    print('Compiling faithful direct evaluator: '+a.variant, flush=True)
    trace, signs = make_rollout(env, policy, axes)(model, keys, jnp.zeros(2, jnp.float32))
    trace = jax.tree.map(np.asarray, trace)
    values = summarise(trace, np.asarray(signs))
    finite = all(np.isfinite(v).all() for v in trace.values())
    np.savez(a.output/'outcomes.npz', **values)
    result = dict(state='MEASURED' if finite else 'NONFINITE', variant=a.variant,
        n=200, steps=steps, finite=bool(finite), flags=flags[a.variant], precision='highest',
        devices=[str(d) for d in jax.devices()], jax_version=jax.__version__,
        checkpoint_sha256=hashlib.sha256(Path('runs/e3p1-s1/c_touch/policy_108134400.pkl').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path('experiments/evaluate_e3.py').read_bytes()).hexdigest(),
        bank_sha256={k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in bank.items()},
        success_count=int(values['success'].sum()), outcomes=values['success'].tolist(),
        sign=np.asarray(signs).tolist())
    (a.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('outcomes','sign')}), flush=True)


if __name__ == '__main__': main()
