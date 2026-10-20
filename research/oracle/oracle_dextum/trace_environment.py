"""Cloud CPU shape/field audit of the real v2 rotation environment.

This traces reset and step through JAX without training or changing physics.
It is not the matched-transition detector, throughput or resume qualification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    if not os.environ.get('CONTAINER_ID'):
        raise RuntimeError('Cloud-only interface audit')
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
    if digest != 'a2e0c670506cb69267bac11e80fa24c467a782cca3bedb901290adc64444c924':
        raise ValueError('This initial interface audit is registered on accepted v2 only')
    os.environ.update(DEXTUM_KIT_XML=str(args.model), DEXTUM_SENSOR_LEVEL='Ideal',
                      DEXTUM_LEGACY_PREFIX='1')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    report = dict(state='RUNNING', model_sha256=digest, training_attempts=0,
                  source_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                      for f in Path(__file__).parent.glob('*.py')})
    try:
        import jax
        import jax.numpy as j
        from .environment import OracleRotation
        from .networks import capacity_contract
        env = OracleRotation('oracle_structured_full')
        abstract = jax.eval_shape(env.reset, jax.random.PRNGKey(0))
        step = jax.eval_shape(env.step, abstract, j.zeros(env.action_size))
        reset_shapes = [(tuple(x.shape), str(x.dtype)) for x in jax.tree.leaves(abstract)]
        step_shapes = [(tuple(x.shape), str(x.dtype)) for x in jax.tree.leaves(step)]
        if jax.tree.structure(abstract) != jax.tree.structure(step) or reset_shapes != step_shapes:
            raise ValueError('Reset and step pytrees/shapes differ')
        shapes = {key: tuple(value.shape) for key, value in abstract.obs.items()}
        capacity = capacity_contract(shapes, env.action_size)
        report.update(state='PASS', observation_shapes=shapes, capacity=capacity,
            action_size=env.action_size, reset_step_pytree_match=True,
            scope='Actual accepted model and environment traced through reset/step; no physics transition or policy optimization measured')
    except BaseException as exc:
        report.update(state='FAIL', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['wall_s'] = time.time()-started
        (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
