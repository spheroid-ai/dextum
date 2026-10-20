"""Registered O1 rotation experiments on the unchanged v2 task/mechanics.

This command never rents compute or waives a qualification. Audit, throughput,
fresh-process resume, and training use the same frozen encoder/environment.
"""
import argparse
import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if not os.environ.get('CONTAINER_ID'):
        raise RuntimeError('Cloud only')
    parser = argparse.ArgumentParser()
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--mode', choices=('audit', 'qualify', 'resume_check', 'train'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    reg = json.loads(args.registration.read_text())
    assert reg['authority'] == 'user_oracle_dextum_full_state_o1_rotation'
    root = Path(reg['base_release'])
    for name, digest in reg['base_source_sha256'].items():
        assert sha(root/name) == digest, name
    for name, digest in reg['oracle_source_sha256'].items():
        assert sha(Path(__file__).parent/name) == digest, name
    for package, version in reg['package_versions'].items():
        assert importlib.metadata.version(package) == version, package
    assert not os.environ.get('XLA_FLAGS', '')
    assert sha(reg['model']) == reg['model_sha256']
    os.environ.update(DEXTUM_KIT_XML=reg['model'], DEXTUM_SENSOR_LEVEL='Ideal',
                      DEXTUM_LEGACY_PREFIX='1')
    import jax
    import jax.numpy as j
    import numpy as np
    from brax.io import model
    from brax.training.agents.ppo import networks
    from pass1 import faithful
    from .calibration import require_pass as require_training_pass
    from pass1c import resumable_ppo
    from pass1c.compare import MonitorGuard
    from pass1c.environment import EPISODE
    from .environment import OracleRotation
    from .networks import network_factory, capacity_contract
    job = reg['cases'][args.case]
    assert job['seed'] in (1, 2, 3)
    assert job['initialization'] == 'fresh_random_parameters_optimizer_normalizer'
    assert 'checkpoint' not in job
    env = OracleRotation(job['arm'])
    jax.config.update('jax_default_matmul_precision', 'highest')
    assert any(d.platform == 'gpu' for d in jax.devices())
    abstract = jax.eval_shape(env.reset, jax.random.PRNGKey(0))
    shapes = {key: tuple(value.shape) for key, value in abstract.obs.items()}
    assert {key: list(value) for key, value in shapes.items()} == reg['observation_shapes']
    capacity = capacity_contract(shapes, env.action_size)
    assert capacity == reg['capacity']
    factory = network_factory(job['arm'], shapes, capacity)
    net = factory(shapes, env.action_size)
    if args.mode == 'audit':
        from .calibration import audit
        result = audit(env, args.output, sha(args.registration), factory)
        result.update(oracle_registration_sha256=sha(args.registration), oracle_case=args.case,
            native_cpu_warmup='Omitted non-gating legacy diagnostic under the GPU-only work order')
        (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        require_training_pass(result)
        return
    detector = json.loads(Path(job['detector']).read_text())
    require_training_pass(detector)
    assert detector['oracle_registration_sha256'] == sha(args.registration)
    assert detector['oracle_case'] == args.case
    assert detector['model_sha256'] == reg['model_sha256']
    qualification = Path(job['qualification'])
    if args.mode == 'train':
        rate = json.loads((qualification/'full_state/throughput.json').read_text())
        replay = json.loads((Path(job['resume_check'])/'full_state/process_resume.json').read_text())
        assert rate['metadata']['registration_sha256'] == sha(args.registration)
        assert rate['metadata']['case'] == args.case
        assert rate['updates'] == 10 and rate['includes_checkpoints']
        assert rate['projected_rounded_budget_s'] <= reg['maximum_projected_budget_s']
        assert replay['state'] == 'PASS' and replay['fresh_process']
        assert replay['checkpoint'] == str(qualification/'full_state/resume_test.pkl.gz')
    args.output.mkdir(parents=True, exist_ok=bool(args.resume))
    config = {**reg['configuration'], 'seed': job['seed']}
    assert config['episode_length'] == EPISODE
    metadata = dict(registration_sha256=sha(args.registration), case=args.case,
                    configuration=config, source_sha256=reg['oracle_source_sha256'],
                    model_sha256=reg['model_sha256'],
                    checkpoint_interval_updates=reg.get('checkpoint_interval_updates',1))
    record = dict(state='RUNNING', case=job, mode=args.mode, started_at=time.time(),
        registration_sha256=sha(args.registration), observation_shapes=reg['observation_shapes'],
        capacity=capacity, training_attempts=0)
    started = time.monotonic()

    def progress(steps, metrics):
        values = {k: float(np.asarray(v)) for k, v in metrics.items() if np.asarray(v).size == 1}
        if not all(np.isfinite(v) for v in values.values()):
            raise RuntimeError('Nonfinite PPO metric')
        row = dict(steps=int(steps), elapsed_s=time.monotonic()-started, metrics=values)
        with (args.output/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps(dict(case=args.case, steps=int(steps))), flush=True)

    def save(steps, unused, params):
        assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(params))
        path = args.output/f'policy_{int(steps)}.pkl'
        if not path.exists(): model.save_params(path, params)

    options = dict(directory=str(args.output/'full_state'), metadata=metadata,
        resume=str(args.resume) if args.resume else None,
        checkpoint_interval_updates=reg.get('checkpoint_interval_updates',1),
        qualify=args.mode == 'qualify', monitor=None)
    if args.mode == 'resume_check':
        assert args.resume is None
        options.update(resume=str(qualification/'full_state/resume_test.pkl.gz'),
                       expected_next=str(qualification/'full_state/expected_next.pkl.gz'))
    if args.mode == 'train':
        options['monitor'] = MonitorGuard(env, networks.make_inference_fn(net), args.output/'monitor')
    try:
        resumable_ppo.train(environment=env, network_factory=factory,
            wrap_env_fn=functools.partial(faithful.wrap_for_training, full_reset=True),
            randomization_fn=env.randomize, progress_fn=progress, policy_params_fn=save,
            restore_params=None, run_evals=False, full_state_options=options, **config)
        record.update(state='COMPLETE', training_attempts=int(args.mode == 'train'))
    except BaseException as exc:
        record.update(state='EXECUTION_FAILURE', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['wall_s'] = time.monotonic()-started
        (args.output/'report.json').write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__': main()
