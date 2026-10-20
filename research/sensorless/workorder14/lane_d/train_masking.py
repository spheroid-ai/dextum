"""Lane D1: v2 20M masking continuations with unchanged PPO and monitor.

All measurements require a GPU host. Each case has its own actual-path
detector, checkpoint-inclusive throughput and fresh-process exact resume.
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


def require_detector(record):
    calibrated = record.get('criterion_version') == 'work_order_2026_10_14_calibrated'
    limit = max(1e-4, 10 * record['direct_direct_median']) if calibrated else 1e-4
    assert record['state'] == 'PASS' and record['finite']
    assert record['actual_ppo_training_path']
    assert record['full_reset_exercised'] and record['full_reset_preserves_physics']
    assert record['matched_step_median'] <= limit


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
    assert reg['state'] == 'REGISTERED' and reg['authority'] == 'work_order_2026_10_14_lane_D1'
    root = Path(__file__).resolve().parents[2]
    for name, digest in reg['source_sha256'].items():
        assert sha(root / name) == digest, name
    for name, digest in reg['inputs_sha256'].items():
        assert sha(name) == digest, name
    for package, version in reg['package_versions'].items():
        assert importlib.metadata.version(package) == version, package
    assert not os.environ.get('XLA_FLAGS', '')
    job = reg['cases'][args.case]
    assert job['seed'] == 1 and job['arm'] == 'c_touch'
    assert job['model_sha256'] == reg['accepted_model_sha256'] == sha(job['model'])
    assert sha(job['checkpoint']) == job['checkpoint_sha256']
    config = dict(job['configuration'])
    assert config['seed'] == job['seed'] and config['num_envs'] == 8192
    assert config['num_timesteps'] == 20000000 and config['num_evals'] == 7
    level = 'Realistic'
    assert job['sensor_level'] == level
    os.environ.update(DEXTUM_KIT_XML=job['model'], DEXTUM_LEGACY_PREFIX='1', DEXTUM_SENSOR_LEVEL=level)
    # CPU is initialized only so JAX host diagnostic callbacks have a device.
    # CUDA remains mandatory/default for every simulation and optimizer array.
    os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
    import jax
    assert jax.default_backend() == 'gpu' and all(device.platform == 'gpu' for device in jax.devices())
    import jax.numpy as j
    import numpy as np
    import vendor.aero_official.vision_env
    from brax.io import model
    from brax.training.agents.ppo import networks
    from pass1 import faithful
    from pass1.qualification import require_training_pass
    from pass1c import resumable_ppo
    from pass1c.environment import CRITIC_SIZE
    from pass1c.masking import BlindRotation
    from pass1c.followup import network_factory
    from pass1c.compare import MonitorGuard
    assert job['intervention'] in ({'mask': 'none'}, {'mask': 'normal_only'}, {'mask': 'palm'})
    env = BlindRotation(job['arm'], intervention=job['intervention'])
    jax.config.update('jax_default_matmul_precision', 'highest')
    assert jax.default_backend() == 'gpu' and all(device.platform == 'gpu' for device in jax.devices())
    width = 3 * env.frame
    factory = network_factory(width)
    net = factory(observation_size=dict(state=width, privileged_state=CRITIC_SIZE), action_size=env.action_size)
    digest = sha(args.registration)
    if args.mode == 'audit':
        # The existing calibrated instrument compares three GPU paths. The
        # inherited native-CPU warm-up is non-gating and excluded by the
        # user's no-CPU-run instruction; no detector criterion is changed.
        from workorder14.lane_d.calibration import audit
        result = audit(env, args.output, digest)
        path = args.output / 'result.json'
        result.update(lane_d_case=args.case, registration_sha256=digest,
                      instance_id=os.environ['CONTAINER_ID'],
                      native_cpu_warmup='Omitted: non-gating legacy diagnostic; user requested no CPU runs')
        path.write_text(json.dumps(result, indent=2) + '\n')
        require_detector(result)
        return
    detector = json.loads(Path(job['detector']).read_text())
    require_detector(detector)
    assert detector['registration_sha256'] == digest and detector['lane_d_case'] == args.case
    assert detector['model_sha256'] == job['model_sha256']
    assert detector['instance_id'] == os.environ['CONTAINER_ID']
    qualification = Path(job['qualification'])
    metadata = dict(registration_sha256=digest, case=args.case, configuration=config,
                    source_sha256=reg['source_sha256'], instance_id=os.environ['CONTAINER_ID'],
                    checkpoint_sha256=job['checkpoint_sha256'], model_sha256=job['model_sha256'])
    if args.mode == 'train':
        rate = json.loads((qualification / 'throughput.json').read_text())
        replay = json.loads((qualification / 'resume_test.json').read_text())
        process = json.loads(Path(job['fresh_process_resume']).read_text())
        assert replay['state'] == 'PASS' and replay['optimizer_state_nonzero']
        assert process['state'] == 'PASS' and process['fresh_process']
        assert process['loss']['maximum_absolute_difference'] == 0
        assert process['full_next_state']['maximum_absolute_difference'] == 0
        assert process['checkpoint'] == str(qualification / 'resume_test.pkl.gz')
        assert rate['metadata'] == replay['metadata']
        assert all(rate['metadata'][key] == value for key, value in metadata.items())
        assert rate['updates'] == 10 and rate['includes_checkpoints'] and rate['optimizer_updates_are_discarded']
        assert rate['projected_rounded_budget_s'] <= job['maximum_projected_budget_s'] == 9600
    if args.mode == 'resume_check':
        assert args.resume == qualification / 'resume_test.pkl.gz'
    elif args.mode == 'qualify':
        assert args.resume is None
    args.output.mkdir(parents=True, exist_ok=bool(args.resume))
    params = model.load_params(job['checkpoint'])
    layers = params[1]['params']
    first = next(key for key in layers if 'kernel' in layers[key])
    old_width = layers[first]['kernel'].shape[0]
    assert old_width == width == 1089

    def restore(fresh):
        del fresh
        # Same architecture, current layout, parameters and normalizer. Each
        # continuation starts a fresh optimizer, as the earlier masking row.
        return params

    started = time.monotonic()
    report = dict(state='RUNNING', job=job, registration_sha256=digest, actor_inputs=width,
                  model_version='v2', instance_id=os.environ['CONTAINER_ID'])
    (args.output / ('continuation_' + str(int(time.time())) + '.json' if args.resume else 'config.json')).write_text(
        json.dumps(report, indent=2) + '\n')
    def progress(steps, metrics):
        values = {key: float(np.asarray(value)) for key, value in metrics.items() if np.asarray(value).size == 1}
        assert all(np.isfinite(value) for value in values.values())
        row = dict(steps=int(steps), elapsed_s=time.monotonic() - started, metrics=values)
        with (args.output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)

    def save(steps, make_policy, values):
        del make_policy
        assert all(np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(values))
        path = args.output / f'policy_{int(steps)}.pkl'
        if not path.exists():
            model.save_params(path, values)

    monitor = MonitorGuard(env, networks.make_inference_fn(net), args.output / 'monitor') if args.mode == 'train' else None
    options = dict(directory=str(args.output / 'full_state'), metadata=metadata,
        resume=str(args.resume) if args.resume else None, qualify=args.mode == 'qualify', monitor=monitor)
    if args.mode == 'resume_check':
        options['expected_next'] = str(qualification / 'expected_next.pkl.gz')
    try:
        resumable_ppo.train(environment=env, network_factory=factory,
            wrap_env_fn=functools.partial(faithful.wrap_for_training, full_reset=True),
            randomization_fn=env.randomize, progress_fn=progress, policy_params_fn=save,
            restore_params=restore, run_evals=False, full_state_options=options, **config)
        report.update(state='COMPLETE', training_attempts=int(args.mode == 'train' and job['intervention']['mask'] != 'none'))
    except BaseException as exc:
        report.update(state='EXECUTION_FAILURE', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['wall_s'] = time.monotonic() - started
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
