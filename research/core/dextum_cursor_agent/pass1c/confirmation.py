"""Seed-2/3 preparation: resumable Ideal teachers and own-teacher adaptation.

This entry point requires a separate frozen registration and qualifications.
It does not rent compute, launch itself, or reuse the old-physics teachers.
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


def width_for_arm(arm):
    return {'c_touch': 489, 'c_notouch': 159}[arm]


def main():
    # The kit observation layout is fixed when environment modules import.
    os.environ['DEXTUM_LEGACY_PREFIX'] = '1'
    import jax
    import numpy as np
    import vendor.aero_official.vision_env
    from brax.io import model
    from brax.training.agents.ppo import networks
    from pass1 import faithful
    from pass1.qualification import require_training_pass
    from pass1c import resumable_ppo
    from pass1c.followup import network_factory
    from pass1c.compare import MonitorGuard

    p = argparse.ArgumentParser()
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--case', required=True)
    p.add_argument('--mode', choices=('audit', 'qualify', 'train'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    a = p.parse_args()
    reg = json.loads(a.registration.read_text())
    assert reg['authority'] == 'review_section16_confirmation'
    for package, version in reg['package_versions'].items():
        assert importlib.metadata.version(package) == version, package
    assert not os.environ.get('XLA_FLAGS', '')
    root = Path(__file__).resolve().parents[1]
    for name, digest in reg['source_sha256'].items():
        assert sha(root / name) == digest, name
    for name, digest in reg['evidence_sha256'].items():
        assert sha(name) == digest, name
    job = reg['cases'][a.case]
    assert job['seed'] in (2, 3) and job['arm'] in ('c_touch', 'c_notouch')
    for prerequisite in reg['prerequisites']:
        observed = json.loads(Path(prerequisite['path']).read_text())
        assert observed['state'] == prerequisite['state'], prerequisite['path']
    config = dict(job['configuration'])
    assert config['seed'] == job['seed'] and config['num_envs'] == 8192
    teacher = job['phase'] == 'teacher'
    assert job['phase'] in ('teacher', 'adaptation')
    if teacher:
        from envs.blind_rotation import BlindRotation, CRITIC_SIZE
        from mujoco_playground._src.manipulation.aero_hand.rotate_z import domain_randomize
        env = BlindRotation(job['arm'])
        env.randomize = domain_randomize
        # Match the actual seed-1 teacher budget and 11-update reset cadence.
        assert config['num_timesteps'] == 108134400 and config['num_evals'] == 31
        assert width_for_arm(job['arm']) == 3*env.frame
    else:
        os.environ.update(DEXTUM_KIT_XML=job['model'], DEXTUM_LEGACY_PREFIX='1',
                          DEXTUM_SENSOR_LEVEL='Realistic' if job['arm'] == 'c_touch' else 'Ideal')
        assert sha(job['model']) == job['model_sha256']
        from pass1c.environment import BlindRotation, CRITIC_SIZE
        env = BlindRotation(job['arm'])
        assert config['num_timesteps'] == 20000000 and config['num_evals'] == 7
    jax.config.update('jax_default_matmul_precision', 'highest')
    assert any(device.platform == 'gpu' for device in jax.devices())
    width = 3 * env.frame
    factory = network_factory(width)
    net = factory(observation_size={'state': width, 'privileged_state': CRITIC_SIZE},
                  action_size=env.action_size)
    if a.mode == 'audit':
        from pass1 import audit_training
        if teacher:
            audit_training.BlindRotation = lambda arm: env
            audit_training.network_factory = lambda arm: factory
            options = []
        else:
            from pass1c import environment, train
            environment.BlindRotation = lambda arm: env
            train.network_factory = lambda arm: factory
            options = ['--kit']
        sys.argv = ['audit_training', '--section13', '--variant', 'barrier', '--probe',
                    '--output', str(a.output), *options]
        audit_training.main()
        path = a.output / 'result.json'
        result = json.loads(path.read_text())
        result.update(confirmation_case=a.case, registration_sha256=sha(a.registration))
        path.write_text(json.dumps(result, indent=2) + '\n')
        require_training_pass(result)
        return
    detector = json.loads(Path(job['detector']).read_text())
    require_training_pass(detector)
    assert detector['registration_sha256'] == sha(a.registration)
    assert detector['confirmation_case'] == job.get('qualification_case', a.case)
    if not teacher:
        assert detector['model_sha256'] == job['model_sha256']
    if a.mode == 'train':
        q = Path(job['qualification'])
        rate = json.loads((q / 'throughput.json').read_text())
        resume = json.loads((q / 'resume_test.json').read_text())
        assert resume['state'] == 'PASS'
        assert rate['updates'] == 10 and rate['includes_checkpoints']
        assert rate['optimizer_updates_are_discarded']
        assert resume['metadata'] == rate['metadata']
        assert rate['metadata']['registration_sha256'] == sha(a.registration)
        assert rate['metadata']['case'] == job.get('qualification_case', a.case)
        assert resume['optimizer_state_nonzero']
        # Teachers may span two capped children; adaptations must fit one.
        # These bounds must also appear in the registration before measurement.
        limit = 9600 if teacher else 4800
        assert job['maximum_projected_budget_s'] == limit
        assert rate['projected_rounded_budget_s'] <= limit
    a.output.mkdir(parents=True, exist_ok=bool(a.resume))
    if a.mode == 'train':
        (a.output/'runtime_gate.json').write_text(json.dumps(dict(
            state='PASS', projected_budget_s=rate['projected_rounded_budget_s'],
            registered_limit_s=limit, maximum_children=2 if teacher else 1,
            child_cap_s=7200, measured_updates=rate['updates'],
            underlying_one_child_label=rate['state'],
            interpretation='The frozen generic throughput report labels a 4800s one-child limit; '
                           'the registered full-budget teacher may use two capped children.'
                           if teacher else 'Original one-child criterion.'), indent=2)+'\n')
    restore = None
    if not teacher:
        parent = Path(job['teacher_run'])
        report = json.loads((parent / 'report.json').read_text())
        assert report['state'] == 'COMPLETE'
        assert report['job']['phase'] == 'teacher'
        assert (report['job']['arm'], report['job']['seed']) == (job['arm'], job['seed'])
        assert report['registration_sha256'] == sha(a.registration)
        checkpoint = parent / 'policy_108134400.pkl'
        params = model.load_params(checkpoint)
        layers = params[1]['params']
        first = next(k for k in layers if 'kernel' in layers[k])
        old_width = layers[first]['kernel'].shape[0]
        assert width >= old_width and old_width % 3 == 0

        def restore(fresh):
            import jax.numpy as j
            from pass1c.warm_start import expand
            expanded = expand(params, old_width // 3, width // 3, fresh) if width != old_width else params
            old_net = network_factory(old_width)(
                observation_size={'state': old_width, 'privileged_state': CRITIC_SIZE},
                action_size=env.action_size)
            old_fn = networks.make_inference_fn(old_net)(params, deterministic=True)
            new_fn = networks.make_inference_fn(net)(expanded, deterministic=True)
            x = jax.random.uniform(jax.random.PRNGKey(72), (16, 3, old_width//3), minval=-2, maxval=2)
            y = j.pad(x, ((0, 0), (0, 0), (0, (width-old_width)//3)))
            def action(fn, values):
                return fn({'state': values.reshape(16, -1), 'privileged_state': j.zeros((16, CRITIC_SIZE))},
                          jax.random.PRNGKey(0))[0]
            error = float(j.max(j.abs(action(old_fn, x)-action(new_fn, y))))
            parity = dict(state='PASS' if error < 1e-5 else 'FAIL', maximum_action_error=error,
                          old_width=old_width, new_width=width)
            (a.output/'warm_parity.json').write_text(json.dumps(parity, indent=2)+'\n')
            assert parity['state'] == 'PASS'
            return expanded

        (a.output / 'teacher.json').write_text(json.dumps(dict(path=str(checkpoint),
            sha256=sha(checkpoint), seed=job['seed'], arm=job['arm']), indent=2) + '\n')
    metadata = dict(registration_sha256=sha(a.registration), case=a.case,
                    configuration=config, source_sha256=reg['source_sha256'])
    started = time.monotonic()
    record = dict(state='RUNNING', job=job, registration_sha256=sha(a.registration),
                  actor_inputs=width, started_at=time.time())
    (a.output / 'config.json').write_text(json.dumps(record, indent=2) + '\n')

    def progress(steps, metrics):
        values = {key: float(np.asarray(value)) for key, value in metrics.items()
                  if np.asarray(value).size == 1}
        assert all(np.isfinite(value) for value in values.values())
        row = dict(steps=int(steps), elapsed_s=time.monotonic()-started, metrics=values)
        with (a.output / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(dict(steps=int(steps), elapsed_s=row['elapsed_s'])), flush=True)

    def save(steps, make_policy, params):
        del make_policy
        assert all(np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(params))
        path = a.output / f'policy_{int(steps)}.pkl'
        if not path.exists():
            model.save_params(path, params)

    monitor = MonitorGuard(env, networks.make_inference_fn(net), a.output / 'monitor') if a.mode == 'train' else None
    try:
        resumable_ppo.train(environment=env, network_factory=factory,
            wrap_env_fn=functools.partial(faithful.wrap_for_training, full_reset=True),
            randomization_fn=env.randomize, progress_fn=progress, policy_params_fn=save,
            restore_params=restore, run_evals=False,
            full_state_options=dict(directory=str(a.output / 'full_state'), metadata=metadata,
                resume=str(a.resume) if a.resume else None, qualify=a.mode == 'qualify', monitor=monitor), **config)
        record.update(state='COMPLETE', training_attempts=0 if a.mode == 'qualify' else 1,
                      continuation_of_same_attempt=bool(a.resume))
    except BaseException as exc:
        record.update(state='EXECUTION_FAILURE', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['wall_s'] = time.monotonic()-started
        (a.output / 'report.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    if not os.environ.get('CONTAINER_ID'):
        raise RuntimeError('Vast only')
    main()
