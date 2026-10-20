"""Lane D2, 20M visited transitions, aggregate DAgger and exact continuation."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if not os.environ.get('CONTAINER_ID'):
        raise RuntimeError('GPU cloud only')
    os.environ['JAX_PLATFORMS'] = 'cuda,cpu'  # CPU device for host callbacks; model execution must remain on GPU.
    p = argparse.ArgumentParser()
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--mode', choices=('audit', 'qualify', 'resume_check', 'train', 'export'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    a = p.parse_args()
    reg = json.loads(a.registration.read_text())
    assert reg['authority'] == 'work_order_2026_10_14_lane_D2' and reg['state'] == 'REGISTERED'
    root = Path(__file__).resolve().parents[2]
    for name, digest in reg['source_sha256'].items():
        assert sha(root / name) == digest, name
    for name, digest in reg['inputs_sha256'].items():
        assert sha(name) == digest, name
    for name, version in reg['package_versions'].items():
        assert importlib.metadata.version(name) == version, name
    assert not os.environ.get('XLA_FLAGS', '')
    os.environ.update(DEXTUM_KIT_XML=reg['model'], DEXTUM_LEGACY_PREFIX='1', DEXTUM_SENSOR_LEVEL='Realistic')
    import jax
    import jax.numpy as j
    import numpy as np
    from brax.io import model
    from brax.training.agents.ppo import networks
    import vendor.aero_official.vision_env  # Establish pinned playground/menagerie paths before env import.
    from pass1c.environment import CRITIC_SIZE
    from pass1c.followup import network_factory
    from pass1c import resume_loop
    from pass1c.compare import MonitorGuard
    from workorder14.lane_d.sensorless import SensorlessRotation
    from workorder14.lane_d.dagger_core import Dagger
    from workorder14.lane_d import learner
    assert jax.default_backend() == 'gpu' and all(d.platform == 'gpu' for d in jax.devices())
    jax.config.update('jax_default_matmul_precision', 'highest')
    teacher_params = model.load_params(reg['teacher'])
    net = network_factory(1089)(observation_size=dict(state=1089, privileged_state=CRITIC_SIZE), action_size=31)
    env = SensorlessRotation(networks.make_inference_fn(net)(teacher_params, deterministic=True))
    metadata = dict(registration_sha256=sha(a.registration), instance_id=os.environ['CONTAINER_ID'],
                    model_sha256=sha(reg['model']), teacher_sha256=sha(reg['teacher']),
                    source_sha256=reg['source_sha256'], configuration=reg['configuration'])
    if a.mode == 'audit':
        from workorder14.lane_d.dagger_detector import audit
        audit(env, a.output, metadata)
        return
    if a.mode == 'export':
        from workorder14.lane_d.export_student import export
        export(Path(reg['training']) / 'student.pkl', a.output, env.mj_model.actuator_ctrlrange)
        return
    detector = json.loads(Path(reg['detector']).read_text())
    assert detector['state'] == 'PASS' and detector['metadata'] == metadata
    config = reg['configuration']
    assert config['environment_steps'] == 20000000 and config['worlds'] == 8192
    assert config['aggregate_retention'] == 'all_collected_examples_no_eviction'
    runner = Dagger(env, config)
    qualification = Path(reg['qualification'])
    if a.mode == 'train':
        throughput = json.loads((qualification / 'throughput.json').read_text())
        replay = json.loads(Path(reg['fresh_process_resume']).read_text())
        assert throughput['updates'] == 10 and throughput['includes_checkpoints']
        assert throughput['optimizer_state_nonzero']
        assert throughput['metadata'] == replay['metadata'] == metadata
        assert throughput['projected_rounded_budget_s'] <= 9600
        assert replay['state'] == 'PASS' and replay['fresh_process']
        assert replay['loss']['maximum_absolute_difference'] == 0
        assert replay['full_next_state']['maximum_absolute_difference'] == 0
    a.output.mkdir(parents=True, exist_ok=bool(a.resume))
    full = a.output / 'full_state'
    full.mkdir(exist_ok=True)
    if a.mode == 'resume_check':
        checkpoint = qualification / 'resume_test.pkl.gz'
        expected = resume_loop.load(qualification / 'expected_next.pkl.gz', metadata)
        payload = resume_loop.load(checkpoint, metadata)
        actual, actual_loss = runner.advance(payload)
        row = dict(metadata=metadata, fresh_process=True, checkpoint=str(checkpoint),
                   loss=resume_loop.compare(expected['loss'], actual_loss),
                   full_next_state=resume_loop.compare(expected['payload'], actual))
        row['state'] = 'PASS' if row['loss']['passed'] and row['full_next_state']['passed'] and (
            row['loss']['maximum_absolute_difference'] == row['full_next_state']['maximum_absolute_difference'] == 0) else 'FAIL'
        (full / 'process_resume.json').write_text(json.dumps(row, indent=2) + '\n')
        assert row['state'] == 'PASS'
        return
    payload = resume_loop.load(a.resume, metadata) if a.resume else runner.initial()
    payload = runner.prepare(payload)
    save = lambda path, value: resume_loop.save(path, value, metadata)
    report = dict(state='RUNNING', metadata=metadata, model_version='v2',
                  aggregate_capacity=runner.capacity, dataset_bytes=runner.capacity * 100 * 4,
                  started_at=time.time(), continuation_of_same_attempt=bool(a.resume))
    (a.output / ('continuation_' + str(int(time.time())) + '.json' if a.resume else 'config.json')).write_text(
        json.dumps(report, indent=2) + '\n')
    started = time.monotonic()
    if a.mode == 'qualify':
        payload, _ = runner.advance(payload)
        payload = runner.prepare(payload)
        save(full / 'resume_test.pkl.gz', payload)
        expected, loss = runner.advance(payload)
        save(full / 'expected_next.pkl.gz', dict(payload=expected, loss=loss))
        del expected, loss
        # This state is measurement-only, discarded after the ten timed updates.
        times = []
        for _ in range(10):
            tick = time.monotonic()
            payload, loss = runner.advance(payload)
            payload = runner.prepare(payload)
            save(full / 'latest.pkl', payload)
            times.append(time.monotonic() - tick)
            assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(loss))
        rate = 10 * runner.block / sum(times)
        result = dict(state='PASS' if runner.capacity / rate <= 9600 else 'TOO_SLOW', metadata=metadata,
            updates=10, includes_checkpoints=True, optimizer_updates_are_discarded=True,
            env_steps_per_s=rate, projected_rounded_budget_s=runner.capacity / rate,
            rounded_budget_steps=runner.capacity, timings_s=times, training_attempts=0,
            optimizer_state_nonzero=any(np.any(np.asarray(x) != 0) for x in jax.tree.leaves(payload['optimizer'])))
        (full / 'throughput.json').write_text(json.dumps(result, indent=2) + '\n')
        assert result['state'] == 'PASS'
        return
    monitor = MonitorGuard(env, learner.policy, a.output / 'monitor')
    eval_updates = set(reg['evaluation_updates'])
    try:
        save(full / 'latest.pkl', payload)
        if not a.resume:
            monitor(0, payload['params'])
        while payload['count'] < runner.capacity:
            tick = time.monotonic()
            payload, loss = runner.advance(payload)
            if payload['count'] < runner.capacity:
                payload = runner.prepare(payload)
            save(full / 'latest.pkl', payload)
            assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(loss))
            metrics = dict(steps=payload['count'], updates=payload['updates'],
                           aggregate_examples=payload['count'],
                           loss=float(loss['loss']), elapsed_s=time.monotonic() - started,
                           checkpoint_inclusive_update_s=time.monotonic() - tick)
            with (a.output / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(metrics) + '\n')
            print(json.dumps(metrics), flush=True)
            if payload['updates'] in eval_updates or payload['count'] == runner.capacity:
                model.save_params(a.output / f"student_{payload['count']}.pkl", payload['params'])
                monitor(payload['count'], payload['params'])
        model.save_params(a.output / 'student.pkl', payload['params'])
        report.update(state='COMPLETE', steps=payload['count'], training_attempts=1)
    except BaseException as exc:
        report.update(state='EXECUTION_FAILURE', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['wall_s'] = time.monotonic() - started
        (a.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
