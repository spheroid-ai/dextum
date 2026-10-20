"""C9: frozen v2 actors on the registered E3 physical-variation bank.

Evaluation only. The accepted direct rollout and randomizer are reused. Every
saved trace must reproduce byte-for-byte on an immediate same-input re-run.
No training, engine, solver, monitor or physical-model criterion is changed.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + '.tmp')
    with pending.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--registration-sha256', required=True)
    p.add_argument('--lease', type=Path, required=True)
    p.add_argument('--arm', choices=('c_touch', 'c_notouch'), required=True)
    a = p.parse_args()
    assert sha(a.registration) == a.registration_sha256
    reg = json.loads(a.registration.read_text())
    lease = json.loads(a.lease.read_text())
    boot = dict(x.split(b'=', 1) for x in Path('/proc/1/environ').read_bytes().split(b'\0') if b'=' in x)
    assert boot[b'CONTAINER_ID'].decode() == str(reg['instance_id']) == str(lease['instance_id'])
    assert reg['state'] == 'REGISTERED_EVALUATION_ONLY'
    assert reg['authority'] == 'user_section25_C9'
    assert reg['deadline'] == lease['deadline']
    assert reg['model_version'] == 'v2' and reg['n'] == 200
    assert not os.environ.get('XLA_FLAGS', '')
    os.environ.update(CONTAINER_ID=str(reg['instance_id']), JAX_PLATFORMS='cuda,cpu',
        DEXTUM_KIT_XML=reg['model'], DEXTUM_LEGACY_PREFIX='1',
        DEXTUM_SENSOR_LEVEL='Realistic' if a.arm == 'c_touch' else 'Ideal')
    os.environ.pop('DEXTUM_DENSE_LAYOUT', None)
    root = Path(__file__).resolve().parents[2]
    for relative, digest in reg['source_sha256'].items():
        assert sha(root / relative) == digest, relative
    for path, digest in reg['input_sha256'].items():
        assert sha(path) == digest, path
    for name, version in reg['package_versions'].items():
        assert importlib.metadata.version(name) == version, name

    import jax
    import jax.numpy as j
    import numpy as np
    import vendor.aero_official.vision_env  # noqa: F401
    from brax.io import model
    from brax.training.agents.ppo import networks
    from pass1c.environment import BlindRotation, CRITIC_SIZE, WARMUP
    from pass1c.followup import network_factory
    from experiments.evaluate_e3 import CONDITIONS, HELD_OUT, make_rollout

    assert jax.default_backend() == 'gpu'
    jax.config.update('jax_default_matmul_precision', 'highest')
    assert json.loads(json.dumps(CONDITIONS)) == reg['conditions']
    assert HELD_OUT == reg['held_out']
    env = BlindRotation(a.arm)
    width = 3 * env.frame
    assert width == reg['actor_widths'][a.arm]
    net = network_factory(width)(observation_size=dict(state=width, privileged_state=CRITIC_SIZE),
                                 action_size=env.action_size)
    factory = networks.make_inference_fn(net)
    rng = jax.random.split(jax.random.PRNGKey(2026), 200)
    dr = jax.random.split(jax.random.PRNGKey(7), 200)
    keys = jax.vmap(jax.random.split)(rng)[..., 1]
    with np.load(reg['shared_bank']) as bank:
        for name, value in (('reset_keys', rng), ('dr_keys', dr), ('direct_reset_keys', keys)):
            assert name in bank and bool(j.array_equal(j.asarray(bank[name]), value)), name
    randomized, axes = env.randomize(env.mjx_model, dr)

    @jax.jit
    def evaluate(params, models, offset):
        inference = factory(params, deterministic=True)
        policy = lambda obs: inference(obs, jax.random.PRNGKey(0))[0]
        return make_rollout(env, policy, axes)(models, keys, offset)

    @jax.jit
    def summarize(trace, sign):
        # Same first-drop and success-onset semantics as evaluate_e3.summarise.
        done = trace['done'] > .5
        end = j.where(j.any(done, axis=0), j.argmax(done, axis=0), done.shape[0] - 1)
        alive = j.arange(done.shape[0])[:, None] <= end[None, :]
        onset = (trace['success'] > 0) & alive
        success = j.any(onset, axis=0)
        return dict(success=success, dropped=j.any(done, axis=0), sign=sign,
            success_step=j.where(success, j.argmax(onset, axis=0) + 1 - WARMUP, -1),
            within_steps=j.sum((trace['within'] > 0) & alive, axis=0),
            final_error=trace['error'][end, j.arange(200)])

    def array_digest(tree):
        digest = hashlib.sha256()
        for name, value in sorted(tree.items()):
            value = np.asarray(value)
            digest.update(name.encode() + b'\0')
            digest.update(value.dtype.str.encode() + b'\0')
            digest.update(json.dumps(value.shape).encode() + b'\0')
            digest.update(value.tobytes(order='C'))
        return digest.hexdigest()

    def physical_condition(condition):
        changed = {}
        if 'friction' in condition:
            ids = np.flatnonzero(env.mj_model.geom_bodyid != 0)
            changed['geom_friction'] = randomized.geom_friction.at[:, ids, 0].set(condition['friction'])
        if 'gain' in condition:
            gain = condition['gain']
            changed['actuator_gainprm'] = randomized.actuator_gainprm.at[:, :, 0].multiply(gain)
            changed['actuator_biasprm'] = randomized.actuator_biasprm.at[:, :, 1].multiply(gain)
        if 'mass' in condition:
            cube = env.cube_body
            inertia = j.asarray(env.mj_model.body_inertia[cube]) * condition['mass'] / env.mj_model.body_mass[cube]
            changed['body_mass'] = randomized.body_mass.at[:, cube].set(condition['mass'])
            changed['body_inertia'] = randomized.body_inertia.at[:, cube].set(inertia)
        return randomized.tree_replace(changed), tuple(changed)

    output = Path(reg['output']) / a.arm
    output.mkdir(parents=True, exist_ok=False)
    result = dict(state='RUNNING', run_id=reg['run_id'], model_version='v2', arm=a.arm,
        registration_sha256=a.registration_sha256, cases={}, training_attempts=0,
        task_scope=reg['scope'], analysis_gate='same-input bitwise trace repeat and frozen boundary identities')
    atomic(output / 'RESULT.json', result)
    try:
        for case in [row for row in reg['cases'] if row['arm'] == a.arm]:
            assert case['seed'] in (1, 2, 3) and case['steps'] == 20316160
            assert sha(case['checkpoint']) == case['checkpoint_sha256']
            params = model.load_params(case['checkpoint'])
            case_dir = output / case['case']
            case_dir.mkdir(exist_ok=False)
            row = dict(state='RUNNING', **case, conditions={})
            result['cases'][case['case']] = row
            for name, condition in CONDITIONS.items():
                if time.time() + reg['archive_reserve_s'] >= reg['deadline']:
                    result['state'] = 'RESOURCE_STOP'
                    result['next_case_condition'] = [case['case'], name]
                    atomic(output / 'RESULT.json', result)
                    return
                started = time.monotonic()
                models, changed = physical_condition(condition)
                offset = j.asarray(condition.get('offset', (0., 0.)), j.float32)
                trace, sign = evaluate(params, models, offset)
                jax.block_until_ready((trace, sign))
                assert all(bool(j.all(j.isfinite(value))) for value in trace.values())
                first = array_digest({**trace, 'sign': sign})
                second_trace, second_sign = evaluate(params, models, offset)
                jax.block_until_ready((second_trace, second_sign))
                second = array_digest({**second_trace, 'sign': second_sign})
                repeat = dict(state='PASS' if first == second else 'REPEAT_HASH_STOP',
                    first_sha256=first, second_sha256=second, same_compiled_program=True,
                    parameters_sha256=case['checkpoint_sha256'], changed_model_fields=list(changed))
                slug = str(list(CONDITIONS).index(name)).zfill(2)
                atomic(case_dir / (slug + '_repeat.json'), repeat)
                if first != second:
                    result.update(state='REPEAT_HASH_STOP', stopped_case=case['case'], stopped_condition=name)
                    atomic(output / 'RESULT.json', result)
                    raise RuntimeError('Same-input trace hashes differ: ' + case['case'] + '/' + name)
                values = summarize(trace, sign)
                count = int(j.sum(values['success']))
                drops = int(j.sum(values['dropped']))
                arrays = {key: np.asarray(value) for key, value in values.items()}
                path = case_dir / (slug + '_per_start.npz')
                np.savez_compressed(path, **arrays, reset_keys=np.asarray(rng), dr_keys=np.asarray(dr),
                                    direct_reset_keys=np.asarray(keys))
                trace_path = case_dir / (slug + '_trace.npz')
                np.savez_compressed(trace_path, **{k: np.asarray(v) for k, v in trace.items()}, sign=np.asarray(sign))
                row['conditions'][name] = dict(n=200, success=count, drops=drops,
                    wall_s=time.monotonic() - started, repeat=repeat,
                    per_start=str(path), per_start_sha256=sha(path),
                    trace=str(trace_path), trace_sha256=sha(trace_path))
                atomic(output / 'RESULT.json', result)
                print(json.dumps(dict(case=case['case'], condition=name, success=count, drops=drops,
                                      wall_s=row['conditions'][name]['wall_s'])), flush=True)
                del trace, second_trace, values, arrays
            row['state'] = 'COMPLETE'
            atomic(output / 'RESULT.json', result)
            del params
            gc.collect()
        result['state'] = 'COMPLETE'
        atomic(output / 'RESULT.json', result)
    except BaseException as exc:
        if result['state'] == 'RUNNING':
            result.update(state='EXECUTION_FAILURE', error=f'{type(exc).__name__}: {exc}')
            atomic(output / 'RESULT.json', result)
        raise


if __name__ == '__main__':
    main()
