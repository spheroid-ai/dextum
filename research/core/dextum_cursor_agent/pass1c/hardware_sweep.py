"""Section 16 isolated sensor errors, fixed Ideal actor and 59% cutoff."""
import argparse
import functools
import hashlib
import json
import os
import time
from pathlib import Path


def main():
    import jax
    import jax.numpy as j
    import numpy as np
    import vendor.aero_official.vision_env  # registers the pinned environment
    from brax.io import model
    from brax.training.agents.ppo import networks
    from pass1c.train import make_network
    from pass1c.ideal_errors import BlindRotation, FIELDS, make_rollout
    from experiments.evaluate_e3 import summarise
    from pass0.review_gate import exact_mcnemar

    p = argparse.ArgumentParser()
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    reg = json.loads(a.registration.read_text())
    assert reg['authority'] == 'user_review_section16_isolated_errors'
    assert reg['fixed_reference_count'] == 128 and reg['crossing_count'] == 118
    assert not os.environ.get('XLA_FLAGS', '')
    for name, expected in reg['input_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    checkpoint = Path(reg['checkpoint'])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == reg['checkpoint_sha256']
    assert hashlib.sha256(Path(os.environ['DEXTUM_KIT_XML']).read_bytes()).hexdigest() == reg['model_sha256']
    os.environ['DEXTUM_SENSOR_LEVEL'] = 'Ideal'
    a.output.mkdir(parents=True, exist_ok=False)
    jax.config.update('jax_default_matmul_precision', 'highest')
    params = model.load_params(checkpoint)
    factory = functools.partial(networks.make_inference_fn(make_network('c_touch')), deterministic=True)
    env = BlindRotation()
    dr = jax.random.split(jax.random.PRNGKey(7), 200)
    keys = jax.vmap(jax.random.split)(jax.random.split(jax.random.PRNGKey(2026), 200))[..., 1]
    randomized, axes = env.randomize(env.mjx_model, dr)

    @jax.jit
    def evaluate(parameters, models, reset_keys, errors):
        infer = factory(parameters)
        policy = lambda obs: infer(obs, jax.random.PRNGKey(0))[0]
        return make_rollout(env, policy, axes)(models, reset_keys, errors)

    report = dict(state='RUNNING', authority=reg['authority'], n=200,
                  checkpoint_sha256=reg['checkpoint_sha256'], model_sha256=reg['model_sha256'],
                  fixed_reference_percent=64., fixed_cutoff_percent=59., optimizer_updates=0,
                  one_executable_all_levels=True, rows=[], thresholds={},
                  hardware_disposition='Proposed for review; acceptance defaults unchanged')

    def save():
        tmp = a.output / 'result.json.tmp'
        tmp.write_text(json.dumps(report, indent=2) + '\n')
        tmp.replace(a.output / 'result.json')

    def measure(label, field=None, level=0.):
        before = time.monotonic()
        errors = j.zeros(5, j.float32)
        if field is not None:
            errors = errors.at[FIELDS.index(field)].set(level)
        trace, sign = evaluate(params, randomized, keys, errors)
        trace = jax.tree.map(np.asarray, trace)
        if not all(np.isfinite(v).all() for v in trace.values()):
            raise RuntimeError('Nonfinite isolated sensor sweep')
        values = summarise(trace, np.asarray(sign))
        x = values['success']
        np.savez_compressed(a.output / (label + '.npz'), **values,
                            reset_keys=np.asarray(keys), dr_keys=np.asarray(dr), errors=np.asarray(errors))
        row = dict(label=label, field=field, level=level, count=int(x.sum()),
                   success_percent=float(x.mean() * 100), drops=int(values['dropped'].sum()),
                   wall_s=time.monotonic()-before, below_fixed_cutoff=bool(x.sum() <= 118))
        return row, x

    row, baseline = measure('ideal_baseline')
    report['fresh_ideal_count'] = row['count']
    report['rows'].append(row)
    save()
    print(json.dumps(row), flush=True)
    for field, grid in reg['grids'].items():
        observations = [(0., int(baseline.sum()))]
        for index, level in enumerate(grid):
            if level == 0:
                continue
            row, outcomes = measure(field + '_' + str(index), field, level)
            b = int(np.sum(outcomes & ~baseline))
            c = int(np.sum(~outcomes & baseline))
            row.update(variant_only=b, baseline_only=c, exact_mcnemar_p=exact_mcnemar(b, c),
                       change_from_fresh_ideal_points=(row['count']-int(baseline.sum()))/2)
            report['rows'].append(row)
            observations.append((level, row['count']))
            save()
            print(json.dumps(row), flush=True)
        crossing = next((i for i, (_, count) in enumerate(observations) if count <= 118), None)
        nonmonotonic = any(b[1] > a[1] for a, b in zip(observations, observations[1:]))
        if int(baseline.sum()) <= 118:
            threshold = dict(state='UNIDENTIFIABLE_BASELINE_AT_OR_BELOW_CUTOFF')
        elif crossing is None:
            threshold = dict(state='NO_CROSSING_TESTED', lower_bound=observations[-1][0])
        else:
            threshold = dict(state='FIRST_TESTED_CROSSING', level=observations[crossing][0],
                             previous_passing_level=observations[crossing-1][0])
        threshold.update(nonmonotonic_observed=nonmonotonic, unit=reg['units'][field],
                         fixed_cutoff_count=118, interpolation=False)
        report['thresholds'][field] = threshold
        save()
    report['state'] = 'COMPLETE'
    save()


if __name__ == '__main__':
    if not os.environ.get('CONTAINER_ID'):
        raise RuntimeError('Vast only')
    main()
