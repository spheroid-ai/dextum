"""Registered compiler-floor detector on the actual DAgger collection code."""
import os
if not os.environ.get('CONTAINER_ID'):
    raise RuntimeError('GPU cloud only')
os.environ['JAX_PLATFORMS'] = 'cuda,cpu'  # CPU device for host callbacks; model execution must remain on GPU.
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from experiments.evaluate_e3 import step_one
from pass0.faithful import MODEL_KEY
from workorder14.lane_d.dagger_core import wrapped_environment, make_collect
from workorder14.lane_d import learner


def audit(env, output, metadata, *, collector_factory=make_collect, collection_kind='dagger'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    assert jax.default_backend() == 'gpu'
    worlds, unroll, seed = 16, 40, 1
    wrapped = wrapped_environment(env, worlds, seed)
    keys = jax.random.split(jax.random.PRNGKey(2026), worlds)
    state = jax.jit(wrapped.reset)(keys)
    models = state.info[MODEL_KEY]
    _, axes = env.randomize(env.mjx_model, jax.random.split(jax.random.PRNGKey(seed + 71000), worlds))
    params = learner.initialize(jax.random.PRNGKey(1), 93, 7, zero_output=True)
    collect = collector_factory(wrapped, worlds, unroll, taps=True)
    _, _, traces = collect(params, state, jax.random.PRNGKey(197), j.array(.5))
    jax.block_until_ready(traces)
    changed = models.tree_replace({'body_mass': models.body_mass.at[:, env.cube_body].multiply(1.1)})
    forced = state.replace(info={**state.info, MODEL_KEY: changed, 'steps': j.full_like(state.info['steps'], 299)})
    reset, _, forced_traces = collect(params, forced, jax.random.PRNGKey(197), j.array(.5))
    jax.block_until_ready(reset)
    exercised = bool(np.all(np.asarray(forced_traces['done'][0]) > .5))
    preserved = all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(
        jax.tree.leaves(changed), jax.tree.leaves(reset.info[MODEL_KEY])))

    def outputs(model, state, action):
        advanced = step_one(env, model, state, action)
        return advanced.data.qpos, advanced.data.ctrl, advanced.done

    direct_a = jax.jit(jax.vmap(outputs, in_axes=(axes, 0, 0)))
    direct_b = jax.jit(outputs)
    single_models = [jax.tree.map(lambda value, axis: value if axis is None else value[index],
                                 models, axes, is_leaf=lambda x: x is None) for index in range(worlds)]
    errors, floors, controls = [], [], []
    for tick in range(unroll):
        before = jax.tree.map(lambda value: value[tick], traces['state'])
        action = traces['action'][tick]
        bulk = direct_a(models, before, action)
        for index in range(worlds):
            if float(traces['done'][tick, index]) > .5 or float(bulk[2][index]) > .5:
                continue
            single = direct_b(single_models[index], jax.tree.map(lambda value: value[index], before), action[index])
            errors.append(np.asarray(j.abs(traces['qpos'][tick, index] - bulk[0][index])))
            floors.append(np.asarray(j.abs(single[0] - bulk[0][index])))
            controls.append(float(j.max(j.abs(traces['ctrl'][tick, index] - bulk[1][index]))))
    assert errors
    error = np.max(np.asarray(errors), axis=-1)
    floor = np.max(np.asarray(floors), axis=-1)
    median, numerical_floor = float(np.median(error)), float(np.median(floor))
    limit = max(1e-4, 10 * numerical_floor)
    finite = bool(np.isfinite(error).all() and np.isfinite(floor).all())
    result = dict(state='PASS' if finite and exercised and preserved and median <= limit else 'FAIL',
        metadata=metadata, finite=finite, matched_step_median=median,
        matched_step_p95=float(np.percentile(error, 95)), matched_step_max=float(error.max()),
        direct_direct_median=numerical_floor, calibrated_limit=limit,
        comparable_transitions=len(errors), maximum_control_error=max(controls),
        full_reset_exercised=exercised, full_reset_preserves_physics=preserved,
        actual_collection_path=True, collection_kind=collection_kind,
        collection_module=collector_factory.__module__ + '.' + collector_factory.__name__,
        direct_compilations=['jit vmap batch16', 'independent jit batch1'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        criterion_version='work_order_2026_10_14', optimizer_steps=0, training_attempts=0)
    np.savez_compressed(output / 'errors.npz', collection_direct=np.asarray(errors),
                        direct_direct=np.asarray(floors), controls=np.asarray(controls))
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    if result['state'] != 'PASS':
        raise RuntimeError('Actual DAgger collection detector failed; no path or guard change authorized')
    return result
