"""GPU-only D2/D3 boundary tests, before any imitation training budget."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    if not os.environ.get('CONTAINER_ID'):
        raise RuntimeError('GPU cloud only')
    os.environ['JAX_PLATFORMS'] = 'cuda,cpu'  # CPU device for host callbacks; model execution must remain on GPU.
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--teacher', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert hashlib.sha256(a.model.read_bytes()).hexdigest() == 'a2e0c670506cb69267bac11e80fa24c467a782cca3bedb901290adc64444c924'
    assert hashlib.sha256(a.teacher.read_bytes()).hexdigest() == '1c1dba7e9c222d5c9279517989f2323b2f383e1448df8d7f1e657a15a440c6a7'
    os.environ.update(DEXTUM_KIT_XML=str(a.model), DEXTUM_LEGACY_PREFIX='1', DEXTUM_SENSOR_LEVEL='Realistic')
    import jax
    import jax.numpy as j
    import numpy as np
    from brax.io import model
    from brax.training.agents.ppo import networks
    import vendor.aero_official.vision_env  # Same pinned source bootstrap as the accepted teacher.
    from pass1c.environment import BlindRotation, CRITIC_SIZE
    from pass1c.followup import network_factory
    from dextum import contact_space as cs
    from workorder14.lane_d.sensorless import SensorlessRotation, STUDENT_WIDTH
    from workorder14.lane_d.inferred_touch import InferredTouchRotation
    assert jax.default_backend() == 'gpu' and all(d.platform == 'gpu' for d in jax.devices())
    params = model.load_params(a.teacher)
    net = network_factory(1089)(observation_size=dict(state=1089, privileged_state=CRITIC_SIZE), action_size=31)
    teacher = networks.make_inference_fn(net)(params, deterministic=True)
    student_env = SensorlessRotation(teacher)
    key = jax.random.PRNGKey(41014)
    reset = jax.jit(student_env.reset)(key)
    assert reset.obs['state'].shape == (STUDENT_WIDTH,)
    assert reset.obs['expert_action'].shape == (7,)
    assert reset.info['history'].shape == (1089,)
    # No G/tactile columns enter student state, even when teacher history changes.
    altered_history = reset.info['history'].reshape(3, 363).at[:, 18:].set(4.25).reshape(-1)
    changed = reset.replace(info={**reset.info, 'history': altered_history})
    measured = student_env._student_observation(changed, initial=True)
    np.testing.assert_array_equal(np.asarray(reset.obs['state']), np.asarray(measured.obs['state']))
    active = reset.replace(info={**reset.info, 'tick': j.array(25, j.int32)})
    infer = teacher(dict(state=active.info['history'], privileged_state=active.obs['privileged_state']), key)[0]
    _, expected = cs.step(active.info['g'], infer, active.info['sense'], student_env.lo, student_env.hi, reflex=True)
    _, target = student_env.teacher(active)
    reconstructed = j.asarray(cs.HOME) + j.asarray(cs.SCALE) * target
    label_error = float(j.max(j.abs(expected - reconstructed)))
    assert label_error < 1e-7
    # Same requested motor command under a deliberately changed touch reading.
    alternate_sense = cs.Sense(j.full(5, 8.), j.full(5, 3.), j.ones(5, bool), j.ones(5, bool))
    other = active.replace(info={**active.info, 'sense': alternate_sense})
    action = j.array([.1, -.2, .3, -.4, .2, -.1, .4], j.float32)
    advance = jax.jit(student_env.step)
    first, second = advance(active, action), advance(other, action)
    np.testing.assert_array_equal(np.asarray(first.info['motor_targets']), np.asarray(second.info['motor_targets']))
    np.testing.assert_array_equal(np.asarray(first.data.ctrl), np.asarray(second.data.ctrl))
    # D3 overwrites packet, legacy features and G's sense with its fixed estimate.
    estimated = j.zeros(200).at[:60:3].set(.7).at[100:120].set(.2)
    inferred = InferredTouchRotation(lambda causal: estimated)
    start = jax.jit(inferred.reset)(key)
    noisy_truth = start.replace(info={**start.info, 'sensor_window': j.full(200, 4.),
        'sensor': {**start.info['sensor'], 'packet': j.full(200, 4.)}})
    # Compare the two truth inputs through the same compiled boundary. Reset
    # and eager re-estimation have different lowering/fusion and are not the
    # intervention whose independence this test is meant to establish.
    estimate = jax.jit(lambda state: inferred._estimate(state, initial=True))
    baseline, replaced = estimate(start), estimate(noisy_truth)
    np.testing.assert_array_equal(np.asarray(baseline.obs['state']), np.asarray(replaced.obs['state']))
    for before, after in ((start, baseline), (noisy_truth, replaced)):
        np.testing.assert_array_equal(np.asarray(before.info['rng']), np.asarray(after.info['rng']))
        np.testing.assert_array_equal(np.asarray(before.info['history'][:18]),
                                      np.asarray(after.info['history'][:18]))
    np.testing.assert_array_equal(np.asarray(baseline.info['sensorless_history']),
                                  np.asarray(replaced.info['sensorless_history']))
    np.testing.assert_array_equal(np.asarray(baseline.info['sensor_window']),
                                  np.asarray(replaced.info['sensor_window']))
    np.testing.assert_array_equal(np.asarray(baseline.info['sensor']['packet']),
                                  np.asarray(replaced.info['sensor']['packet']))
    for x, y in zip(jax.tree.leaves((baseline.info['g'], baseline.info['sense'])),
                    jax.tree.leaves((replaced.info['g'], replaced.info['sense']))):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    # Preserve the old cross-compilation comparison as a reported diagnostic,
    # with reductions on the GPU. It neither admits nor rejects training.
    eager = inferred._estimate(noisy_truth, initial=True)
    reset_eager_diagnostic = dict(
        maximum_actor_difference=float(j.max(j.abs(start.obs['state'] - eager.obs['state']))),
        differing_actor_entries=int(j.sum(start.obs['state'] != eager.obs['state'])),
        actor_entries=int(start.obs['state'].size), gated=False,
        scope='Whole jitted reset versus eager re-estimation; different compilation boundaries')
    result = dict(state='PASS', device=str(jax.devices()), checks=6,
        teacher_motor_label_max_error=label_error,
        reset_eager_diagnostic=reset_eager_diagnostic,
        independence_comparison='Same compiled estimator, identical state except perturbed true touch packet/window',
        student_actor_width=STUDENT_WIDTH, student_action_width=7,
        tests=['student interface shapes', 'student ignores G/touch history',
               'teacher contact-to-motor label parity', 'student command independent of teacher touch',
               'inferred touch replaces actor legacy packet and G routes',
               'touch substitution preserves sampled proprioception and its RNG'],
        qualification_scope='Interface checks only; actual collection detector/resume/throughput still required')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
