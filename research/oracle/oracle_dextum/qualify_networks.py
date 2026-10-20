"""Bounded GPU checks of O1 encoder contracts, with no policy learning."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import unittest


class EncoderContracts(unittest.TestCase):
    def test_durable_checkpoint_boundaries(self):
        from pass1c.resume_loop import checkpoint_due
        # Legacy callers still save every update.
        self.assertTrue(all(checkpoint_due(i, i, i*80, 10000, 1) for i in range(1, 12)))
        # Amortized writes retain the evaluation, final and qualification boundaries.
        self.assertFalse(checkpoint_due(4, 4, 320, 10000, 5))
        self.assertTrue(checkpoint_due(5, 5, 400, 10000, 5))
        self.assertTrue(checkpoint_due(44, 0, 3520, 10000, 5))
        self.assertTrue(checkpoint_due(12, 12, 960, 960, 5))
        self.assertTrue(checkpoint_due(12, 12, 960, 10000, 5, qualification_end=True))
        for invalid in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                checkpoint_due(1, 1, 80, 10000, invalid)

    @staticmethod
    def fixture():
        import jax
        import jax.numpy as j
        from .networks import dummy_observation
        shapes = dict(records=(3, 72, 27), state=675, ceiling=2400, privileged_state=2608)
        obs = dummy_observation(shapes, leading=(2,))
        record = jax.random.normal(jax.random.PRNGKey(12), (2, 3, 72, 27))*.1
        element = j.repeat(j.arange(12), 6)
        classes = j.array([0]*5+[1]*5+[2, 4])[element]
        instances = j.array(list(range(5))+list(range(5))+[0, 0])[element]
        record = record.at[..., 11:16].set(jax.nn.one_hot(classes, 5))
        record = record.at[..., 16:21].set(jax.nn.one_hot(instances, 5))
        record = record.at[..., 21:27].set(jax.nn.one_hot(j.tile(j.arange(6), 12), 6))
        return shapes, dict(obs, records=record)

    def test_structured_permutation_and_privilege_isolation(self):
        import numpy as np
        import jax
        from .networks import Actor
        _, obs = self.fixture()
        module = Actor('oracle_structured_full', 62)
        params = module.init(jax.random.PRNGKey(0), obs)
        apply = jax.jit(module.apply)
        baseline = apply(params, obs)
        p = jax.random.permutation(jax.random.PRNGKey(33), 72)
        shuffled = dict(obs, records=obs['records'][..., p, :])
        np.testing.assert_allclose(apply(params, shuffled), baseline, atol=2e-6, rtol=1e-5)
        hidden = dict(obs, ceiling=obs['ceiling']+123., privileged_state=obs['privileged_state']+999.)
        np.testing.assert_array_equal(apply(params, hidden), baseline)
        self.assertTrue(np.isfinite(baseline).all())

    def test_capacity_and_ppo_interfaces(self):
        import numpy as np
        import jax
        import jax.numpy as j
        from brax.training.acme import running_statistics, specs
        from brax.training.agents.ppo.networks import make_inference_fn
        from .networks import capacity_contract, network_factory, parameter_count
        from .observations_mjx import O1_ARMS
        shapes, obs = self.fixture()
        capacity = capacity_contract(shapes, 31)
        stats = running_statistics.init_state({
            key: specs.Array((shape,) if isinstance(shape, int) else shape, j.dtype('float32'))
            for key, shape in shapes.items()})
        for arm in O1_ARMS:
            net = network_factory(arm, shapes, capacity)(shapes, 31)
            actor = net.policy_network.init(jax.random.PRNGKey(0))
            value = net.value_network.init(jax.random.PRNGKey(1))
            count = parameter_count(actor)
            self.assertLessEqual(abs(count-capacity['structured_parameters'])/capacity['structured_parameters'], .01)
            action, _ = make_inference_fn(net)((stats, actor, value), deterministic=True)(obs, jax.random.PRNGKey(0))
            self.assertEqual(action.shape, (2, 31))
            self.assertTrue(np.isfinite(action).all())
            self.assertEqual(net.value_network.apply(stats, value, obs).shape, (2,))
            gradient = jax.grad(lambda parameters: j.sum(net.policy_network.apply(stats, parameters, obs)**2))(actor)
            self.assertTrue(all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(gradient)))
            self.assertGreater(sum(float(j.sum(j.abs(x))) for x in jax.tree.leaves(gradient)), 0.)
        self.capacity = capacity

    def test_training_normalizer_preserves_tensor_features(self):
        import numpy as np
        import jax
        import jax.numpy as j
        from brax.training.acme import running_statistics, specs
        from pass1c.resumable_ppo import observation_statistics_specs
        from .networks import dummy_observation
        shapes = dict(records=(3, 72, 27), state=687, ceiling=4091,
                      privileged_state=4299)
        # The production reset path carries device and environment axes.
        reset = dummy_observation(shapes, leading=(1, 2))
        stats = running_statistics.init_state(observation_statistics_specs(reset))
        self.assertEqual(stats.mean['records'].shape, (3, 72, 27))
        # Learn-step input has sequence and time batch axes. Distinct record
        # values ensure this checks reduction axes, not only shape acceptance.
        batch = dummy_observation(shapes, leading=(2, 4))
        values = j.arange(2*4*3*72*27, dtype=j.float32).reshape(2, 4, 3, 72, 27)/1000
        batch['records'] = values
        updated = jax.jit(running_statistics.update)(stats, batch)
        np.testing.assert_allclose(updated.mean['records'], np.mean(np.asarray(values), axis=(0, 1)),
                                   atol=2e-6, rtol=1e-6)
        self.assertEqual(float(updated.count), 8.)
        # Legacy vector-only normalizers must be bitwise unchanged.
        vectors = {k: v for k, v in reset.items() if k != 'records'}
        old = running_statistics.init_state({k: specs.Array(v.shape[-1:], j.dtype('float32'))
                                             for k, v in vectors.items()})
        new = running_statistics.init_state(observation_statistics_specs(vectors))
        for a, b in zip(jax.tree.leaves(old), jax.tree.leaves(new)):
            np.testing.assert_array_equal(a, b)


def main():
    if not os.environ.get('CONTAINER_ID'):
        raise RuntimeError('Numerical checks run only on cloud')
    import jax
    assert jax.default_backend() == 'gpu', 'GPU-only work order'
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    started = time.time()
    with (args.output/'tests.txt').open('w') as f:
        result = unittest.TextTestRunner(stream=f, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(EncoderContracts))
    report = dict(state='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), wall_s=time.time()-started,
        source_sha256=sources, training_attempts=0,
        scope='Synthetic input contract, permutation behavior, capacity match, PPO interfaces and finite gradients; no control capability or physics qualification')
    (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    if not result.wasSuccessful(): raise SystemExit(1)


if __name__ == '__main__': main()
