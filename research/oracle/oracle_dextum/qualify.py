"""Cloud-only adapter qualification; no learning and no representation verdict."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time
import unittest


class Contracts(unittest.TestCase):
    def test_tokens_order_events_and_episode_reset(self):
        from .schema import ContactKey, ContactRecord, OracleFrame
        from .tokenizer import Tokenizer, flatten_losslessly
        a = ContactRecord(ContactKey('object', 1, 'tip', 0), 1., (1., 0., 0.),
                          (.025, 0., 0.), 0., .2, 2)
        b = replace(a, key=ContactKey('object', 2, 'jaw', 0))
        frame = OracleFrame(0., (b, a), 'test', 'hold')
        left, right = Tokenizer(), Tokenizer()
        self.assertEqual(left.encode(frame), right.encode(replace(frame, contacts=(a, b))))
        self.assertEqual(flatten_losslessly(frame), frame.to_dict())
        released = left.encode(replace(frame, time_s=.005, contacts=(a,)))
        self.assertEqual([(t.value, t.contact) for t in released if t.kind == 'event'],
                         [(0, a.key), (2, b.key)])
        with self.assertRaises(ValueError): left.encode(frame)
        left.reset()
        self.assertEqual(left.encode(frame), Tokenizer().encode(frame))
        self.assertLessEqual(len(Tokenizer.vocabulary), 4096)

    def test_unloaded_contact_is_not_an_onset(self):
        from .schema import ContactKey, ContactRecord, OracleFrame
        from .tokenizer import Tokenizer
        c = ContactRecord(ContactKey('object', 0, 'tip', 0), 0., (0., 0., 0.),
                          (0., 0., 0.), 1., .2, 1)
        tokens = Tokenizer().encode(OracleFrame(0., (c,), 'test'))
        self.assertEqual([t.value for t in tokens if t.kind == 'event'], [3])
        self.assertEqual([t.value for t in tokens if t.kind == 'slip'], [0])

    def test_duplicate_contacts_and_nonfinite_rejected(self):
        from .schema import ContactKey, ContactRecord, OracleFrame
        c = ContactRecord(ContactKey('object', 0, 'tip', 0), 1., (1., 0., 0.),
                          (0., 0., 0.), 0., .2, 1)
        with self.assertRaises(ValueError): OracleFrame(0., (c, c), 'test')
        with self.assertRaises(ValueError): replace(c, normal_n=float('nan'))
        with self.assertRaises(ValueError): replace(c.key, element_class='index_joint')

    def test_cube_face_ids_and_edge_ambiguity(self):
        from .patches import CubePatches
        p = CubePatches()
        for axis in range(3):
            for side in range(2):
                point = [0., 0., 0.]; point[axis] = (2*side-1)*.025
                a = p.assign(point)
                self.assertEqual(a.patch_id, 2*axis+side)
                self.assertEqual(a.distance_m, 0.)
                self.assertFalse(a.boundary_ambiguous)
        self.assertTrue(p.assign([.025, .025, 0.]).boundary_ambiguous)
        self.assertAlmostEqual(p.assign([.026, 0., 0.]).distance_m, .001)

    def test_unknown_edges_never_become_paths(self):
        from .lattice import Edge, plan
        unmeasured = Edge('a', 'b', 'hold', 0, 0, 1., 'controller', 'bank', '0')
        self.assertEqual(plan({'a', 'b'}, [unmeasured], 'a', 'b').status, 'UNREACHABLE')
        self.assertEqual(plan({'a', 'b'}, [], 'a', 'a').status, 'AT_GOAL')
        self.assertEqual(plan({'a'}, [], 'a', 'missing').status, 'UNKNOWN_NODE')
        good = replace(unmeasured, trials=16, successes=16)
        self.assertEqual(plan({'a', 'b'}, [good], 'a', 'b').edges, (good,))
        marginal = replace(good, successes=10)
        self.assertEqual(plan({'a', 'b'}, [marginal], 'a', 'b').status, 'MARGINAL_PATH')
        self.assertEqual(plan({'a', 'b'}, [marginal], 'a', 'b', allow_marginal=False).status,
                         'UNREACHABLE')

    def test_energy_units_signed_work_and_no_fake_minimum(self):
        from .native import EnergyLedger
        ledger = EnergyLedger()
        for _ in range(100):
            ledger.add(dict(actuator_w=2., external_w=-1., friction_signed_w=-.5), .01)
        r = ledger.report(.25)
        self.assertAlmostEqual(r['actuator_signed_j'], 2.)
        self.assertAlmostEqual(r['friction_signed_j'], -.5)
        self.assertAlmostEqual(r['energy_residual_j'], .75)
        self.assertIsNone(r['dexterity_score'])
        with self.assertRaises(ValueError): ledger.add(dict(actuator_w=float('nan')), .01)


class NativeAdapter(unittest.TestCase):
    @staticmethod
    def fixture(reverse=False):
        import mujoco
        floor = '<geom name="floor" type="plane" size="1 1 .1" friction=".5 .01 .001"/>'
        cube = '<body name="object" pos="0 0 .026"><freejoint/>' \
               '<geom name="object_geom" type="box" size=".025 .025 .025" mass=".1" friction=".5 .01 .001"/></body>'
        model = mujoco.MjModel.from_xml_string('<mujoco><option timestep=".002"/>'
            '<worldbody>'+(''.join([cube, floor]) if reverse else ''.join([floor, cube]))+
            '</worldbody></mujoco>')
        data = mujoco.MjData(model)
        for _ in range(1000): mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        return model, data

    def test_contact_force_frame_both_geom_orders(self):
        import numpy as np
        from .native import ContactOracle, ObjectSurface
        from .patches import CubePatches
        for reverse in (False, True):
            model, data = self.fixture(reverse)
            obj = ObjectSurface('object', model.body('object').id,
                                (model.geom('object_geom').id,), CubePatches())
            oracle = ContactOracle(model, [obj], {model.geom('floor').id: ('environment', 0)})
            before = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy())
            frame = oracle.read(data, task_id='static_fixture', quantum='hold')
            self.assertGreater(len(frame.contacts), 0)
            force = np.sum([c.force_object_n for c in frame.contacts], axis=0)
            self.assertAlmostEqual(force[2], .1*9.81, delta=.002)
            self.assertLess(np.linalg.norm(force[:2]), 1e-5)
            self.assertTrue(all(c.key.patch_id == 4 for c in frame.contacts))
            for a, b in zip(before, (data.qpos, data.qvel, data.ctrl)):
                np.testing.assert_array_equal(a, b)

    def test_sliding_contact_removes_energy(self):
        import mujoco
        from .native import power_sample
        model, data = self.fixture()
        data.qvel[0] = .2
        mujoco.mj_forward(model, data)
        power = power_sample(model, data)
        self.assertLess(power['friction_signed_w'], -1e-4)
        self.assertEqual(power['actuator_w'], 0.)

    def test_unmapped_contact_is_explicit_failure(self):
        from .native import ContactOracle, ObjectSurface
        from .patches import CubePatches
        model, data = self.fixture()
        obj = ObjectSurface('object', model.body('object').id,
                            (model.geom('object_geom').id,), CubePatches())
        with self.assertRaises(ValueError): ContactOracle(model, [obj], {}).read(data, task_id='test')


class SummariesAndMeasurements(unittest.TestCase):
    def test_window_is_time_weighted_and_causal(self):
        from .schema import ContactKey, ContactRecord, OracleFrame
        from .window import QuantumWindow
        c = ContactRecord(ContactKey('cube', 0, 'tip', 0), 1., (1., 0., 0.),
                          (0., 0., 0.), 0., .2, 1)
        initial = OracleFrame(0., (c,), 'task', 'hold')
        window = QuantumWindow(initial, [('duration_s', 2.)])
        window.advance(replace(initial, time_s=.5, contacts=(replace(c, normal_n=3.),)))
        window.advance(replace(initial, time_s=2., contacts=(replace(c, normal_n=3., slip_speed_m_s=.1),)))
        summary = window.finish()
        self.assertAlmostEqual(summary.mean_total_normal_n, 2.5)
        self.assertEqual(summary.slip_onsets, 1)
        self.assertEqual(summary.contact_onsets, 0)
        self.assertIn(('energy', 0), summary.atoms())
        with self.assertRaises(ValueError): window.advance(replace(initial, time_s=3., quantum='roll'))

    def test_builder_requires_observed_target_and_valid_entry(self):
        from .builder import build_cell, Trial
        def connector(**kwargs):
            seed = kwargs['seed']
            return Trial(True, 'wrong' if seed == 0 else 'goal', 1., seed == 1, seed != 2, str(seed))
        result = build_cell(connector=connector, object_id='cube', hand_id='fixture',
            source='start', target='goal', quantum='hold', parameters={}, parameter_id='0',
            seeds=list(range(16)), controller_sha256='fixture_controller', entry_bank_sha256='fixture_bank')
        self.assertEqual(result['edge']['successes'], 13)
        self.assertEqual(len(result['trials']), 16)
        self.assertLess(result['success_interval_95'][0], 13/16)
        self.assertGreater(result['success_interval_95'][1], 13/16)


class CompiledAdapter(unittest.TestCase):
    def test_mjx_force_reading_and_lossless_control(self):
        import numpy as np
        import jax
        import jax.numpy as j
        from mujoco import mjx
        from types import SimpleNamespace
        from .observations_mjx import CubeContactOracle, records_for_arm
        model, data = NativeAdapter.fixture()
        object_geom = np.zeros(model.ngeom, bool); object_geom[model.geom('object_geom').id] = True
        touch = SimpleNamespace(object_body=model.body('object').id,
            object_geom=object_geom, pad_of_geom=np.full(model.ngeom, -1), frame_bodies=np.zeros(11, int))
        oracle = CubeContactOracle(model, touch)
        m, d = mjx.put_model(model), mjx.put_data(model, data)
        result = jax.jit(oracle.read)(m, d)
        values = np.asarray(result['records'])
        self.assertEqual(values.shape, (72, 27))
        self.assertEqual(int(result['unmapped_contacts']), 0)
        self.assertTrue(np.isfinite(values).all())
        self.assertAlmostEqual(float(values[:, 0].sum()), .981, delta=.002)
        self.assertAlmostEqual(float(values[:, 3].sum()), .981, delta=.002)
        permutation = jax.random.permutation(jax.random.PRNGKey(63), 72)
        inverse = j.argsort(permutation)
        shuffled = records_for_arm(result['records'], permutation, 'flat_same_information_full')
        np.testing.assert_array_equal(np.asarray(shuffled[inverse]), values)
        removed = records_for_arm(result['records'], permutation, 'patch_identity_removed_full')
        np.testing.assert_array_equal(np.asarray(removed[:, :-6]), np.asarray(shuffled[:, :-6]))
        self.assertTrue(np.all(np.asarray(removed[:, -6:]) == 0))
        inactive = d.replace(_impl=d._impl.replace(contact=d._impl.contact.replace(
            dist=j.full_like(d._impl.contact.dist, 1.))))
        inactive_values = np.asarray(jax.jit(oracle.read)(m, inactive)['records'])
        # Deliberately retain stale solved forces while making every contact
        # inactive: no physical values or candidate counts may leak through.
        np.testing.assert_array_equal(inactive_values[:, :11], 0.)

    def test_full_state_moving_palm_against_native_jacobian(self):
        import mujoco
        import numpy as np
        import jax
        import jax.numpy as j
        from mujoco import mjx
        from .observations_mjx import palm_frame_object_state
        model = mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0"/>
          <worldbody>
            <body name="palm" pos=".2 -.3 .4"><freejoint/>
              <geom type="box" size=".04 .03 .01" mass="1"/></body>
            <body name="object" pos=".3 -.25 .5"><freejoint/>
              <geom type="box" size=".025 .025 .025" mass=".1"/></body>
          </worldbody></mujoco>''')
        data = mujoco.MjData(model)
        data.qpos[3:7] = np.array([.8, .1, -.3, .5])/np.linalg.norm([.8, .1, -.3, .5])
        data.qpos[10:14] = np.array([.6, -.4, .2, .3])/np.linalg.norm([.6, -.4, .2, .3])
        data.qvel[:] = [.1, -.2, .3, .5, -.6, .7, -.3, .4, .2, -.4, .1, .8]
        mujoco.mj_forward(model, data)
        palm, obj = model.body('palm').id, model.body('object').id
        def twist(body):
            jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacp, jacr, data.xpos[body], body)
            return jacp@data.qvel, jacr@data.qvel
        vp, wp = twist(palm); vo, wo = twist(obj)
        delta = data.xpos[obj]-data.xpos[palm]
        rot = data.xmat[palm].reshape(3, 3)
        expected_linear = (vo-vp-np.cross(wp, delta))@rot
        expected_angular = (wo-wp)@rot
        d = mjx.put_data(model, data)
        compiled = jax.jit(lambda d: palm_frame_object_state(
            d, obj, palm, j.asarray(model.body_rootid)))
        actual = np.asarray(compiled(d))
        self.assertEqual(actual.shape, (13,))
        np.testing.assert_allclose(actual[:3], delta@rot, atol=1e-6)
        np.testing.assert_allclose(actual[7:10], expected_linear, atol=1e-6)
        np.testing.assert_allclose(actual[10:13], expected_angular, atol=1e-6)
        represented_rotation = np.empty(9)
        mujoco.mju_quat2Mat(represented_rotation, actual[3:7].astype(float))
        np.testing.assert_allclose(represented_rotation.reshape(3, 3),
            rot.T@data.xmat[obj].reshape(3, 3), atol=1e-6)
        changed_sign = d.replace(xquat=d.xquat.at[obj].multiply(-1))
        np.testing.assert_array_equal(np.asarray(compiled(changed_sign)), actual)

    def test_arm_information_contract_and_pose_alias_separation(self):
        import numpy as np
        import jax
        import jax.numpy as j
        from .observations_mjx import interface_for_arm
        records = j.arange(72*27, dtype=j.float32).reshape(72, 27)/1000
        first = dict(records=records, object_state=j.arange(13, dtype=j.float32),
                     physical_parameters=j.arange(21, dtype=j.float32), unmapped_contacts=j.array(0))
        second = dict(first, object_state=first['object_state'].at[3:7].set(j.array([.9, 0, 0, .1])))
        permutation = jax.random.permutation(jax.random.PRNGKey(84), 72)
        structured = interface_for_arm(first, permutation, 'oracle_structured_full')
        flat = interface_for_arm(first, permutation, 'flat_same_information_full')
        for key in first: np.testing.assert_array_equal(structured[key], flat[key])
        inverse = j.argsort(permutation)
        np.testing.assert_array_equal(flat['records'][inverse], records)
        # The ablation is exactly pose/velocity removal, not a renamed full arm.
        a = interface_for_arm(first, permutation, 'oracle_contact_only')
        b = interface_for_arm(second, permutation, 'oracle_contact_only')
        for key in first: np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(a['records'], structured['records'])
        self.assertFalse(np.array_equal(structured['object_state'],
            interface_for_arm(second, permutation, 'oracle_structured_full')['object_state']))
        removed = interface_for_arm(first, permutation, 'patch_identity_removed_full')
        np.testing.assert_array_equal(removed['object_state'], structured['object_state'])
        np.testing.assert_array_equal(removed['records'][:, :-6], structured['records'][:, :-6])


def main():
    if not os.environ.get('CONTAINER_ID'):
        raise RuntimeError('Numerical qualification runs on an authorized cloud host only')
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compiled', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(Path(__file__).parent.glob('*.py'))}
    (args.output/'source_manifest.json').write_text(json.dumps(sources, indent=2)+'\n')
    started = time.time()
    classes = [Contracts, NativeAdapter, SummariesAndMeasurements]
    if args.compiled: classes.append(CompiledAdapter)
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in classes)
    with (args.output/'tests.txt').open('w') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = dict(state='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors),
                  source_sha256=sources, wall_s=time.time()-started,
                  training_attempts=0, representation_claims_tested=[],
                  scope='Schema, causal tokens, empirical lattice builder, native force/power; '
                        'with --compiled: contact extraction, palm-frame state and O1 arm information contract')
    (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
    if not result.wasSuccessful(): raise SystemExit(1)


if __name__ == '__main__':
    main()
