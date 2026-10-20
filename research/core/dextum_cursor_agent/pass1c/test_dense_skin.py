"""Sensor boundary checks, run only on the registered cloud host."""
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import jax
import jax.numpy as j
import numpy as np
import mujoco
from pass1c.dense_skin import Sensors, assign
from pass1c.sensors import Sensors as BaseSensors
from pass1c.kit_model import TruthTouch
from pass1c.warm_start import calibrate


class DenseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = mujoco.MjModel.from_xml_path(os.environ['DEXTUM_KIT_XML'])
        cad = json.loads((Path('/root/dextum/kit_b')/'geometry.json').read_text())
        cls.base = BaseSensors(cls.m, cad['cells'], TruthTouch(cls.m))
        cls.dense = Sensors(cls.m, cad['cells'], TruthTouch(cls.m))
        cls.dense.configure_dense(cls.m, json.loads(Path(os.environ['DEXTUM_DENSE_LAYOUT']).read_text()))
        calibrate(cls.dense, cls.m)

    def test_rotations_and_groups(self):
        s = self.dense
        np.testing.assert_allclose(np.swapaxes(s.dense_rot, 1, 2)@s.dense_rot,
            np.broadcast_to(np.eye(3), (s.nd, 3, 3)), atol=1e-7)
        np.testing.assert_allclose(np.linalg.det(s.dense_rot), 1, atol=1e-7)
        self.assertEqual(set(s.dense_group), {5, 6, 7, 8, 9})

    def test_nearest_cell_never_crosses_pad(self):
        positions = j.array([[0.,0,0],[.0001,0,0],[1.,0,0]])
        groups = j.array([5,6,5])
        got = assign(j.array([[0.,0,0]]), j.array([6]), positions, groups)
        np.testing.assert_array_equal(got, [1])

    def test_optical_and_pvdf_packets_unchanged(self):
        b, s = self.base, self.dense
        b.level=s.level='Realistic'
        key=jax.random.PRNGKey(17)
        old, new=b.initial(key),s.initial(key)
        raw=j.tile(j.array([1.,.2,.1]),(40,1))
        expanded=j.concatenate((raw[:20],j.zeros((s.nd,3))))
        for _ in range(8):
            old=b.advance(old,raw,j.ones(40)*.02,False)
            new=s.advance(new,expanded,j.ones(20+s.nd)*.02,False)
            for start,stop in [(0,100),(120,180)]:
                np.testing.assert_allclose(old['packet'][start:stop],new['packet'][start:stop],atol=1e-7)

    def test_dense_delay_and_no_privileged_readout(self):
        s=self.dense;s.level='Realistic'
        state=s.initial(jax.random.PRNGKey(23))
        raw=j.zeros((20+s.nd,3)).at[20].set(j.array([2.,.4,.3]))
        first=s.advance(state,raw,j.zeros(20+s.nd),False)
        second=s.advance(first,raw,j.zeros(20+s.nd),False)
        third=s.advance(second,raw,j.zeros(20+s.nd),False)
        np.testing.assert_array_equal(first['packet'][200:],0)
        np.testing.assert_array_equal(second['packet'][200:],0)
        self.assertGreater(float(j.sum(third['packet'][200:]**2)),1)
        # Reading can be reconstructed from a packet without simulator data.
        empty=s.reading(j.zeros(s.features),None)
        np.testing.assert_array_equal(empty.f_n,0)
        np.testing.assert_array_equal(empty.f_t,0)
        loaded=s.reading(third['packet'],None)
        self.assertGreater(float(loaded.f_n.sum()),1)
        self.assertGreater(float(loaded.f_t.sum()),.2)

    def test_ideal_pooling_preserves_applied_load(self):
        s=self.dense;s.level='Ideal'
        state=s.initial(jax.random.PRNGKey(33))
        raw=j.zeros((20+s.nd,3)).at[20,0].set(2.)
        out=s.advance(state,raw,j.zeros(20+s.nd),False)
        np.testing.assert_allclose(out['packet'][200:203],[2.,0,0],atol=1e-7)
        expected=2*s.dense_to_old[0,2,2]
        self.assertAlmostEqual(float(out['packet'][100:120].sum()),expected,places=5)

    def test_environment_scan_shapes(self):
        from pass1c.dense_skin import BlindRotation
        env=BlindRotation()
        state=jax.eval_shape(env.reset,jax.random.PRNGKey(42))
        next_state=jax.eval_shape(env.step,state,j.zeros(env.action_size))
        self.assertEqual(state.obs['state'].shape,(3*env.frame,))
        self.assertEqual(state.info['sensor_window'].shape,(env.sensors.features,))
        self.assertEqual(jax.tree.structure(state),jax.tree.structure(next_state))
        for old,new in zip(jax.tree.leaves(state),jax.tree.leaves(next_state)):
            self.assertEqual(old.shape,new.shape)
            self.assertEqual(old.dtype,new.dtype)

    def test_rezero_uses_noisy_observations_across_both_modalities(self):
        s=self.dense;s.level='Realistic'
        state=s.initial(jax.random.PRNGKey(55))
        state.update(open_ticks=j.array(99),dense_gain=j.ones((s.nd,3)),gain=j.ones((40,3)))
        raw=j.zeros((20+s.nd,3)).at[20,0].set(.95)
        def noise(key,shape,*args,**kwargs):
            del key,args,kwargs
            # Dense force reads below .9N while its latent signal is above it.
            return j.full(shape,-20.) if shape==(s.nd,3) else j.zeros(shape)
        with patch('jax.random.normal',side_effect=noise):
            out=s.advance(state,raw,j.zeros(20+s.nd),True)
            self.assertEqual(int(out['rezero']),1)
            # A loaded optical cell must prevent the same global re-zero.
            loaded=raw.at[0,0].set(2.)
            out=s.advance(state,loaded,j.zeros(20+s.nd),True)
            self.assertEqual(int(out['rezero']),0)


if __name__=='__main__':
    unittest.main()
