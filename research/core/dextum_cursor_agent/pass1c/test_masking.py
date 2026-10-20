"""Cloud-only sensor intervention checks before the masking phase."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import jax
import jax.numpy as j
import numpy as np
from pass1c.masking import Sensors,mask_packet,BlindRotation,BaseEnvironment
from pass1c.sensors import Sensors as OriginalSensors


def sensor(config):
    s=Sensors.__new__(Sensors)
    s.level='Realistic';s.coupling=np.eye(20,dtype=np.float32)
    s.cell_pos=np.arange(120,dtype=np.float32).reshape(40,3)*.0001
    s.membership=np.eye(40,11,dtype=np.float32)
    s.configure(config)
    return s


class MaskTests(unittest.TestCase):
    def test_shear_removed_from_packet_ratios_and_controller(self):
        s=sensor({});p=j.ones(200);masked=mask_packet(p,'shear')
        np.testing.assert_array_equal(np.asarray(masked[:60]).reshape(20,3)[:,0],1)
        np.testing.assert_array_equal(np.asarray(masked[:60]).reshape(20,3)[:,1:],0)
        np.testing.assert_array_equal(np.asarray(masked[60:100]),0)
        np.testing.assert_array_equal(np.asarray(s.reading(masked,None).f_t),0)

    def test_normal_only_preserves_normal_and_removes_event_channels(self):
        p=mask_packet(j.ones(200),'normal_only')
        np.testing.assert_array_equal(np.asarray(p[100:120]),1)
        np.testing.assert_array_equal(np.asarray(p[120:160]),0)
        self.assertFalse(np.asarray(sensor({}).reading(p,None).slip).any())

    def test_barometric_removal_includes_age(self):
        for group,force,age in [('palm',slice(100,112),slice(180,192)),('proximal',slice(112,120),slice(192,200))]:
            p=mask_packet(j.ones(200),group)
            np.testing.assert_array_equal(np.asarray(p[force]),0)
            np.testing.assert_array_equal(np.asarray(p[age]),0)
            np.testing.assert_array_equal(np.asarray(p[:60]),1)

    def test_legacy_location_duplicate_removed(self):
        env=BlindRotation.__new__(BlindRotation);env.touch=True;env.sensors=SimpleNamespace(group='location')
        with patch.object(BaseEnvironment,'_frame',return_value=j.ones(363)):
            result=np.asarray(env._frame(None,None,None))
        legacy=result[53:163].reshape(11,10)
        np.testing.assert_array_equal(legacy[:,4:7],0)
        np.testing.assert_array_equal(legacy[:,:4],1)
        np.testing.assert_array_equal(legacy[:,7:],1)

    def test_configured_default_sensor_sequence_parity(self):
        s=sensor({'detection_floor_n':.3,'optical_noise_n':.05,'direction_degrees':10,'pvdf_false_alarm':.05,'delay_ms':10})
        raw=j.sin(j.arange(120,dtype=j.float32).reshape(40,3))+.5;speed=j.full(40,.01)
        initial=s.initial(jax.random.PRNGKey(4))
        def rollout(method):
            def step(state,t):
                state=method(s,state,raw*(1+t/50),speed,False)
                return state,state['packet']
            return jax.jit(lambda:jax.lax.scan(step,initial,j.arange(64)) )()
        old=rollout(OriginalSensors.advance);new=rollout(Sensors.configured_advance)
        for x,y in zip(jax.tree.leaves(old),jax.tree.leaves(new)):
            np.testing.assert_allclose(np.asarray(x),np.asarray(y),rtol=0,atol=1e-7)

    def test_floor_intervention_reveals_weak_contacts(self):
        def run(floor):
            s=sensor({'detection_floor_n':floor,'optical_noise_n':0,'delay_ms':0})
            state=s.initial(jax.random.PRNGKey(4));state['gain']=j.ones((40,3))
            raw=j.zeros((40,3)).at[:,0].set(.1)
            return s.advance(state,raw,j.zeros(40),False)['packet'][:60].reshape(20,3)[:,0]
        np.testing.assert_array_equal(np.asarray(run(.3)),0)
        self.assertTrue((np.asarray(run(.02))>.09).all())

    def test_floor_isolated_from_other_ideal_sensor_channels(self):
        def run(floor):
            s=sensor({'ideal_detection_floor_n':floor});s.level='Ideal'
            return s.advance(s.initial(jax.random.PRNGKey(4)),
                j.zeros((40,3)).at[:,0].set(.1),j.zeros(40),False)['packet']
        clean,cut=run(0),run(.3)
        np.testing.assert_allclose(np.asarray(clean[:60]).reshape(20,3)[:,0],.1)
        np.testing.assert_array_equal(np.asarray(cut[:60]),0)
        np.testing.assert_array_equal(np.asarray(clean[135:155]),np.asarray(cut[135:155]))

    def test_optical_floor_does_not_change_barometric_floor(self):
        s=sensor({'ideal_optical_detection_floor_n':.3});s.level='Ideal'
        packet=s.advance(s.initial(jax.random.PRNGKey(4)),
            j.zeros((40,3)).at[:,0].set(.1),j.zeros(40),False)['packet']
        np.testing.assert_array_equal(np.asarray(packet[:60]),0)
        np.testing.assert_allclose(np.asarray(packet[100:120]),.1)

    def test_packet_rate_holds_between_samples(self):
        s=sensor({'rate_hz':20});s.level='Ideal'
        state=s.initial(jax.random.PRNGKey(4));packets=[]
        for tick in range(12):
            state=s.advance(state,j.zeros((40,3)).at[:,0].set(1+tick/10),j.zeros(40),False)
            packets.append(np.asarray(state['packet'][:60]).reshape(20,3)[:,0])
        for tick in range(10):np.testing.assert_allclose(packets[tick],1)
        np.testing.assert_allclose(packets[10],2)
        np.testing.assert_array_equal(packets[10],packets[11])

    def test_location_noise_limit_and_channel_isolation(self):
        def run(mm):
            s=sensor({'location_noise_mm':mm,'delay_ms':0})
            return np.asarray(s.advance(s.initial(jax.random.PRNGKey(4)),
                j.zeros((40,3)).at[:,0].set(1),j.zeros(40),False)['packet'])
        clean,noisy=run(0),run(5)
        np.testing.assert_array_equal(clean[:120],noisy[:120])
        np.testing.assert_array_equal(clean[135:],noisy[135:])
        delta=(noisy[120:135]-clean[120:135]).reshape(5,3)*.02
        self.assertTrue((np.linalg.norm(delta,axis=1)<=.010001).all())


if __name__=='__main__':unittest.main()
