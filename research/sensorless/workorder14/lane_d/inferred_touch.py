"""D3 evaluated teacher: replace every policy/G touch route with an estimate.

The unchanged physical sensor runs only as an evaluation target. Neither its
window, legacy prefix nor its last packet can enter the actor or motor G.
"""
import os
if not os.environ.get('CONTAINER_ID'):
    raise RuntimeError('GPU cloud only')
os.environ['JAX_PLATFORMS'] = 'cuda,cpu'  # CPU device for host callbacks; model execution must remain on GPU.
import jax
import jax.numpy as j
from pass1c.environment import BlindRotation, HISTORY
from dextum import contact_space as cs
from vendor.aero_official.vision_oracle import (
    TagOracle, initial_sensor, sample_qualified_cameras, advance_sensor, features)
from workorder14.lane_d.sensorless import STUDENT_FRAME, STUDENT_WIDTH


def constrain_packet(packet):
    """Observation-domain ranges, fixed before fitting; no physical truth."""
    optical = packet[:60].reshape(20, 3)
    optical = j.concatenate((j.clip(optical[:, :1], 0, 10), j.clip(optical[:, 1:], -4, 4)), axis=1)
    packet = packet.at[:60].set(optical.ravel())
    packet = packet.at[100:120].set(j.clip(packet[100:120], 0, 20))
    packet = packet.at[135:155].set(j.maximum(packet[135:155], 0))
    packet = packet.at[155:160].set(j.clip(packet[155:160], 0, 1))
    packet = packet.at[160:200].set(j.clip(packet[160:200], 0, 1))
    return packet


class InferredTouchRotation(BlindRotation):
    def __init__(self, encoder):
        super().__init__('c_touch')
        self.encoder = encoder
        self.tag_oracle = TagOracle(self.mj_model)

    def _estimate(self, state, *, initial=False):
        info = dict(state.info)
        # Use only observable non-tactile columns and the simulated camera.
        vision = features(info['vision_packet'], info['vision_sensor']['age'])
        frame = j.clip(j.concatenate((info['history'][:18], vision)), -5, 5)
        causal = j.tile(frame, HISTORY) if initial else j.concatenate((frame,
            info['sensorless_history'][:(HISTORY - 1) * STUDENT_FRAME]))
        predicted = constrain_packet(self.encoder(causal))
        info.update(sensorless_history=causal, true_touch_window=info['sensor_window'],
                    true_touch_packet=info['sensor']['packet'], sensor_window=predicted,
                    sensor={**info['sensor'], 'packet': predicted})
        info['sense'] = cs.sense_touch(self.sensors.reading(predicted, state.data))
        # Preserve the existing noisy proprioception/task sample and its RNG.
        # Calling _get_obs again would draw a second measurement and advance
        # the official observation RNG solely because touch was substituted.
        from pass1c.warm_start import prefix
        frame = j.clip(j.concatenate((info['history'][:18],
            cs.features(info['g'], info['sense']), prefix(self.sensors, predicted), predicted)), -5, 5)
        # BaseRotation already shifted history. Replace its newest frame;
        # retain the older frames which were estimated at their own times.
        info['history'] = j.tile(frame, HISTORY) if initial else j.concatenate((frame, info['history'][self.frame:]))
        return state.replace(info=info, obs={**state.obs, 'state': info['history']})

    def reset(self, rng):
        state = super().reset(rng)
        key, camera = jax.random.split(jax.random.fold_in(rng, 20261014))
        sensor = initial_sensor()
        info = {**state.info, 'vision_rng': key, 'vision_cameras': sample_qualified_cameras(camera),
                'vision_sensor': sensor, 'vision_packet': sensor['last'],
                'sensorless_history': j.zeros(STUDENT_WIDTH),
                'true_touch_window': j.zeros(200), 'true_touch_packet': j.zeros(200)}
        return self._estimate(state.replace(info=info), initial=True)

    def step(self, state, action):
        advanced = super().step(state, action)
        info = dict(advanced.info)
        key, capture_key, delay_key = jax.random.split(info['vision_rng'], 3)
        palm = self.tag_oracle.palm_id
        packet, _ = self.tag_oracle.capture(capture_key, info['vision_cameras'],
            advanced.data.xpos[self.cube_body], advanced.data.xquat[self.cube_body],
            advanced.data.geom_xpos, advanced.data.geom_xmat, advanced.data.xpos[palm], advanced.data.xquat[palm])
        sensor, packet = advance_sensor(info['vision_sensor'], packet,
                                        jax.random.bernoulli(delay_key, .2).astype(j.int32))
        info.update(vision_rng=key, vision_sensor=sensor, vision_packet=packet)
        return self._estimate(advanced.replace(info=info))
