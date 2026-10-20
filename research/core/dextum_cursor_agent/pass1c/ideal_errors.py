"""Section 16: one sensor error on an otherwise Ideal packet.

The five error parameters are dynamic state inputs, so every grid point uses
one compiled evaluator. No contact model, actor, reward or compiler option is
changed. The neutral configuration reproduces the existing Ideal transducer.
"""
import os
if not os.environ.get('CONTAINER_ID'):
    raise RuntimeError('Vast only')
import jax
import jax.numpy as j
import numpy as np
import vendor.aero_official.vision_env  # pinned MuJoCo Playground import path
from pass1c.sensors import Sensors as BaseSensors
from pass1c.environment import BlindRotation as BaseEnvironment
from pass1b.sensors import DT, FEATURES


# optical floor N, delay in 5-ms ticks, optical noise SD N,
# shear-direction half-width degrees, nominal false-alarm probability / 50 ms
FIELDS = ('optical_floor_n', 'delay_ticks', 'optical_noise_n',
          'direction_degrees', 'pvdf_false_alarm')


class Sensors(BaseSensors):
    def initial(self, key):
        state = super().initial(key)
        state.update(errors=j.zeros(5), queue=j.zeros((10, FEATURES)))
        return state

    def advance(self, state, raw, speed, hand_open):
        del hand_open  # No gain/drift/bias errors in the isolated Ideal experiment.
        s = dict(state)
        keys = jax.random.split(s['key'], 8)
        s['key'] = keys[0]
        floor, delay, sigma, direction, false_alarm = s['errors']
        # Keep the original key and array layout for the optical noise source.
        optical = raw[:20] + jax.random.normal(keys[2], (40, 3))[:20] * sigma
        angle = jax.random.uniform(keys[3], (20,), minval=-1., maxval=1.) * direction * j.pi / 180
        x, y = optical[:, 1], optical[:, 2]
        optical = j.stack((j.maximum(optical[:, 0], 0),
                           x * j.cos(angle) - y * j.sin(angle),
                           x * j.sin(angle) + y * j.cos(angle)), -1)
        baro = j.maximum(raw[20:, 0], 0)
        normal = j.concatenate((optical[:, 0], baro))
        engaged = normal > j.concatenate((j.full(20, floor), j.zeros(20)))
        optical = j.where(engaged[:20, None], optical, 0)
        baro = j.where(engaged[20:], baro, 0)
        s['age'] = j.where(engaged, s['age'] + DT, 0)
        total = raw[:20, 0].reshape(5, 4).sum(1)
        hp = (s['hp'] + total - s['previous']) * j.exp(-2 * j.pi * 1.6 * DT)
        s['hp'], s['previous'] = hp, total
        slide = speed[:20].reshape(5, 4).sum(1)
        low = j.maximum(0, .02 * hp ** 2 + .02 * slide)
        high = j.maximum(0, .05 * slide)
        peak = j.maximum(0, j.abs(hp) * .1)
        event = (j.abs(total - state['previous']) > .3) | (slide > .01)
        # Same nominal per-tick proxy as the registered Realistic transducer;
        # true event detection remains Ideal, only false alarms are injected.
        onset = (event | (jax.random.uniform(keys[5], (5,)) < false_alarm * DT / .05)).astype(j.float32)
        pvdf = j.stack((low, high, peak, onset), -1)
        slip = ((high > .003) & (normal[:20].reshape(5, 4).sum(1) > .3)).astype(j.float32)
        weights = optical[:, 0].reshape(5, 4)
        pos = j.asarray(self.cell_pos[:20]).reshape(5, 4, 3)
        location = (weights[:, :, None] * pos).sum(1) / j.maximum(weights.sum(1)[:, None], 1e-6)
        ratios = optical[:, 1:] / j.maximum(optical[:, :1], .3)
        packet = j.concatenate((optical.ravel(), ratios.ravel(), baro, location.ravel() / .02,
                                pvdf.ravel(), slip, j.minimum(s['age'], 1)))
        ticks = j.clip(delay.astype(j.int32), 0, 10)
        s['packet'] = j.where(ticks == 0, packet, state['queue'][10 - ticks])
        s['queue'] = j.concatenate((state['queue'][1:], packet[None]))
        return s


class BlindRotation(BaseEnvironment):
    def __init__(self):
        if os.environ.get('DEXTUM_SENSOR_LEVEL') != 'Ideal':
            raise ValueError('Isolated-error sweeps require an Ideal sensor base')
        super().__init__('c_touch')
        self.sensors.__class__ = Sensors


def make_rollout(env, policy, axes):
    """The faithful direct loop with a dynamic transducer parameter vector."""
    import functools
    from experiments.evaluate_e3 import reset_one, step_one
    from pass1c.environment import EPISODE
    reset = jax.vmap(functools.partial(reset_one, env), in_axes=[axes, 0, None])
    step = jax.vmap(functools.partial(step_one, env), in_axes=[axes, 0, 0])

    def run(models, keys, errors):
        state = reset(models, keys, j.zeros(2, j.float32))
        sensor = dict(state.info['sensor'])
        sensor['errors'] = j.broadcast_to(errors, (keys.shape[0], 5))
        state = state.replace(info={**state.info, 'sensor': sensor})

        def body(state, _):
            nxt = step(models, state, policy(state.obs))
            m = nxt.metrics
            return nxt, dict(done=nxt.done, success=m['e3/success'], within=m['e3/within'],
                             error=m['e3/abs_error_deg'])
        _, trace = jax.lax.scan(body, state, None, length=EPISODE)
        return trace, state.info['turn_sign']
    return run
