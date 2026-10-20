"""§11.3(4a): virtual dense proximal/palm force cells on unchanged physics.

The original 200 packet channels retain their positions; dense 3-axis readings
are appended. Barometric channels are replaced by projections of measured
dense forces, so neither the actor nor G receives privileged contact truth.
"""
import json
import os
from pathlib import Path
if not os.environ.get('CONTAINER_ID'):
    raise RuntimeError('Vast only')
import jax
import jax.numpy as j
import numpy as np
from mujoco.mjx._src import support
import vendor.aero_official.vision_env
from pass1c.sensors import Sensors as BaseSensors, PAD_NAMES
from pass1c.environment import BlindRotation as BaseEnvironment
from pass1b.sensors import DT


def assign(positions, pad, cell_positions, cell_pad):
    distance = j.sum((positions[:, None]-cell_positions[None])**2, -1)
    distance = j.where(pad[:, None] == cell_pad[None], distance, j.inf)
    return j.argmin(distance, -1)


class Sensors(BaseSensors):
    def configure_dense(self, m, layout):
        import mujoco
        cells = layout['cells']
        self.nd = len(cells)
        self.features = 200+3*self.nd
        self.dense_pos = np.array([c['position_mm'] for c in cells])/1000
        self.dense_rot = np.array([c['rotation'] for c in cells])
        self.dense_body = np.array([m.body(c['body']).id for c in cells])
        self.dense_group = np.array([PAD_NAMES.index(c['module']) for c in cells])
        self.dense_old = np.array([c['old_cell_id'] for c in cells])-20
        self.dense_pool = (self.dense_old[:, None] == np.arange(20)).astype(np.float32)
        self.dense_membership = self.membership[self.dense_old+20]
        # Reference body/site rotations are static calibration, not a runtime
        # privileged signal. Old sites may attach through fixed kit children.
        for parent, site in zip(self.dense_body, self.sites[self.dense_old+20]):
            child = int(m.site_bodyid[site])
            while child != parent:
                assert child != 0 and m.body_jntnum[child] == 0
                child = int(m.body_parentid[child])
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        old_world = d.site_xmat[self.sites[self.dense_old+20]].reshape(-1, 3, 3)
        dense_world = d.xmat[self.dense_body].reshape(-1, 3, 3)@self.dense_rot
        self.dense_to_old = np.einsum('nji,njk->nik', old_world, dense_world)
        distance = np.linalg.norm(self.dense_pos[:, None]-self.dense_pos[None], axis=-1)
        same = self.dense_group[:, None] == self.dense_group[None]
        adjacent = same & (distance>1e-8) & (distance<=.0045)
        # Total cross-talk is 5% of the adjacent-cell average, on all axes.
        self.dense_coupling = adjacent/np.maximum(adjacent.sum(1, keepdims=True), 1)

    def initial(self, key):
        s = super().initial(key)
        gain_key = jax.random.fold_in(key, 4004)
        s.update(dense_gain=jax.random.uniform(gain_key, (self.nd, 3), minval=.95, maxval=1.05),
            dense_bias=j.zeros((self.nd, 3)), dense_zero=j.zeros((self.nd, 3)),
            dense_age=j.zeros(self.nd), dense_open=j.array(0, j.int32),
            dense_queue=j.zeros((2, 4*self.nd)), packet=j.zeros(self.features))
        return s

    def raw(self, model, data):
        c = data._impl.contact
        geom = j.maximum(c.geom, 0)
        wrench = j.zeros((c.pos.shape[0], 6), data.qpos.dtype)
        for dim in np.unique(c.dim):
            force, ids = support.contact_force_dim(model, data, int(dim))
            wrench = wrench.at[ids].set(force)
        world = j.einsum('ci,cij->cj', wrench[:, :3], c.frame)
        rotations = data.xmat[self.dense_body]@j.asarray(self.dense_rot)
        positions = data.xpos[self.dense_body]+j.einsum('nij,nj->ni', data.xmat[self.dense_body], j.asarray(self.dense_pos))
        positions = j.concatenate((data.site_xpos[self.sites[:20]], positions))
        rotations = j.concatenate((data.site_xmat[self.sites[:20]], rotations))
        groups = j.concatenate((j.repeat(j.arange(5), 4), j.asarray(self.dense_group)))
        count = 20+self.nd
        forces, speed = j.zeros((count, 3)), j.zeros(count)
        active = j.all(c.geom>=0, -1) & (c.efc_address>=0)
        for side, sign in ((0, 1.), (1, -1.)):
            pad = j.asarray(self.lookup)[geom[:, side]]
            cell = assign(c.pos, pad, positions, groups)
            mask = (cell[:, None] == j.arange(count)) & active[:, None] & (pad[:, None]>=0)
            body = j.asarray(self.body)[geom[:, side]]
            other = j.asarray(self.body)[geom[:, 1-side]]
            def vel(b):
                return data.cvel[b, 3:]+j.cross(data.cvel[b, :3], c.pos-data.subtree_com[j.asarray(self.roots)[b]])
            rel = vel(other)-vel(body)
            tang = rel-j.sum(rel*c.frame[:, 0], -1, keepdims=True)*c.frame[:, 0]
            speed += mask.T@(j.linalg.norm(tang, axis=-1)*j.maximum(wrench[:, 0], 0))
            local = j.einsum('nji,nj->ni', rotations, mask.astype(world.dtype).T@(world*sign))
            forces += local[:, j.array([2, 0, 1])]
        return forces, speed

    def pooled_normal(self, measured):
        xyz = measured[:, j.array([1, 2, 0])]
        old = j.einsum('nij,nj->ni', j.asarray(self.dense_to_old), xyz)
        return j.asarray(self.dense_pool).T@j.maximum(old[:, 2], 0)

    def advance(self, state, raw, speed, hand_open):
        # Keep the optical transducer, keys, PVDF and slip implementation exact.
        old_raw = j.concatenate((raw[:20], j.zeros((20, 3))))
        old_speed = j.concatenate((speed[:20], j.zeros(20)))
        s = super().advance(state, old_raw, old_speed, hand_open)
        key = jax.random.fold_in(state['key'], 4004)
        drift_key, noise_key = jax.random.split(key)
        bias = state['dense_bias']+jax.random.normal(drift_key, (self.nd, 3))*(.1*j.sqrt(DT/600))
        signal = raw[20:]*state['dense_gain']+bias
        measured = signal-state['dense_zero']+jax.random.normal(noise_key, (self.nd, 3))*.01
        measured += .05*(j.asarray(self.dense_coupling)@measured)
        measured = measured.at[:, 0].set(j.clip(measured[:, 0], 0, 20))
        measured = measured.at[:, 1:].set(j.clip(measured[:, 1:], -20, 20))
        ideal = self.level == 'Ideal'
        if ideal:
            measured = raw[20:].at[:, 0].set(j.maximum(raw[20:, 0], 0))
            bias = j.zeros_like(bias)
        # Non-varied Realistic floor and delay retain their registered values.
        engaged = measured[:, 0] > (0. if ideal else .3)
        measured = j.where(engaged[:, None], measured, 0)
        age = j.where(engaged, state['dense_age']+DT, 0)
        # One observable open/unloaded decision across the actual sensor set.
        # The obsolete barometer noise generated by the compatibility packet
        # must not decide re-zero, nor may noiseless latent force do so.
        optical_keys=jax.random.split(state['key'],8)
        optical_signal=raw[:20]*state['gain'][:20]+s['bias'][:20]
        optical_measured=optical_signal-state['zero'][:20]+jax.random.normal(optical_keys[2],(40,3))[:20]*.05
        unloaded=(j.max(j.abs(optical_measured[:,0]))<.9)&(j.max(j.abs(measured[:,0]))<.9)
        opened=j.where(hand_open&unloaded,state['open_ticks']+1,0)
        rezero=(opened==100)&(not ideal)
        s.update(dense_bias=bias,dense_age=age,dense_open=opened,open_ticks=opened,
            dense_zero=j.where(rezero,signal,state['dense_zero']),
            zero=state['zero'].at[:20].set(j.where(rezero,optical_signal,state['zero'][:20])),
            rezero=state['rezero']+rezero.astype(j.int32))
        current = j.concatenate((measured.ravel(), j.minimum(age, 1)))
        delayed = current if ideal else state['dense_queue'][0]
        s['dense_queue'] = j.concatenate((state['dense_queue'][1:], current[None]))
        measured = delayed[:3*self.nd].reshape(self.nd, 3)
        old_age = j.max(j.where(j.asarray(self.dense_pool)>0, delayed[3*self.nd:, None], 0), axis=0)
        packet = s['packet'].at[100:120].set(self.pooled_normal(measured)).at[180:200].set(old_age)
        s['packet'] = j.concatenate((packet, measured.ravel()))
        return s

    def reading(self, packet, data):
        base = super().reading(packet, data)
        dense = packet[200:].reshape(self.nd, 3)
        shear = j.asarray(self.dense_membership).T@j.linalg.norm(dense[:, 1:], axis=-1)
        return base._replace(f_t=base.f_t+shear)


class BlindRotation(BaseEnvironment):
    def __init__(self, arm='c_touch', layout=None):
        if arm != 'c_touch':
            raise ValueError('Dense-skin intervention applies to C-touch only')
        super().__init__(arm)
        self.sensors.__class__ = Sensors
        layout = layout or json.loads(Path(os.environ['DEXTUM_DENSE_LAYOUT']).read_text())
        self.sensors.configure_dense(self.mj_model, layout)
        self.frame += 3*self.sensors.nd

    def reset(self, rng):
        state = super().reset(rng)
        info = dict(state.info)
        info['sensor_window'] = j.zeros(self.sensors.features)
        return state.replace(info=info)

    def _frame(self, official_state, info, reading):
        # Base reset first creates its historical 200-vector; pad only that
        # initialization. All subsequent packets already have the new width.
        if info['sensor_window'].shape[-1] == 200:
            info = {**info, 'sensor_window':j.pad(info['sensor_window'], (0, 3*self.sensors.nd))}
        return super()._frame(official_state, info, reading)
