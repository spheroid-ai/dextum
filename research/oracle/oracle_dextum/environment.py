"""O1 observation adapters around the existing rotation and hold mechanics.

No reward, termination, stepping sequence, reset distribution or C connector
is changed. The privileged experiment uses Ideal sensing in every arm. The
old sensor observation remains internal to the connector; the actor gets the
explicitly declared oracle interface below. Training qualification is required
on this new observation/compilation path before learning.
"""
import os
if not os.environ.get('CONTAINER_ID'):
    raise RuntimeError('Oracle environment numerical execution is cloud-only')

import jax
import jax.numpy as j
import numpy as np
import vendor.aero_official.vision_env  # pinned Playground import path
from dextum import contact_space as cs, touch_mjx
from pass1c.environment import BlindRotation, HISTORY, WARMUP
from .observations_mjx import CubeContactOracle, O1_ARMS, interface_for_arm


def cube_geometry(dtype):
    """Six named planar patches, outward normals, half-edge and adjacency.

    Opposite faces share no edge; a face is adjacent to the other four. The
    flat arm receives this same explicit table, not just contact magnitudes.
    """
    normals = j.repeat(j.eye(3, dtype=dtype), 2, axis=0)*j.tile(j.array([-1., 1.], dtype), 3)[:, None]
    # Curvature class flat/convex/concave and radii reciprocal (zero for flat).
    descriptors = j.concatenate((j.tile(j.array([[1., 0., 0.]], dtype), (6, 1)),
                                 normals, j.zeros((6, 2), dtype)), axis=-1)
    faces = j.arange(6)
    adjacency = (faces[:, None]//2 != faces[None, :]//2).astype(dtype)
    return j.concatenate((descriptors.ravel(), adjacency.ravel(), j.array([.025], dtype)))


class OracleObservationMixin:
    def configure_oracle(self, arm, task):
        if arm not in O1_ARMS:
            raise ValueError(arm)
        if os.environ.get('DEXTUM_SENSOR_LEVEL') != 'Ideal':
            raise ValueError('O1 is a representation test with the same Ideal connector in every arm')
        self.oracle_arm, self.oracle_task = arm, task
        self.oracle = CubeContactOracle(self.mj_model, self.touch_model)
        # A single frozen permutation, shared across arms, environments and
        # seeds. The flat control is not made to learn episode-wise sorting.
        self.oracle_permutation = j.asarray(np.random.default_rng(6301).permutation(72), j.int32)
        self.oracle_record_shape = (HISTORY, 72, 27)

    def _ceiling_raw(self, state):
        """Explicit current simulator/connector state; no future RNG draws.

        This block exposes hidden hand coordinates and numerical warm-start
        state. Contacts are supplied through the shared oracle records. It is
        a finite-state privileged ceiling, not an oracle of future outcomes.
        The full inventory is recorded with the environment contract.
        """
        data, model, info = state.data, self.mjx_model, state.info
        arrays = [data.qpos, data.qvel, data.act, data.ctrl, data.qacc_warmstart,
                  data.qfrc_applied, data.xfrc_applied,
                  model.body_mass, model.body_ipos, model.body_iquat, model.body_inertia,
                  model.geom_friction, model.geom_solref, model.geom_solimp,
                  model.actuator_gainprm, model.actuator_biasprm, model.actuator_gear]
        # The same complete current state is the critic input for every arm.
        arrays += jax.tree.leaves(info['g']) + jax.tree.leaves(info['sense'])
        # Sensor memory is causal. Exclude random keys, which are not physical
        # state and must never allow a policy to predict a future noise draw.
        arrays += [value for key, value in sorted(info['sensor'].items()) if key != 'key']
        return j.concatenate([j.asarray(value, j.float32).ravel() for value in arrays])

    def _oracle_observation(self, state, reset=False):
        info = dict(state.info)
        full = self.oracle.read_full(self.mjx_model, state.data)
        interface = interface_for_arm(full, self.oracle_permutation, self.oracle_arm)
        records = interface['records']
        reading = self.touch_model.read(self.mjx_model, state.data)
        engaged = reading.engaged
        margin = j.min(j.where(engaged, reading.margin, j.inf))
        margin = j.where(j.any(engaged), margin, 0.)
        grip = j.array([j.sum(reading.f_n), j.sum(engaged), margin,
                       touch_mjx.grasp_quality(reading)], j.float32)
        # Existing measured proprioception, issued goal and contact connector
        # state (including measured aggregate forces): common to all arms.
        common = info['history'][:53]
        intended = info.get('intended', j.zeros(11, bool)).astype(j.float32)
        issued_goal = j.where(info['tick'] >= WARMUP, info['goal'], j.zeros(4))
        if self.oracle_task == 'rotation':
            task_memory = j.array([info['hold']/40., info['success'], 0., 0.], j.float32)
            if self.oracle_arm == 'oracle_contact_only':
                # This goal-hold counter is derived from pose. Removing the
                # pose block while retaining it would leak the omitted field.
                task_memory = j.zeros_like(task_memory)
        else:
            task_memory = j.array([info['hold_intact'], info['hold_valid'],
                                   info['hold_success'], info['hold_dropped']], j.float32)
        pvdf = info['sensor_window'][135:155]
        visibility = j.array([1., 0., 1.], j.float32)
        # Use physical scaling only; no actor clipping or learned normalizer
        # can silently destroy the information-equivalence control.
        object_scale = j.array([.1]*3+[1.]*4+[1.]*3+[10.]*3)
        frame = j.concatenate((common, intended, issued_goal, task_memory,
            interface['object_state']/object_scale, interface['physical_parameters'],
            grip/j.array([10., 11., 1., 1.]), visibility, pvdf,
            info['touch_age'].astype(j.float32)/20., cube_geometry(j.float32)))
        if reset:
            record_history = j.tile(records[None], (HISTORY, 1, 1))
            global_history = j.tile(frame[None], (HISTORY, 1))
        else:
            record_history = j.concatenate((records[None], info['oracle_records'][:-1]))
            global_history = j.concatenate((frame[None], info['oracle_global'][:-1]))
        info.update(oracle_records=record_history, oracle_global=global_history,
                    oracle_unmapped_contacts=full['unmapped_contacts'],
                    oracle_base_obs=state.obs)
        raw = self._ceiling_raw(state)
        observation = dict(records=record_history, state=global_history.ravel(),
            ceiling=raw if self.oracle_arm == 'privileged_ceiling' else j.zeros_like(raw),
            privileged_state=j.concatenate((state.obs['privileged_state'], raw)))
        return state.replace(obs=observation, info=info)

    def reset(self, rng):
        return self._oracle_observation(super().reset(rng), reset=True)

    def step(self, state, action):
        # The existing environments construct their own critic observation and
        # preserve the internal sensor history. No oracle field is an action.
        result = super().step(state.replace(obs=state.info['oracle_base_obs']), action)
        return self._oracle_observation(result)


class OracleRotation(OracleObservationMixin, BlindRotation):
    def __init__(self, oracle_arm):
        super().__init__('c_touch')
        self.configure_oracle(oracle_arm, 'rotation')


