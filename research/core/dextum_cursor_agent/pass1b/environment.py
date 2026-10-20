"""Kit B Realistic variant; original E3 reward/task retained.

E3: goal rotation of the cube about z by +/-30 deg, without vision.

The cube drops onto the upturned palm from the official random start and settles
for WARMUP steps under the home command (actions ignored). At the end of the
warm-up a goal is issued: the resting orientation turned by +30 or -30 deg about
world z. The actor is told the goal once, through the interface: the goal's face
angle relative to the hand (period 90 deg, the cube's symmetry) and the turn
direction. It never sees the cube pose. Success (spec §6.1, OR1's test): twist
error within 10 deg and tilt within 30 deg for 40 consecutive steps (2 s). The
episode continues after success with the goal held until the 14 s timeout; a
drop ends it.

The four spec arms differ only in the action space (A: official tendon targets;
C: contact commands through map G) and in the actor's touch inputs. Physics,
randomisation, reward and critic are shared. The reference arm a_task is
A-no-touch trained on the task terms alone (current practice: no DEXTUM terms).

Actor frame (3 frames): proprioception 15 (as E2) | goal 3 | C arms: G state 35 |
touch arms: 11 pads x 10. Critic: official privileged 81 | true touch 77 | task 15 | G state 35.
"""
import os
if not os.environ.get("CONTAINER_ID"): raise RuntimeError("Vast.ai only")
import jax
import jax.numpy as jnp
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.aero_hand.rotate_z import CubeRotateZAxis

from dextum import contact_space as cs, touch_mjx
from pass1b import kit_model, sensors as kit_sensors
from pass1b.randomization import make_randomizer
from envs.grasp_edge import (ARMS as SPEC_ARMS, DROP_Z, FIRM_N, GRIP, PALM, PROPRIO, QUALITY_N3, SENSOR_ACTUATOR,
                             STILL_ANGULAR, STILL_SPEED, THUMB_PADS, zero_reading)

ARMS = {**SPEC_ARMS, "a_task": ("tendon", False)}
DEXTUM_TERMS = ("dextum", "margin", "slip", "quality")
TASK_ONLY = {"a_task"}

WARMUP = 20
ACTIVE = 280
EPISODE = WARMUP + ACTIVE
HOLD = 40
TURN = np.pi / 6
TOLERANCE = np.pi / 18
TILT_COS2 = float(np.cos(np.radians(30.0) / 2) ** 2)    # w^2 + z^2 of the error quaternion at 30 deg tilt
HISTORY = 3
GAMMA = 0.97
GOAL_INPUTS = 3

# Task: OR1's goal-yaw terms. dextum: 0.5 x (gamma phi' - phi) over the version-1 rotation graph.
# margin and quality are paid only at the goal; slip, drop and smooth as registered for E2.
REWARD = dict(progress=4.0, approach=0.05, holding=0.1, settling=-0.0025, time=-0.01, success=5.0, dextum=0.5,
              margin=0.1, slip=-1.0, quality=0.05, drop=-5.0, smooth=-0.05)
NODES = ("n6", "gait", "n3", "n2", "rest")    # graph distance 0..4 to N6; dropped is 5

OFFICIAL_CRITIC = 81
TASK_CRITIC = 10 + len(NODES)
CRITIC_SIZE = OFFICIAL_CRITIC + touch_mjx.CRITIC_SIZE + TASK_CRITIC + cs.STATE_FEATURES
METRICS = ("success", "success_time", "drop", "warmup_drop", "within", "abs_error_deg", "turned_deg", "grip_pads",
           "slip_onsets", "grip_slips", "quality", "contact_flags") + NODES + tuple("reward_" + k for k in REWARD)


def frame_size(arm):
    space, touch = ARMS[arm]
    return PROPRIO + GOAL_INPUTS + (cs.STATE_FEATURES if space == "contact" else 0) + \
        (kit_sensors.FEATURES if touch else 0)


def quat_mul(a, b):
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return jnp.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def conj(q):
    return q * jnp.array([1.0, -1.0, -1.0, -1.0])


def z_turn(angle):
    return jnp.stack([jnp.cos(angle / 2), jnp.zeros_like(angle), jnp.zeros_like(angle), jnp.sin(angle / 2)], -1)


def twist_error(current, goal):
    """Signed turn about world z still needed to reach the goal (OR1's measure) and whether tilt is <= 30 deg."""
    e = quat_mul(goal, conj(current))
    angle = 2 * jnp.arctan2(e[..., 3], e[..., 0])
    upright = e[..., 0] ** 2 + e[..., 3] ** 2 >= TILT_COS2
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle)), upright


def face_angle(xmat):
    """Horizontal direction of a cube side axis (rad); meaningful modulo 90 deg."""
    up = jnp.argmax(jnp.abs(xmat[2]))
    side = xmat[:, (up + 1) % 3]
    return jnp.arctan2(side[1], side[0])


class BlindRotation(CubeRotateZAxis):
    def __init__(self, arm="c_touch"):
        if arm not in ARMS:
            raise ValueError(f"Unknown arm {arm}; choose from {sorted(ARMS)}")
        super().__init__(config_overrides={"episode_length": EPISODE})
        self.arm = arm
        self.space, self.touch = ARMS[arm]
        cells = kit_model.install(self)
        self.randomize = make_randomizer(self.mj_model, cells)
        self.sensors = kit_sensors.Sensors(self.mj_model, cells, self.touch_model)
        lo, hi = cs.bounds(self.mj_model.actuator_ctrlrange)
        self.lo, self.hi = jnp.asarray(lo), jnp.asarray(hi)
        self.estimator = cs.TendonEstimator() if (self.space == "contact" and not self.touch) else None
        self.cube_body = self.mj_model.body("cube").id
        self.frame = frame_size(arm)

    @property
    def action_size(self):
        return cs.ACTION_SIZE if self.space == "contact" else 7

    # ---------------------------------------------------------------- features
    def _goal_inputs(self, info):
        g = 4.0 * info["goal_face"]
        goal = jnp.stack([jnp.cos(g), jnp.sin(g), info["turn_sign"]])
        return jnp.where(info["tick"] >= WARMUP, goal, 0.0)

    def _frame(self, official_state, info, reading):
        sensors = (official_state[:7] - cs.HOME[SENSOR_ACTUATOR]) / cs.SCALE[SENSOR_ACTUATOR]
        clock = (info["tick"] - WARMUP).astype(jnp.float32) / ACTIVE
        parts = [sensors, info["u"], clock[None], self._goal_inputs(info)]
        if self.space == "contact":
            parts.append(cs.features(info["g"], info["sense"]))
        if self.touch:
            parts.append(info["sensor_window"])
        return jnp.clip(jnp.concatenate(parts), -5.0, 5.0)

    def _critic(self, privileged, info, reading, task):
        g = cs.features(info["g"], info["sense"]) if self.space == "contact" else jnp.zeros(cs.STATE_FEATURES)
        return jnp.concatenate((privileged, touch_mjx.critic_features(reading), task, g))

    # ---------------------------------------------------------------- env API
    def reset(self, rng):
        state = super().reset(rng)
        info = state.info
        info["rng"], key = jax.random.split(info["rng"])
        sign = jnp.where(jax.random.bernoulli(key), 1.0, -1.0)
        info.update(tick=jnp.zeros((), jnp.int32), hold=jnp.zeros((), jnp.int32), success=jnp.zeros((), bool),
                    turn_sign=sign, goal=jnp.array([1.0, 0.0, 0.0, 0.0]), goal_face=jnp.zeros(()),
                    error=jnp.zeros(()), phi=jnp.full((), -4.0), g=cs.initial_state(), sense=cs.no_sense(),
                    press_length=jnp.asarray(cs.HOME[cs.PRESS]), touch_age=jnp.zeros(touch_mjx.N_PADS, jnp.int32),
                    grip_slip=jnp.zeros(touch_mjx.N_PADS, bool), n3=jnp.zeros((), bool),
                    u=jnp.zeros(7), last_act=jnp.zeros(7))
        info["sensor"] = self.sensors.initial(jax.random.fold_in(rng, 20261011))
        info["sensor_window"] = jnp.zeros(kit_sensors.FEATURES)
        reading = zero_reading()
        frame = self._frame(state.obs["state"][:14], info, reading)
        info["history"] = jnp.tile(frame, HISTORY)
        critic = self._critic(state.obs["privileged_state"], info, reading, jnp.zeros(TASK_CRITIC))
        metrics = {"e3/" + k: jnp.zeros(()) for k in METRICS}
        return state.replace(obs=dict(state=info["history"], privileged_state=critic), metrics=metrics, info=info)

    def step(self, state, action):
        info = dict(state.info)
        tick = info["tick"]
        warm = tick < WARMUP
        active = ~warm
        home = jnp.asarray(cs.HOME)
        if self.space == "contact":
            g, cmd = cs.step(info["g"], action, info["sense"], self.lo, self.hi, reflex=self.touch)
        else:
            g = info["g"]
            cmd = jnp.clip(home + jnp.asarray(cs.SCALE) * action, self.lo, self.hi)
        g = jax.tree.map(lambda a, b: jnp.where(warm, a, b), cs.initial_state(), g)
        cmd = jnp.where(warm, home, cmd)
        def substep(carry, _):
            data, sensor = carry
            data = mjx_env.step(self.mjx_model, data, cmd, 1)
            raw, speed = self.sensors.raw(self.mjx_model, data)
            # Open declaration from measured tendon length, not contact truth.
            opened = jnp.all(data.actuator_length[:4] > self.hi[:4] - .003)
            sensor = self.sensors.advance(sensor, raw, speed, opened)
            return (data, sensor), sensor["packet"]
        (data, info["sensor"]), packets = jax.lax.scan(substep, (state.data, info["sensor"]), None, length=self.n_substeps)
        window = jnp.mean(packets, axis=0)
        # Keep transient peaks/onsets and slip events over the 50-ms window.
        window = window.at[135:160].set(jnp.max(packets[:,135:160], axis=0))
        info["sensor_window"] = window
        u = (cmd - home) / jnp.asarray(cs.SCALE)

        reading = self.touch_model.read(self.mjx_model, data)
        info["touch_age"] = jnp.where(reading.engaged, info["touch_age"] + 1, 0)
        if self.space == "contact" and self.touch:
            sense = cs.sense_touch(self.sensors.reading(info["sensor"]["packet"], data))
        elif self.space == "contact":
            info["rng"], key = jax.random.split(info["rng"])
            sense, info["press_length"] = self.estimator(data.actuator_length, info["press_length"], cmd, key)
        else:
            sense = cs.no_sense()

        # Goal and twist error. The goal is issued as the warm-up ends.
        tick1 = tick + 1
        issued = tick1 >= WARMUP
        q = data.qpos[self._cube_qids][3:7]
        issue = tick1 == WARMUP
        goal = jnp.where(issue, quat_mul(z_turn(info["turn_sign"] * TURN), q), info["goal"])
        goal_face = jnp.where(issue, face_angle(data.xmat[self.cube_body]) + info["turn_sign"] * TURN,
                              info["goal_face"])
        error, upright = twist_error(q, goal)
        previous = jnp.where(issue, error, info["error"])
        omega = self.get_cube_angvel(data)
        drop = data.qpos[self._cube_qids][2] < DROP_Z
        within = (jnp.abs(error) < TOLERANCE) & upright & ~drop & (tick1 > WARMUP)
        hold = jnp.where(within, info["hold"] + 1, 0)
        onset = (hold >= HOLD) & ~info["success"]

        # Contact state and the version-1 rotation graph N1 -> N2 -> N3 -> N4/N5 -> N6 (spec §6.2-6.3).
        engaged = reading.engaged
        grip = jnp.asarray(GRIP)
        n_grip = jnp.sum(engaged & grip)
        firm = reading.f_n >= FIRM_N
        digit_firm = jnp.asarray(cs.MEMBERSHIP) @ firm.astype(jnp.float32) > 0
        fingers_firm = jnp.sum(digit_firm[:4])
        still = (jnp.linalg.norm(self.get_cube_linvel(data)) < STILL_SPEED) & (jnp.linalg.norm(omega) < STILL_ANGULAR)
        quality = touch_mjx.grasp_quality(reading)
        n3 = (jnp.sum(firm) >= 3) & (firm[THUMB_PADS[0]] | firm[THUMB_PADS[1]]) & (quality >= QUALITY_N3)
        n6 = within & (jnp.sum(firm) >= 3) & still
        gait = digit_firm[4] & (fingers_firm >= 2) & (fingers_firm <= 3) & (jnp.abs(omega[2]) >= STILL_ANGULAR)
        n2 = engaged[PALM] & (n_grip >= 2) & still
        dist = jnp.select([n6, gait, n3, n2], [0, 1, 2, 3], 4)
        phi = jnp.where(drop, -5.0, -dist.astype(jnp.float32))

        grip_slip = reading.slip & engaged & grip
        slip_onsets = jnp.sum(grip_slip & ~info["grip_slip"])
        grip_slips = jnp.where(info["n3"], slip_onsets, 0)
        margin_ok = jnp.all(jnp.where(engaged & grip, reading.margin > 0, True)) & (n_grip >= 2)

        terms = dict(progress=REWARD["progress"] * (jnp.abs(previous) - jnp.abs(error)),
                     approach=REWARD["approach"] * jnp.clip(jnp.sign(error) * omega[2], -3.0, 3.0) * ~within,
                     holding=REWARD["holding"] * within,
                     settling=REWARD["settling"] * jnp.sum(omega ** 2) * within,
                     time=REWARD["time"] * ~within,
                     success=REWARD["success"] * onset,
                     dextum=REWARD["dextum"] * (GAMMA * phi - info["phi"]),
                     margin=REWARD["margin"] * (margin_ok & within),
                     slip=REWARD["slip"] * grip_slips,
                     quality=REWARD["quality"] * quality * ((jnp.sum(engaged) >= 3) & within),
                     drop=REWARD["drop"] * drop,
                     smooth=REWARD["smooth"] * jnp.sum((u - info["u"]) ** 2))
        if self.arm in TASK_ONLY:
            terms.update({k: jnp.zeros(()) for k in DEXTUM_TERMS})
        terms = {k: jnp.where(active, v, 0.0).astype(jnp.float32) for k, v in terms.items()}
        reward = sum(terms.values())

        info.update(tick=tick1, hold=hold, success=info["success"] | onset, goal=goal, goal_face=goal_face,
                    error=error, phi=phi, g=g, sense=sense, grip_slip=grip_slip, n3=n3 & active, u=u, last_act=u,
                    motor_targets=cmd)
        official = self._get_obs(data, info, jnp.zeros(14))
        frame = self._frame(official["state"], info, reading)
        history = jnp.concatenate((frame, info["history"][:(HISTORY - 1) * self.frame]))
        info["history"] = history
        g4 = 4.0 * goal_face
        task = jnp.concatenate((jnp.stack([jnp.sin(error), jnp.cos(error), within.astype(jnp.float32), hold / HOLD,
                                           info["success"].astype(jnp.float32), (tick1 - WARMUP) / ACTIVE,
                                           info["turn_sign"], jnp.cos(g4), jnp.sin(g4), omega[2] / 3.0]),
                                (dist == jnp.arange(len(NODES))).astype(jnp.float32))).astype(jnp.float32)
        task = jnp.where(issued, task, 0.0)
        critic = self._critic(official["privileged_state"], info, reading, task)

        f = lambda x: jnp.where(active, x, 0.0).astype(jnp.float32)
        metrics = dict(state.metrics)
        metrics.update({"e3/success": onset.astype(jnp.float32),
                        "e3/success_time": f(onset * (tick1 - WARMUP) * self.dt),
                        "e3/drop": f(drop), "e3/warmup_drop": (drop & warm).astype(jnp.float32),
                        "e3/within": f(within), "e3/abs_error_deg": f(jnp.degrees(jnp.abs(error))),
                        "e3/turned_deg": f(jnp.degrees(jnp.abs(previous) - jnp.abs(error))),
                        "e3/grip_pads": f(n_grip), "e3/slip_onsets": f(slip_onsets), "e3/grip_slips": f(grip_slips),
                        "e3/quality": f(quality), "e3/contact_flags": f(jnp.sum(sense.contact))})
        metrics.update({"e3/" + k: f(dist == i) for i, k in enumerate(NODES)})
        metrics.update({"e3/reward_" + k: v for k, v in terms.items()})
        return state.replace(data=data, obs=dict(state=history, privileged_state=critic), reward=reward,
                             done=drop.astype(jnp.float32), metrics=metrics, info=info)


def make_env(arm="c_touch"):
    return BlindRotation(arm=arm)
