"""E2: one DEXTUM edge, rest -> grasp (N1 -> N3), without vision.

The cube drops onto the upturned palm from the official random start and settles
for WARMUP steps under the home command (actions ignored). The policy then has
ACTIVE steps. Success: spec N3 (>= 3 firm pads including the thumb, grasp
quality >= 0.3) with the cube >= 5 mm above its resting height, still, and no
engaged grip pad slipping, for HOLD consecutive steps (the spec's 5 mm lift
verification). The episode continues after success; a drop ends it.

Arms differ only in the action space (A: official tendon targets; C: contact
commands through map G) and in the actor's touch inputs. Physics,
randomisation, reward and critic are shared.

Actor frame (3 frames): proprioception 15 = noisy tendon lengths and abduction
(official sensors, normalised like the action) | tendon command | clock;
C arms add G's state 35; touch arms add 11 pads x 10.
Critic: official privileged 81 | true touch 77 | task 10 | G state 35.
"""
import jax
import jax.numpy as jnp
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.aero_hand.rotate_z import CubeRotateZAxis

from dextum import contact_space as cs, touch_mjx

ARMS = {"a_touch": ("tendon", True), "a_notouch": ("tendon", False),
        "c_touch": ("contact", True), "c_notouch": ("contact", False)}
WARMUP = 10
ACTIVE = 100
EPISODE = WARMUP + ACTIVE
HOLD = 20
LIFT_M = 0.005
STILL_SPEED, STILL_ANGULAR = 0.01, 0.25
FIRM_N = 1.0
QUALITY_N3 = 0.3
DROP_Z = -0.05
HISTORY = 3
GAMMA = 0.97

# Per control step, OR1's scale; drop and smoothness are the official values times dt.
REWARD = dict(lift=4.0, success=5.0, progress=0.5, margin=0.1, slip=-1.0, quality=0.05, drop=-5.0, smooth=-0.05)

PALM = touch_mjx.PAD_NAMES.index("palm")
THUMB_PADS = (touch_mjx.PAD_NAMES.index("thumb_tip"), touch_mjx.PAD_NAMES.index("thumb_base"))
GRIP = np.array([p != PALM for p in range(touch_mjx.N_PADS)])
SENSOR_ACTUATOR = np.array([0, 1, 2, 3, 5, 6, 4])   # official sensor order: 6 tendons, then abduction

PROPRIO = 15
OFFICIAL_CRITIC = 81
TASK_CRITIC = 10
CRITIC_SIZE = OFFICIAL_CRITIC + touch_mjx.CRITIC_SIZE + TASK_CRITIC + cs.STATE_FEATURES
METRICS = ("success", "drop", "warmup_drop", "n3", "n2", "held", "lifted", "slip_onsets", "grip_slips", "grip_pads",
           "quality", "contact_flags", "lift_mm") + tuple("reward_" + k for k in REWARD)


def frame_size(arm):
    space, touch = ARMS[arm]
    return PROPRIO + (cs.STATE_FEATURES if space == "contact" else 0) + (touch_mjx.ACTOR_SIZE if touch else 0)


def zero_reading():
    p = touch_mjx.N_PADS
    z3, z = jnp.zeros((p, 3)), jnp.zeros(p)
    no = jnp.zeros(p, bool)
    return touch_mjx.Reading(force=z3, f_n=z, f_t=z, centroid=z3, point_obj=z3, normal_obj=z3, slip=no, margin=z, engaged=no)


class GraspEdge(CubeRotateZAxis):
    def __init__(self, arm="c_touch"):
        if arm not in ARMS:
            raise ValueError(f"Unknown arm {arm}; choose from {sorted(ARMS)}")
        super().__init__(config_overrides={"episode_length": EPISODE})
        self.arm = arm
        self.space, self.touch = ARMS[arm]
        self.touch_model = touch_mjx.TouchModel(self.mj_model)
        lo, hi = cs.bounds(self.mj_model.actuator_ctrlrange)
        self.lo, self.hi = jnp.asarray(lo), jnp.asarray(hi)
        self.estimator = cs.TendonEstimator() if (self.space == "contact" and not self.touch) else None
        self.frame = frame_size(arm)

    @property
    def action_size(self):
        return cs.ACTION_SIZE if self.space == "contact" else 7

    # ---------------------------------------------------------------- features
    def _frame(self, official_state, info, reading):
        sensors = (official_state[:7] - cs.HOME[SENSOR_ACTUATOR]) / cs.SCALE[SENSOR_ACTUATOR]
        clock = (info["tick"] - WARMUP).astype(jnp.float32) / ACTIVE
        parts = [sensors, info["u"], clock[None]]
        if self.space == "contact":
            parts.append(cs.features(info["g"], info["sense"]))
        if self.touch:
            parts.append(touch_mjx.actor_features(reading, info["touch_age"]))
        return jnp.clip(jnp.concatenate(parts), -5.0, 5.0)

    def _critic(self, privileged, info, reading, task):
        g = cs.features(info["g"], info["sense"]) if self.space == "contact" else jnp.zeros(cs.STATE_FEATURES)
        return jnp.concatenate((privileged, touch_mjx.critic_features(reading), task, g))

    # ---------------------------------------------------------------- env API
    def reset(self, rng):
        state = super().reset(rng)
        info = state.info
        info.update(tick=jnp.zeros((), jnp.int32), rest_z=jnp.zeros(()), hold=jnp.zeros((), jnp.int32),
                    success=jnp.zeros((), bool), g=cs.initial_state(), sense=cs.no_sense(),
                    press_length=jnp.asarray(cs.HOME[cs.PRESS]), touch_age=jnp.zeros(touch_mjx.N_PADS, jnp.int32),
                    grip_slip=jnp.zeros(touch_mjx.N_PADS, bool), n3=jnp.zeros((), bool), phi=jnp.full((), -2.0),
                    psi=jnp.zeros(()), u=jnp.zeros(7), last_act=jnp.zeros(7))
        reading = zero_reading()
        frame = self._frame(state.obs["state"][:14], info, reading)
        info["history"] = jnp.tile(frame, HISTORY)
        critic = self._critic(state.obs["privileged_state"], info, reading, jnp.zeros(TASK_CRITIC))
        metrics = {"e2/" + k: jnp.zeros(()) for k in METRICS}
        return state.replace(obs=dict(state=info["history"], privileged_state=critic), metrics=metrics, info=info)

    def step(self, state, action):
        info = dict(state.info)
        tick = info["tick"]
        warm = tick < WARMUP
        home = jnp.asarray(cs.HOME)
        if self.space == "contact":
            g, cmd = cs.step(info["g"], action, info["sense"], self.lo, self.hi, reflex=self.touch)
        else:
            g = info["g"]
            cmd = jnp.clip(home + jnp.asarray(cs.SCALE) * action, self.lo, self.hi)
        g = jax.tree.map(lambda a, b: jnp.where(warm, a, b), cs.initial_state(), g)
        cmd = jnp.where(warm, home, cmd)
        data = mjx_env.step(self.mjx_model, state.data, cmd, self.n_substeps)
        u = (cmd - home) / jnp.asarray(cs.SCALE)

        reading = self.touch_model.read(self.mjx_model, data)
        info["touch_age"] = jnp.where(reading.engaged, info["touch_age"] + 1, 0)
        if self.space == "contact" and self.touch:
            sense = cs.sense_touch(reading)
        elif self.space == "contact":
            info["rng"], key = jax.random.split(info["rng"])
            sense, info["press_length"] = self.estimator(data.actuator_length, info["press_length"], cmd, key)
        else:
            sense = cs.no_sense()

        tick1 = tick + 1
        active = ~warm
        z = data.qpos[self._cube_qids][2]
        rest_z = jnp.where(tick1 == WARMUP, z, info["rest_z"])
        lift = jnp.where(tick1 >= WARMUP, z - rest_z, 0.0)
        still = (jnp.linalg.norm(self.get_cube_linvel(data)) < STILL_SPEED) & \
                (jnp.linalg.norm(self.get_cube_angvel(data)) < STILL_ANGULAR)
        engaged = reading.engaged
        firm = reading.f_n >= FIRM_N
        grip = jnp.asarray(GRIP)
        n_grip = jnp.sum(engaged & grip)
        quality = touch_mjx.grasp_quality(reading)
        n3 = (jnp.sum(firm) >= 3) & (firm[THUMB_PADS[0]] | firm[THUMB_PADS[1]]) & (quality >= QUALITY_N3)
        n2 = engaged[PALM] & (n_grip >= 2) & still
        dist = jnp.where(n3, 0, jnp.where(n2, 1, 2))
        drop = z < DROP_Z
        grip_slip = reading.slip & engaged & grip
        slip_onsets = jnp.sum(grip_slip & ~info["grip_slip"])
        grip_slips = jnp.where(info["n3"], slip_onsets, 0)    # slip events of a formed grasp; sliding on approach is not one
        margin_ok = jnp.all(jnp.where(engaged & grip, reading.margin > 0, True)) & (n_grip >= 2)
        holding = n3 & (lift >= LIFT_M) & still & ~jnp.any(grip_slip) & active
        hold = jnp.where(holding, info["hold"] + 1, 0)
        onset = (hold >= HOLD) & ~info["success"]
        psi = jnp.where(n3, jnp.clip(lift / LIFT_M, 0.0, 1.0), 0.0)
        phi = jnp.where(drop, -3.0, -dist.astype(jnp.float32))

        terms = dict(lift=REWARD["lift"] * (psi - info["psi"]),
                     success=REWARD["success"] * onset,
                     progress=REWARD["progress"] * (GAMMA * phi - info["phi"]),
                     margin=REWARD["margin"] * margin_ok,
                     slip=REWARD["slip"] * grip_slips,
                     quality=REWARD["quality"] * quality * (jnp.sum(engaged) >= 3),
                     drop=REWARD["drop"] * drop,
                     smooth=REWARD["smooth"] * jnp.sum((u - info["u"]) ** 2))
        terms = {k: jnp.where(active, v, 0.0).astype(jnp.float32) for k, v in terms.items()}
        reward = sum(terms.values())

        info.update(tick=tick1, rest_z=rest_z, hold=hold, success=info["success"] | onset, g=g, sense=sense,
                    grip_slip=grip_slip, n3=n3 & active, phi=phi, psi=psi, u=u, last_act=u, motor_targets=cmd)
        official = self._get_obs(data, info, jnp.zeros(14))
        frame = self._frame(official["state"], info, reading)
        history = jnp.concatenate((frame, info["history"][:(HISTORY - 1) * self.frame]))
        info["history"] = history
        task = jnp.concatenate((jnp.clip(lift / LIFT_M, -5.0, 5.0)[None], psi[None],
                                (dist == jnp.arange(3)).astype(jnp.float32), (hold / HOLD)[None],
                                info["success"].astype(jnp.float32)[None],
                                ((tick1 - WARMUP) / ACTIVE)[None].astype(jnp.float32),
                                still.astype(jnp.float32)[None], (n_grip / 5.0)[None]))
        critic = self._critic(official["privileged_state"], info, reading, task)

        f = lambda x: (x & active).astype(jnp.float32) if x.dtype == bool else jnp.where(active, x, 0.0).astype(jnp.float32)
        metrics = dict(state.metrics)
        metrics.update({"e2/success": onset.astype(jnp.float32), "e2/drop": f(drop),
                   "e2/warmup_drop": (drop & warm).astype(jnp.float32), "e2/n3": f(n3), "e2/n2": f(n2),
                   "e2/held": f(holding), "e2/lifted": f(lift >= LIFT_M), "e2/slip_onsets": f(slip_onsets),
                   "e2/grip_slips": f(grip_slips),
                   "e2/grip_pads": f(n_grip), "e2/quality": f(quality),
                   "e2/contact_flags": f(jnp.sum(sense.contact)), "e2/lift_mm": f(1000.0 * lift)})
        metrics.update({"e2/reward_" + k: v for k, v in terms.items()})
        return state.replace(data=data, obs=dict(state=history, privileged_state=critic), reward=reward,
                             done=drop.astype(jnp.float32), metrics=metrics, info=info)


def make_env(arm="c_touch"):
    return GraspEdge(arm=arm)
