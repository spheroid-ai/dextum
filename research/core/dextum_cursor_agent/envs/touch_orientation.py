"""Goal-yaw rotation with camera blackouts, cube pushes and optional ideal touch.

Built on the frozen OR1 curriculum (stage 0: 15-60 deg yaw, +/-10 deg, 2 s hold),
whose reward, success, drop and timeout rules are unchanged. Every arm shares
the same physics, perturbations and critic; arms differ only in actor inputs.

Actor frame (current first, three frames):
    base 34 = motors 14 | camera features 13 | goal error 6 | clock 1
    touch    = 11 pads x 10 (touch='ideal' only)
Critic: base 89 | true touch 11 x 7 | perturbation state 6.
"""
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from vendor.aero_official.goal_yaw import WARMUP
from vendor.aero_official.orientation_curriculum import OrientationCurriculum, STAGES

from dextum import touch_mjx

BASE_ACTOR = 34
BASE_CRITIC = 89
PERTURB_CRITIC = 6
HISTORY = 3
FREEZE, DROP = 0, 1


@dataclass(frozen=True)
class Perturbations:
    blackout_prob: float = 0.3
    blackout_ticks: tuple = (10, 60)          # 0.5-3 s at 20 Hz
    blackout_start: tuple = (WARMUP + 2, WARMUP + 120)
    freeze_prob: float = 0.5
    push_prob: float = 0.2
    push_newton: tuple = (0.5, 2.0)
    push_ticks: int = 4                       # 0.2 s
    push_start: tuple = (WARMUP + 2, WARMUP + 120)


@dataclass(frozen=True)
class Forced:
    """Evaluation perturbations. A start is a tick, a (lo, hi) range drawn from
    the reset key (so shared across arms), or -1 to disable."""
    blackout_start: object = -1
    blackout_ticks: int = 0
    blackout_mode: int = DROP
    push_start: object = -1
    push_newton: float = 0.0


def _tick(key, start):
    if isinstance(start, tuple):
        return jax.random.randint(key, (), start[0], start[1] + 1).astype(jnp.int32)
    return jnp.array(start, jnp.int32)


def actor_frame_size(touch):
    return BASE_ACTOR + (touch_mjx.ACTOR_SIZE if touch == "ideal" else 0)


CRITIC_SIZE = BASE_CRITIC + touch_mjx.CRITIC_SIZE + PERTURB_CRITIC


class TouchOrientation(OrientationCurriculum):
    def __init__(self, touch="none", stage=0, perturb=Perturbations(), forced=None):
        if touch not in ("none", "ideal"):
            raise ValueError(f"Unknown touch arm: {touch}")
        super().__init__(STAGES[stage], pose_mode="camera")
        self.touch = touch
        self.perturb = perturb
        self.forced = forced
        self.touch_model = touch_mjx.TouchModel(self.mj_model)
        self.cube_body = self.mj_model.body("cube").id
        self.frame = actor_frame_size(touch)

    # ---------------------------------------------------------------- schedule
    def _schedule(self, key):
        p = self.perturb
        k = jax.random.split(key, 8)
        if self.forced is not None:
            f = self.forced
            angle = jax.random.uniform(k[6], minval=0.0, maxval=2 * jnp.pi)
            start = _tick(k[1], f.blackout_start)
            on = (start >= 0) & (f.blackout_ticks > 0)
            return dict(
                blackout_start=jnp.where(on, start, -1).astype(jnp.int32),
                blackout_end=jnp.where(on, start + f.blackout_ticks, -1).astype(jnp.int32),
                blackout_mode=jnp.array(f.blackout_mode, jnp.int32),
                push_start=_tick(k[5], f.push_start),
                push_force=f.push_newton * jnp.array([jnp.cos(angle), jnp.sin(angle), 0.0]))
        has_blackout = jax.random.bernoulli(k[0], p.blackout_prob)
        start = jax.random.randint(k[1], (), p.blackout_start[0], p.blackout_start[1] + 1)
        length = jax.random.randint(k[2], (), p.blackout_ticks[0], p.blackout_ticks[1] + 1)
        mode = jnp.where(jax.random.bernoulli(k[3], p.freeze_prob), FREEZE, DROP)
        has_push = jax.random.bernoulli(k[4], p.push_prob)
        push_start = jax.random.randint(k[5], (), p.push_start[0], p.push_start[1] + 1)
        angle = jax.random.uniform(k[6], minval=0.0, maxval=2 * jnp.pi)
        magnitude = jax.random.uniform(k[7], minval=p.push_newton[0], maxval=p.push_newton[1])
        return dict(
            blackout_start=jnp.where(has_blackout, start, -1).astype(jnp.int32),
            blackout_end=jnp.where(has_blackout, start + length, -1).astype(jnp.int32),
            blackout_mode=mode.astype(jnp.int32),
            push_start=jnp.where(has_push, push_start, -1).astype(jnp.int32),
            push_force=magnitude * jnp.array([jnp.cos(angle), jnp.sin(angle), 0.0]))

    def _blackout_active(self, info, tick):
        s = info["schedule"]
        return (s["blackout_start"] >= 0) & (tick >= s["blackout_start"]) & (tick < s["blackout_end"])

    def _push_active(self, info, tick):
        s = info["schedule"]
        return (s["push_start"] >= 0) & (tick >= s["push_start"]) & (tick < s["push_start"] + self.perturb.push_ticks)

    # ---------------------------------------------------------------- features
    def _touch(self, data, age):
        reading = self.touch_model.read(self.mjx_model, data)
        age = jnp.where(reading.engaged, age + 1, 0)
        return reading, age

    def _compose(self, obs, info, reading):
        tick = info["age"]
        blackout = self._blackout_active(info, tick)
        push = self._push_active(info, tick)
        freeze = blackout & (info["schedule"]["blackout_mode"] == FREEZE)
        perturb = jnp.concatenate((jnp.stack([blackout, freeze, push]).astype(jnp.float32),
                                   jnp.where(push, info["schedule"]["push_force"], 0.0)))
        critic = jnp.concatenate((obs["privileged_state"], touch_mjx.critic_features(reading), perturb))
        frame = obs["state"]
        if self.touch == "ideal":
            frame = jnp.concatenate((frame, touch_mjx.actor_features(reading, info["touch_age"])))
        return frame, critic

    # ---------------------------------------------------------------- env API
    def reset(self, key):
        state = super().reset(key)
        info = state.info
        info["schedule"] = self._schedule(jax.random.fold_in(key, 9091))
        info["frozen_packet"] = info["packet"]
        info["frozen_sensor"] = info["sensor"]
        info["touch_age"] = jnp.zeros(touch_mjx.N_PADS, jnp.int32)
        reading, info["touch_age"] = self._touch(state.data, info["touch_age"])
        frame, critic = self._compose(state.obs, info, reading)
        info["dextum_history"] = jnp.tile(frame, HISTORY)
        for name in ("blackout", "push", "engaged_pads", "slipping_pads", "touch_force", "shear", "quality"):
            state.metrics["dextum_" + name] = jnp.zeros(())
        return state.replace(obs=dict(state=info["dextum_history"], privileged_state=critic))

    def _advance_physics(self, state, action):
        tick = state.info["age"] + 1
        push = self._push_active(state.info, tick)
        xfrc = state.data.xfrc_applied.at[self.cube_body, :3].set(
            jnp.where(push, state.info["schedule"]["push_force"], 0.0))
        return super()._advance_physics(state.replace(data=state.data.replace(xfrc_applied=xfrc)), action)

    def step(self, state, action):
        info = state.info
        tick = info["age"] + 1
        blackout = self._blackout_active(info, tick)
        starting = blackout & ~self._blackout_active(info, info["age"])
        # Freeze holds the last pre-blackout packet; drop disables the cameras.
        frozen_packet = jax.tree.map(lambda a, b: jnp.where(starting, a, b), info["packet"], info["frozen_packet"])
        frozen_sensor = jax.tree.map(lambda a, b: jnp.where(starting, a, b), info["sensor"], info["frozen_sensor"])
        drop = blackout & (info["schedule"]["blackout_mode"] == DROP)
        enabled = info["cameras"]["enabled"]
        cameras = dict(info["cameras"], enabled=enabled & ~drop)
        base = dict(state=state.obs["state"][:BASE_ACTOR], privileged_state=state.obs["privileged_state"][:BASE_CRITIC])
        state = state.replace(obs=base, info=dict(info, cameras=cameras))
        state = super().step(state, action)
        info = state.info
        info["cameras"] = dict(info["cameras"], enabled=enabled)
        freeze = blackout & (info["schedule"]["blackout_mode"] == FREEZE)
        info["packet"] = jax.tree.map(lambda a, b: jnp.where(freeze, a, b), frozen_packet, info["packet"])
        info["sensor"] = jax.tree.map(lambda a, b: jnp.where(freeze, a, b), frozen_sensor, info["sensor"])
        info["frozen_packet"], info["frozen_sensor"] = frozen_packet, frozen_sensor
        q = state.data.qpos[self._cube_qids][3:7]
        base = dict(state=state.obs["state"][:14], privileged_state=state.obs["privileged_state"][:81])
        obs = self.observations(base, info, q)
        reading, info["touch_age"] = self._touch(state.data, info["touch_age"])
        frame, critic = self._compose(obs, info, reading)
        history = jnp.concatenate((frame, info["dextum_history"][:(HISTORY - 1) * self.frame]))
        info["dextum_history"] = history
        state.metrics.update(
            dextum_blackout=blackout.astype(jnp.float32),
            dextum_push=self._push_active(info, info["age"]).astype(jnp.float32),
            dextum_engaged_pads=jnp.sum(reading.engaged).astype(jnp.float32),
            dextum_slipping_pads=jnp.sum(reading.slip).astype(jnp.float32),
            dextum_touch_force=jnp.sum(reading.f_n),
            dextum_shear=jnp.sum(reading.f_t),
            dextum_quality=touch_mjx.grasp_quality(reading))
        return state.replace(obs=dict(state=history, privileged_state=critic), info=info)


def make_env(touch="none", **kwargs):
    return TouchOrientation(touch=touch, **kwargs)
