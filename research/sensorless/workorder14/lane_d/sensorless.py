"""Sensorless motor interface and privileged teacher labels for D2.

Physics, sensors, reward and goal logic are the accepted v2 E3 implementation.
The student executes seven motor targets directly. The teacher's contact-space
G is used only to label the current state; it cannot modify the student command.
The copied step below differs at the motor boundary and observation return only.
Numerical execution requires a GPU cloud host.
"""
import os
if not os.environ.get('CONTAINER_ID'):
    raise RuntimeError('GPU cloud only')
os.environ['JAX_PLATFORMS'] = 'cuda,cpu'  # CPU device for host callbacks; model execution must remain on GPU.
from pass1c.environment import *
from pass1c.environment import BlindRotation as BaseRotation
from vendor.aero_official.vision_oracle import (
    TagOracle, initial_sensor, sample_qualified_cameras, advance_sensor, features)

STUDENT_FRAME = 31  # 15 proprio/command/clock + 3 task + 13 measured vision
STUDENT_WIDTH = HISTORY * STUDENT_FRAME


class SensorlessRotation(BaseRotation):
    """Accepted physical task, direct actuator action, causal sensorless actor."""
    def __init__(self, teacher_inference):
        super().__init__('c_touch')
        self.teacher_frame = self.frame
        self.frame = STUDENT_FRAME
        self.teacher_inference = teacher_inference
        self.tag_oracle = TagOracle(self.mj_model)

    @property
    def action_size(self):
        return 7

    def teacher(self, state):
        action = self.teacher_inference(dict(state=state.info['history'],
            privileged_state=state.obs['privileged_state']), jax.random.PRNGKey(0))[0]
        g, command = cs.step(state.info['g'], action, state.info['sense'],
                             self.lo, self.hi, reflex=True)
        warm = state.info['tick'] < WARMUP
        command = jnp.where(warm, jnp.asarray(cs.HOME), command)
        target = jnp.clip((command - jnp.asarray(cs.HOME)) / jnp.asarray(cs.SCALE), -1, 1)
        return g, target

    def _student_observation(self, state, *, initial=False):
        info = dict(state.info)
        vision = features(info['vision_packet'], info['vision_sensor']['age'])
        # The first 18 teacher-frame entries contain proprioception, previous
        # command, clock and the issued task. G and all touch start at index18.
        frame = jnp.clip(jnp.concatenate((info['history'][:18], vision)), -5, 5)
        history = jnp.tile(frame, HISTORY) if initial else jnp.concatenate((frame,
            info['student_history'][:(HISTORY - 1) * STUDENT_FRAME]))
        info['student_history'] = history
        obs = dict(state=history, privileged_state=state.obs['privileged_state'])
        result = state.replace(obs=obs, info=info)
        # Labels are kept in a separately named field. The deployable student
        # accepts only `state`, never the label or teacher/G hidden history.
        _, target = self.teacher(result)
        return result.replace(obs={**obs, 'expert_action': target})

    def reset(self, rng):
        state = super().reset(rng)
        key, camera = jax.random.split(jax.random.fold_in(rng, 20261014))
        sensor = initial_sensor()
        info = {**state.info, 'vision_rng': key, 'vision_cameras': sample_qualified_cameras(camera),
                'vision_sensor': sensor, 'vision_packet': sensor['last'],
                'student_history': jnp.zeros(STUDENT_WIDTH)}
        return self._student_observation(state.replace(info=info), initial=True)

    def _after_motor_step(self, state):
        info = dict(state.info)
        key, capture_key, delay_key = jax.random.split(info['vision_rng'], 3)
        palm = self.tag_oracle.palm_id
        packet, _ = self.tag_oracle.capture(capture_key, info['vision_cameras'],
            state.data.xpos[self.cube_body], state.data.xquat[self.cube_body],
            state.data.geom_xpos, state.data.geom_xmat, state.data.xpos[palm], state.data.xquat[palm])
        sensor, packet = advance_sensor(info['vision_sensor'], packet,
                                        jax.random.bernoulli(delay_key, .2).astype(jnp.int32))
        info.update(vision_rng=key, vision_sensor=sensor, vision_packet=packet)
        return self._student_observation(state.replace(info=info))

    def step(self, state, action):
        info = dict(state.info)
        tick = info["tick"]
        warm = tick < WARMUP
        active = ~warm
        home = jnp.asarray(cs.HOME)
        # Expert G generates a supervised target only. The physical command
        # is exactly the seven-dimensional student action, with no force reflex.
        g, _ = self.teacher(state)
        cmd = jnp.clip(home + jnp.asarray(cs.SCALE) * action, self.lo, self.hi)
        g = g._replace(cmd=cmd)
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
        history = jnp.concatenate((frame, info["history"][:(HISTORY - 1) * self.teacher_frame]))
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
        result = state.replace(data=data, obs=dict(state=history, privileged_state=critic), reward=reward,
                               done=drop.astype(jnp.float32), metrics=metrics, info=info)
        return self._after_motor_step(result)

