"""Carry a mixed-axis MJX model as rollout state, outside per-env auto-reset.

Randomized arrays have an environment axis; nominal arrays are shared. The
model is simulator state only, never an actor observation or PPO transition.
Historical source and prior qualification releases remain unchanged.
"""
import contextlib

import jax
from brax.envs.wrappers import training
from mujoco_playground import wrapper

MODEL_KEY = "pass0_physics"


class StatefulRandomization(wrapper.Wrapper):
    def __init__(self, env, randomization_fn):
        super().__init__(env)
        self.model, self.axes = randomization_fn(env.mjx_model)

    @contextlib.contextmanager
    def model_context(self, model):
        env = self.env.unwrapped
        old = env._mjx_model
        env._mjx_model = model
        try:
            yield self.env
        finally:
            env._mjx_model = old

    def reset(self, rng):
        def one(model, key):
            with self.model_context(model) as env:
                return env.reset(key)
        # Brax initially calls reset with only RNG arguments. Prevent captured
        # model arrays being folded there; subsequent resets use the live model.
        model = jax.lax.optimization_barrier(self.model)
        return jax.vmap(one, in_axes=(self.axes, 0))(model, rng)

    def step(self, state, action):
        def one(model, s, a):
            with self.model_context(model) as env:
                return env.step(s, a)
        return jax.vmap(one, in_axes=(self.axes, 0, 0))(self.model, state, action)


class StatefulAutoReset(wrapper.Wrapper):
    """Keep shared model arrays outside the per-environment reset selection."""
    def __init__(self, env, physics, full_reset=True):
        super().__init__(wrapper.BraxAutoResetWrapper(env, full_reset=full_reset))
        self.physics = physics

    def reset(self, rng):
        state = self.env.reset(rng)
        return state.replace(info={**state.info, MODEL_KEY: self.physics.model})

    def step(self, state, action):
        info = dict(state.info)
        model = info.pop(MODEL_KEY)
        old = self.physics.model
        self.physics.model = model
        try:
            result = self.env.step(state.replace(info=info), action)
        finally:
            self.physics.model = old
        return result.replace(info={**result.info, MODEL_KEY: model})


def wrap_for_training(env, episode_length=1000, action_repeat=1,
                      randomization_fn=None, full_reset=True, **kwargs):
    if kwargs.get("vision", False):
        raise ValueError("Pass 0 uses sensor packets, not a runtime renderer")
    if randomization_fn is None:
        raise ValueError("Pass 0 requires explicit physics randomization")
    physics = StatefulRandomization(env, randomization_fn)
    episodic = training.EpisodeWrapper(physics, episode_length, action_repeat)
    return StatefulAutoReset(episodic, physics, full_reset=full_reset)
