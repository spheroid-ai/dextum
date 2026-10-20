"""Barrier both reset physics and carried physics across joint compilation."""
import jax
from brax.envs.wrappers import training
from pass0.faithful import MODEL_KEY, StatefulRandomization
from pass0.faithful import StatefulAutoReset as PreviousAutoReset


class StatefulAutoReset(PreviousAutoReset):
    def reset(self, rng):
        model = jax.lax.optimization_barrier(self.physics.model)
        old = self.physics.model
        self.physics.model = model
        try:
            state = self.env.reset(rng)
        finally:
            self.physics.model = old
        return state.replace(info={**state.info, MODEL_KEY: model})


def wrap_for_training(env, episode_length=1000, action_repeat=1,
                      randomization_fn=None, full_reset=True, **kwargs):
    if kwargs.get('vision', False) or randomization_fn is None:
        raise ValueError('Explicit randomized physics and sensor packets required')
    physics = StatefulRandomization(env, randomization_fn)
    episodic = training.EpisodeWrapper(physics, episode_length, action_repeat)
    return StatefulAutoReset(episodic, physics, full_reset=full_reset)
