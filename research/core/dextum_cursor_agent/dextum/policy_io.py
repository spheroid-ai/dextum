"""Networks, frozen input scaling and zero-initialised warm starts.

Input scaling is a constant affine map (§16.1 D2): motor and privileged channels
use the working vision policy's normaliser statistics, frozen at warm start; the
camera/goal channels keep their fixed physical scaling; touch and perturbation
channels arrive pre-scaled and clipped to +/-5. Brax's running statistics are
still updated by PPO but never read.
"""
import copy
import functools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from brax.io import model
from brax.training.agents.ppo import networks

from envs.touch_orientation import BASE_ACTOR, BASE_CRITIC, CRITIC_SIZE, HISTORY, actor_frame_size
from vendor.aero_official.orientation_policy import fixed_scaling

HIDDEN = (512, 256, 128)
ACTIONS = 7


def frozen_scaling(source_params, touch):
    """Constant (mean, std) for actor and critic inputs from a 34/89 checkpoint."""
    stats = source_params[0]
    if stats.mean["state"].shape[-1] != BASE_ACTOR or stats.mean["privileged_state"].shape[-1] != BASE_CRITIC:
        raise ValueError("Source must be the single-frame 34/89 vision policy")
    start, mean, std = fixed_scaling(BASE_ACTOR)
    base_mean = np.asarray(mean).copy()
    base_std = np.asarray(std).copy()
    base_mean[:start] = np.asarray(stats.mean["state"])[:start]
    base_std[:start] = np.asarray(stats.std["state"])[:start]
    extra = actor_frame_size(touch) - BASE_ACTOR
    frame_mean = np.concatenate((base_mean, np.zeros(extra)))
    frame_std = np.concatenate((base_std, np.ones(extra)))
    cstart, cmean, cstd = fixed_scaling(BASE_CRITIC)
    critic_mean = np.zeros(CRITIC_SIZE)
    critic_std = np.ones(CRITIC_SIZE)
    critic_mean[:BASE_CRITIC] = np.asarray(cmean)
    critic_std[:BASE_CRITIC] = np.asarray(cstd)
    critic_mean[:cstart] = np.asarray(stats.mean["privileged_state"])[:cstart]
    critic_std[:cstart] = np.asarray(stats.std["privileged_state"])[:cstart]
    return dict(actor_mean=np.tile(frame_mean, HISTORY).astype(np.float32),
                actor_std=np.tile(frame_std, HISTORY).astype(np.float32),
                critic_mean=critic_mean.astype(np.float32), critic_std=critic_std.astype(np.float32))


def _preprocess(x, stats, *, scaling):
    del stats
    size = x.shape[-1]
    if size == scaling["actor_mean"].shape[0]:
        mean, std = scaling["actor_mean"], scaling["actor_std"]
    elif size == scaling["critic_mean"].shape[0]:
        mean, std = scaling["critic_mean"], scaling["critic_std"]
    else:
        raise ValueError(f"Unexpected observation width {size}")
    return (x - jnp.asarray(mean)) / jnp.asarray(std)


def network_factory(scaling):
    if scaling["actor_mean"].shape == scaling["critic_mean"].shape:
        raise ValueError("Actor and critic widths must differ for shape dispatch")

    def factory(*args, **kwargs):
        kwargs["preprocess_observations_fn"] = functools.partial(_preprocess, scaling=scaling)
        kwargs.setdefault("policy_hidden_layer_sizes", HIDDEN)
        kwargs.setdefault("value_hidden_layer_sizes", HIDDEN)
        kwargs.setdefault("policy_obs_key", "state")
        kwargs.setdefault("value_obs_key", "privileged_state")
        return networks.make_ppo_networks(*args, **kwargs)
    return factory


def make_network(scaling):
    return network_factory(scaling)(
        observation_size={"state": scaling["actor_mean"].shape[0], "privileged_state": CRITIC_SIZE},
        action_size=ACTIONS)


def _widen(kernel, rows, placement):
    """New first-layer kernel with old rows at `placement`, zeros elsewhere."""
    new = jnp.zeros((rows, kernel.shape[1]), kernel.dtype)
    return new.at[placement].set(kernel)


def warm_start(source_params, touch):
    """Same mean action and value as the source on the current-frame base inputs."""
    if source_params[1]["params"]["hidden_0"]["kernel"].shape[0] != BASE_ACTOR:
        raise ValueError("Warm start expects the single-frame vision checkpoint")
    frame = actor_frame_size(touch)
    actor_rows = frame * HISTORY
    new = copy.deepcopy(source_params)
    first = new[1]["params"]["hidden_0"]
    first["kernel"] = _widen(first["kernel"], actor_rows, jnp.arange(BASE_ACTOR))
    critic_first = new[2]["params"]["hidden_0"]
    critic_first["kernel"] = _widen(critic_first["kernel"], CRITIC_SIZE, jnp.arange(BASE_CRITIC))
    stats = new[0]

    def pad(values, key, width, fill):
        old = np.asarray(values[key])
        if key == "state":
            block = np.concatenate((old, np.full(frame - BASE_ACTOR, fill, old.dtype)))
            return jnp.asarray(np.tile(block, HISTORY))
        return jnp.asarray(np.concatenate((old, np.full(width - old.shape[0], fill, old.dtype))))

    fields = {}
    for name, fill in (("mean", 0.0), ("std", 1.0), ("summed_variance", 0.0)):
        values = getattr(stats, name)
        fields[name] = {"state": pad(values, "state", actor_rows, fill),
                        "privileged_state": pad(values, "privileged_state", CRITIC_SIZE, fill)}
    return (stats.replace(**fields), *new[1:])


def save_scaling(path, scaling):
    np.savez(path, **scaling)


def load_scaling(path):
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


class Policy:
    """Deterministic actor for evaluation; loads a run checkpoint and its scaling."""

    def __init__(self, checkpoint, scaling=None):
        checkpoint = Path(checkpoint)
        self.params = model.load_params(checkpoint)
        self.scaling = scaling or load_scaling(checkpoint.parent / "scaling.npz")
        network = make_network(self.scaling)
        self.infer = networks.make_inference_fn(network)(self.params, deterministic=True)

    def __call__(self, obs):
        return self.infer(obs, jax.random.PRNGKey(0))[0]


def describe(path):
    return json.loads(Path(path).read_text())
