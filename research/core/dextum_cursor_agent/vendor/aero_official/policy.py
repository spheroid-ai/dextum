"""Released Brax policy, with its original normalization and action distribution."""
from pathlib import Path
import json

import jax
import jax.numpy as jnp
import numpy as np
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks

ROOT = Path(__file__).resolve().parents[3]
OFFICIAL = ROOT / "upstream/aero-official"
PLAYGROUND = ROOT / "upstream/aero-playground"
MENAGERIE = ROOT / "upstream/aero-menagerie"
CHECKPOINT = OFFICIAL / "ros2/src/aero_hand_open_rl/aero_hand_open_rl/ppo_TetheriaCubeRotateZAxisTendon_20250926_152920"
TASK = PLAYGROUND / "mujoco_playground/_src/manipulation/aero_hand"
TRAIN_SCALE = np.array([.02, .02, .02, .02, .7, .003, .012])
DEPLOY_SCALE = np.array([.026, .026, .026, .024, .7, .003, .012])
DEPLOY_HOME = np.array([.083, .082, .082, .086, .75, .035, .1])
SENSORS = ("len_if", "len_mf", "len_rf", "len_pf", "len_th1", "len_th2", "len_th_abd")


class ReleasedPolicy:
    def __init__(self, checkpoint=CHECKPOINT):
        # Only load trusted official or our own training checkpoints, not arbitrary pickles.
        self.params = model.load_params(checkpoint)
        config_path = Path(checkpoint).parent/"config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        self.actor_kind = config.get("actor", {}).get("kind", "official_mlp")
        if self.actor_kind == "official_mlp":
            factory = networks.make_ppo_networks
        elif self.actor_kind == "mpaper_token":
            from .mpaper_actor import make_networks
            factory = make_networks
        else:
            raise ValueError(f"Unknown checkpoint actor: {self.actor_kind}")
        self.network = factory(
            observation_size={"state": 14, "privileged_state": 81}, action_size=7,
            preprocess_observations_fn=running_statistics.normalize,
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
            policy_obs_key="state", value_obs_key="privileged_state",
        )
        self.inference = jax.jit(networks.make_inference_fn(self.network)(
            self.params, deterministic=True))
        self.key = jax.random.PRNGKey(14)

    def __call__(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape[-1] != 14 or not np.isfinite(observation).all():
            raise ValueError("Actor requires exactly 14 finite observable values")
        action = np.asarray(self.inference({"state": jnp.asarray(observation)}, self.key)[0])
        if not np.isfinite(action).all() or np.max(np.abs(action)) > 1.00001:
            raise RuntimeError("Invalid released-policy action")
        return action
