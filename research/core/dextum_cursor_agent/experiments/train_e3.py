"""PPO for one E3 arm (goal rotation +/-30 deg, no vision), from scratch.

Same PPO, network sizes, input scaling and checkpointing as E2 (`train_e2.py`);
only the environment differs.
"""
import argparse
import functools
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "disable")
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import jax
import numpy as np
from brax.io import model
from brax.training.agents.ppo import networks
from brax.training.agents.ppo import train as ppo

import vendor.aero_official.vision_env  # noqa: F401
from dextum import contact_space
from envs import blind_rotation as e3
from envs.blind_rotation import ARMS, CRITIC_SIZE, EPISODE, HISTORY, BlindRotation, frame_size
from experiments.train_e2 import HIDDEN, KEEP_EVERY, _preprocess, sha256
from mujoco_playground import wrapper
from mujoco_playground._src.manipulation.aero_hand.rotate_z import domain_randomize
from mujoco_playground.config import manipulation_params

SOURCES = ("envs/blind_rotation.py", "envs/grasp_edge.py", "dextum/contact_space.py", "dextum/touch_mjx.py",
           "dextum/identification_v1.json", "experiments/train_e3.py", "experiments/train_e2.py")
SHOWN = ("eval/episode_e3/success", "eval/episode_e3/drop", "eval/episode_e3/within", "eval/episode_e3/abs_error_deg",
         "episode/e3/success", "eval/episode_reward")


def network_factory(arm):
    actor_width = HISTORY * frame_size(arm)
    if actor_width == CRITIC_SIZE:
        raise ValueError("Actor and critic widths must differ for shape dispatch")

    def factory(*args, **kwargs):
        kwargs["preprocess_observations_fn"] = functools.partial(_preprocess, actor_width=actor_width)
        kwargs.setdefault("policy_hidden_layer_sizes", HIDDEN)
        kwargs.setdefault("value_hidden_layer_sizes", HIDDEN)
        kwargs.setdefault("policy_obs_key", "state")
        kwargs.setdefault("value_obs_key", "privileged_state")
        return networks.make_ppo_networks(*args, **kwargs)
    return factory


def make_network(arm):
    env_actions = contact_space.ACTION_SIZE if ARMS[arm][0] == "contact" else 7
    return network_factory(arm)(observation_size={"state": HISTORY * frame_size(arm), "privileged_state": CRITIC_SIZE},
                                action_size=env_actions)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=tuple(ARMS), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--envs", type=int, default=8192)
    p.add_argument("--evals", type=int, default=None)
    p.add_argument("--smoke", action="store_true", help="CPU-sized batches for a pipeline check")
    a = p.parse_args()

    jax.config.update("jax_default_matmul_precision", "highest")
    if not a.smoke and not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("Full training requires a GPU; use --smoke for a CPU check")
    if a.output.exists():
        raise FileExistsError(f"Refusing to overwrite {a.output}")
    a.output.mkdir(parents=True)

    config = manipulation_params.brax_ppo_config("AeroCubeRotateZAxis").to_dict()
    config.pop("network_factory")
    config.update(episode_length=EPISODE, num_envs=a.envs, num_timesteps=a.steps, seed=a.seed,
                  log_training_metrics=True, training_metrics_steps=327680, deterministic_eval=True,
                  num_eval_envs=128)
    if a.smoke:
        config.update(num_envs=a.envs, batch_size=a.envs // 2, num_minibatches=2, unroll_length=10,
                      num_eval_envs=4, training_metrics_steps=a.envs * 10, num_resets_per_eval=0)
    per_iteration = config["batch_size"] * config["num_minibatches"] * config["unroll_length"]
    config["num_evals"] = a.evals or max(2, a.steps // (10 * per_iteration) + 1)

    env = BlindRotation(arm=a.arm)
    record = dict(state="STARTING", arm=a.arm, action_space=env.space, touch=env.touch, configuration=config,
                  actor_inputs=HISTORY * env.frame, critic_inputs=CRITIC_SIZE, actions=env.action_size,
                  reward=e3.REWARD, task=dict(warmup=e3.WARMUP, active=e3.ACTIVE, hold=e3.HOLD, turn=e3.TURN,
                                              tolerance=e3.TOLERANCE, tilt_cos2=e3.TILT_COS2, gamma=e3.GAMMA),
                  sources={s: sha256(HERE / s) for s in SOURCES}, device=str(jax.devices()), smoke=a.smoke,
                  started_at=time.time(), scaling="actor fixed + clip 5; critic running stats, std floor 1e-2, clip 5")
    (a.output / "config.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    started = time.monotonic()
    checkpoints = []

    def progress(steps, metrics):
        values = {str(k): float(np.asarray(v)) for k, v in metrics.items() if np.asarray(v).size == 1}
        bad = [k for k, v in values.items() if not np.isfinite(v)]
        if bad:
            raise RuntimeError(f"Nonfinite PPO metrics: {bad[:5]}")
        row = dict(steps=int(steps), elapsed_s=time.monotonic() - started, metrics=values)
        with (a.output / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(dict(steps=int(steps), t=round(row["elapsed_s"]),
                              **{k.replace("eval/episode_", "eval_").replace("episode/", "train_"): round(values[k], 3)
                                 for k in SHOWN if k in values})), flush=True)

    def save(steps, make_policy, params):
        del make_policy
        if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(params)):
            raise RuntimeError("Nonfinite parameters")
        path = a.output / f"policy_{int(steps)}.pkl"
        model.save_params(path, params)
        if checkpoints and (len(checkpoints) - 1) % KEEP_EVERY:
            (a.output / checkpoints[-1]["path"]).unlink(missing_ok=True)
            checkpoints[-1]["kept"] = False
        checkpoints.append(dict(steps=int(steps), elapsed_s=time.monotonic() - started, path=path.name, kept=True))
        (a.output / "checkpoints.json").write_text(json.dumps(checkpoints, indent=2) + "\n")

    try:
        ppo.train(environment=env, network_factory=network_factory(a.arm),
                  wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training, full_reset=True),
                  randomization_fn=domain_randomize, progress_fn=progress, policy_params_fn=save, **config)
        record.update(state="COMPLETE")
        if len(checkpoints) >= 3:
            (s0, t0), (s1, t1) = [(c["steps"], c["elapsed_s"]) for c in checkpoints[-2:]]
            record["steady_decisions_per_s"] = (s1 - s0) / max(t1 - t0, 1e-9)
    except BaseException as exc:
        record.update(state="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record.update(wall_s=time.monotonic() - started, checkpoints=checkpoints)
        (a.output / "report.json").write_text(json.dumps(record, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
