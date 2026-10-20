"""Fresh official Brax PPO, with bounded qualification and machine-readable logs."""
import argparse
import functools
import inspect
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
# The proprioceptive task renders no images. Avoid a graphics-driver dependency.
os.environ.setdefault("MUJOCO_GL","disable")

import jax
import numpy as np
from brax.io import model
from brax.training.agents.ppo import networks, train as ppo
from etils import epath

from .policy import PLAYGROUND, MENAGERIE
from .protocol import PROTOCOLS, get_protocol, configure_precision


def main(*, actor_factory=None, actor_metadata=None, randomization_factory=None,
         restore_path=None, config_overrides=None, registration=None,
         environment_factory=None, checkpoint_hook=None, wrap_env_factory=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--timing-updates",type=int,default=50)
    parser.add_argument("--steps",type=int,default=None)
    parser.add_argument("--seed",type=int,default=1)
    parser.add_argument("--envs",type=int,default=8192)
    parser.add_argument("--protocol",choices=tuple(PROTOCOLS),required=True,
                        help="Required: distinguish documented quick-start from paper DR recipe")
    args = parser.parse_args()
    protocol = get_protocol(args.protocol)
    precision = configure_precision(protocol, jax)
    if not any(d.platform=="gpu" for d in jax.devices()):
        raise RuntimeError("Fresh PPO training requires the qualified GPU")
    if args.output.exists():
        raise FileExistsError("Use a new run directory; never overwrite a run")
    args.output.mkdir(parents=True)
    sys.path.insert(0,str(PLAYGROUND))
    from mujoco_playground._src import mjx_env
    mjx_env.MENAGERIE_PATH = epath.Path(MENAGERIE)
    from mujoco_playground import wrapper
    from mujoco_playground._src.manipulation.aero_hand.rotate_z import CubeRotateZAxis, domain_randomize
    from mujoco_playground.config import manipulation_params
    config = manipulation_params.brax_ppo_config("AeroCubeRotateZAxis").to_dict()
    factory = config.pop("network_factory")
    if args.envs!=8192:
        raise ValueError("8192 environments required; smaller variant must be registered separately")
    timing = args.steps is None
    if timing:
        config["num_timesteps"] = args.timing_updates*config["unroll_length"]*config["batch_size"]*config["num_minibatches"]
        config["num_evals"] = 3
    else:
        config["num_timesteps"] = args.steps
    config.update(seed=args.seed,run_evals=not timing,log_training_metrics=True,
                  training_metrics_steps=327680)
    config.update(config_overrides or {})
    restored = model.load_params(restore_path) if restore_path else None
    record = dict(state="STARTING",configuration=config,network_factory=factory,
                  timing_only=timing,domain_randomization=protocol.domain_randomization,
                  protocol=protocol.name,protocol_reference=protocol.reference,device=str(jax.devices()),
                  matmul_precision=precision,
                  actor=actor_metadata or {"kind":"official_mlp"},
                  registration=registration,
                  source="official pinned AeroCubeRotateZAxis",
                  inherited_ppo_defaults={k:str(v.default) for k,v in inspect.signature(ppo.train).parameters.items()
                    if k in ("clipping_epsilon","gae_lambda","max_grad_norm","num_eval_envs","normalize_advantage")},
                  deviations=[f"Timing uses {args.timing_updates} PPO batches, 3 reporting points, no evaluation rollouts"] if timing else [])
    if restored is not None:
        import hashlib
        record["warm_start"] = dict(path=str(restore_path),
            sha256=hashlib.sha256(Path(restore_path).read_bytes()).hexdigest(),
            restored="actor, critic, observation normalizer", optimizer="fresh Adam", registration=registration)
    (args.output/"config.json").write_text(json.dumps(record,indent=2)+"\n")
    started = time.monotonic()
    initial = None
    callback_records = []

    def progress(steps,metrics):
        values = {str(k):float(np.asarray(v)) for k,v in metrics.items() if np.asarray(v).size==1}
        if not all(np.isfinite(v) for v in values.values()):
            raise RuntimeError("Nonfinite PPO metrics")
        row = dict(steps=int(steps),elapsed_s=time.monotonic()-started,metrics=values)
        with (args.output/"metrics.jsonl").open("a") as f:
            f.write(json.dumps(row)+"\n")
        print(json.dumps(row),flush=True)

    def checkpoint(steps,make_policy,params):
        nonlocal initial
        if initial is None and restored is not None:
            assert jax.tree.structure(params) == jax.tree.structure(restored)
            for a,b in zip(jax.tree.leaves(params),jax.tree.leaves(restored)):
                np.testing.assert_array_equal(a,b)
            record["warm_start"]["initial_checkpoint_exact"] = True
        leaves = [np.asarray(x).copy() for x in jax.tree.leaves(params[1])]
        if not all(np.isfinite(x).all() for x in jax.tree.leaves(params)):
            raise RuntimeError("Nonfinite checkpoint")
        if initial is None:
            initial=leaves
        delta=float(np.sqrt(sum(np.square(a-b).sum() for a,b in zip(leaves,initial))))
        path=args.output/f"policy_{int(steps)}.pkl"
        model.save_params(path,params)
        loaded=model.load_params(path)
        for a,b in zip(jax.tree.leaves(params),jax.tree.leaves(loaded)):
            np.testing.assert_array_equal(a,b)
        callback_records.append(dict(steps=int(steps),elapsed_s=time.monotonic()-started,
                                     policy_parameter_delta_l2=delta,checkpoint_roundtrip=True))
        (args.output/"checkpoints.json").write_text(json.dumps(callback_records,indent=2)+"\n")
        if checkpoint_hook is not None:
            checkpoint_hook(int(steps), path)

    try:
        restore_args = {"restore_params": restored} if restored is not None else {}
        ppo.train(environment=(environment_factory or CubeRotateZAxis)(),network_factory=functools.partial(
            actor_factory or networks.make_ppo_networks,**factory),wrap_env_fn=wrap_env_factory or wrapper.wrap_for_brax_training,
            randomization_fn=(randomization_factory or domain_randomize) if protocol.domain_randomization else None,
            progress_fn=progress,policy_params_fn=checkpoint,**restore_args,**config)
        assert callback_records[-1]["policy_parameter_delta_l2"]>1e-8
        record.update(state="COMPLETE",wall_s=time.monotonic()-started,checkpoints=callback_records)
        if len(callback_records)>=3:
            # Exclude the first compilation/training chunk; include measured
            # reset and checkpoint overhead in the steady-state projection.
            previous,last=callback_records[-2:]
            sps=(last["steps"]-previous["steps"])/(last["elapsed_s"]-previous["elapsed_s"])
            record.update(steady_decisions_per_s=sps,projected_300m_hours=300e6/sps/3600)
    except BaseException as exc:
        record.update(state="FAILED",wall_s=time.monotonic()-started,checkpoints=callback_records,
                      error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (args.output/"report.json").write_text(json.dumps(record,indent=2)+"\n")


if __name__ == "__main__":
    main()
