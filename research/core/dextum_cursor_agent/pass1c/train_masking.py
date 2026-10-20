"""Registered 20M-step rotation masking continuations from the Realistic actor.

The qualified collection, learning, physics and full-state resume code is reused.
Sensor interventions are isolated in pass1c.masking and recorded in metadata.
"""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "disable")
if not os.environ.get("CONTAINER_ID"):
    raise RuntimeError("Run Pass 1 on Vast.ai only; local numerical imports are disabled")
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import jax
import numpy as np
from brax.io import model
from brax.training.agents.ppo import networks
from brax.training.agents.ppo import train as ppo
from pass1c import resumable_ppo

import vendor.aero_official.vision_env  # noqa: F401
from dextum import contact_space
from pass1c import environment as e3
from pass1c.environment import ARMS, CRITIC_SIZE, EPISODE, HISTORY, BlindRotation, frame_size
from experiments.train_e2 import HIDDEN, KEEP_EVERY, _preprocess, sha256
from pass1 import faithful
from pass1.qualification import require_training_pass
from mujoco_playground._src.manipulation.aero_hand.rotate_z import domain_randomize
from mujoco_playground.config import manipulation_params

SOURCES = ("envs/blind_rotation.py", "envs/grasp_edge.py", "dextum/contact_space.py", "dextum/touch_mjx.py",
           "dextum/identification_v1.json", "experiments/train_e3.py", "experiments/train_e2.py", "pass1/train_e3.py", "pass1c/train_masking.py", "pass1c/masking.py", "pass1c/environment.py", "pass1c/sensors.py", "pass1b/sensors.py", "pass1c/build_model.py", "pass1c/kit_model.py", "pass1c/randomization.py", "pass1/faithful.py", "pass1/qualification.py", "pass0/faithful.py")
SOURCES = (*SOURCES, 'pass1c/compare.py', 'pass1c/criteria.py')
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
    p.add_argument("--qualification", type=Path, required=True)
    p.add_argument("--qualification-registration", type=Path, required=True)
    p.add_argument("--envs", type=int, default=8192)
    p.add_argument("--evals", type=int, default=None)
    p.add_argument("--smoke", action="store_true", help="CPU-sized batches for a pipeline check")
    p.add_argument('--warm-start', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--qualify-resume-throughput', action='store_true')
    p.add_argument('--evaluator-agreement', type=Path, required=True)
    p.add_argument('--throughput-qualification', type=Path)
    p.add_argument('--sensor-level', choices=['Ideal','Realistic'], default='Realistic')
    p.add_argument('--intervention', type=Path, required=True)
    p.add_argument('--matched-registration', type=Path,
                   help='Review section14 geometry result and frozen matched-batch authorization')
    a = p.parse_args()

    if not os.environ.get("CONTAINER_ID"):
        raise RuntimeError("Run Pass 1 on Vast.ai only")
    qualification = json.loads(a.qualification.read_text())
    require_training_pass(qualification)
    if qualification.get('criterion_version') not in ('review_section13','review_section15'):
        raise RuntimeError('Final collection-path qualification under review §13/§15 required')
    os.environ['DEXTUM_SENSOR_LEVEL']=a.sensor_level
    matched=json.loads(a.matched_registration.read_text()) if a.matched_registration else None
    if not matched or matched.get('authority')!='review_section15_masking':
        raise RuntimeError('Unknown matched-batch authority')
    if matched and (matched.get('state')!='AUTHORIZED' or not matched.get('geometry_accepted')):
        raise RuntimeError('The matched hand has no accepted geometry disposition')
    if a.sensor_level!='Realistic' and not matched:
        raise RuntimeError('The warm-start arms train at the Realistic sensor level')
    current_model_hash=sha256(Path(os.environ['DEXTUM_KIT_XML']))
    agreement=json.loads(a.evaluator_agreement.read_text())
    if qualification.get('model_sha256')!=current_model_hash:
        raise RuntimeError('D5 must qualify the exact corrected model')
    if matched:
        if matched['model_sha256']!=current_model_hash:
            raise RuntimeError('Model differs from the frozen matched hand')
        if [a.arm,a.sensor_level,a.seed] not in matched['authorized_runs']:
            raise RuntimeError('Run is outside the registered matched batch')
        if a.steps!=20000000 or a.envs!=8192:
            raise RuntimeError('Matched controls retain the 20M / 8192 protocol')
    elif agreement.get('model_sha256')!=current_model_hash:
        raise RuntimeError('D5 paired report model differs')
    if qualification.get('source_sha256',{}).get('ppo')!=sha256(HERE/'pass1c/resumable_ppo.py'):
        raise RuntimeError('D5 must qualify the current PPO collection/update implementation')
    if agreement.get('criterion_version')!='review_section13' or agreement.get('sensor_level')!='Ideal':
        raise RuntimeError('The reported §13 paired policy test at Ideal sensor level is required')
    if not a.qualify_resume_throughput:
        if not a.throughput_qualification:raise RuntimeError('Resume and throughput qualification required')
        q=json.loads(a.throughput_qualification.read_text())
        r=json.loads(a.throughput_qualification.with_name('resume_test.json').read_text())
        if q['state']!='PASS' or r['state']!='PASS':raise RuntimeError('Resume or throughput qualification failed')
        if q['metadata']['model_sha256']!=current_model_hash:
            if not matched or matched['geometry_source_sha256']!=q['metadata']['model_sha256']:
                raise RuntimeError('Throughput model differs without the section14 dimension-only correction')
        for source in ('pass1c/resumable_ppo.py','pass1c/resume_loop.py','pass1c/environment.py','pass1c/warm_start.py'):
            if q['metadata']['sources'][source]!=sha256(HERE/source):raise RuntimeError('Throughput source differs: '+source)
    if qualification.get('xla_flags','') != os.environ.get('XLA_FLAGS',''):
        raise RuntimeError('Compiler configuration differs from the qualified run')
    registration = json.loads(a.qualification_registration.read_text())
    wrapper_hash = hashlib.sha256((HERE / 'pass1/faithful.py').read_bytes()).hexdigest()
    if registration['input_sha256']['dextum_cursor_agent/pass1/faithful.py'] != wrapper_hash:
        raise RuntimeError('Training wrapper differs from the qualified source release')
    if qualification['wrapper_sha256'] != wrapper_hash:
        raise RuntimeError('Evaluator result wrapper hash differs from training')
    jax.config.update("jax_default_matmul_precision", "highest")
    if not a.smoke and not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("Full training requires a GPU; use --smoke for a CPU check")
    if a.output.exists() and not a.resume:
        raise FileExistsError(f"Refusing to overwrite {a.output}")
    a.output.mkdir(parents=True,exist_ok=bool(a.resume))

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

    intervention=json.loads(a.intervention.read_text())
    if intervention not in matched['authorized_interventions']:
        raise RuntimeError('Sensor intervention is not registered')
    if sha256(a.warm_start)!=matched['warm_start_sha256']:
        raise RuntimeError('Masking must use the registered full-sensor Realistic checkpoint')
    from pass1c.masking import BlindRotation as MaskEnvironment
    env = MaskEnvironment(arm=a.arm,intervention=intervention)
    ideal_params=model.load_params(a.warm_start)
    def initial_parameters(fresh_normalizer):
        # Same actor/critic layout; retain all learned parameters and normalization.
        layers=ideal_params[1]['params']
        first=next(k for k in layers if 'kernel' in layers[k])
        if layers[first]['kernel'].shape[0]!=HISTORY*env.frame:
            raise ValueError('Realistic masking checkpoint must match the actor layout')
        return ideal_params
    record = dict(state="STARTING", arm=a.arm, action_space=env.space, touch=env.touch, configuration=config,
                  actor_inputs=HISTORY * env.frame, critic_inputs=CRITIC_SIZE, actions=env.action_size,
                  reward=e3.REWARD, task=dict(warmup=e3.WARMUP, active=e3.ACTIVE, hold=e3.HOLD, turn=e3.TURN,
                                              tolerance=e3.TOLERANCE, tilt_cos2=e3.TILT_COS2, gamma=e3.GAMMA),
                  sources={s: sha256(HERE / s) for s in SOURCES}, device=str(jax.devices()), smoke=a.smoke,
                  started_at=time.time(), scaling="actor fixed + clip 5; critic running stats, std floor 1e-2, clip 5")
    record.update(sensor_level=a.sensor_level,intervention=intervention,hand_model='Corrected sensorised hand: 10 pads, 22 kit geoms',
                  matched_registration_sha256=sha256(a.matched_registration) if matched else None,
                  review_required_control=intervention.get('mask')=='none',
                  registered_ablation=intervention.get('mask')!='none')
    record['physics_qualification'] = dict(path=str(a.qualification),
        sha256=hashlib.sha256(a.qualification.read_bytes()).hexdigest(), wrapper_sha256=wrapper_hash)
    (a.output / "config.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    started = time.monotonic()
    checkpoints = json.loads((a.output/'checkpoints.json').read_text()) if a.resume and (a.output/'checkpoints.json').exists() else []
    metadata=dict(arm=a.arm,sensor_level=a.sensor_level,intervention=intervention,configuration=config,model_sha256=current_model_hash,
                  warm_start_sha256=sha256(a.warm_start),sources={s:sha256(HERE/s) for s in (*SOURCES,'pass1c/resumable_ppo.py','pass1c/resume_loop.py','pass1c/warm_start.py')})

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
        if any(c['steps']==int(steps) for c in checkpoints):return
        model.save_params(path, params)
        if checkpoints and (len(checkpoints) - 1) % KEEP_EVERY:
            (a.output / checkpoints[-1]["path"]).unlink(missing_ok=True)
            checkpoints[-1]["kept"] = False
        checkpoints.append(dict(steps=int(steps), elapsed_s=time.monotonic() - started, path=path.name, kept=True))
        (a.output / "checkpoints.json").write_text(json.dumps(checkpoints, indent=2) + "\n")

    monitor=None
    if not a.qualify_resume_throughput:
        from pass1c.compare import MonitorGuard
        monitor=MonitorGuard(env,networks.make_inference_fn(make_network(a.arm)),a.output/'monitor')
    try:
        resumable_ppo.train(environment=env, network_factory=network_factory(a.arm),
                  wrap_env_fn=functools.partial(faithful.wrap_for_training, full_reset=True),
                  randomization_fn=env.randomize, progress_fn=progress, policy_params_fn=save,
                  restore_params=initial_parameters,run_evals=False,
                  full_state_options=dict(directory=str(a.output/'full_state'),metadata=metadata,
                      resume=str(a.resume) if a.resume else None,qualify=a.qualify_resume_throughput,
                      monitor=monitor),**config)
        record.update(state="COMPLETE")
        if len(checkpoints) >= 3:
            (s0, t0), (s1, t1) = [(c["steps"], c["elapsed_s"]) for c in checkpoints[-2:]]
            record["steady_decisions_per_s"] = (s1 - s0) / max(t1 - t0, 1e-9)
    except BaseException as exc:
        record.update(state="FAILED", error=f"{type(exc).__name__}: {exc}")
        if type(exc).__name__=='MonitorGapError':
            record.update(state='EXECUTION_FAILURE',failure_cause='monitor_faithful_gap',attempts_consumed=0)
        raise
    finally:
        record.update(wall_s=time.monotonic() - started, checkpoints=checkpoints)
        (a.output / "report.json").write_text(json.dumps(record, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
