"""Offline hold/regrasp starts from a fixed final actor; no policy training.

The masks need actual contact-state entries, not slices named after quanta.
This records the first eligible entry per independent episode, with its
physics draw and checkpoint provenance. Truth here is an offline label.
"""
import os
if not os.environ.get('CONTAINER_ID'):
    raise RuntimeError('Vast.ai only')
import argparse
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as j
import numpy as np
from pass1c.environment import BlindRotation, EPISODE, WARMUP, DROP_Z
from pass1c.evaluate import load_policy
from dextum.touch_mjx import grasp_quality
from experiments.evaluate_e3 import reset_one, step_one


def eligibility(reading, linear_speed, angular_speed, height):
    """Existing quality proxy and spec quantum entry predicates; truth only."""
    # Keep the palm and proximal pads in the support set. Three non-collinear
    # point contacts are needed by the existing quality proxy.
    quality = grasp_quality(reading)
    engaged = reading.engaged
    safe = j.all(j.where(engaged, (reading.margin >= .3) & ~reading.slip, True))
    formed = (j.sum(engaged) >= 3) & (quality >= .3) & safe & (height >= DROP_Z)
    hold = formed & (linear_speed < .01) & (angular_speed < .25)
    # Middle/ring are G digits 1/2 in the registered Pass-0 script.
    candidates = []
    for tip, proximal in ((2, 6), (3, 7)):
        retained = engaged.at[tip].set(False).at[proximal].set(False)
        remaining = reading._replace(engaged=retained)
        candidates.append(formed & engaged[tip] & (grasp_quality(remaining) >= .2))
    valid = j.stack(candidates)
    moving = j.where(valid[0], 2, 3)
    point = reading.point_obj[moving]
    axis = j.argmax(j.abs(point))
    source_patch = 2 * axis + (point[axis] >= 0).astype(j.int32)
    # Closest adjacent cube face by the contact centroid; opposite faces
    # are never a one-regrasp target. Feasibility remains to be measured.
    other = j.where(j.arange(3) != axis, j.abs(point), -j.inf)
    target_axis = j.argmax(other)
    target_patch = 2 * target_axis + (point[target_axis] >= 0).astype(j.int32)
    return hold, j.any(valid), moving, source_patch, target_patch, quality


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--episodes', type=int, default=256)
    parser.add_argument('--seed', type=int, default=11026)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    jax.config.update('jax_default_matmul_precision', 'highest')
    env = BlindRotation('c_touch')
    policy, steps = load_policy(args.run, 'c_touch')
    record = json.loads((args.run/'c_touch/checkpoints.json').read_text())
    checkpoint = args.run/'c_touch'/next(c['path'] for c in record if c['steps'] == steps)
    registration = dict(seed=args.seed, episodes=args.episodes, actor_steps=steps,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        selection='First eligible entry per episode, independently for hold and regrasp',
        quality='Existing frictional point-contact grasp-quality proxy; no new force-closure claim',
        hold_entry='>=3 pads, quality>=0.3, engaged margins>=0.3N, no slip; speed<0.01m/s, angular<0.25rad/s',
        regrasp_entry='Same formed grasp; middle or ring fingertip engaged; quality>=0.2 with that digit removed',
        regrasp_goal='Nearest adjacent cube face; selected offline, feasibility unmeasured',
        privileged_fields='Offline reset bank and labels only; never actor observations',
        training_attempts=0, optimizer_steps=0)
    (args.output/'registration.json').write_text(json.dumps(registration, indent=2)+'\n')
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.episodes)
    dr_keys = jax.random.split(jax.random.PRNGKey(args.seed+1), args.episodes)
    models, axes = env.randomize(env.mjx_model, dr_keys)
    reset = jax.vmap(lambda m,k: reset_one(env,m,k,j.zeros(2)), in_axes=(axes,0))
    step = jax.vmap(lambda m,s,a: step_one(env,m,s,a), in_axes=(axes,0,0))

    def describe(model, state):
        reading = env.touch_model.read(model, state.data)
        return eligibility(reading, j.linalg.norm(env.get_cube_linvel(state.data)),
            j.linalg.norm(env.get_cube_angvel(state.data)), state.data.qpos[env._cube_qids][2])
    describe_batch = jax.vmap(describe, in_axes=(axes,0))

    @jax.jit
    def run(models, keys):
        state = reset(models, keys)
        def body(carry, _):
            state, alive = carry
            state = step(models, state, policy(state.obs))
            hold, regrasp, moving, source, target, quality = describe_batch(models, state)
            eligible = alive & (state.info['tick'] > WARMUP)
            row = dict(hold=hold & eligible, regrasp=regrasp & eligible,
                moving=moving, source_patch=source, target_patch=target, quality=quality,
                qpos=state.data.qpos, qvel=state.data.qvel, ctrl=state.data.ctrl,
                act=state.data.act, info=state.info, obs=state.obs)
            return (state, alive & ~state.done.astype(bool)), row
        return jax.lax.scan(body, (state,j.ones(args.episodes,bool)), None, length=EPISODE)[1]

    started = time.monotonic()
    print('Collecting fixed-policy quantum entries', flush=True)
    trace = jax.tree.map(np.asarray, run(models, keys))
    summary = dict(state='MEASURED', episodes=args.episodes, elapsed_s=time.monotonic()-started,
                   training_attempts=0, optimizer_steps=0, quanta={})
    for quantum in ('hold','regrasp'):
        valid = trace[quantum].any(0)
        episode = np.flatnonzero(valid)
        tick = trace[quantum][:,episode].argmax(0)
        chosen = jax.tree.map(lambda x:x[tick,episode], trace)
        # Named flat arrays avoid executable pickle in the bank format.
        arrays = {}
        for path,value in jax.tree_util.tree_flatten_with_path(chosen)[0]:
            name='/'.join(str(getattr(k,'key',getattr(k,'idx',k))) for k in path)
            arrays[name]=value
        arrays.update(episode=episode,tick=tick,reset_keys=np.asarray(keys)[episode],dr_keys=np.asarray(dr_keys)[episode])
        np.savez_compressed(args.output/(quantum+'_entries.npz'), **arrays)
        summary['quanta'][quantum]=dict(eligible_episodes=int(valid.sum()),total_episodes=args.episodes,
            eligible_transitions=int(trace[quantum].sum()),
            note='Entry candidates only; no execution success or masking classification')
    (args.output/'result.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
