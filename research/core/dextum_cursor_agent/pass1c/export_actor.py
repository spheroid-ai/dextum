"""Remote trusted-checkpoint export, with executable inference and G parity."""
import argparse
import json
import os
from pathlib import Path
if not os.environ.get("CONTAINER_ID"):
    raise RuntimeError("Export and numeric checks run on Vast only")

import jax
import jax.numpy as jnp
import numpy as np
from brax.io import model
from brax.training.agents.ppo import networks
from dextum import contact_space as cs
from pass1c.train import make_network
from pass1c.environment import HISTORY,frame_size,ARMS
from pass1c.numpy_actor import Actor, ContactMotor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--arm", choices=("c_touch","c_notouch","a_task"), required=True)
    a = p.parse_args()
    actor_width=HISTORY*frame_size(a.arm)
    action_size=31 if ARMS[a.arm][0]=="contact" else 7
    jax.config.update("jax_default_matmul_precision", "highest")
    params = model.load_params(a.checkpoint)
    dense = []
    def collect(x, path=()):
        if hasattr(x, "items"):
            if "kernel" in x and "bias" in x:
                dense.append((path, np.asarray(x["kernel"]), np.asarray(x["bias"])))
            else:
                for k, v in x.items():
                    collect(v, (*path, str(k)))
    collect(params[1])
    remaining = list(dense)
    layers = []
    width = actor_width
    while remaining:
        candidates = [d for d in remaining if d[1].shape[0] == width]
        assert len(candidates) == 1, [(d[0], d[1].shape) for d in remaining]
        layer = candidates[0]
        remaining = [d for d in remaining if d[0] != layer[0]]
        layers.append(layer)
        width = layer[1].shape[1]
    assert width == 2*action_size, "Expected means and scales"
    # The deterministic mode is tanh(mean); the second half only controls noise.
    path, w, b = layers[-1]
    layers[-1] = path, w[:, :action_size], b[:action_size]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"layers": np.array(len(layers))}
    for i, (_, w, b) in enumerate(layers):
        arrays.update({f"w{i}": w, f"b{i}": b})
    np.savez(a.output, **arrays)
    exported = Actor(a.output)
    infer = jax.jit(networks.make_inference_fn(make_network(a.arm))(params, deterministic=True))
    rng = np.random.default_rng(91023)
    observations = rng.normal(size=(64, actor_width)).astype(np.float32)*3
    expected = np.asarray(infer({"state": jnp.asarray(observations)}, jax.random.PRNGKey(0))[0])
    error = float(np.max(np.abs(expected-exported(observations))))
    ranges = np.stack((cs.HOME-cs.SCALE, cs.HOME+cs.SCALE), axis=1)
    port = ContactMotor(ranges, reflex=a.arm=="c_touch")
    state = cs.initial_state()
    lo, hi = map(jnp.asarray, cs.bounds(ranges))
    advance = jax.jit(lambda state,action,sense,lo,hi: cs.step(state,action,sense,lo,hi,reflex=a.arm=="c_touch"))
    motor_error = 0.0
    for _ in range(128):
        action = rng.uniform(-1, 1, 31).astype(np.float32)
        normal = rng.uniform(0, 5, 11)
        shear = rng.uniform(0, 2, 11)
        slip = rng.uniform(size=11) < .1
        force, tang, slips, contact = port.sense(normal, shear, slip)
        sense = cs.Sense(*map(jnp.asarray, (force, tang, slips, contact)))
        state, command = advance(state, jnp.asarray(action), sense, lo, hi)
        motor_error = max(motor_error, float(np.max(np.abs(np.asarray(command)-port.step(action, normal, shear, slip)))))
    result = dict(actor_max_error=error, motor_max_error=motor_error,
                  state="PASS" if error < 1e-4 and motor_error < 1e-5 else "HARD_STOP",
                  runtime_dependencies=["numpy"], simulator_at_runtime=False,
                  checkpoint=str(a.checkpoint), arm=a.arm,actor_width=actor_width,actions=action_size, activation="swish", action="tanh of mean")
    a.output.with_suffix(".audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)
    assert result["state"] == "PASS"


if __name__ == "__main__":
    main()
