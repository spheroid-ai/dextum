"""Plain-MuJoCo probe of the Aero hand for E2 design: tendon direction, resting
contacts, closing programs and the tension-to-force relation. CPU, seconds."""
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disable")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

import vendor.aero_official.vision_env  # noqa: F401  (puts the pinned playground on sys.path)
from dextum import touch_mjx
from mujoco_playground._src.manipulation.aero_hand import aero_hand_constants as consts
from mujoco_playground._src.manipulation.aero_hand.rotate_z import CubeRotateZAxis

env = CubeRotateZAxis()
m = env.mj_model
touch = touch_mjx.TouchModel(m)
CUBE = m.body("cube").id
ACT = consts.ACTUATOR_NAMES
HOME = m.keyframe("home")
LO, HI = m.actuator_ctrlrange.T
JOINTS = [m.joint(n).qposadr[0] for n in consts.JOINT_NAMES]


def fresh(cube_pos=(0.1, 0.0, 0.05)):
    d = mujoco.MjData(m)
    d.qpos[:] = HOME.qpos
    d.qpos[16:19] = cube_pos
    d.qpos[19:23] = (1, 0, 0, 0)
    d.ctrl[:] = HOME.ctrl
    mujoco.mj_forward(m, d)
    return d


def run(d, ctrl, seconds):
    d.ctrl[:] = ctrl
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)


def pads(d):
    out = np.zeros(touch_mjx.N_PADS)
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = c.geom1, c.geom2
        p1, p2 = touch.pad_of_geom[g1], touch.pad_of_geom[g2]
        if p1 >= 0 and touch.object_geom[g2]:
            pad = p1
        elif p2 >= 0 and touch.object_geom[g1]:
            pad = p2
        else:
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        out[pad] += max(f6[0], 0.0)
    return out


def show(tag, d):
    f = pads(d)
    on = ", ".join(f"{n}={v:.2f}" for n, v in zip(touch_mjx.PAD_NAMES, f) if v > 0.05)
    tension = d.actuator_force.copy()
    print(f"{tag}: cube z={d.qpos[18]*1000:.1f} mm xy=({d.qpos[16]*1000:.0f},{d.qpos[17]*1000:.0f}) | pads N: {on or '-'}")
    print(f"    tension: " + " ".join(f"{a.split('_')[1]}={t:.2f}" for a, t in zip(ACT, tension)))


print("timestep", m.opt.timestep, "gravity", m.opt.gravity, "ctrl lo", LO.round(4), "hi", HI.round(4))
d = fresh()
run(d, HOME.ctrl, 1.0)
show("rest (home ctrl, 1 s)", d)
print("palm_position sensor", env.get_palm_position(d).round(4) if hasattr(env, "get_palm_position") else None)

print("\n-- finger direction (index tendon) --")
for value in (LO[0], 0.07, 0.09, 0.10, HI[0]):
    d = fresh((0.3, 0.0, 0.2))   # cube far away
    ctrl = HOME.ctrl.copy(); ctrl[0] = value
    run(d, ctrl, 1.0)
    q = d.qpos[JOINTS[:3]]
    print(f"index ctrl {value:.4f}: mcp/pip/dip = {np.round(q, 3)} tension {d.actuator_force[0]:.2f} len {d.actuator_length[0]:.4f}")

print("\n-- thumb direction --")
for i, name in ((4, "abd"), (5, "th1"), (6, "th2")):
    for value in (LO[i], HOME.ctrl[i], HI[i]):
        d = fresh((0.3, 0.0, 0.2))
        ctrl = HOME.ctrl.copy(); ctrl[i] = value
        run(d, ctrl, 1.0)
        q = d.qpos[JOINTS[12:16]]
        tip = d.geom_xpos[m.geom("th_tip").id]
        print(f"{name} ctrl {value:.4f}: thumb abd/cmc/mcp/ip = {np.round(q, 3)} tip {np.round(tip*1000, 0)} mm")

d = fresh()
run(d, HOME.ctrl, 1.0)
REST_Z = d.qpos[18]
print("\nfingertips at rest (mm):", {g: np.round(d.geom_xpos[m.geom(g).id] * 1000).tolist() for g in consts.FINGERTIP_NAMES})


def ramp(d, target, seconds=1.0, hold=1.0):
    start = d.ctrl.copy()
    n = int(seconds / m.opt.timestep)
    for k in range(n):
        d.ctrl[:] = start + (target - start) * (k + 1) / n
        mujoco.mj_step(m, d)
    run(d, target, hold)


def grasp_stats(d):
    f = pads(d)
    names = touch_mjx.PAD_NAMES
    thumb = f[names.index("thumb_tip")] + f[names.index("thumb_base")]
    fingers = [f[names.index(n)] for n in names if n.split("_")[0] in ("index", "middle", "ring", "little")]
    digits = {}
    for n, v in zip(names, f):
        digits[n.split("_")[0]] = digits.get(n.split("_")[0], 0.0) + v
    firm_fingers = sum(digits[k] >= 0.3 for k in ("index", "middle", "ring", "little"))
    speed = np.linalg.norm(d.qvel[16:19])
    return dict(lift_mm=(d.qpos[18] - REST_Z) * 1000, palm=f[names.index("palm")], thumb=thumb,
                firm_fingers=firm_fingers, speed=speed, z=d.qpos[18], digits=digits)


print("\n-- random slow closures from rest: looking for thumb + 2 fingers holding the cube off the palm --")
rng = np.random.default_rng(0)
found = []
for trial in range(3000):
    d = fresh()
    run(d, HOME.ctrl, 0.5)
    target = rng.uniform(LO, HI)
    ramp(d, target, 1.0, 1.0)
    s = grasp_stats(d)
    if s["z"] < -0.05:
        continue
    held = s["thumb"] >= 0.3 and s["firm_fingers"] >= 2 and s["speed"] < 0.01
    if held and s["palm"] < 0.05:
        found.append((s["lift_mm"], target, s))
print(f"lifted thumb+2-finger holds: {len(found)} / 3000")
for lift, target, s in sorted(found, key=lambda x: -x[0])[:8]:
    print(f"  lift {lift:5.1f} mm  palm {s['palm']:.2f} N  thumb {s['thumb']:.2f} N  digits " +
          " ".join(f"{k}={v:.1f}" for k, v in s['digits'].items() if v > 0.05) + f"  ctrl {np.round(target, 4).tolist()}")
