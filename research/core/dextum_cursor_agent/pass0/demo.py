"""Remote native-MuJoCo Pass 0 demo with an observable-only controller boundary.

Ideal camera and 11 lumped Ideal tactile pads are explicit stand-ins. This is
not pixel perception, Kit B realism, or a tactile-odometry success claim.
"""
import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET
if not os.environ.get("CONTAINER_ID"):
    raise RuntimeError("Run this demo on Vast.ai only")

import mujoco
import numpy as np
import vendor.aero_official.vision_env  # Configures the frozen asset roots.
from mujoco_playground._src.manipulation.aero_hand import base, aero_hand_constants
from dextum.touch_mjx import TouchModel
from pass0.assets import ASSETS, generate, mesh
from pass0.numpy_actor import Actor, ContactMotor, HOME, actor_frame
from pass0.runtime import Packet, TaskLoop, reorientation_dag, scripted_action
from pass0.review_gate import require_review_pass


def scene(asset):
    resources = base.get_assets()
    root = ET.fromstring(resources["reorientation_cube.xml"])
    body = root.find(".//body[@name='cube']")
    for geom in list(body.findall("geom")):
        if geom.get("name") != "cube":
            body.remove(geom)
    collision = body.find("geom[@name='cube']")
    vertices, faces = mesh(asset)
    ET.SubElement(root.find("asset"), "mesh", name="pass0_object",
                  vertex=" ".join(str(x) for v in vertices for x in v),
                  face=" ".join(str(i) for f in faces for i in f))
    collision.attrib.pop("size", None)
    collision.set("type", "mesh")
    collision.set("mesh", "pass0_object")
    collision.set("mass", str(asset["mass_g"]/1000))
    collision.set("rgba", ".25 .6 .9 1")
    collision.set("group", "2")
    resources["reorientation_cube.xml"] = ET.tostring(root)
    model = mujoco.MjModel.from_xml_string(Path(aero_hand_constants.CUBE_XML).read_text(), assets=resources)
    model.opt.timestep = .01
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 720
    return model


def reading(model, data, touch):
    normal, shear = np.zeros(11), np.zeros(11)
    world_force, weighted_point = np.zeros((11, 3)), np.zeros((11, 3))
    slip = np.zeros(11, bool)
    cube = touch.object_body
    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom
        p1, p2 = touch.pad_of_geom[g1], touch.pad_of_geom[g2]
        first = p1 >= 0 and touch.object_geom[g2]
        second = p2 >= 0 and touch.object_geom[g1]
        if not (first or second):
            continue
        pad = p1 if first else p2
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        f_n, f_t = max(0.0, force[0]), np.linalg.norm(force[1:3])
        frame = contact.frame.reshape(3, 3)
        world_force[pad] += (1 if first else -1)*force[:3] @ frame
        normal[pad] += f_n
        shear[pad] += f_t
        weighted_point[pad] += f_n*contact.pos
        pad_body = model.geom_bodyid[g1 if first else g2]
        def velocity(body):
            cv = data.cvel[body]
            return cv[3:] + np.cross(cv[:3], contact.pos-data.subtree_com[model.body_rootid[body]])
        relative = velocity(cube)-velocity(pad_body)
        tangent = relative - np.dot(relative, frame[0])*frame[0]
        slip[pad] |= f_n > .05 and np.linalg.norm(tangent) > .01
    force_xyz, centroid = np.zeros((11, 3)), np.zeros((11, 3))
    for i, body in enumerate(touch.frame_bodies):
        rotation = data.xmat[body].reshape(3, 3)
        force_xyz[i] = world_force[i] @ rotation
        if normal[i] > 0:
            centroid[i] = (weighted_point[i]/normal[i]-data.xpos[body]) @ rotation
    return normal, shear, slip, force_xyz, centroid


def camera_angle(data, cube):
    rotation = data.xmat[cube].reshape(3, 3)
    up = np.argmax(np.abs(rotation[2]))
    side = rotation[:, (up+1) % 3]
    return math.degrees(math.atan2(side[1], side[0]))


def quat_mul(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return np.array([w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                     w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v])


def trial(asset, seed, sign, blackout, actor):
    model = scene(asset)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    cube = model.body("cube").id
    joint = model.joint("cube_freejoint").id
    qstart = model.jnt_qposadr[joint]
    rng = np.random.default_rng(seed)
    # Independent held-out start bank. No feedback from its outcomes into scripts.
    limits = model.jnt_range[model.jnt_type != mujoco.mjtJoint.mjJNT_FREE]
    data.qpos[:qstart] = np.clip(data.qpos[:qstart]+.1*rng.normal(size=qstart), limits[:, 0], limits[:, 1])
    data.qpos[qstart:qstart+3] = np.array([.1, 0, .05])+rng.uniform(-.01, .01, 3)
    quaternion = rng.normal(size=4)
    data.qpos[qstart+3:qstart+7] = quaternion/np.linalg.norm(quaternion)
    data.qvel[:] = 0
    data.ctrl[:] = HOME
    data.mocap_pos[:] = -100
    mujoco.mj_forward(model, data)
    touch = TouchModel(model)
    motor = ContactMotor(model.actuator_ctrlrange)
    loop = TaskLoop(reorientation_dag(f"reorient by {sign*30} degrees"), blackout=blackout)
    history = deque(maxlen=3)
    age = np.zeros(11, int)
    target = None
    true_hold = best_hold = 0
    first_contact_tick = None
    success = dropped = False
    trace = []
    sensor_names = ("len_if", "len_mf", "len_rf", "len_pf", "len_th1", "len_th2", "len_th_abd")
    yaw_error = 180.0
    for tick in range(300):
        normal, shear, slip, force_xyz, centroid = reading(model, data, touch)
        age = np.where(normal>.3, age+1, 0)
        motors = np.array([data.sensor(n).data[0] for n in sensor_names])
        # Camera is an explicitly declared Ideal sensor stand-in; the backend
        # consumes geometry, the runtime sees only this measurement packet.
        camera_valid = not blackout or first_contact_tick is None
        yaw = camera_angle(data, cube) if camera_valid else None
        if first_contact_tick is None and np.any(normal > .3):
            first_contact_tick = tick
            theta = math.radians(sign*30)/2
            target = quat_mul([math.cos(theta), 0, 0, math.sin(theta)], data.xquat[cube])
        packet = Packet(tick, tuple(normal), tuple(shear), tuple(slip), tuple(motors),
                        yaw, asset["family"] if camera_valid else None, camera_valid)
        choice = loop.update(packet)
        quantum = choice["quantum"]
        frame = actor_frame(motors, motor, normal, shear, slip, force_xyz, centroid, age,
                            (loop.boundary_yaw or 0)+sign*30, sign, max(0, tick-20))
        if not history:
            history.extend([frame.copy() for _ in range(3)])
        else:
            history.appendleft(frame)
        # E3 policy only where its cube-trained rotation interface fits. Other
        # families exercise the explicitly labelled scripts without retraining.
        learned = asset["family"] == "cube" and quantum == "pivot" and actor is not None
        action = actor(np.concatenate(list(history))) if learned else scripted_action(
            quantum, choice.get("edge_tick", 0), sign, choice.get("attempts", 0))
        command = HOME if tick < 20 else motor.step(action, normal, shear, slip)
        if tick < 20:
            motor.cmd = HOME.copy()
        data.ctrl[:] = command
        for _ in range(5):
            mujoco.mj_step(model, data)
        dropped = bool(data.qpos[qstart+2] < -.05)
        if target is not None:
            error = quat_mul(target, data.xquat[cube]*np.array([1, -1, -1, -1]))
            angle = 2*math.atan2(error[3], error[0])
            yaw_error = abs(math.degrees(math.atan2(math.sin(angle), math.cos(angle))))
            tilt_ok = error[0]**2+error[3]**2 >= math.cos(math.radians(30)/2)**2
            within = yaw_error < 10 and tilt_ok and not dropped
            true_hold = true_hold+1 if within else 0
            best_hold = max(best_hold, true_hold)
            success |= true_hold >= 40
        trace.append(dict(tick=tick, node=choice["node"], quantum=quantum, learned=learned,
                          progress=choice.get("progress", 0), goal_observed=loop.goal_observed,
                          camera_valid=camera_valid, evaluation_only_yaw_error_deg=yaw_error,
                          contact_count=int((normal>.3).sum()), command=list(map(float, command))))
        if dropped or not np.isfinite(data.qpos).all():
            break
    return dict(asset=asset["id"], seed=seed, turn_deg=sign*30, blackout=blackout,
                success=bool(success), dropped=dropped, max_hold_s=best_hold*.05,
                goal_observed=loop.goal_observed, final_error_deg=yaw_error,
                first_contact_tick=first_contact_tick, replans=loop.replans,
                trace=list(loop.trace), steps=trace)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--qualification", type=Path, required=True)
    p.add_argument("--actor", type=Path, required=True)
    p.add_argument("--starts", type=int, default=16)
    a = p.parse_args()
    require_review_pass(json.loads(a.qualification.read_text()))
    assert json.loads(a.actor.with_suffix('.audit.json').read_text())["state"] == "PASS", "Actor/G audit hard stop"
    a.output.mkdir(parents=True, exist_ok=False)
    generate(a.output/"assets")
    actor = Actor(a.actor)
    results = []
    for asset in ASSETS:
        for blackout in (True, False):
            trials = []
            for i in range(a.starts):
                trial_result = trial(asset, 2026101000+i, 1 if i % 2 == 0 else -1, blackout, actor)
                path = a.output/f"{asset['id']}_{'blackout' if blackout else 'vision'}_{i:02d}.json"
                path.write_text(json.dumps(trial_result)+"\n")
                trials.append(trial_result)
            row = dict(asset=asset["id"], blackout=blackout, n=len(trials),
                       success=sum(t["success"] for t in trials)/len(trials),
                       drops=sum(t["dropped"] for t in trials)/len(trials),
                       goal_observed=sum(t["goal_observed"] for t in trials)/len(trials),
                       mean_replans=sum(t["replans"] for t in trials)/len(trials))
            results.append(row)
            print(json.dumps(row), flush=True)
            (a.output/"results.json").write_text(json.dumps(dict(state="RUNNING", rows=results), indent=2)+"\n")
    result = dict(state="COMPLETE", rows=results, sensor_level="Ideal 11 lumped pads",
                  vision="Ideal shape/face-angle packet stand-in; no learned pixel encoder",
                  odometry="Unavailable stand-in; post-contact blind pose stays unknown",
                  physics="Native C MuJoCo; nominal model; independent held-out starts",
                  scope="Pass 0 plumbing demonstration, not T5/T6/T8 qualification or Kit B competence")
    (a.output/"results.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
