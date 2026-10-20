"""Registered contact-only aliasing probe; not an O1 PPO success comparison."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path


def main():
    if not os.environ.get('CONTAINER_ID'):
        raise RuntimeError('Cloud numerical diagnostic only')
    import mujoco
    import numpy as np
    from .native import ContactOracle, ObjectSurface
    from .patches import CubePatches
    from .tokenizer import Tokenizer
    p = argparse.ArgumentParser()
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    registration = json.loads(args.registration.read_text())
    assert registration['name'] == 'pose_free_contact_alias_probe_01'
    for name, digest in registration['source_sha256'].items():
        assert hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() == digest
    args.output.mkdir(parents=True, exist_ok=False)
    traces, vectors, tokens, physical_success = [], [], [], []
    for yaw_deg in registration['yaw_degrees']:
        a = np.radians(yaw_deg)/2
        xml = '<mujoco><option timestep=".002"/><worldbody>' \
              '<geom name="support" type="plane" size="1 1 .1" friction=".5 .01 .001"/>' \
              '<body name="object" pos="0 0 .026" quat="'+f'{np.cos(a)} 0 0 {np.sin(a)}'+'">' \
              '<freejoint/><geom name="object_geom" type="box" size=".025 .025 .025" mass=".1"/>' \
              '</body></worldbody></mujoco>'
        model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model)
        for _ in range(registration['settle_steps']): mujoco.mj_step(model, data)
        oracle = ContactOracle(model, [ObjectSurface('cube', model.body('object').id,
            (model.geom('object_geom').id,), CubePatches())],
            {model.geom('support').id: ('palm', 0)})
        encoder = Tokenizer(); trace, packed, sequence, within = [], [], [], []
        for _ in range(registration['control_ticks']):
            for __ in range(registration['physics_steps_per_tick']): mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            frame = oracle.read(data, task_id='issued_yaw_plus_30', quantum='hold')
            trace.append(frame.to_dict()); sequence.append([asdict(t) for t in encoder.encode(frame)])
            packed.append([[c.normal_n, *c.force_object_n, *c.point_object_m,
                c.slip_speed_m_s, c.friction, c.contact_count,
                c.patch_assignment_distance_m, int(c.patch_boundary_ambiguous)] for c in frame.contacts])
            q = data.qpos[3:7]
            yaw = 2*np.arctan2(q[3], q[0])
            error = np.arctan2(np.sin(np.radians(registration['goal_degrees'])-yaw),
                               np.cos(np.radians(registration['goal_degrees'])-yaw))
            within.append(abs(error) < np.radians(registration['yaw_tolerance_degrees']))
        traces.append(trace); vectors.append(np.asarray(packed)); tokens.append(sequence)
        physical_success.append(bool(all(within)))
    matching_keys = all([[c['key'] for c in f['contacts']] for f in traces[0]] ==
                        [[c['key'] for c in f['contacts']] for f in t] for t in traces[1:])
    same_shape = vectors[0].shape == vectors[1].shape
    maximum = float(np.max(np.abs(vectors[0]-vectors[1]))) if same_shape else None
    alias = matching_keys and tokens[0] == tokens[1] and same_shape and \
        maximum <= registration['continuous_absolute_tolerance'] and len(set(physical_success)) > 1
    result = dict(state='MEASURED', observation_alias_detected=alias,
        continuous_max_absolute_difference=maximum, continuous_shape=list(vectors[0].shape),
        identical_contact_keys=matching_keys, identical_typed_tokens=tokens[0] == tokens[1],
        physical_yaw_hold_success=physical_success, yaw_degrees=registration['yaw_degrees'],
        registration_sha256=hashlib.sha256(args.registration.read_bytes()).hexdigest(),
        training_attempts=0, optimizer_steps=0,
        scope='Passive plane used as a canonical palm support; same mass, friction, goal and full contact-only history. No Aero controller or registered O1 policy comparison.',
        interpretation='An alias is a counterexample for this pose-free aggregate interface on the fixture. The full DEXTUM state explicitly includes pose; no failure of that richer state is implied.')
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    (args.output/'traces.json').write_text(json.dumps(traces)+'\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
