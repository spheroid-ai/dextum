"""Measure fingertip indentation for §16's proposed stiffness intervention.

Identification fixture only, not a policy test or a bench measurement. The
frozen hand is never overwritten. Numerical work runs on the cloud instance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    import mujoco
    import numpy as np
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    full = mujoco.MjModel.from_xml_path(str(a.model))
    report = dict(state='MEASURING', model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),
                  fixture='Fixed fingertip sphere, 50g translating flat indenter, gravity zero, 1/2/5N normal loads, 10s each',
                  scope='Native fixture identification; MJX and changed-hand collection qualification still required before training',
                  settings=dict(timestep=.005, iterations=5, ls_iterations=8, cone='elliptic'), rows=[])
    def values(x):
        return ' '.join(str(float(v)) for v in np.atleast_1d(x))
    for finger in ('thumb', 'index', 'middle', 'ring', 'pinky'):
        g = full.geom('kit_pad_contact_'+finger+'_tip').id
        assert full.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
        for multiplier in (1., 5.):
            solref = full.geom_solref[g].copy()
            assert solref[0] < 0
            solref[0] *= multiplier
            root = ET.Element('mujoco')
            ET.SubElement(root, 'option', timestep='.005', gravity='0 0 0', solver='Newton',
                          iterations='5', ls_iterations='8', cone='elliptic', integrator='implicitfast')
            world = ET.SubElement(root, 'worldbody')
            common = dict(friction=values(full.geom_friction[g]), solref=values(solref),
                          solimp=values(full.geom_solimp[g]), condim=str(int(full.geom_condim[g])))
            radius = float(full.geom_size[g, 0])
            ET.SubElement(world, 'geom', type='sphere', size=str(radius), **common)
            body = ET.SubElement(world, 'body', pos=values([0, 0, radius+.004]))
            ET.SubElement(body, 'joint', type='slide', axis='0 0 1')
            ET.SubElement(body, 'geom', type='box', size='.004 .004 .004', mass='.05', **common)
            m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
            for force in (1., 2., 5.):
                d = mujoco.MjData(m)
                d.qfrc_applied[0] = -force
                trace = []
                for _ in range(2000):
                    mujoco.mj_step(m, d)
                    trace.append((float(d.qpos[0]), float(d.qfrc_constraint[0])))
                trace = np.asarray(trace)
                settled = trace[-400:]
                indentation_mm = float(-np.mean(settled[:, 0])*1000)
                row = dict(finger=finger, multiplier=multiplier, solref=solref.tolist(), force_N=force,
                           settled_indentation_mm=indentation_mm,
                           secant_stiffness_N_per_mm=force/indentation_mm,
                           settled_drift_mm=float(np.ptp(settled[:, 0])*1000),
                           settled_force_error_N=float(np.max(np.abs(settled[:, 1]-force))),
                           finite=bool(np.isfinite(trace).all()))
                report['rows'].append(row)
                np.savez_compressed(a.output/f'{finger}_{multiplier}_{force}.npz', trace=trace)
                print(json.dumps(row), flush=True)
    report['state'] = 'MEASURED'
    report['limitations'] = [
        'Contact stiffness is mass-normalized; this fixture does not establish a universal material stiffness.',
        'The hardware indentation curve remains unmeasured.',
        'No policy, sensor acceptance or training attempt is changed by this diagnostic.']
    (a.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    if not os.environ.get('CONTAINER_ID'):
        raise RuntimeError('Vast only')
    main()
