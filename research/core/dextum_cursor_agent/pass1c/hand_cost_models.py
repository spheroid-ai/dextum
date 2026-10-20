"""§16 dimension-only hand variants, with fixed-fixture stiffness identification."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    import mujoco
    import numpy as np
    from pass1c.geometry_audit import ray_exit, shifted, summary
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    source = ET.parse(a.model)
    full = mujoco.MjModel.from_xml_path(str(a.model))
    gid = full.geom('kit_pad_contact_index_tip').id
    def values(x): return ' '.join(str(float(v)) for v in np.atleast_1d(x))

    def measure(stiffness, force):
        root = ET.Element('mujoco')
        ET.SubElement(root, 'option', timestep='.005', gravity='0 0 0', solver='Newton',
                      iterations='5', ls_iterations='8', cone='elliptic', integrator='implicitfast')
        world = ET.SubElement(root, 'worldbody')
        common = dict(solref=values([-stiffness, full.geom_solref[gid, 1]]),
                      solimp=values(full.geom_solimp[gid]), friction=values(full.geom_friction[gid]),
                      condim=str(int(full.geom_condim[gid])))
        radius = float(full.geom_size[gid, 0])
        ET.SubElement(world, 'geom', type='sphere', size=str(radius), **common)
        body = ET.SubElement(world, 'body', pos=values([0, 0, radius+.004]))
        ET.SubElement(body, 'joint', type='slide', axis='0 0 1')
        ET.SubElement(body, 'geom', type='box', size='.004 .004 .004', mass='.05', **common)
        m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        d = mujoco.MjData(m); d.qfrc_applied[0] = -force
        trace = []
        for _ in range(2000):
            mujoco.mj_step(m, d)
            trace.append((d.qpos[0], d.qfrc_constraint[0]))
        trace = np.asarray(trace)
        settled = trace[-400:]
        depth = float(-np.mean(settled[:, 0])*1000)
        row = dict(coefficient=stiffness, force_N=force, indentation_mm=depth,
                   secant_N_per_mm=force/depth, drift_mm=float(np.ptp(settled[:, 0])*1000),
                   force_error_N=float(np.max(np.abs(settled[:, 1]-force))),
                   finite=bool(np.isfinite(trace).all()))
        with (a.output/'identification.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        assert row['finite'] and row['drift_mm'] < .01 and row['force_error_N'] < .05, row
        return row

    # Identify one fixed physical response, without evaluating any policy.
    lo, hi = 10000., 40000.
    assert measure(lo, 5.)['secant_N_per_mm'] < 10 < measure(hi, 5.)['secant_N_per_mm']
    for _ in range(20):
        mid = (lo+hi)/2
        if measure(mid, 5.)['secant_N_per_mm'] < 10: lo = mid
        else: hi = mid
    coefficient = (lo+hi)/2
    curve = [measure(coefficient, f) for f in (1., 2., 5.)]
    assert abs(curve[-1]['secant_N_per_mm']-10) < .01
    root = source.getroot()
    for g in root.iter('geom'):
        if g.get('name', '').startswith('kit_pad_contact_'):
            old = [float(v) for v in g.get('solref').split()]
            g.set('solref', values([-coefficient, old[1]]))
    stiff = a.output/'stiff_10.xml'
    source.write(stiff, encoding='unicode')
    thin_tree = copy.deepcopy(source)
    thin_root = thin_tree.getroot()
    geoms = {g.get('name'): g for g in thin_root.iter('geom')}
    cad = json.loads(a.geometry.read_text())
    geometry = []
    for part in cad['parts']:
        if part.get('layer') != 'skin' or not part.get('module', '').endswith('_tip'): continue
        name = part['module']; g = geoms['kit_pad_contact_'+name]
        vertices = np.asarray(part['vertices']); count = len(vertices)//2
        used = np.unique(np.asarray(part['faces']).ravel() % count)
        lower, upper = vertices[used], vertices[used+count]
        band = part['band_mm']; normals = (upper-lower)/(band[1]-band[0])
        base = lower-band[0]*normals
        before = ray_exit(base, normals, g)
        # Keep each sphere's radius/curvature. Minimal inward offset satisfies
        # the same +1mm approximation allowance for the 3.5mm stack.
        lo, hi = 0., 6.
        assert np.nanmax(ray_exit(base, normals, shifted(g, hi))) <= 4.5
        for _ in range(40):
            mid = (lo+hi)/2
            if np.nanmax(ray_exit(base, normals, shifted(g, mid))) > 4.5: lo = mid
            else: hi = mid
        g.attrib.update(shifted(g, hi).attrib)
        after = ray_exit(base, normals, g)
        cells = [c for c in cad['cells'] if c['module'] == name]
        cn = np.array([c['normal'] for c in cells])
        cb = np.array([c['position_mm'] for c in cells])-5.25*cn
        cell_depth = ray_exit(cb, cn, g)
        # Record coverage instead of selecting a geometry by policy success.
        admissible = bool(np.isfinite(cell_depth).all() and (cell_depth >= 2.5).all()
                          and (cell_depth <= 4.5001).all())
        for c in cells:
            site = next(s for s in thin_root.iter('site') if s.get('name') == 'kit_site_'+c['name'])
            pos = np.fromstring(site.get('pos'), sep=' ')-np.asarray(c['normal'])*.00175
            site.set('pos', values(pos))
        geometry.append(dict(pad=name, inward_offset_mm=hi, before=summary(before), after=summary(after),
                             cell_protrusion_mm=cell_depth.tolist(), admissible=admissible))
    assert len(geometry) == 5
    thin = a.output/'thin_stiff_10.xml'
    thin_tree.write(thin, encoding='unicode')
    report = dict(state='PREPARED_NEEDS_QUALIFICATION',
                  source_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),
                  identified_solref_stiffness=coefficient, fixture_curve=curve,
                  target='10 N/mm secant at 5N with 50g indenter; full curve reported, not a universal material constant',
                  models={p.stem:hashlib.sha256(p.read_bytes()).hexdigest() for p in (stiff, thin)},
                  thin_nominal_stack_mm=3.5, thin_allowance_mm=1., thin_geometry=geometry,
                  thin_geometry_admissible=all(x['admissible'] for x in geometry),
                  preserved='Hand, cube, masses/inertias, solver, contact damping/impedance, reward, task; only pad stiffness and thin-tip offsets/site positions changed',
                  next='MJX indentation check and final collection detector before training; masking must finish first')
    (a.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    if not os.environ.get('CONTAINER_ID'): raise RuntimeError('Vast only')
    main()
