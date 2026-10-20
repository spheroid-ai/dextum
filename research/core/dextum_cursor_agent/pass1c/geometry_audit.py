"""Review §14: one dimensional audit, with only primitive offsets corrected.

Uses the saved CAD skin's paired offset vertices to recover the exact fitted
substrate and its outward normals. No simulation or policy outcomes enter it.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

if not os.environ.get('CONTAINER_ID'):
    raise RuntimeError('Numerical geometry audit runs on Vast only')
import mujoco
import numpy as np


def ray_exit(base, normal, geom):
    """Outer intersection in mm; NaN means the pad does not cover that ray."""
    kind = geom.get('type')
    radius = float(geom.get('size').split()[0]) * 1000
    center = np.fromstring(geom.get('pos', '0 0 0'), sep=' ') * 1000
    if kind == 'sphere':
        d = base - center
        b = np.sum(d * normal, axis=1)
        disc = b*b - (np.sum(d*d, axis=1) - radius*radius)
        return np.where(disc >= 0, -b + np.sqrt(np.maximum(disc, 0)), np.nan)
    if kind == 'box':
        assert geom.get('quat') is None
        half = np.fromstring(geom.get('size'), sep=' ') * 1000
        with np.errstate(divide='ignore', invalid='ignore'):
            t0 = (center - half - base) / normal
            t1 = (center + half - base) / normal
        lo = np.min(np.stack([t0, t1]), axis=0).max(axis=1)
        hi = np.max(np.stack([t0, t1]), axis=0).min(axis=1)
        return np.where(hi >= lo, hi, np.nan)
    assert kind == 'capsule'
    ends = np.fromstring(geom.get('fromto'), sep=' ').reshape(2, 3) * 1000
    axis = ends[1] - ends[0]
    length = np.linalg.norm(axis)
    axis /= length
    d = base - ends[0]
    dn, da = normal @ axis, d @ axis
    pn, pd = normal - dn[:, None]*axis, d - da[:, None]*axis
    a, b = (pn*pn).sum(1), (pn*pd).sum(1)
    disc = b*b - a*((pd*pd).sum(1) - radius*radius)
    candidates = []
    for sign in [-1, 1]:
        t = (-b + sign*np.sqrt(np.maximum(disc, 0))) / np.maximum(a, 1e-30)
        z = da + t*dn
        candidates.append(np.where((disc >= 0) & (z >= 0) & (z <= length), t, np.nan))
    for k, endpoint in enumerate(ends):
        delta = base - endpoint
        b = (delta*normal).sum(1)
        disc = b*b - ((delta*delta).sum(1) - radius*radius)
        for sign in [-1, 1]:
            t = -b + sign*np.sqrt(np.maximum(disc, 0))
            z = da + t*dn
            valid = z <= 0 if k == 0 else z >= length
            candidates.append(np.where((disc >= 0) & valid, t, np.nan))
    values = np.stack(candidates)
    return np.max(np.where(np.isnan(values), -np.inf, values), axis=0)


def shifted(g, inward_mm):
    h = ET.Element('geom', dict(g.attrib))
    key = 'fromto' if h.get('type') == 'capsule' else 'pos'
    p = np.fromstring(h.get(key), sep=' ').reshape(-1, 3)
    p[:, 1] += inward_mm / 1000
    h.set(key, ' '.join(map(str, p.ravel())))
    return h


def summary(values):
    covered = np.isfinite(values) & (values >= 0)
    v = values[covered]
    return dict(samples=len(values), covered=int(covered.sum()),
                min_mm=float(v.min()), median_mm=float(np.median(v)),
                p95_mm=float(np.percentile(v, 95)), max_mm=float(v.max()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--parameters', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cad = json.loads(a.geometry.read_text())
    params = json.loads(a.parameters.read_text())
    assert hashlib.sha256(a.parameters.read_bytes()).hexdigest() == cad['parameters_sha256']
    tree = ET.parse(a.model)
    geoms = {g.get('name'): g for g in tree.getroot().iter('geom')}
    rows, traces = [], {}
    for part in cad['parts']:
        if part.get('layer') != 'skin':
            continue
        name = part['module']
        g = geoms['kit_pad_contact_' + name]
        vertices = np.asarray(part['vertices'])
        count = len(vertices)//2
        used = np.unique(np.asarray(part['faces']).ravel() % count)
        lower, upper = vertices[used], vertices[used + count]
        band = part['band_mm']
        normal = (upper-lower)/(band[1]-band[0])
        np.testing.assert_allclose(np.linalg.norm(normal, axis=1), 1, atol=1e-8)
        base = lower-band[0]*normal
        family = 'fingertip' if name.endswith('_tip') else ('palm' if name == 'palm' else 'pad')
        thickness = max(layer['band_mm'][1] for layer in params[family]['layers'])
        before = ray_exit(base, normal, g)
        stats = summary(before)
        correction = 0.
        if stats['max_mm'] > thickness + 1:
            # Move the existing primitive inward without changing its curvature,
            # contact options, sites, masses, collision masks, or hand geometry.
            low, high = 0., stats['max_mm'] + 1
            for _ in range(50):
                mid = (low+high)/2
                v = ray_exit(base, normal, shifted(g, mid))
                peak = np.max(np.where(np.isfinite(v), v, -np.inf))
                if peak > thickness: low = mid
                else: high = mid
            correction = high
            g.attrib.update(shifted(g, correction).attrib)
        after = ray_exit(base, normal, g)
        cells = [c for c in cad['cells'] if c['module'] == name]
        cn = np.array([c['normal'] for c in cells])
        cb = np.array([c['position_mm'] for c in cells])-thickness*cn
        rows.append(dict(pad=name,shape=g.get('type'),radius_mm=(float(g.get('size').split()[0])*1000 if g.get('type') != 'box' else None),
            specified_thickness_mm=thickness,before=stats,after=summary(after),
            inward_offset_correction_mm=correction,cell_protrusion_after_mm=ray_exit(cb,cn,g).tolist(),
            contact_solref=g.get('solref'),contact_solimp=g.get('solimp'),final_geom=dict(g.attrib)))
        traces[name+'_base_mm']=base;traces[name+'_normal']=normal
        traces[name+'_before_mm']=before;traces[name+'_after_mm']=after
    assert len(rows)==10
    xml=a.output/'kit_b_training.xml'
    if any(r['inward_offset_correction_mm'] for r in rows): tree.write(xml,encoding='unicode')
    else: xml.write_bytes(a.model.read_bytes())
    old=mujoco.MjModel.from_xml_path(str(a.model));new=mujoco.MjModel.from_xml_path(str(xml))
    for f in ('geom_type','geom_size','geom_contype','geom_conaffinity','geom_friction','geom_solref','geom_solimp','body_mass','body_inertia','site_pos','site_quat'):
        np.testing.assert_array_equal(getattr(old,f),getattr(new,f))
    result=dict(state='COMPLETE',authority='review_section14',changed=any(r['inward_offset_correction_mm'] for r in rows),pads=rows,
        method='Outer primitive intersection along each outward CAD fitted-substrate normal, all skin mesh nodes used by triangles. CAD substrate bridges printed-link recesses as specified by the kit generator; measurements are normal thickness above that substrate, not sphere diameter.',
        contact_stiffness_placeholder_N_per_mm=2,shape_sites_masses_contact_options_unchanged=True,
        source_xml_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),model_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
        parameters_sha256=hashlib.sha256(a.parameters.read_bytes()).hexdigest())
    np.savez_compressed(a.output/'measurements.npz',**traces)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
