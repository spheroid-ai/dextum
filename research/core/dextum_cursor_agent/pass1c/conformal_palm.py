"""User-authorized §14 palm correction from the saved kit CAD, before training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

if not os.environ.get('CONTAINER_ID'):
    raise RuntimeError('Run numerical geometry work on Vast only')
import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from pass1c.geometry_audit import ray_exit, summary


def hull_exit(points, normals, vertices):
    hull = ConvexHull(vertices)
    a, b = hull.equations[:, :3], hull.equations[:, 3]
    offset = points @ a.T + b
    direction = normals @ a.T
    t = -offset / np.where(abs(direction) > 1e-12, direction, 1)
    upper = np.min(np.where(direction > 1e-12, t, np.inf), axis=1)
    lower = np.max(np.where(direction < -1e-12, t, -np.inf), axis=1)
    outside = np.any((abs(direction) <= 1e-12) & (offset > 1e-7), axis=1)
    return np.where((upper >= lower) & ~outside, upper, np.nan)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cad = json.loads(Path('/root/dextum/kit_b/geometry.json').read_text())
    skin = next(p for p in cad['parts'] if p.get('module') == 'palm' and p.get('layer') == 'skin')
    vertices = np.asarray(skin['vertices'])
    vertices = vertices[np.unique(np.asarray(skin['faces']).ravel())]
    vertices = vertices[ConvexHull(vertices).vertices]
    measurements = np.load(args.run/'geometry/measurements.npz')
    base, normal = measurements['palm_base_mm'], measurements['palm_normal']
    full = hull_exit(base, normal, vertices)
    full_summary = summary(full)
    cells = [c for c in cad['cells'] if c['module'] == 'palm']
    cn = np.array([c['normal'] for c in cells])
    cb = np.array([c['position_mm'] for c in cells]) - 5.25*cn
    # Deterministic inscribed hull: axis extrema, then the source vertex with
    # largest distance outside the current approximation. This is CAD fitting,
    # independent of policies, contacts, solver options, or training outcomes.
    selected = sorted(set(np.r_[np.argmin(vertices, axis=0), np.argmax(vertices, axis=0)].tolist()))
    while len(selected) < 32:
        h = ConvexHull(vertices[selected])
        dist = (vertices @ h.equations[:, :3].T + h.equations[:, 3]).max(axis=1)
        if float(dist.max()) < .10: break
        selected.append(int(np.argmax(dist)))
    chosen = vertices[selected]
    fit = hull_exit(base, normal, chosen)
    cell_fit = hull_exit(cb, cn, chosen)
    row = dict(state='MEASURED',full_hull_vertices=len(vertices),fitted_hull_vertices=len(chosen),
               full_hull=full_summary,fitted_hull=summary(fit),cell_protrusion_mm=cell_fit.tolist())
    (args.output/'dimension_check.json').write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps(row),flush=True)
    if not (np.isfinite(cell_fit).all() and np.max(abs(cell_fit-5.25)) <= 1
            and np.nanmax(fit) <= 6.25 and np.isfinite(fit).sum() >= .95*len(fit)):
        raise RuntimeError('One convex palm hull does not represent the specified skin within tolerance')
    tree=ET.parse(args.source)
    root=tree.getroot()
    palm=next(g for g in root.iter('geom') if g.get('name')=='kit_pad_contact_palm')
    mesh_name='kit_palm_conformal_section14'
    # The exact 32-or-fewer vertices are shared by native and MJX hull builders.
    ET.SubElement(root.find('asset'),'mesh',name=mesh_name,
        vertex=' '.join(map(str,(chosen*.001).ravel())),maxhullvert='32')
    for key in ('size','pos','quat','fromto'):palm.attrib.pop(key,None)
    palm.set('type','mesh');palm.set('mesh',mesh_name)
    thumb_fit=next(r for r in json.loads((args.run/'geometry_size_offset_fit.json').read_text()) if r['pad']=='thumb_tip')
    assert thumb_fit['success'] and thumb_fit['min_constraint']>=-1e-6
    thumb=next(g for g in root.iter('geom') if g.get('name')=='kit_pad_contact_thumb_tip')
    thumb.attrib.update(thumb_fit['geometry'])
    xml=args.output/'kit_b_training.xml';tree.write(xml,encoding='unicode')
    old=mujoco.MjModel.from_xml_path(str(args.source));new=mujoco.MjModel.from_xml_path(str(xml))
    for f in ('body_mass','body_inertia','body_ipos','body_iquat','site_pos','site_quat',
              'geom_contype','geom_conaffinity','geom_friction','geom_solref','geom_solimp'):
        np.testing.assert_array_equal(getattr(old,f),getattr(new,f))
    for gid in range(old.ngeom):
        if old.geom(gid).name in ('kit_pad_contact_palm','kit_pad_contact_thumb_tip'):continue
        for f in ('geom_type','geom_size','geom_pos','geom_quat'):
            np.testing.assert_array_equal(getattr(old,f)[gid],getattr(new,f)[gid])
    assert old.ngeom==new.ngeom and old.nsite==new.nsite
    # Check the compiled mesh coordinates too, including MuJoCo's recentering.
    gid=new.geom('kit_pad_contact_palm').id;mid=new.geom_dataid[gid]
    start=new.mesh_vertadr[mid];size=new.mesh_vertnum[mid]
    rotation=np.zeros(9);mujoco.mju_quat2Mat(rotation,new.geom_quat[gid])
    compiled=(new.mesh_vert[start:start+size] @ rotation.reshape(3,3).T + new.geom_pos[gid])*1000
    compiled_fit=hull_exit(base,normal,compiled)
    valid=np.isfinite(fit)&np.isfinite(compiled_fit)
    np.testing.assert_allclose(compiled_fit[valid],fit[valid],atol=1e-4)
    result=dict(state='PASS',authority='review_section14_followup_approved_conformal_palm',
        source_xml_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        model_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),geometry_accepted=True,
        palm=row,thumb=thumb_fit,compiled_palm=summary(compiled_fit),
        unchanged='Other pads, original hand, cube, sites, masses/inertias, filters, contact parameters and solver',
        kit_pad_colliders=10,cell_sites=40,collision_shape='One inscribed conformal convex palm hull, <=32 vertices',
        limitations=['Thumb primitive covers 1514/2804 sampled CAD skin rays; four sensing sites remain within 1 mm of the surface',
                    'Stiffness remains the unmeasured 2 N/mm placeholder'])
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(args.output/'palm_fit.npz',vertices_mm=chosen,base_mm=base,normal=normal,
                        full_hull_protrusion_mm=full,fitted_protrusion_mm=fit)
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
