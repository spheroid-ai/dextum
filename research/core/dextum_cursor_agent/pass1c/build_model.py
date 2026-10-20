"""Section 10: ten outer pad hulls; forty sensing sites; original hand intact."""
import os,sys,json,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Vast.ai only')
import numpy as np
import mujoco
from mujoco import mjx
from mujoco.mjx._src import collision_driver

PAD_NAMES=('thumb_tip','index_tip','middle_tip','ring_tip','pinky_tip',
           'palm','index_pad','middle_pad','ring_pad','pinky_pad')

def primitives(source,out):
    """Review section 11 item 0 correction; output never replaces old XML."""
    out.mkdir(parents=True,exist_ok=False)
    tree=ET.parse(source);doc=tree.getroot();old=mujoco.MjModel.from_xml_path(str(source))
    cells=json.loads(Path('/root/dextum/kit_b/geometry.json').read_text())['cells']
    rows=[]
    for name in PAD_NAMES:
        body=next(b for b in doc.iter('body') if b.get('name')=='kit_'+name)
        g=next(g for g in body.findall('geom') if g.get('name')=='kit_pad_contact_'+name)
        g.attrib.pop('mesh',None);g.set('contype','2');g.set('conaffinity','4')
        selected=[c for c in cells if c['module']==name]
        pos=np.array([c['position_mm'] for c in selected])*.001
        normal=np.array([c['normal'] for c in selected])
        if name.endswith('_tip'):
            # Least-squares sphere centre/radius from the measured CAD surface
            # points and normals: position = centre + radius * normal.
            dp=pos-pos.mean(0);dn=normal-normal.mean(0)
            radius=float(np.sum(dp*dn)/np.sum(dn*dn))
            assert np.isfinite(radius) and radius>0,(name,radius)
            centre=pos.mean(0)-radius*normal.mean(0)
            g.set('type','sphere');g.set('pos',' '.join(map(str,centre)));g.set('size',str(radius))
            fit=np.linalg.norm(pos-centre,axis=1)-radius
        elif name.endswith('_pad'):
            radius=float(selected[0]['footprint_mm'][0])*.0005
            # Both endpoints sit one radius behind the sensing surface.
            endpoints=pos-radius*normal
            g.set('type','capsule');g.set('fromto',' '.join(map(str,endpoints.reshape(-1))));g.set('size',str(radius))
            fit=np.zeros(2)
        else:
            # Box covers the same skin bounds, in the fixed module's frame.
            gid=old.geom('kit_pad_contact_'+name).id;mid=old.geom_dataid[gid]
            start=old.mesh_vertadr[mid];count=old.mesh_vertnum[mid]
            rotation=np.zeros(9);mujoco.mju_quat2Mat(rotation,old.geom_quat[gid])
            vertices=old.mesh_vert[start:start+count]@rotation.reshape(3,3).T+old.geom_pos[gid]
            lo,hi=vertices.min(0),vertices.max(0)
            g.set('type','box');g.set('pos',' '.join(map(str,(lo+hi)/2)));g.set('size',' '.join(map(str,(hi-lo)/2)))
            fit=pos[:,1]-lo[1]
        rows.append(dict(pad=name,geom=dict(g.attrib),site_surface_residual_mm=(1000*fit).tolist()))
    # Leave hand collision masks intact. Only objects and world geoms gain
    # the receiver bit; MuJoCo's OR rule then excludes every pad/hand pair.
    receiver=[]
    for parent in doc.iter():
        for g in parent.findall('geom'):
            name=g.get('name','')
            if not name:continue
            try:gid=old.geom(name).id
            except KeyError:continue
            if old.geom_bodyid[gid] in (0,old.body('cube').id) and (old.geom_contype[gid] or old.geom_conaffinity[gid]):
                g.set('contype',str(int(old.geom_contype[gid])|4));receiver.append(name)
    # The object geom can be unnamed; its containing cube body is explicit.
    cube=next(b for b in doc.iter('body') if b.get('name')=='cube')
    for g in cube.findall('geom'):
        if g.get('contype','1')!='0' or g.get('conaffinity','1')!='0':
            g.set('contype',str(int(g.get('contype','1'))|4))
    xml=out/'kit_b_training.xml';tree.write(xml,encoding='unicode')
    m=mujoco.MjModel.from_xml_path(str(xml))
    for field in ('body_mass','body_inertia','body_ipos','body_iquat','site_pos','site_quat'):
        np.testing.assert_array_equal(getattr(old,field),getattr(m,field))
    kit=[m.geom('kit_pad_contact_'+name).id for name in PAD_NAMES]
    active=np.flatnonzero((m.geom_contype!=0)|(m.geom_conaffinity!=0))
    for a in kit:
        assert m.geom_type[a]!=mujoco.mjtGeom.mjGEOM_MESH
        for b in active:
            allowed=(int(m.geom_contype[a])&int(m.geom_conaffinity[b])) or (int(m.geom_contype[b])&int(m.geom_conaffinity[a]))
            if allowed:assert m.geom_bodyid[b] in (0,m.body('cube').id),(m.geom(a).name,m.geom(b).name)
    assert sum(m.geom_bodyid[g]==m.body('cube').id for g in active)<=8
    groups=collision_driver._geom_groups(m)
    result=dict(state='BUILT',authority='review section11 item0',pads=rows,receiver_geoms=receiver,
        collision_groups=len(groups),candidate_geom_pairs=sum(map(len,groups.values())),
        masses_inertias_sites_exact=True,hand_masks_unchanged=True,pad_hand_collisions=False,
        source_xml_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),xml_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
        caveat='Primitive geometry is an approximation; surface residuals reported, no unmeasured geometry accuracy claim')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

def main():
    if len(sys.argv)>2 and sys.argv[1]=='--primitives':
        primitives(Path(sys.argv[2]),Path(sys.argv[3]));return
    root=Path('/root/dextum/kit_b');out=Path(sys.argv[1]);out.mkdir(exist_ok=False,parents=True)
    tree=ET.parse(root/'kit_b.xml');doc=tree.getroot()
    original=mujoco.MjModel.from_xml_path(str(root/'kit_b.xml'))
    assets=doc.find('asset');meshes={n.get('name'):n for n in assets.findall('mesh')}
    removed=[]
    for parent in doc.iter():
        for g in list(parent.findall('geom')):
            name=g.get('name','')
            if name.startswith(('kit_contact_','kit_shell_contact_')) or name=='kit_hub_collision':
                removed.append(name);parent.remove(g)
    for name in PAD_NAMES:
        body=next(b for b in doc.iter('body') if b.get('name')=='kit_'+name)
        stem=name[:-4] if name.endswith('_tip') else name
        visual=next(g for g in body.findall('geom') if g.get('name')=='kit_visual_'+stem+'_skin')
        source=meshes[visual.get('mesh')]
        mesh=ET.SubElement(assets,'mesh',dict(source.attrib))
        mesh.set('name','kit_pad_hull_'+name)
        mesh.set('maxhullvert','32')
        ET.SubElement(body,'geom',dict(name='kit_pad_contact_'+name,type='mesh',mesh=mesh.get('name'),
            group='3',contype='1',conaffinity='1',condim='3',mass='0',rgba='.05 .09 .09 .25',
            friction='1 0.005 0.0001',solref='-2000 -90',solimp='0.9 0.95 0.001'))
    # Absolute asset references preserve the untouched originals.
    for n in list(assets.findall('mesh'))+list(assets.findall('texture')):
        if n.get('file') and not Path(n.get('file')).is_absolute():n.set('file',str(root/'assets'/n.get('file')))
    for n in doc.findall('compiler'):
        n.attrib.pop('meshdir',None);n.attrib.pop('texturedir',None)
    xml=out/'kit_b_training.xml';tree.write(xml,encoding='unicode')
    m=mujoco.MjModel.from_xml_path(str(xml))
    active=np.flatnonzero((m.geom_contype!=0)|(m.geom_conaffinity!=0))
    names=[m.geom(int(i)).name for i in active]
    kit=[n for n in names if n.startswith('kit_')]
    assert sorted(kit)==sorted('kit_pad_contact_'+n for n in PAD_NAMES)
    sites=[m.site(i).name for i in range(m.nsite) if m.site(i).name.startswith('kit_site_')]
    assert len(sites)==40 and not any(n.startswith('kit_contact_') for n in names)
    # Compare all existing hand collider fields by name, not shifted ids.
    fields=('geom_type','geom_contype','geom_conaffinity','geom_condim','geom_size','geom_pos','geom_quat','geom_friction','geom_solref','geom_solimp')
    kept=[]
    retained_new=[int(k) for k in active if not m.geom(int(k)).name.startswith('kit_')]
    retained_old=[int(i) for i in np.flatnonzero((original.geom_contype!=0)|(original.geom_conaffinity!=0))
                  if not original.geom(int(i)).name.startswith('kit_')]
    assert len(retained_new)==len(retained_old)
    for i,k in zip(retained_old,retained_new):
        name=original.geom(int(i)).name
        assert m.geom(k).name==name
        for f in fields:np.testing.assert_array_equal(getattr(original,f)[i],getattr(m,f)[k])
        kept.append(name or ('unnamed_in_'+original.body(int(original.geom_bodyid[i])).name))
    for i in range(original.nbody):
        k=m.body(original.body(i).name).id if i else 0
        for f in ('body_mass','body_inertia','body_ipos','body_iquat'):
            np.testing.assert_array_equal(getattr(original,f)[i],getattr(m,f)[k])
    for name in sites:
        for f in ('site_pos','site_quat'):
            np.testing.assert_array_equal(getattr(original,f)[original.site(name).id],getattr(m,f)[m.site(name).id])
    groups=collision_driver._geom_groups(m)
    result=dict(state='PASS',kit_colliders=10,cell_sites=len(sites),removed_colliders=removed,
        retained_nonkit_colliders=kept,collision_groups=len(groups),geom_pairs=sum(map(len,groups.values())),
        original_hand_fields_exact=True,masses_inertias_sites_exact=True,
        hull='Outer skin mesh convex hull, MuJoCo maxhullvert=32; one per pad',
        xml_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),source_xml_sha256=hashlib.sha256((root/'kit_b.xml').read_bytes()).hexdigest(),
        limits=['Contact stiffness remains the unmeasured placeholder','No training or task result'])
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
