"""Freeze the approved CAD-derived palm and dimension-corrected thumb."""
import argparse,hashlib,json,os
from pathlib import Path
import xml.etree.ElementTree as ET
if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Vast only')
import numpy as np
import mujoco
from pass1c.conformal_palm import hull_exit

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    partition=json.loads((a.run/'palm_accepted_geometry.json').read_text())
    assert partition['max_mm']<=6.25
    assert all(4.25<=x<=6.25 for x in partition['cells_mm'])
    assert partition['coverage']/partition['total']>=.95
    tree=ET.parse(a.source);root=tree.getroot();body=next(b for b in root.iter('body') if b.get('name')=='kit_palm')
    palm=next(g for g in body.findall('geom') if g.get('name')=='kit_pad_contact_palm')
    attrs=dict(palm.attrib);body.remove(palm)
    for k,vertices in enumerate(partition['parts']):
        mesh='kit_palm_conformal_'+str(k)
        ET.SubElement(root.find('asset'),'mesh',name=mesh,vertex=' '.join(map(str,(np.array(vertices)*.001).ravel())),maxhullvert='32')
        g=dict(attrs)
        for key in ('size','pos','quat','fromto'):g.pop(key,None)
        g.update(type='mesh',mesh=mesh,name='kit_pad_contact_palm'+('_part_'+str(k) if k else ''))
        ET.SubElement(body,'geom',g)
    thumb_fit=next(r for r in json.loads((a.run/'geometry_size_offset_fit.json').read_text()) if r['pad']=='thumb_tip')
    assert thumb_fit['success']
    thumb=next(g for g in root.iter('geom') if g.get('name')=='kit_pad_contact_thumb_tip')
    thumb.attrib.update(thumb_fit['geometry'])
    xml=a.output/'kit_b_training.xml';tree.write(xml,encoding='unicode')
    old=mujoco.MjModel.from_xml_path(str(a.source));new=mujoco.MjModel.from_xml_path(str(xml))
    for f in ('body_mass','body_inertia','body_ipos','body_iquat','site_pos','site_quat'):
        np.testing.assert_array_equal(getattr(old,f),getattr(new,f))
    new_ids=[i for i in range(new.ngeom) if not new.geom(i).name.startswith('kit_pad_contact_palm_part_')]
    assert len(new_ids)==old.ngeom
    for i,j in zip(range(old.ngeom),new_ids):
        assert old.geom(i).name==new.geom(j).name
        fields=['geom_contype','geom_conaffinity','geom_friction','geom_solref','geom_solimp']
        if old.geom(i).name not in ('kit_pad_contact_palm','kit_pad_contact_thumb_tip'):
            fields+=['geom_type','geom_size','geom_pos','geom_quat']
        for f in fields:np.testing.assert_array_equal(getattr(old,f)[i],getattr(new,f)[j])
    m=np.load(a.run/'geometry/measurements.npz');base=m['palm_base_mm'];normal=m['palm_normal']
    values=[];palm_ids=[]
    for gid in range(new.ngeom):
        if not new.geom(gid).name.startswith('kit_pad_contact_palm'):continue
        palm_ids.append(gid);mid=new.geom_dataid[gid];start=new.mesh_vertadr[mid];count=new.mesh_vertnum[mid]
        mat=np.zeros(9);mujoco.mju_quat2Mat(mat,new.geom_quat[gid])
        v=(new.mesh_vert[start:start+count]@mat.reshape(3,3).T+new.geom_pos[gid])*1000
        values.append(hull_exit(base,normal,v))
    vals=np.array(values);maximum=np.max(np.where(np.isfinite(vals),vals,-np.inf),axis=0)
    assert maximum.max()<=6.2501
    # Exercise actual sensor routing on every piece using the unchanged sites.
    from pass1c.kit_model import TruthTouch
    from pass1c.sensors import Sensors,assignments
    cells=json.loads(Path('/root/dextum/kit_b/geometry.json').read_text())['cells']
    sensors=Sensors(new,cells,TruthTouch(new));data=mujoco.MjData(new);mujoco.mj_forward(new,data)
    sites=data.site_xpos[sensors.sites]
    checks=0
    for gid in palm_ids:
        ids,eligible=assignments(sites[20:32],np.full(12,gid),sites,sensors.lookup)
        np.testing.assert_array_equal(np.asarray(ids),np.arange(20,32));assert np.asarray(eligible).all();checks+=12
    result=dict(state='PASS',authority='review_section14_followup_approved_conformal_palm',geometry_accepted=True,
        source_xml_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),model_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
        palm_parts=len(palm_ids),palm_max_protrusion_mm=float(maximum.max()),palm_covered_samples=int((maximum>0).sum()),
        palm_samples=len(maximum),palm_cell_protrusion_mm=partition['cells_mm'],thumb=thumb_fit,
        cell_sites=40,pads=10,kit_collision_geoms=9+len(palm_ids),sensor_route_checks=checks,
        unchanged='Original hand/cube, other pad geometry, sites, mass/inertia, contact options, solver and sensor noise model',
        limitations=['Conformal palm represented by adjoining convex pieces, all sharing one palm sensor map',
                    'Thumb sphere covers 1514/2804 sampled skin rays; four sites within the dimensional tolerance',
                    '2 N/mm stiffness is unmeasured'])
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
