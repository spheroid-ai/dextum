"""Rigidly share congruent Kit B hulls so MJX compiles fewer mesh-pair kernels."""
import os,sys,json,itertools,xml.etree.ElementTree as ET,hashlib
from pathlib import Path
if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Vast only')
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
root=Path('/root/dextum/kit_b');base=Path('/root/dextum/dextum_cursor_agent/runs/pass1_realistic_01/collision_adaptive');out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
tree=ET.parse(root/'kit_b.xml');doc=tree.getroot();asset=doc.find('asset');nodes={n.attrib['name']:n for n in asset.findall('mesh')}
geoms=[g for g in doc.iter('geom') if g.attrib.get('name','').startswith('kit_') and g.attrib.get('contype','0')!='0' and 'mesh' in g.attrib]
perms=[]
for perm in itertools.permutations(range(3)):
 for signs in itertools.product((-1,1),repeat=3):
  P=np.eye(3)[:,perm]*np.array(signs)
  if np.linalg.det(P)>.5:perms.append(P)
protos=[];rows=[]
for g in geoms:
 name=g.attrib['mesh'];node=nodes[name]
 full=trimesh.load(root/'assets'/node.attrib['file'],force='mesh',process=True).convex_hull;full.apply_scale(np.fromstring(node.attrib.get('scale','1 1 1'),sep=' '))
 v=np.asarray(full.vertices);center=full.center_mass;_,axes=np.linalg.eigh(np.cov((v-center).T));axes=axes[:,::-1]
 if np.linalg.det(axes)<0:axes[:,-1]*=-1
 canonical=(v-center)@axes;vol=full.volume;chosen=None
 for k,proto in enumerate(protos):
  if abs(proto['mesh'].volume-vol)/vol>.05:continue
  for P in perms:
   aligned=canonical@P
   if np.max(np.abs(np.ptp(aligned,axis=0)-proto['extent']))>.001:continue
   _,d,_=trimesh.proximity.closest_point(proto['mesh'],aligned)
   # Existing prototype must remain inside/near the current true hull too.
   if d.max()>.0005:continue
   current=trimesh.Trimesh(vertices=aligned,faces=full.faces,process=False)
   _,reverse,_=trimesh.proximity.closest_point(current,proto['mesh'].vertices)
   error=max(float(d.max()),float(reverse.max()))
   if error<=.0005:chosen=(k,P,error);break
  if chosen is not None:break
 if chosen is None:
  m=trimesh.load(base/(name+'.stl'),force='mesh',process=True);m.vertices=(np.asarray(m.vertices)-center)@axes
  path=out/('shared_'+str(len(protos))+'.stl');m.export(path)
  _,d,_=trimesh.proximity.closest_point(m,canonical)
  chosen=(len(protos),np.eye(3),float(d.max()))
  protos.append(dict(mesh=m,extent=np.ptp(m.vertices,axis=0),path=path))
  ET.SubElement(asset,'mesh',dict(name='kit_shared_'+str(chosen[0]),file=str(path)))
 k,P,error=chosen
 q=np.fromstring(g.attrib.get('quat','1 0 0 0'),sep=' ');rot=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();pos=np.fromstring(g.attrib.get('pos','0 0 0'),sep=' ')
 qnew=Rotation.from_matrix(rot@axes@P).as_quat()[[3,0,1,2]]
 assert not any(x in g.attrib for x in ('euler','axisangle','xyaxes','zaxis'))
 g.set('pos',' '.join(map(str,pos+rot@center)));g.set('quat',' '.join(map(str,qnew)));g.set('mesh','kit_shared_'+str(k))
 rows.append(dict(geom=g.attrib['name'],source_mesh=name,prototype=k,surface_deviation_mm=error*1000,volume_relative_error=float(abs(protos[k]['mesh'].volume-vol)/vol)))
 print(json.dumps(rows[-1]),flush=True)
for n in doc.findall('./asset/mesh')+doc.findall('./asset/texture'):
 if 'file' in n.attrib and not Path(n.attrib['file']).is_absolute():n.set('file',str(root/'assets'/n.attrib['file']))
for n in doc.findall('./compiler'):n.attrib.pop('meshdir',None);n.attrib.pop('texturedir',None)
xml=out/'kit_b_training.xml';tree.write(xml,encoding='unicode')
import mujoco
from mujoco.mjx._src.collision_driver import _geom_groups
m=mujoco.MjModel.from_xml_path(str(xml));original=mujoco.MjModel.from_xml_path(str(root/'kit_b.xml'))
result=dict(state='MEASURED',geoms=len(geoms),prototypes=len(protos),original_collision_groups=len(_geom_groups(original)),shared_collision_groups=len(_geom_groups(m)),max_deviation_mm=max(x['surface_deviation_mm'] for x in rows),max_volume_relative_error=max(x['volume_relative_error'] for x in rows),rows=rows,xml_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),scope='Rigidly shared physical collision prototypes; same named40 sensing geoms, original visual references/inertias/softness; no collisions removed')
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
