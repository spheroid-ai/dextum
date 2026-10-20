"""Measured collision-hull simplification for the Kit B training model."""
import os,sys,json,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Vast only')
import numpy as np
import trimesh
from scipy.spatial import ConvexHull,cKDTree
root=Path('/root/dextum/kit_b');out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
tree=ET.parse(root/'kit_b.xml');doc=tree.getroot();meshes={n.attrib['name']:n for n in doc.findall('./asset/mesh')}
# Only named kit collision meshes are replaced; original hand and visuals stay.
collision_geoms=[g for g in doc.iter('geom') if g.attrib.get('name','').startswith('kit_') and g.attrib.get('contype','0')!='0' and 'mesh' in g.attrib]
names=sorted({g.attrib['mesh'] for g in collision_geoms})
rows=[]
for name in names:
 node=meshes[name];source=root/'assets'/node.attrib['file'];m=trimesh.load(source,force='mesh',process=True).convex_hull
 m.apply_scale(np.fromstring(node.attrib.get('scale','1 1 1'),sep=' '))
 v=np.asarray(m.vertices);chosen=[]
 for axis in np.eye(3):
  for sign in (-1,1):chosen.append(int(np.argmax(v@(axis*sign))))
 chosen=list(dict.fromkeys(chosen))
 # Farthest point sampling on the original convex vertices; hull stays inside.
 while len(chosen)<min(16,len(v)):
  d=cKDTree(v[chosen]).query(v)[0];chosen.append(int(np.argmax(d)))
 h=trimesh.Trimesh(vertices=v[chosen],faces=ConvexHull(v[chosen]).simplices,process=True).convex_hull
 _,dist,_=trimesh.proximity.closest_point(h,v)
 error=float(dist.max()*1000)
 # v0 collision exporter uses a 0.5-mm geometric tolerance, not a physics gate.
 # Add the worst represented original support until within it or 32 vertices.
 while (error>.5 or abs(h.volume-m.volume)/m.volume>.05) and len(chosen)<min(128,len(v)):
  chosen.append(int(np.argmax(dist)));h=trimesh.Trimesh(vertices=v[chosen],faces=ConvexHull(v[chosen]).simplices,process=True).convex_hull
  _,dist,_=trimesh.proximity.closest_point(h,v);error=float(dist.max()*1000)
 target=out/(name+'.stl');h.export(target)
 new=ET.SubElement(doc.find('asset'),'mesh',dict(node.attrib))
 new.set('name','training_'+name);new.set('file',str(target));new.set('scale','1 1 1');new.set('maxhullvert',str(len(chosen)))
 for g in collision_geoms:
  if g.attrib['mesh']==name:g.set('mesh','training_'+name)
 rows.append(dict(name=name,source_vertices=len(v),vertices=len(h.vertices),max_surface_deviation_mm=error,volume_relative_error=float(abs(h.volume-m.volume)/m.volume),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),new_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
# Retain all untouched mesh paths and textures as absolute references.
for node in doc.findall('./asset/mesh')+doc.findall('./asset/texture'):
 if 'file' in node.attrib and not Path(node.attrib['file']).is_absolute():node.set('file',str(root/'assets'/node.attrib['file']))
for node in doc.findall('./compiler'):node.attrib.pop('meshdir',None);node.attrib.pop('texturedir',None)
xml=out/'kit_b_training.xml';tree.write(xml,encoding='unicode')
result=dict(state='MEASURED',meshes=len(rows),rows=rows,max_deviation_mm=max(x['max_surface_deviation_mm'] for x in rows),max_volume_relative_error=max(x['volume_relative_error'] for x in rows),xml_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),scope='All Kit B collision hulls with adaptive support, at most128; original hand collisions, meshes, inertias, contact softness and visual geometry preserved; original collision meshes retained')
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
