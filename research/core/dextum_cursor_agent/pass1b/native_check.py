"""Sensor-bearing XML finite/coverage probe, not a new qualification gate."""
import os,sys,json,time
from pathlib import Path
if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Vast only')
import numpy as np
import mujoco
root=Path('/root/dextum/kit_b');out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
m=mujoco.MjModel.from_xml_path(str(root/'kit_b.xml'));m.opt.timestep=.005
d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,m.key('home').id)
cells=json.loads((root/'geometry.json').read_text())['cells'];ids={m.geom('kit_contact_'+c['name']).id:i for i,c in enumerate(cells)}
force=np.zeros(6);maxload=np.zeros(40);counts=np.zeros(40,int);rows=[]
for step in range(1000):
 mujoco.mj_step(m,d)
 for k in range(d.ncon):
  c=d.contact[k];mujoco.mj_contactForce(m,d,k,force)
  for g in c.geom:
   if g in ids:
    i=ids[g];counts[i]+=force[0]>.05;maxload[i]=max(maxload[i],float(force[0]))
 if step%100==0:rows.append(dict(step=step,finite=bool(np.isfinite(d.qpos).all()),cube_z=float(d.xpos[m.body('cube').id,2]),contacts=d.ncon))
result=dict(state='FINITE' if all(r['finite'] for r in rows) else 'NONFINITE',timestep=.005,nominal_home_only=True,rows=rows,cell_names=[c['name'] for c in cells],contact_samples=counts.tolist(),max_load=maxload.tolist(),limits=['No task success or native/MJX agreement claimed','Home-key start differs from official random bank'])
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
