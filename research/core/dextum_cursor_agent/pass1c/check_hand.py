"""Compare the ten-pad model's hand collision set with the actual Ideal model."""
import os,sys,json
from pathlib import Path
if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Vast.ai only')
import numpy as np
import mujoco
import vendor.aero_official.vision_env
from envs.blind_rotation import BlindRotation

def descriptors(m):
    result={};counts={}
    for i in np.flatnonzero((m.geom_contype!=0)|(m.geom_conaffinity!=0)):
        name=m.geom(int(i)).name;body=m.body(int(m.geom_bodyid[i])).name
        if name.startswith('kit_') or body.startswith('kit_'):continue
        n=counts.get((body,name),0);counts[body,name]=n+1
        result[body,name,n]=int(i)
    return result

root=Path(sys.argv[1]);a=BlindRotation('c_touch').mj_model;b=mujoco.MjModel.from_xml_path(str(root/'model/kit_b_training.xml'))
x,y=descriptors(a),descriptors(b);assert x.keys()==y.keys(),(x.keys()-y.keys(),y.keys()-x.keys())
fields=('geom_type','geom_contype','geom_conaffinity','geom_condim','geom_size','geom_pos','geom_quat','geom_friction','geom_solref','geom_solimp')
differences=[]
for key,i in x.items():
    for field in fields:
        if not np.array_equal(getattr(a,field)[i],getattr(b,field)[y[key]]):differences.append(dict(geom=key,field=field,ideal=getattr(a,field)[i].tolist(),kit=getattr(b,field)[y[key]].tolist()))
result=dict(state='PASS',colliders=len(x),fields=list(fields),differences=differences,
            bitwise_equal_to_ideal=not differences,
            interpretation='Collision set matches Ideal. Numeric differences are diagnostic; unchanged source-Kit fields are checked exactly by build_model.',
            reference='envs.blind_rotation.BlindRotation, actual Ideal baseline',optimizer_steps=0)
(root/'ORIGINAL_HAND_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
