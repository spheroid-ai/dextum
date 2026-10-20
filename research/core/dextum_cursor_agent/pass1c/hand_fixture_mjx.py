"""Native/MJX indentation qualification of the registered fingertip fixture."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    import mujoco
    from mujoco import mjx
    import jax
    import jax.numpy as j
    import numpy as np
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    full=mujoco.MjModel.from_xml_path(str(a.model))
    gid=full.geom('kit_pad_contact_index_tip').id
    def values(x):return ' '.join(str(float(v)) for v in np.atleast_1d(x))
    root=ET.Element('mujoco')
    ET.SubElement(root,'option',timestep='.005',gravity='0 0 0',solver='Newton',iterations='5',ls_iterations='8',cone='elliptic',integrator='implicitfast')
    world=ET.SubElement(root,'worldbody')
    common=dict(solref=values(full.geom_solref[gid]),solimp=values(full.geom_solimp[gid]),friction=values(full.geom_friction[gid]),condim='3')
    radius=float(full.geom_size[gid,0])
    ET.SubElement(world,'geom',type='sphere',size=str(radius),**common)
    body=ET.SubElement(world,'body',pos=values([0,0,radius+.004]))
    ET.SubElement(body,'joint',type='slide',axis='0 0 1')
    ET.SubElement(body,'geom',type='box',size='.004 .004 .004',mass='.05',**common)
    text=ET.tostring(root,encoding='unicode');(a.output/'fixture.xml').write_text(text)
    native=mujoco.MjModel.from_xml_string(text);model=mjx.put_model(native)
    def rollout(force):
        initial=mjx.make_data(native).replace(qfrc_applied=j.array([-force]))
        def step(d,_):
            d=mjx.step(model,d)
            return d,j.stack((d.qpos[0],d.qfrc_constraint[0]))
        return jax.lax.scan(step,initial,None,length=2000)[1]
    device=jax.jit(jax.vmap(rollout))(j.array([1.,2.,5.]))
    device=np.asarray(device)
    rows=[]
    for i,force in enumerate((1.,2.,5.)):
        d=mujoco.MjData(native);d.qfrc_applied[0]=-force
        trace=[]
        for _ in range(2000):
            mujoco.mj_step(native,d);trace.append((d.qpos[0],d.qfrc_constraint[0]))
        trace=np.asarray(trace);x=device[i,-400:];y=trace[-400:]
        row=dict(force_N=force,native_indentation_mm=float(-y[:,0].mean()*1000),mjx_indentation_mm=float(-x[:,0].mean()*1000),
            mjx_drift_mm=float(np.ptp(x[:,0])*1000),mjx_force_error_N=float(np.max(abs(x[:,1]-force))),finite=bool(np.isfinite(device[i]).all()))
        row['passed']=bool(row['finite'] and row['mjx_drift_mm']<.01 and row['mjx_force_error_N']<.05 and abs(row['native_indentation_mm']-row['mjx_indentation_mm'])<.01)
        rows.append(row);np.savez_compressed(a.output/f'{force}N.npz',native=trace,mjx=device[i])
    result=dict(state='PASS' if all(r['passed'] for r in rows) else 'FAIL',model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),
        rows=rows,devices=str(jax.devices()),target='10N/mm secant at5N;1/2N response also reported;fixed50g indenter',training_attempts=0)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    assert result['state']=='PASS'


if __name__=='__main__':
    if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Vast only')
    main()
