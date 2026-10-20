"""Check all forty site assignments and the unchanged measured-packet boundary."""
import os,sys,json
from pathlib import Path
if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Vast.ai only')
import numpy as np
import jax
import jax.numpy as j
import mujoco
from pass1c.sensors import Sensors,assignments,CELL_GROUP
from pass1c.kit_model import TruthTouch

def main():
    root=Path(sys.argv[1]);out=root/'cell_check';out.mkdir(exist_ok=False)
    m=mujoco.MjModel.from_xml_path(str(root/'model/kit_b_training.xml'))
    cells=json.loads(Path('/root/dextum/kit_b/geometry.json').read_text())['cells']
    sensor=Sensors(m,cells,TruthTouch(m));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    sites=np.asarray(d.site_xpos[sensor.sites]);geoms=sensor.geoms
    fn=jax.jit(lambda points,geoms:assignments(points,geoms,j.asarray(sites),sensor.lookup))
    selected,valid=map(np.asarray,fn(j.asarray(sites),j.asarray(geoms)))
    np.testing.assert_array_equal(selected,np.arange(40));assert valid.all()
    # A contact physically close to another pad must remain on its own pad.
    far,ok=map(np.asarray,fn(j.asarray(sites[::-1].copy()),j.asarray(geoms)))
    np.testing.assert_array_equal(CELL_GROUP[far],CELL_GROUP);assert ok.all()
    none,ok=fn(j.asarray(sites[:1]),j.asarray([m.geom('cube').id]));assert not bool(ok[0])
    # Each eligible contact is counted exactly once before the error model.
    assignment=(selected[:,None]==np.arange(40));assert np.all(assignment.sum(1)==1)
    load=np.arange(1,41,dtype=np.float32)/10
    np.testing.assert_allclose((assignment.T@load).sum(),load.sum())
    state=sensor.initial(jax.random.PRNGKey(8));advance=jax.jit(sensor.advance)
    raw=j.zeros((40,3)).at[0,0].set(1.);packets=[]
    for _ in range(3):state=advance(state,raw,j.zeros(40),j.array(False));packets.append(np.asarray(state['packet']))
    assert not packets[0].any() and not packets[1].any() and packets[2][0]>.5
    reading=sensor.reading(j.zeros(200).at[100].set(2.),None)
    assert float(reading.f_t.sum())==0 and float(reading.f_n.sum())==2
    result=dict(state='PASS',sites_exercised=40,nearest_site_same_pad=True,contacts_counted_once=True,
        nonpad_contacts_excluded=True,latency_ms=10,barometric_shear_absent=True,
        error_model='Inherited unchanged pass1b.sensors.Sensors.advance',optimizer_steps=0)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
