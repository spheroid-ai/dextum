"""Synthetic calibration measurements of the registered sensor transfer model."""
import os,sys,json
from pathlib import Path
if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Vast only')
os.environ['JAX_PLATFORMS']='cpu'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import jax
import jax.numpy as j
import mujoco
from pass1b.kit_model import TruthTouch,KIT
from pass1b.sensors import Sensors
p=Path(sys.argv[1]);p.mkdir(exist_ok=False)
m=mujoco.MjModel.from_xml_path(str(KIT/'kit_b.xml'));sensors=Sensors(m,json.loads((KIT/'geometry.json').read_text())['cells'],TruthTouch(m))
raw=j.zeros((40,3)).at[:,0].set(5.).at[:20,1].set(1.)
@jax.jit
def run(key):
 s=sensors.initial(key)
 def step(s,_):
  s=sensors.advance(s,raw,j.zeros(40),j.array(False));return s,s['packet']
 return jax.lax.scan(step,s,None,length=10020)
s,x=run(jax.random.PRNGKey(91));x=np.asarray(x)[20:];opt=x[:,:60].reshape(-1,20,3);bar=x[:,100:120]
onset=x[:,135:155].reshape(-1,5,4)[:,:,3];windows=onset.reshape(-1,10,5).max(1)
angles=np.degrees(np.abs(np.arctan2(opt[:,:,2],opt[:,:,1])))
result=dict(state='MEASURED',optical_normal_noise_mean_sd=float(opt[:,:,0].std(0).mean()),barometric_noise_mean_sd=float(bar.std(0).mean()),shear_direction_abs_degrees=dict(median=float(np.median(angles)),p95=float(np.quantile(angles,.95)),maximum=float(angles.max())),pvdf_false_onset_per_50ms=windows.mean(0).tolist(),limits=['Synthetic static 5-N force/1-N shear check; hardware remains unmeasured','Direction includes both additive force noise and the bounded orientation perturbation; total error need not stay within perturbation bound','No task or tactile-benefit conclusion'])
(p/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
