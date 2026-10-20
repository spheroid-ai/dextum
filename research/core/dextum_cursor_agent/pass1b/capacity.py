import os,sys,json
from pathlib import Path
if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Vast only')
os.environ['JAX_PLATFORMS']='cpu'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import vendor.aero_official.vision_env
import jax
from mujoco import mjx
from pass1b.environment import BlindRotation
p=Path(sys.argv[1]);p.mkdir(exist_ok=False)
e=BlindRotation();d=mjx.make_data(e.mj_model)
def size(tree):return sum(x.size*x.dtype.itemsize for x in jax.tree.leaves(tree) if hasattr(x,'dtype'))
x=dict(model_bytes=size(e.mjx_model),data_bytes=size(d),ncon=d._impl.contact.pos.shape[0],substeps=e.n_substeps,contacts_shape=list(d._impl.contact.pos.shape),raw_data_gib_8192=size(d)*8192/2**30)
(p/'result.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x),flush=True)
