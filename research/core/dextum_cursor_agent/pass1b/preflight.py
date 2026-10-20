"""Cloud-only finite rollout and sensor-interface checks; no new D5 criteria."""
import os,sys,json,time
from pathlib import Path
if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Vast only')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import vendor.aero_official.vision_env
import jax
import jax.numpy as j
import numpy as np
from pass1b.environment import BlindRotation
from experiments.evaluate_e3 import swapped

def main():
    p=Path(sys.argv[1]);p.mkdir(exist_ok=False);start=time.monotonic()
    jax.config.update('jax_default_matmul_precision','highest')
    env=BlindRotation();print('model',env.mj_model.ngeom,env.mj_model.nbody,env.n_substeps,flush=True)
    sensor=env.sensors;s=sensor.initial(jax.random.PRNGKey(8))
    raw=j.zeros((40,3)).at[0,0].set(1.);speed=j.zeros(40)
    test=jax.jit(lambda s:sensor.advance(s,raw,speed,j.array(False)))
    packets=[]
    for _ in range(3):s=test(s);packets.append(np.asarray(s['packet']))
    assert not packets[0].any() and not packets[1].any() and packets[2][0]>.5
    # Barometric shear channels do not reach the policy or G.
    a=sensor.reading(j.zeros(200).at[100].set(2.),None)
    assert np.asarray(a.f_t).sum()==0 and np.asarray(a.f_n).sum()==2
    print('sensor_boundary_pass',flush=True)
    @jax.jit
    def reset(model,key):
        with swapped(env,model):return env.reset(key)
    @jax.jit
    def step(model,state,action):
        with swapped(env,model):return env.step(state,action)
    print('reset_compile',flush=True)
    state=reset(env.mjx_model,jax.random.PRNGKey(11));jax.block_until_ready(state)
    print('step_compile',flush=True)
    rows=[]
    for i in range(30):
        state=step(env.mjx_model,state,j.zeros(env.action_size));jax.block_until_ready(state)
        assert np.isfinite(np.asarray(state.data.qpos)).all()
        assert np.isfinite(np.asarray(state.obs['state'])).all()
        row=dict(step=i+1,z=float(state.data.qpos[env._cube_qids][2]),load=float(state.info['sense'].force.sum()),reward=float(state.reward),done=float(state.done))
        rows.append(row)
        if (i+1)%5==0: print(json.dumps(row),flush=True)
    record=dict(state='PASS',device=str(jax.devices()),seconds=time.monotonic()-start,actor_width=state.obs['state'].shape[0],sensor_cells=40,substeps=env.n_substeps,rows=rows,
        limits=['Finite interface smoke only; not physics requalification or capability evaluation','PVDF feature envelope proxy, not resolved vibration'])
    (p/'result.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
if __name__=='__main__':main()
