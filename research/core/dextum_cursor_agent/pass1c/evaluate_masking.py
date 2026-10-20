"""Frozen masked policies on the registered E3 physical-variation bank."""
import argparse,hashlib,json,os,time
from pathlib import Path


def main():
    import jax
    import jax.numpy as j
    import numpy as np
    import vendor.aero_official.vision_env
    from brax.io import model
    from brax.training.agents.ppo import networks
    from pass1c.train import make_network
    from pass1c.masking import BlindRotation
    from experiments.evaluate_e3 import CONDITIONS,HELD_OUT,make_rollout,summarise
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    config=json.loads((a.case/'config.json').read_text())
    assert json.loads((a.case/'report.json').read_text())['state']=='COMPLETE'
    reg=json.loads(Path(config['physics_qualification']['path']).with_name('MATCHED_REGISTRATION.json').read_text())
    assert reg['authority']=='review_section15_masking'
    assert hashlib.sha256(Path(os.environ['DEXTUM_KIT_XML']).read_bytes()).hexdigest()==reg['model_sha256']
    assert not os.environ.get('XLA_FLAGS','')
    os.environ['DEXTUM_SENSOR_LEVEL']='Realistic';jax.config.update('jax_default_matmul_precision','highest')
    env=BlindRotation('c_touch',config['intervention'])
    checkpoint=a.case/'policy_20316160.pkl';params=model.load_params(checkpoint)
    make_policy=networks.make_inference_fn(make_network('c_touch'))
    randomized,axes=env.randomize(env.mjx_model,jax.random.split(jax.random.PRNGKey(7),200))
    keys=jax.vmap(jax.random.split)(jax.random.split(jax.random.PRNGKey(2026),200))[...,1]
    @jax.jit
    def evaluate(parameters,models,offset):
        inference=make_policy(parameters,deterministic=True)
        policy=lambda obs:inference(obs,jax.random.PRNGKey(0))[0]
        return make_rollout(env,policy,axes)(models,keys,offset)
    a.output.mkdir(parents=True,exist_ok=False);rows={};arrays={}
    for name,condition in CONDITIONS.items():
        started=time.monotonic();m=randomized;mj=env.mj_model
        if 'friction' in condition:
            ids=np.flatnonzero(mj.geom_bodyid!=0)
            m=m.tree_replace({'geom_friction':m.geom_friction.at[:,ids,0].set(condition['friction'])})
        if 'gain' in condition:
            f=condition['gain'];m=m.tree_replace({'actuator_gainprm':m.actuator_gainprm.at[:,:,0].multiply(f),
                                                  'actuator_biasprm':m.actuator_biasprm.at[:,:,1].multiply(f)})
        if 'mass' in condition:
            cube=env.cube_body;mass=condition['mass'];inertia=j.asarray(mj.body_inertia[cube])*mass/mj.body_mass[cube]
            m=m.tree_replace({'body_mass':m.body_mass.at[:,cube].set(mass),'body_inertia':m.body_inertia.at[:,cube].set(inertia)})
        trace,sign=evaluate(params,m,j.asarray(condition.get('offset',(0.,0.)),j.float32))
        trace=jax.tree.map(np.asarray,trace);assert all(np.isfinite(v).all() for v in trace.values())
        values=summarise(trace,np.asarray(sign))
        rows[name]=dict(n=200,success=int(values['success'].sum()),drops=int(values['dropped'].sum()),wall_s=time.monotonic()-started)
        arrays.update({name+'/'+k:v for k,v in values.items()})
        np.savez_compressed(a.output/'per_start.npz',**arrays,reset_keys=np.asarray(keys),sign=np.asarray(sign))
        result=dict(state='IN_PROGRESS',scope='E3 cube physical-variation subset; one seed; no F1/F2 or camera claim',
                    intervention=config['intervention'],checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    model_sha256=reg['model_sha256'],conditions=rows,
                    limit='Starts repeat across conditions; pooled trials are not independent')
        (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(condition=name,**rows[name])),flush=True)
    result.update(state='COMPLETE',held_out_macro_success=float(np.mean([rows[k]['success']/200 for k in HELD_OUT])))
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Vast only')
    main()
