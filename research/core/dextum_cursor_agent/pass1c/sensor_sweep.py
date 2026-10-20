"""Registered, fixed-policy sensor counterfactuals on E3; no optimizer updates."""
import argparse,functools,hashlib,json,os,time
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
    from experiments.evaluate_e3 import make_rollout,summarise
    from pass0.review_gate import exact_mcnemar
    p=argparse.ArgumentParser();p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--sweep',choices=['floor','ideal_floor','delay','noise','direction','pvdf','rate','location'],required=True)
    a=p.parse_args();reg=json.loads(a.registration.read_text())
    assert reg['authority']=='review_section15_masking' and reg['matched_decision_complete']
    checkpoint_hash=reg['ideal_checkpoint_sha256'] if a.sweep=='ideal_floor' else reg['warm_start_sha256']
    assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==checkpoint_hash
    assert hashlib.sha256(Path(os.environ['DEXTUM_KIT_XML']).read_bytes()).hexdigest()==reg['model_sha256']
    assert not os.environ.get('XLA_FLAGS','')
    os.environ['DEXTUM_SENSOR_LEVEL']='Ideal' if a.sweep=='ideal_floor' else 'Realistic'
    a.output.mkdir(parents=True,exist_ok=False);jax.config.update('jax_default_matmul_precision','highest')
    params=model.load_params(a.checkpoint)
    factory=functools.partial(networks.make_inference_fn(make_network('c_touch')),deterministic=True)
    dr=jax.random.split(jax.random.PRNGKey(7),200)
    keys=jax.vmap(jax.random.split)(jax.random.split(jax.random.PRNGKey(2026),200))[...,1]
    results=[];baseline=None
    for label,intervention in reg['sweeps'][a.sweep]:
        before=time.monotonic();env=BlindRotation('c_touch',intervention)
        randomized,axes=env.randomize(env.mjx_model,dr)
        @jax.jit
        def direct(parameters,models,reset_keys):
            infer=factory(parameters)
            policy=lambda obs:infer(obs,jax.random.PRNGKey(0))[0]
            return make_rollout(env,policy,axes)(models,reset_keys,j.zeros(2,j.float32))
        trace,sign=direct(params,randomized,keys)
        trace=jax.tree.map(np.asarray,trace);sign=np.asarray(sign)
        finite=all(np.isfinite(x).all() for x in trace.values())
        if not finite:raise RuntimeError('Nonfinite sensor sweep')
        values=summarise(trace,sign);x=values['success']
        if baseline is None:baseline=x.copy()
        b=int(np.sum(x&~baseline));c=int(np.sum(~x&baseline))
        row=dict(label=label,intervention=intervention,n=200,success_count=int(x.sum()),
            success_percent=float(x.mean()*100),drops=int(values['dropped'].sum()),
            change_from_same_path_baseline_points=(int(x.sum())-int(baseline.sum()))/2,
            variant_only=b,baseline_only=c,exact_mcnemar_p=exact_mcnemar(b,c),wall_s=time.monotonic()-before)
        np.savez_compressed(a.output/(label+'.npz'),**values,reset_keys=np.asarray(keys),dr_keys=np.asarray(dr))
        results.append(row)
        (a.output/'result.json').write_text(json.dumps(dict(state='MEASURED',sweep=a.sweep,
            checkpoint_sha256=checkpoint_hash,model_sha256=reg['model_sha256'],
            sensor_level=os.environ['DEXTUM_SENSOR_LEVEL'],
            optimizer_updates=0,baseline_is_fresh_same_path=True,results=results),indent=2)+'\n')
        print(json.dumps(row),flush=True)
        del direct,env,trace
        jax.clear_caches()
    final=json.loads((a.output/'result.json').read_text());final['state']='COMPLETE'
    (a.output/'result.json').write_text(json.dumps(final,indent=2)+'\n')


if __name__=='__main__':
    if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Vast only')
    main()
