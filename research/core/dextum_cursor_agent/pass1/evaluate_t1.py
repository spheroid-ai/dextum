"""Frozen-checkpoint T1 cube physical-variation subset, on Vast only.

The original E3 physical conditions are retained. This is not T1/T6 on the
new F1/F2 shapes, nor a camera-dropout, seed-replication or hardware result.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Vast.ai only')
import jax
import jax.numpy as jnp
import numpy as np
import vendor.aero_official.vision_env
from envs.blind_rotation import BlindRotation
from experiments.evaluate_e3 import CONDITIONS,HELD_OUT,load_policy,physics,make_rollout,summarise
from pass0.review_gate import exact_mcnemar
from pass1.qualification import require_training_pass


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--touch-run',type=Path,required=True);a=p.parse_args()
    require_training_pass(json.loads((a.root/'TRAINING_QUALIFICATION_BARRIER.json').read_text()))
    out=a.root/'t1_cube_physical_subset';out.mkdir(exist_ok=False)
    jax.config.update('jax_default_matmul_precision','highest')
    starts=200;rng=jax.random.split(jax.random.PRNGKey(2026),starts)
    keys=jax.vmap(jax.random.split)(rng)[:,1]
    results={};arrays={}
    for arm in ('c_touch','c_notouch','a_task'):
        run=a.touch_run if arm=='c_touch' else a.root/'controls_01/e3'
        policy,steps=load_policy(run,arm);assert steps==108134400
        records=json.loads((run/arm/'checkpoints.json').read_text());chosen=next(x for x in records if x['steps']==steps)
        checkpoint=run/arm/chosen['path'];env=BlindRotation(arm);rollout=None
        results[arm]=dict(steps=steps,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),conditions={})
        for name,condition in CONDITIONS.items():
            begin=time.monotonic();model,axes=physics(env,starts,7,condition)
            rollout=rollout or make_rollout(env,policy,axes)
            trace,sign=rollout(model,keys,jnp.asarray(condition.get('offset',(0.,0.)),jnp.float32))
            trace=jax.tree.map(np.asarray,trace);assert all(np.isfinite(x).all() for x in trace.values())
            summary=summarise(trace,np.asarray(sign));arrays[arm+'/'+name+'/success']=summary['success']
            arrays[arm+'/'+name+'/dropped']=summary['dropped']
            row=dict(n=starts,success=int(summary['success'].sum()),drops=int(summary['dropped'].sum()),elapsed_s=time.monotonic()-begin)
            results[arm]['conditions'][name]=row;print(json.dumps(dict(arm=arm,condition=name,**row)),flush=True)
            (out/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
            np.savez_compressed(out/'per_start.npz',**arrays)
        del env,policy,rollout,model,trace;gc.collect();jax.clear_caches()
    comparisons=[]
    for control in ('c_notouch','a_task'):
        for name in CONDITIONS:
            x,y=(arrays[arm+'/'+name+'/success'] for arm in ('c_touch',control))
            b,c=int(np.sum(x&~y)),int(np.sum(~x&y))
            comparisons.append(dict(control=control,condition=name,difference_points=100*float(x.mean()-y.mean()),
                                    touch_only=b,control_only=c,exact_mcnemar_p=exact_mcnemar(b,c)))
    means={arm:float(np.mean([results[arm]['conditions'][name]['success']/starts for name in HELD_OUT])) for arm in results}
    result=dict(state='MEASURED',scope='T1 original E3 cube physical-variation subset',n_per_condition=starts,
        reset_seed=2026,reset_split='second wrapper split',dr_seed=7,training_seeds=1,results=results,
        paired_comparisons=comparisons,held_out_macro_success=means,
        touch_minus_notouch_points=100*(means['c_touch']-means['c_notouch']),
        observed_ten_point_target=means['c_touch']>=means['c_notouch']+.1,
        limits=['Same starts reused across conditions: pooled trials are not independent observations',
                'Raw per-condition paired-start p values are descriptive, not seed-level replication',
                'No new F1/F2 geometry, camera-dropout, Realistic Kit B or hardware claim'])
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
