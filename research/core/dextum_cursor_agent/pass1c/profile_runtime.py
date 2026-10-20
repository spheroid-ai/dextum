"""GPU component timing, no optimizer updates; dynamic model arguments."""
import os,sys,time,json,statistics,gc
from pathlib import Path
if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Vast only')
import numpy as np
import jax
import jax.numpy as j
import mujoco
from mujoco import mjx
from mujoco.mjx._src import collision_driver,constraint,solver
import vendor.aero_official.vision_env
from pass1c.environment import BlindRotation
from pass1c.sensors_fast import Sensors as FastSensors
from dextum.contact_space import HOME

def pad_hold(out):
    """Section 12: actual pad contact parameters, isolated 5 N normal loads."""
    import xml.etree.ElementTree as ET
    source=Path(os.environ['DEXTUM_KIT_XML'])
    full=mujoco.MjModel.from_xml_path(str(source))
    rows=[]
    for iterations in range(5,41):
        current=[]
        for g in range(full.ngeom):
            name=full.geom(g).name
            if not name.startswith('kit_pad_contact_'):continue
            kind=mujoco.mjtGeom(int(full.geom_type[g])).name.removeprefix('mjGEOM_').lower()
            size=full.geom_size[g]
            # Canonical orientation, retaining actual primitive dimensions and contact parameters.
            axis=np.array([1.,0,0]) if kind=='capsule' else np.array([0.,0,1.])
            top=size[0] if kind in ('sphere','capsule') else size[2]
            def values(x):return ' '.join(str(float(v)) for v in np.atleast_1d(x))
            xml=ET.Element('mujoco');ET.SubElement(xml,'option',timestep='.005',gravity='0 0 0',solver='Newton',iterations=str(iterations),ls_iterations='8',cone='pyramidal')
            world=ET.SubElement(xml,'worldbody')
            attrs=dict(type=kind,size=values(size[:1] if kind=='sphere' else size[:2] if kind=='capsule' else size),
                       friction=values(full.geom_friction[g]),solref=values(full.geom_solref[g]),solimp=values(full.geom_solimp[g]),condim=str(int(full.geom_condim[g])))
            ET.SubElement(world,'geom',**attrs)
            body=ET.SubElement(world,'body',pos=values(axis*(top+.004)))
            ET.SubElement(body,'joint',type='slide',axis=values(axis))
            ET.SubElement(body,'geom',type='box',size='.004 .004 .004',mass='.05',friction=attrs['friction'],solref=attrs['solref'],solimp=attrs['solimp'],condim=attrs['condim'])
            m=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'))
            d=mujoco.MjData(m);d.qfrc_applied[0]=-5
            md=mjx.put_model(m);dd=mjx.put_data(m,d)
            @jax.jit
            def run(model,data):
                def step(data,_):
                    data=mjx.step(model,data)
                    return data,(data.qpos[0],j.min(data._impl.contact.dist),data.qfrc_constraint[0])
                return jax.lax.scan(step,data,None,length=2000)[1]
            gpu=np.stack([np.asarray(x) for x in run(md,dd)],axis=1)
            cpu=[]
            for _ in range(2000):
                mujoco.mj_step(m,d);cpu.append([d.qpos[0],min((c.dist for c in d.contact),default=0),d.qfrc_constraint[0]])
            item=dict(pad=name,kind=kind,iterations=iterations,paths={})
            for engine,trace in [('native',np.asarray(cpu)),('mjx',gpu)]:
                settled=trace[400:];drift=float(np.ptp(settled[:,0]));growth=float(max(0,settled[0,1]-settled[:,1].min()))
                force_error=float(np.max(np.abs(settled[:,2]-5)))
                item['paths'][engine]=dict(displacement_drift_m=drift,penetration_growth_m=growth,maximum_penetration_m=float(-trace[:,1].min()),force_error_N=force_error,
                    passed=bool(np.isfinite(trace).all() and drift<=1e-5 and growth<=1e-5 and force_error<=.05))
                np.savez_compressed(out/(name+'_'+str(iterations)+'_'+engine+'.npz'),trace=trace)
            item['passed']=all(x['passed'] for x in item['paths'].values());current.append(item)
            print(json.dumps(item),flush=True)
            del run,md,dd;gc.collect();jax.clear_caches()
        rows.extend(current)
        passed=all(x['passed'] for x in current)
        result=dict(state='PASS' if passed else 'HOLD_FAIL',iterations=iterations,ls_iterations=8,rows=rows,fixture='isolated 5 N normal load, not tangential creep or whole-hand stability')
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        if passed:
            tree=ET.parse(source);option=tree.getroot().find('option')
            if option is None:option=ET.SubElement(tree.getroot(),'option')
            option.set('solver','Newton');option.set('iterations',str(iterations));option.set('ls_iterations','8')
            tree.write(out/'kit_b_training.xml',encoding='unicode')
            return
    raise RuntimeError('No passing iteration count through the original 40; do not tune other settings')

def model_cost(out, batch):
    """Review item 0: same fixed states and compiler path for both geometries."""
    from envs.blind_rotation import BlindRotation as Ideal
    from pass1c.collision_batched import install
    import hashlib
    install()
    ideal=Ideal('c_touch').mj_model
    realistic_path=Path(os.environ.get('DEXTUM_KIT_XML','/root/dextum/dextum_cursor_agent/runs/pass1_realistic_02/model/kit_b_training.xml'))
    realistic=mujoco.MjModel.from_xml_path(str(realistic_path))
    models={'ideal':ideal,'realistic':realistic}
    for m in models.values():
        m.opt.timestep=.005
        if os.environ.get('DEXTUM_EQUAL_SOLVER'):
            m.opt.solver=mujoco.mjtSolver.mjSOL_NEWTON
            m.opt.iterations=int(os.environ['DEXTUM_EQUAL_SOLVER']);m.opt.ls_iterations=8
    cube_qid=int(ideal.jnt_qposadr[ideal.body_jntadr[ideal.body('cube').id]])
    shared=[]
    for i in range(batch):
        d=mujoco.MjData(ideal);mujoco.mj_resetDataKeyframe(ideal,d,0)
        d.ctrl[:]=HOME;d.qpos[cube_qid]+=(i%8-3.5)*.001
        for _ in range(40+i%80):mujoco.mj_step(ideal,d)
        shared.append({name:np.asarray(getattr(d,name)).copy() for name in ('qpos','qvel','act','ctrl','time','qacc_warmstart')})
    np.savez(out/'shared_states.npz',**{name:np.stack([d[name] for d in shared]) for name in shared[0]})
    result=dict(item=0,state='RUNNING',batch=batch,engine='MJX-JAX with reviewed collision batching',
                timestep=.005,optimizer_steps=0,training_attempts=0,models={},
                realistic_xml=str(realistic_path),realistic_xml_sha256=hashlib.sha256(realistic_path.read_bytes()).hexdigest())
    def save():(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,m in models.items():
        native=[];states=[]
        for source in shared:
            d=mujoco.MjData(m)
            for field,value in source.items():
                if field=='time':d.time=float(value)
                else:getattr(d,field)[:]=value
            mujoco.mj_forward(m,d);native.append(int(d.ncon));states.append(mjx.put_data(m,d))
        model=mjx.put_model(m);data=jax.tree.map(lambda *x:j.stack(x),*states)
        groups=collision_driver._geom_groups(m)
        row=dict(native_contacts=native,geom_pairs=sum(map(len,groups.values())),collision_groups=len(groups),
                 pad_colliders=[dict(name=m.geom(k).name,body=m.body(int(m.geom_bodyid[k])).name,
                      type=mujoco.mjtGeom(int(m.geom_type[k])).name,contype=int(m.geom_contype[k]),
                      conaffinity=int(m.geom_conaffinity[k])) for k in range(m.ngeom)
                      if (m.geom_contype[k] or m.geom_conaffinity[k]) and
                      (m.geom(k).name.startswith('kit_pad_contact_') if name=='realistic' else m.geom_bodyid[k]!=0 and m.geom_bodyid[k]!=m.body('cube').id)],
                 solver=int(m.opt.solver),iterations=int(m.opt.iterations),ls_iterations=int(m.opt.ls_iterations))
        result['models'][name]=row;save()
        start=time.monotonic();compiled=jax.jit(jax.vmap(mjx.step,in_axes=(None,0))).lower(model,data).compile()
        row['compile_s']=time.monotonic()-start
        value=compiled(model,data);jax.block_until_ready(value)
        timings=[]
        for _ in range(7):
            start=time.monotonic()
            for __ in range(8):value=compiled(model,data)
            jax.block_until_ready(value);timings.append((time.monotonic()-start)/8)
        row.update(physics_ms_per_step=1000*statistics.median(timings),timing_samples_s=timings,
                   selected_contacts=np.sum(np.asarray(value._impl.contact.dist)<0,axis=1).tolist(),
                   finite=bool(np.isfinite(np.asarray(value.qpos)).all()))
        print(name,json.dumps(row),flush=True);save()
        del compiled,model,data,states,value;gc.collect();jax.clear_caches()
    result['cost_ratio']=result['models']['realistic']['physics_ms_per_step']/result['models']['ideal']['physics_ms_per_step']
    result['limit_met']=result['cost_ratio']<=1.5
    result['state']=('CLOSED' if os.environ.get('DEXTUM_EQUAL_SOLVER') else 'PASS' if result['cost_ratio']<=1.5 else 'MODEL_FIX_REQUIRED') if all(x['finite'] for x in result['models'].values()) else 'NONFINITE'
    save();print('ITEM0',result['state'],result['cost_ratio'],flush=True)

def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
    batch=int(sys.argv[2]) if len(sys.argv)>2 else 128
    assert not os.environ.get('XLA_FLAGS')
    jax.config.update('jax_default_matmul_precision','highest')
    assert jax.devices()[0].platform=='gpu'
    if len(sys.argv)>3 and sys.argv[3]=='pad_hold':
        pad_hold(out);return
    if len(sys.argv)>3 and sys.argv[3]=='model_cost':
        model_cost(out,batch);return
    env=BlindRotation('c_touch');model=env.mjx_model
    cells=json.loads(Path('/root/dextum/kit_b/geometry.json').read_text())['cells']
    fast=FastSensors(env.mj_model,cells,env.touch_model)
    result=dict(batch=batch,training_attempts=0,optimizer_steps=0,device=str(jax.devices()),timings={})
    def save():
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    def bench(name,fn,*args):
        before=time.monotonic();compiled=jax.jit(fn).lower(*args).compile()
        compile_s=time.monotonic()-before
        value=compiled(*args);jax.block_until_ready(value)
        times=[]
        for _ in range(7):
            start=time.monotonic()
            for __ in range(8):value=compiled(*args)
            jax.block_until_ready(value);times.append((time.monotonic()-start)/8)
        result['timings'][name]=dict(compile_s=compile_s,median_s=statistics.median(times),min_s=min(times),max_s=max(times))
        print(name,json.dumps(result['timings'][name]),flush=True);save()
        return value
    # Varied native contact states; no optimizer or changed task rollout.
    states=[];native_contacts=[]
    for i in range(batch):
        d=mujoco.MjData(env.mj_model);mujoco.mj_resetDataKeyframe(env.mj_model,d,0)
        d.ctrl[:]=HOME
        d.qpos[env._cube_qids[0]]+=(i%8-3.5)*.001
        for _ in range(40+i%80):mujoco.mj_step(env.mj_model,d)
        native_contacts.append(int(d.ncon))
        states.append(mjx.put_data(env.mj_model,d))
    data=jax.tree.map(lambda *x:j.stack(x),*states)
    del states;gc.collect()
    result['sizes']=dict(ngeom=env.mj_model.ngeom,nv=env.mj_model.nv,ncon=data._impl.contact.pos.shape[1],nefc=data._impl.efc_J.shape[1],solver=int(env.mj_model.opt.solver),iterations=env.mj_model.opt.iterations,ls_iterations=env.mj_model.opt.ls_iterations)
    result['native_contact_counts']=native_contacts
    save();print('SIZES',result['sizes'],flush=True)
    raw=bench('sensor_raw_reference',jax.vmap(env.sensors.raw,in_axes=(None,0)),model,data)
    result['sample_raw_force_max']=float(j.max(j.abs(raw[0])))
    alternate=bench('sensor_raw_scatter',jax.vmap(fast.raw,in_axes=(None,0)),model,data)
    result['raw_parity']={k:dict(max_abs=float(j.max(j.abs(a-b))),mean_abs=float(j.mean(j.abs(a-b)))) for k,a,b in zip(('force','speed'),raw,alternate)}
    sensor=jax.vmap(env.sensors.initial)(jax.random.split(jax.random.PRNGKey(34),batch))
    bench('sensor_noise_packet',jax.vmap(env.sensors.advance),sensor,*raw,j.zeros(batch,bool))
    if len(sys.argv)>3 and sys.argv[3]=='collision_batch':
        from pass1c.collision_batched import collision as grouped
        original=bench('collision_reference',jax.vmap(collision_driver.collision,in_axes=(None,0)),model,data)
        changed=bench('collision_batched',jax.vmap(grouped,in_axes=(None,0)),model,data)
        result['contact_parity']={}
        result['selected_penetrating_contacts']=int(j.sum(original._impl.contact.dist<0))
        for name in ('geom','dist','pos','frame','includemargin','friction','solref','solimp'):
            a=getattr(original._impl.contact,name);b=getattr(changed._impl.contact,name)
            result['contact_parity'][name]=dict(equal=bool(j.all(a==b)),max_abs=float(j.max(j.abs(a-b))),finite=bool(j.all(j.isfinite(b))))
        save()
        # Contact-batching check first; a fused full-step compile is a separate
        # check after the candidate has shown a useful collision speedup.
        if len(sys.argv)<=4 or sys.argv[4]!='full':
            result['state']='COMPLETE_COLLISION_CHECK';save();return
        def optimized_step(m,d):
            reference=collision_driver.collision
            collision_driver.collision=grouped
            try:return mjx.step(m,d)
            finally:collision_driver.collision=reference
        a=bench('step_reference',jax.vmap(mjx.step,in_axes=(None,0)),model,data)
        b=bench('step_batched',jax.vmap(optimized_step,in_axes=(None,0)),model,data)
        result['step_parity']={name:float(j.max(j.abs(getattr(a,name)-getattr(b,name)))) for name in ('qpos','qvel','qacc')}
    else:
        for name,fn in [('collision',collision_driver.collision),('constraint',constraint.make_constraint),('solver',solver.solve),('physics_step',mjx.step)]:
            bench(name,jax.vmap(fn,in_axes=(None,0)),model,data)
    result['state']='COMPLETE';save()

if __name__=='__main__':main()
