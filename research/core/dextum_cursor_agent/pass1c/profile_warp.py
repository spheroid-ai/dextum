"""Isolated native Warp physics-only benchmark; not a training backend change."""
import os,sys,json,time,statistics,hashlib
from pathlib import Path
if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Vast only')
sys.path.insert(0,'/root/dextum/warp_probe_deps')
import numpy as np
import mujoco
import warp as wp
from mujoco.mjx.third_party import mujoco_warp as mw

def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
    xml=Path('/root/dextum/dextum_cursor_agent/runs/pass1_realistic_02/model/kit_b_training.xml')
    m=mujoco.MjModel.from_xml_path(str(xml));m.opt.timestep=.005
    wp.init();wp.set_device('cuda:0')
    model=mw.put_model(m)
    result=dict(state='STARTING',training_attempts=0,optimizer_steps=0,backend='vendored MuJoCo Warp',
                mujoco_version=mujoco.__version__,warp_version=wp.__version__,xml_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
                timestep=m.opt.timestep,iterations=m.opt.iterations,ls_iterations=m.opt.ls_iterations,
                solver=int(m.opt.solver),cone=int(m.opt.cone),timings={},
                limitation='Physics-only, different collision algorithm; not qualified or adopted for training')
    def save():(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    batches=tuple(map(int,sys.argv[2].split(','))) if len(sys.argv)>2 else (8,128)
    for batch in batches:
        native=[]
        cube_qid=m.jnt_qposadr[m.body_jntadr[m.body('cube').id]]
        for i in range(batch):
            d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0)
            d.ctrl[:]=np.array([.09,.09,.09,.09,.75,.035,.1])
            d.qpos[cube_qid]+=(i%8-3.5)*.001
            for _ in range(40+i%80):mujoco.mj_step(m,d)
            native.append(d)
        # Extra capacity avoids silently adopting JAX's top-30 contact rule.
        # Capacity/observed counts are reported, not a faithfulness claim.
        data=mw.put_data(m,native[0],nworld=batch,nconmax=batch*128,njmax=512)
        inputs={}
        for name in ('qpos','qvel','act','qacc_warmstart','ctrl','time'):
            arr=getattr(data,name)
            values=np.stack([np.asarray(getattr(d,name)) for d in native])
            inputs[name]=wp.array(values,dtype=arr.dtype,device=arr.device)
        def reset():
            for name,value in inputs.items():wp.copy(getattr(data,name),value)
        before=time.monotonic();reset();mw.step(model,data);wp.synchronize()
        setup=time.monotonic()-before
        with wp.ScopedCapture() as capture:
            reset();mw.step(model,data)
        wp.capture_launch(capture.graph);wp.synchronize()
        times=[]
        for _ in range(7):
            before=time.monotonic()
            for __ in range(8):wp.capture_launch(capture.graph)
            wp.synchronize();times.append((time.monotonic()-before)/8)
        qpos=data.qpos.numpy();qvel=data.qvel.numpy()
        result['timings'][str(batch)]=dict(median_s=statistics.median(times),min_s=min(times),max_s=max(times),
            physics_steps_per_s=batch/statistics.median(times),setup_s=setup,
            finite=bool(np.isfinite(qpos).all() and np.isfinite(qvel).all()),
            ncon=data.ncon.numpy().tolist(),nefc=data.nefc.numpy().tolist(),ncon_capacity=batch*128,njmax=512,
            native_contact_counts=[int(d.ncon) for d in native])
        if batch==8:np.savez(out/'one_step8.npz',qpos=qpos,qvel=qvel)
        print('WARP',batch,json.dumps({k:v for k,v in result['timings'][str(batch)].items() if k not in ('nefc','native_contact_counts')}),flush=True);save()
        # Destroy the preceding graph outside capture. Rebinding `capture`
        # inside the next ScopedCapture otherwise destroys it during capture.
        del capture
    result['state']='COMPLETE_SPEED_PROBE';save()

def qualify_sensors(out, bank_path):
    """Review §11 item 1, after §13 JAX arms: one fixed-bank measurement."""
    import jax
    import jax.numpy as j
    from mujoco import mjx
    from pass1c.kit_model import TruthTouch, KIT
    from pass1c.sensors import Sensors
    from pass1c.warp_sensors import WarpSensors, warp_fields, contact_wrench
    from pass1c.collision_batched import install
    install()
    out.mkdir(parents=True,exist_ok=False)
    jax.config.update('jax_default_matmul_precision','highest')
    xml=Path(os.environ['DEXTUM_KIT_XML'])
    m=mujoco.MjModel.from_xml_path(str(xml));m.opt.timestep=.005
    assert m.opt.iterations==5 and m.opt.ls_iterations==8
    bank=np.load(bank_path);batch=bank['qpos'].shape[0]
    assert batch==1024
    cells=json.loads((KIT/'geometry.json').read_text())['cells']
    truth=TruthTouch(m);jax_sensor=Sensors(m,cells,truth);warp_sensor=WarpSensors(m,cells,truth)
    result=dict(item=1,state='RUNNING',model_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
        bank_sha256=hashlib.sha256(bank_path.read_bytes()).hexdigest(),worlds=batch,cells=40,
        method='One physics step from every saved item-0 fixed state; identical nominal model, solver and initial dynamic fields',
        sensor_level='Raw before noise; optical 3 axes, barometric normal only',
        contact_threshold_N=.3,minimum_contact_agreement=.99,force_difference_norm='Euclidean per cell; normal only on barometric cells',
        maximum_force_difference_N=.05,minimum_force_agreement=.95,
        versions=dict(mujoco=mujoco.__version__,warp=wp.__version__,jax=jax.__version__,numpy=np.__version__),
        training_attempts=0,optimizer_steps=0)
    def save():(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    native=[];states=[]
    for i in range(batch):
        d=mujoco.MjData(m)
        for field in bank.files:
            if field=='time':d.time=float(bank[field][i])
            else:getattr(d,field)[:]=bank[field][i]
        mujoco.mj_forward(m,d)
        native.append(d);states.append(mjx.put_data(m,d,impl='jax'))
    model=mjx.put_model(m,impl='jax')
    data=jax.tree.map(lambda *xs:j.stack(xs),*states)
    del states  # The stacked data owns its buffers; release per-world copies.
    print('Sensor qualification: JAX fixed-state step and raw contacts',flush=True)
    data=jax.jit(jax.vmap(mjx.step,in_axes=(None,0)))(model,data)
    jf,js=jax.jit(jax.vmap(lambda d:jax_sensor.raw(model,d)))(data)
    jf,js=np.asarray(jf),np.asarray(js)
    # No second physics benchmark: this is contact/sensor agreement only.
    wp.init();wp.set_device('cuda:0')
    wm=mw.put_model(m);wd=mw.put_data(m,native[0],nworld=batch,nconmax=batch*128,njmax=512)
    for field in bank.files:
        arr=getattr(wd,field)
        values=np.stack([np.asarray(getattr(d,field)) for d in native])
        wp.copy(arr,wp.array(values,dtype=arr.dtype,device=arr.device))
    print('Sensor qualification: Warp fixed-state step and raw contacts',flush=True)
    mw.step(wm,wd);wp.synchronize()
    fields=warp_fields(wd,wp)
    wf,ws=jax.jit(lambda f:warp_sensor.raw_batched(int(m.opt.cone),f))(fields)
    wf,ws=np.asarray(wf),np.asarray(ws)
    # Verify the interface decoder against Warp's own public wrench extractor.
    oracle=wp.zeros(batch*128,dtype=wp.spatial_vector,device='cuda:0')
    ids=wp.array(np.arange(batch*128,dtype=np.int32),dtype=wp.int32,device='cuda:0')
    mw.contact_force(wm,wd,ids,False,oracle);wp.synchronize()
    decoded=np.asarray(jax.jit(lambda f:contact_wrench(int(m.opt.cone),f))(fields))
    oracle_values=oracle.numpy()
    result['wrench_decoder']=dict(passed=bool(np.allclose(decoded,oracle_values,rtol=1e-5,atol=1e-5)),
        maximum_absolute_difference=float(np.max(np.abs(decoded-oracle_values))))
    difference=wf-jf;difference[:,20:,1:]=0
    error=np.linalg.norm(difference,axis=-1)
    flags_j=jf[:,:,0]>.3;flags_w=wf[:,:,0]>.3
    active=flags_j|flags_w
    result.update(contact_agreement=float(np.mean(flags_j==flags_w)),
        force_agreement=float(np.mean(error<.05)),
        force_error_median_N=float(np.median(error)),force_error_p95_N=float(np.percentile(error,95)),
        force_error_max_N=float(error.max()),active_cell_count=int(active.sum()),
        active_cell_force_agreement=float(np.mean(error[active]<.05)) if active.any() else None,
        ncon=int(wd.ncon.numpy()[0]),ncon_capacity=batch*128,
        maximum_constraints=int(wd.nefc.numpy().max()),constraint_capacity=512)
    # Existing noise/delay/direction/PVDF/palm transducer, unchanged on both arrays.
    keys=jax.random.split(jax.random.PRNGKey(20261011),batch)
    initial=jax.vmap(jax_sensor.initial)(keys)
    @jax.jit
    def packets(raw,speed):
        def step(s,_):
            s=jax.vmap(jax_sensor.advance)(s,raw,speed,j.zeros(batch,bool))
            return s,s['packet']
        return jax.lax.scan(step,initial,None,length=3)
    _,jp=packets(j.asarray(jf),j.asarray(js));_,wp_packet=packets(j.asarray(wf),j.asarray(ws))
    finite=all(np.isfinite(v).all() for v in [jf,js,wf,ws,np.asarray(jp),np.asarray(wp_packet)])
    result['finite']=bool(finite)
    result['packet_max_difference_diagnostic']=float(np.max(np.abs(np.asarray(jp)-np.asarray(wp_packet))))
    result['capacity_ok']=result['ncon']<result['ncon_capacity'] and result['maximum_constraints']<512
    result['state']='PASS' if (finite and result['wrench_decoder']['passed'] and result['capacity_ok'] and
        result['contact_agreement']>=.99 and result['force_agreement']>=.95) else 'FAIL'
    np.savez_compressed(out/'cells.npz',jax_force=jf,warp_force=wf,jax_speed=js,warp_speed=ws,
        force_difference_N=error,jax_contact=flags_j,warp_contact=flags_w,
        jax_packet=np.asarray(jp),warp_packet=np.asarray(wp_packet))
    save();print(json.dumps(result),flush=True)
    if result['state']!='PASS':raise SystemExit(3)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='qualify_sensors':
        qualify_sensors(Path(sys.argv[2]),Path(sys.argv[3]))
    else:main()

