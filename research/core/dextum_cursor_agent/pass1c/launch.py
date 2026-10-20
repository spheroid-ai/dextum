"""Launch one cloud job with the existing numerical configuration."""
import os,sys,subprocess,json
from pathlib import Path
ROOT=Path('/root/dextum/dextum_cursor_agent/runs/pass1_realistic_02')
RELEASE=Path('/root/dextum/releases/pass1_realistic_02/dextum_cursor_agent')
def launch(name,args):
    ROOT.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CONTAINER_ID='55271331',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        MUJOCO_GL='disable',XLA_PYTHON_CLIENT_PREALLOCATE='false',JAX_COMPILATION_CACHE_DIR='/root/dextum/jax_cache_ten_pads')
    env.pop('XLA_FLAGS',None)
    env['PYTHONPATH']='/root/dextum/asset_deps:'+str(RELEASE)
    env['LD_LIBRARY_PATH']=':'.join(str(p) for p in Path('/root/dextum/.venv/lib/python3.11/site-packages/nvidia').glob('*/lib'))+':/usr/local/cuda/lib64:/usr/local/nvidia/lib64'
    with (ROOT/(name+'.log')).open('xb') as log:
        p=subprocess.Popen(['/root/dextum/.venv/bin/python','-u',*args],cwd=RELEASE,env=env,
            stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
    (ROOT/(name+'_pid.json')).write_text(json.dumps(dict(pid=p.pid,args=args))+'\n');print(p.pid)
if __name__=='__main__':launch(sys.argv[1],sys.argv[2:])
