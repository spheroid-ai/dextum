"""A hard time/RSS cap and exit report for one cloud command."""
import os,sys,time,json,subprocess,signal
from pathlib import Path

def run(root,name,args,seconds,rss_gib=220):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();peak=0;reason=None
    with (root/(name+'.log')).open('xb') as log:
        p=subprocess.Popen([sys.executable,'-u',*args],stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
        while p.poll() is None:
            try:
                lines=Path(f'/proc/{p.pid}/status').read_text().splitlines()
                rss=int(next((x.split()[1] for x in lines if x.startswith('VmRSS:')),'0'));peak=max(peak,rss)
            except FileNotFoundError:continue
            if peak>rss_gib*1024**2 or time.monotonic()-start>seconds:
                reason='rss_cap' if peak>rss_gib*1024**2 else 'time_cap'
                os.killpg(p.pid,signal.SIGKILL);break
            time.sleep(1)
        code=p.wait()
    result=dict(exit_code=code,reason=reason,peak_rss_kib=peak,wall_s=time.monotonic()-start,
                rss_cap_gib=rss_gib,time_cap_s=seconds,pid=p.pid)
    (root/(name+'_exit.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(name,json.dumps(result),flush=True);return result

if __name__=='__main__':
    if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Vast.ai only')
    result=run(sys.argv[1],sys.argv[2],sys.argv[4:],float(sys.argv[3]))
    sys.exit(0 if result['exit_code']==0 else 1)
