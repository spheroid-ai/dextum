"""Temporarily pause the registered trainer; always resume after profiling."""
import os,sys,signal,time,json
from pathlib import Path
from pass1c.bounded import run
root=Path('/root/dextum/dextum_cursor_agent/runs/pass1_realistic_02')
pid=int(sys.argv[1]);name=sys.argv[2]
assert b'pass1c.train' in Path(f'/proc/{pid}/cmdline').read_bytes()
record=dict(trainer_pid=pid,pause_at=time.time(),reason='User requested runtime optimization; no training attempt, optimizer retained in memory')
(root/(name+'_pause.json')).write_text(json.dumps(record,indent=2)+'\n')
os.kill(pid,signal.SIGSTOP)
try:
    time.sleep(3)
    args=['-m','pass1c.profile_runtime',str(root/name),sys.argv[3] if len(sys.argv)>3 else '8']
    if len(sys.argv)>4:args.append(sys.argv[4])
    if len(sys.argv)>4 and sys.argv[4]=='warp':args=['-m','pass1c.profile_warp',str(root/name),sys.argv[3]]
    run(root,name,args,1800)
finally:
    os.kill(pid,signal.SIGCONT)
    record['resumed_at']=time.time()
    (root/(name+'_pause.json')).write_text(json.dumps(record,indent=2)+'\n')
