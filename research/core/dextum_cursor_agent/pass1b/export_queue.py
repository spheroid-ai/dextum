"""Export completed fixed-final actors on CPU; never selects checkpoints by score."""
import os,sys,time,json,subprocess
from pathlib import Path
ROOT=Path('/root/dextum/dextum_cursor_agent/runs/pass1_realistic_01')
stop=float(sys.argv[1]);done=set()
while time.time()<stop-60 and len(done)<3:
 for arm in ('c_touch','c_notouch','a_task'):
  if arm in done:continue
  folder=ROOT/'e3'/arm;report=folder/'report.json'
  if not report.exists():continue
  record=json.loads(report.read_text())
  if record['state']!='COMPLETE':done.add(arm);continue
  checkpoints=json.loads((folder/'checkpoints.json').read_text());last=next(c for c in reversed(checkpoints) if c.get('kept',True))
  target=ROOT/'runtime'/arm/'actor.npz'
  with (ROOT/(arm+'_export.log')).open('xb') as log:
   code=subprocess.call(['/root/dextum/.venv/bin/python','-u','-m','pass1b.export_actor',str(folder/last['path']),str(target),'--arm',arm],env={**os.environ,'JAX_PLATFORMS':'cpu'},stdout=log,stderr=log)
  (ROOT/(arm+'_export.EXIT_CODE')).write_text(str(code)+'\n');done.add(arm)
 time.sleep(15)
