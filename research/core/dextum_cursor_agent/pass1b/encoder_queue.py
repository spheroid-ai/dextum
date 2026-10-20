"""Run the fixed CPU encoder attempt once the corrected graph build finishes."""
import sys,time,json,subprocess,os,hashlib
from pathlib import Path
ROOT=Path('/root/dextum/dextum_cursor_agent/runs/pass1_realistic_01')
start=time.time();result=ROOT/'patches_faces/result.json'
while not result.exists():
    pid=json.loads((ROOT/'patches_faces_pid.json').read_text())['pid']
    if not Path(f'/proc/{pid}').exists() or time.time()-start>1800:raise RuntimeError('Corrected graph build unavailable; encoder not trained')
    time.sleep(10)
record=json.loads(result.read_text())
(ROOT/'encoder_input_registration.json').write_text(json.dumps(dict(time=time.time(),graph_result_sha256=hashlib.sha256(result.read_bytes()).hexdigest(),sanity=record['sanity'],sweep=record['sweep'],decision='Train fixed v0 and report target limitations; count observations do not add hard stops',attempt=1),indent=2)+'\n')
code=subprocess.call(['/root/dextum/.venv/bin/python','-u','pass1b/encoder.py','/root/dextum/dextum_cursor_agent/runs/pass1_audit_01/assets_f1_v0',str(ROOT/'patches_faces'),str(ROOT/'encoder_v0')],env={**os.environ,'JAX_PLATFORMS':'cpu'})
(ROOT/'encoder.EXIT_CODE').write_text(str(code)+'\n')
raise SystemExit(code)
