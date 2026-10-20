"""Section 10 continuation. Local use is limited to stdlib I/O and remote calls."""
from pathlib import Path
import time
from dextum_cursor_agent.pass1b import cloud as old

ROOT=old.ROOT
RECORD=ROOT/'dextum_plan/progress/2026-10-11-ten-pads/cloud'
OUTPUT=ROOT/'dextum_cursor_agent/runs/pass1_realistic_02'
REMOTE='/root/dextum/dextum_cursor_agent/runs/pass1_realistic_02'
RELEASE='/root/dextum/releases/pass1_realistic_02/dextum_cursor_agent'
old.RECORD,old.OUTPUT,old.REMOTE,old.RELEASE=RECORD,OUTPUT,REMOTE,RELEASE
api=old.api
api.RECORD=RECORD
upload,fetch,resume=old.upload,old.fetch,old.resume

def prepare():
    assert not (RECORD/'lease.json').exists()
    prior=api.read(ROOT/'dextum_plan/progress/2026-10-10-pass1-realistic/cloud/lease.json')
    assert prior['state']=='STOPPED'
    lease=dict(instance_id=prior['instance_id'],label=prior['label'],state='PREPARED',
        hourly_usd=prior['hourly_usd'],expires_at=time.time()+8*3600,maximum_hours=8,
        compute_bound_usd=8*prior['hourly_usd'],
        authorization='User review section 10: ten pad colliders, compile-only checks, then second C-touch attempt and other arms/masking',
        original_instance_stopped=True)
    api.write(RECORD/'lease.json',lease)
    return lease
