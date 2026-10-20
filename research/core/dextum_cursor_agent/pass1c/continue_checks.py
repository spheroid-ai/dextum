"""Finish the prescribed dry run, then admit the registered controls."""
import os,sys,time,json
from pathlib import Path
from pass1c.bounded import run
ROOT=Path('/root/dextum/dextum_cursor_agent/runs/pass1_realistic_02')
def record(**values):(ROOT/'CHECK_STATUS.json').write_text(json.dumps(values,indent=2)+'\n')
def main():
    deadline=float(sys.argv[1]);exit_path=ROOT/'compile_training2_exit.json'
    while not exit_path.exists() and time.time()<deadline:time.sleep(5)
    if not exit_path.exists():record(stage='DEADLINE',training_attempts=0);return
    if json.loads(exit_path.read_text())['exit_code']:
        record(stage='TRAINING_COMPILE_FAILED',training_attempts=0);return
    record(stage='COMPILING_EVALUATOR',training_attempts=0)
    result=run(ROOT,'compile_evaluator',['-m','pass1c.compile_check','evaluator',str(ROOT/'compile_evaluator')],1800)
    if result['exit_code']:
        if result['reason'] not in ('rss_cap','time_cap'):
            record(stage='EVALUATOR_CHECK_ERROR',exit=result,training_attempts=0);return
        fallback=run(ROOT,'compile_materialized_evaluator',['-m','pass1c.compile_check','materialized_evaluator',str(ROOT/'compile_materialized_evaluator')],1800)
        record(stage='FALLBACK_RECORDED',original=result,materialized=fallback,training_attempts=0)
        return
    record(stage='COMPILE_CHECKS_PASS',training_attempts=0)
    from pass1c.controls import main as controls
    # Controls independently recheck every result and the admitted model hash.
    sys.argv=['controls',str(deadline)];controls()
if __name__=='__main__':main()
