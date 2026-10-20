"""Fresh-process replay of a frozen scratch qualification checkpoint.

The original scratch trainer and its source registration stay unchanged.
Only the independent check's checkpoint/expected-state options are supplied;
its one replayed optimizer update is discarded, never used for training.
"""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    if not os.environ.get('CONTAINER_ID'):
        raise RuntimeError('Cloud qualification only')
    parser=argparse.ArgumentParser()
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--case',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    record=json.loads(args.registration.read_text())
    qualification=Path(record['cases'][args.case]['qualification'])
    checkpoint=qualification/'resume_test.pkl.gz'
    expected=qualification/'expected_next.pkl.gz'
    assert checkpoint.is_file() and expected.is_file()
    from pass1c import resumable_ppo
    from pass1e import scratch
    original=resumable_ppo.train

    def replay(*positional,**kwargs):
        options=kwargs['full_state_options']
        assert options['qualify'] and options['monitor'] is None
        assert options['resume']==str(checkpoint)
        kwargs['full_state_options']=dict(options,qualify=False,expected_next=str(expected))
        return original(*positional,**kwargs)

    resumable_ppo.train=replay
    sys.argv=['scratch_resume_check','--registration',str(args.registration),
        '--case',args.case,'--mode','qualify','--output',str(args.output),
        '--resume',str(checkpoint)]
    try:
        scratch.main()
    finally:
        resumable_ppo.train=original
    result=json.loads((args.output/'full_state/process_resume.json').read_text())
    assert result['state']=='PASS' and result['fresh_process']
    assert result['checkpoint']==str(checkpoint)


if __name__=='__main__':main()
