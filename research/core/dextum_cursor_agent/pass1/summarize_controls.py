"""Cloud-only comparison on the frozen bank; no checkpoint selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Numerical summaries run on Vast.ai only')
import numpy as np
from pass0.review_gate import exact_mcnemar


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    folders={'c_touch':a.root/'c_touch_reference_evaluation',
             'c_notouch':a.root/'controls_01/c_notouch_evaluation','a_task':a.root/'controls_01/a_task_evaluation'}
    records={};arrays={};bank=None
    for arm,folder in folders.items():
        report=json.loads((folder/'paired_evaluators.json').read_text());assert report['finite']
        assert report['arm']==arm and report['n']==200 and report['steps']==108134400
        with np.load(folder/'start_bank.npz',allow_pickle=False) as f:current={k:f[k] for k in f.files}
        if bank is not None:assert all(np.array_equal(bank[k],current[k]) for k in bank)
        bank=current
        with np.load(folder/'paired_evaluators.npz',allow_pickle=False) as f:arrays[arm]={k:f[k] for k in f.files}
        records[arm]={k:report[k] for k in ('checkpoint_sha256','success_counts','results','evaluator_agreement','absolute_success_gap','paired_counts','exact_mcnemar_p')}
    comparisons=[]
    for arm in ('c_notouch','a_task'):
        for evaluator in ('training','post_training'):
            x=arrays['c_touch'][evaluator+'_success'];y=arrays[arm][evaluator+'_success']
            b=int(np.sum(x&~y));c=int(np.sum(~x&y))
            comparisons.append(dict(reference='c_touch',control=arm,evaluator=evaluator,
                reference_only=b,control_only=c,both_success=int(np.sum(x&y)),both_fail=int(np.sum(~x&~y)),
                difference_points=100*float(x.mean()-y.mean()),exact_mcnemar_p=exact_mcnemar(b,c)))
    result=dict(state='MEASURED',n=200,training_seeds=1,records=records,comparisons=comparisons,
        bank_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in bank.items()},
        limits=['One training seed per arm: paired-start tests do not establish reproducibility across training seeds',
            'C-touch was trained with qualified retry-03; controls with the separately qualified Pass 1 barrier. All three use the same Pass 1 evaluation code',
            'Original arm definitions retained: a_task changes action space and reward as well as touch availability; c_notouch is the touch ablation',
            'Training-distribution cube bank only; no T1, T6, hardware or Realistic Kit B claim'])
    (a.root/'CONTROL_COMPARISON.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
