"""GPU-host report from final paired records; no pooled-seed inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    if not os.environ.get('CONTAINER_ID'):
        raise RuntimeError('GPU cloud only')
    os.environ['JAX_PLATFORMS'] = 'cuda,cpu'  # CPU device for host callbacks; model execution must remain on GPU.
    import jax
    import numpy as np
    from pass0.review_gate import exact_mcnemar
    assert jax.default_backend() == 'gpu'
    p = argparse.ArgumentParser()
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--teacher-evaluation', type=Path, required=True)
    p.add_argument('--no-touch-evaluation', type=Path, action='append', default=[])
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    reg = json.loads(a.registration.read_text())
    root = Path(reg['training'])
    assert json.loads((root / 'report.json').read_text())['state'] == 'COMPLETE'
    checks = [json.loads(p.read_text()) for p in (root / 'monitor').glob('*.json')]
    assert len(checks) == 7 and all(row['state'] == 'PASS' for row in checks)
    final = max(checks, key=lambda row: row['steps'])
    assert final['steps'] == 20316160

    def load(path):
        with np.load(path) as record:
            return {key: record[key].copy() for key in record.files}

    outcome = load(root / 'monitor/20316160.npz')
    teacher = load(a.teacher_evaluation)
    refs = [('teacher', teacher)] + [(f'no_touch_seed{index + 1}', load(path))
                                    for index, path in enumerate(a.no_touch_evaluation)]
    rows = {}
    for name, reference in refs:
        for key in ('reset_keys', 'dr_keys', 'direct_reset_keys'):
            assert np.array_equal(outcome[key], reference[key]), key
        x, y = outcome['faithful'], reference['faithful']
        only, other_only = int(np.sum(x & ~y)), int(np.sum(~x & y))
        rows[name] = dict(n=200, success=int(x.sum()), reference_success=int(y.sum()),
            difference_points=(int(x.sum()) - int(y.sum())) / 2,
            candidate_only=only, reference_only=other_only, exact_mcnemar_p=exact_mcnemar(only, other_only))
        if name.startswith('no_touch'):
            denominator = int(teacher['faithful'].sum()) - int(y.sum())
            rows[name]['fraction_touch_gap_recovered'] = ((int(x.sum()) - int(y.sum())) / denominator
                                                         if denominator else None)
    no_touch = [row['reference_success'] for name, row in rows.items() if name.startswith('no_touch')]
    fraction = None
    if no_touch:
        mean = float(np.mean(no_touch))
        denominator = int(teacher['faithful'].sum()) - mean
        fraction = (int(outcome['faithful'].sum()) - mean) / denominator if denominator else None
    report = dict(state='MEASURED', model_version='v2', run_id=root.parent.name,
        registration_sha256=hashlib.sha256(a.registration.read_bytes()).hexdigest(),
        steps=final['steps'], n=200, success=final['faithful_success'], drops=final['faithful_drops'],
        comparisons=rows, recovered_fraction_against_no_touch_seed_mean=fraction,
        monitor_last_two=sorted(checks, key=lambda row: row['steps'])[-2:],
        scope='One seed; vision-assisted sensorless inference. Matched bank and model; historical baselines have different training budgets and inputs.')
    if reg['authority'].endswith('D3'):
        report['touch_error_per_channel'] = json.loads((root / 'touch_errors.json').read_text())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
