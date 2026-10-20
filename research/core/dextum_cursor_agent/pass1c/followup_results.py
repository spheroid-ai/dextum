"""Report registered dense-skin and hand-cost contrasts from saved outcomes."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    import numpy as np
    from pass0.review_gate import exact_mcnemar

    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    reg = json.loads((a.run / 'REGISTRATION.json').read_text())
    assert reg['authority'] == 'user_review_section16_followups'
    base = a.run.parent
    references = {
        'dense': base / 'pass1_rotation_masking_01/cases/none/c_touch',
        'stiff_10': base / 'pass1_section15_01/cases/c_touch_ideal/c_touch',
        'thin_stiff_10': base / 'pass1_section15_01/cases/c_touch_ideal/c_touch',
    }
    evidence = {}

    def read(path):
        evidence[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    def outcomes(path):
        evidence[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as saved:
            return {key: saved[key].copy() for key in saved.files}

    def contrast(x, y):
        for key in ('reset_keys', 'direct_reset_keys', 'dr_keys', 'sign'):
            assert np.array_equal(x[key], y[key]), key
        xx, yy = x['faithful'], y['faithful']
        b, c = int(np.sum(xx & ~yy)), int(np.sum(~xx & yy))
        return dict(reference_success=int(xx.sum()), variant_success=int(yy.sum()),
                    n=len(xx), gain_points=100 * (int(yy.sum()) - int(xx.sum())) / len(xx),
                    reference_only=b, variant_only=c, exact_mcnemar_p=exact_mcnemar(b, c))

    result = dict(state='IN_PROGRESS', scope='One seed, E3 cube rotation, 20.3M warm starts',
                  registration_sha256=hashlib.sha256((a.run / 'REGISTRATION.json').read_bytes()).hexdigest(),
                  cases={}, limitations=[
                      'Dense-skin comparison uses an equal-budget unmasked continuation from the same Realistic teacher.',
                      'Mechanical controls start from the original bare-hand Ideal teacher, matching the 64% control protocol.',
                      '10 N/mm denotes the measured 5N fixture secant stiffness; damping preserves the baseline damping ratio.',
                      'The thin variant retains collider radii and masses; its maximum protrusion is 4.5mm, including the registered 1mm allowance.',
                      'No hardware stiffness, sensor accuracy, hold/regrasp, or seed-replication claim.',
                  ])
    for name in reg['order']:
        job = reg['cases'][name]
        root = Path(job['training'])
        row = dict(state='PENDING', model_sha256=job['model_sha256'],
                   teacher_sha256=job['checkpoint_sha256'], sensor_level=job['sensor_level'])
        result['cases'][name] = row
        detector = Path(job['detector'])
        if detector.exists():
            measured = read(detector)
            row['detector'] = {k: measured.get(k) for k in
                               ('state', 'matched_step_median', 'matched_step_p95', 'matched_step_max')}
        qualification = Path(job['qualification'])
        for filename in ('throughput.json', 'resume_test.json'):
            path = qualification / filename
            if path.exists():
                q = read(path)
                row[filename.removesuffix('.json')] = {k: q[k] for k in
                    ('state', 'env_steps_per_s', 'projected_rounded_budget_s', 'loss', 'full_next_state') if k in q}
        report = root / 'report.json'
        if not report.exists():
            continue
        report = read(report)
        row.update(state=report['state'], wall_s=report.get('wall_s'),
                   training_attempts=report.get('training_attempts'))
        if row['state'] != 'COMPLETE':
            row['error'] = report.get('error')
            continue
        checks = sorted((root / 'monitor').glob('*.json'), key=lambda x: int(x.stem))
        checks = [read(path) for path in checks]
        assert checks and all(check['state'] == 'PASS' for check in checks)
        assert checks[-1]['steps'] == 20316160
        row['last_two_checks'] = [{k: check[k] for k in
            ('steps', 'internal_success', 'faithful_success', 'exact_mcnemar_p', 'state')} for check in checks[-2:]]
        reference = references[name]
        old = outcomes(reference / 'monitor/20316160.npz')
        new = outcomes(root / 'monitor/20316160.npz')
        assert int(old['faithful'].sum()) == (124 if name == 'dense' else 128)
        row['paired'] = contrast(old, new)
        row['faithful_drops'] = checks[-1]['faithful_drops']
        if name == 'dense':
            previous = base / 'pass1_rotation_masking_01/evaluations/none'
            current = Path(job['t1'])
            if (current / 'result.json').exists():
                q = read(current / 'result.json')
                if q['state'] == 'COMPLETE':
                    x, y = outcomes(previous / 'per_start.npz'), outcomes(current / 'per_start.npz')
                    assert np.array_equal(x['reset_keys'], y['reset_keys'])
                    conditions = {}
                    for condition in q['conditions']:
                        if condition == 'train':
                            continue
                        assert np.array_equal(x[condition + '/sign'], y[condition + '/sign'])
                        xx, yy = x[condition + '/success'], y[condition + '/success']
                        b, c = int(np.sum(xx & ~yy)), int(np.sum(~xx & yy))
                        conditions[condition] = dict(reference_success=int(xx.sum()), variant_success=int(yy.sum()),
                            gain_points=100 * (int(yy.sum()) - int(xx.sum())) / len(xx),
                            reference_only=b, variant_only=c, exact_mcnemar_p=exact_mcnemar(b, c))
                    row['held_out'] = dict(conditions=conditions,
                        macro_gain_points=float(np.mean([r['gain_points'] for r in conditions.values()])),
                        scope='Registered cube physical-variation subset only', pooled_p_value=None,
                        reason='Repeated starts across conditions are not independent trials')
    if all(result['cases'].get(name, {}).get('state') == 'COMPLETE'
           for name in ('stiff_10', 'thin_stiff_10')):
        x = outcomes(Path(reg['cases']['stiff_10']['training'])/'monitor/20316160.npz')
        y = outcomes(Path(reg['cases']['thin_stiff_10']['training'])/'monitor/20316160.npz')
        result['thickness_at_fixed_stiffness'] = dict(
            **contrast(x, y), reference='stiff_10', variant='thin_stiff_10',
            scope='Secondary paired contrast: thinner geometry at the same registered contact parameters')
    result['evidence_sha256'] = evidence
    if all(row['state'] == 'COMPLETE' for row in result['cases'].values()):
        result['state'] = 'TRAINING_COMPLETE'
        if 'held_out' in result['cases']['dense']:
            result['state'] = 'FOLLOWUPS_MEASURED'
    (a.run / 'RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Section 16 follow-up measurements', '', result['scope'] + '.', '',
             '| Variant | Faithful success | Reference | Gain | Exact paired p |',
             '|---|---:|---:|---:|---:|']
    for name, row in result['cases'].items():
        if 'paired' not in row:
            lines.append(f"| {name} | {row['state']} | — | — | — |")
            continue
        c = row['paired']
        lines.append(f"| {name} | {c['variant_success']}/200 | {c['reference_success']}/200 | "
                     f"{c['gain_points']:+.1f} pp | {c['exact_mcnemar_p']:.5g} |")
    if 'thickness_at_fixed_stiffness' in result:
        c = result['thickness_at_fixed_stiffness']
        lines += ['', f"Thinner geometry at fixed contact parameters: {c['gain_points']:+.1f} points "
                  f"against stiff_10, exact paired p={c['exact_mcnemar_p']:.5g} (secondary contrast)."]
    for limitation in result['limitations']:
        lines += ['', limitation]
    lines += ['',
              'Mapping audit: conventional PPO and physical sensor/mechanical interfaces; no Metriplector variant.']
    (a.run / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(state=result['state'], cases={name: row['state'] for name, row in result['cases'].items()})))


if __name__ == '__main__':
    if not os.environ.get('CONTAINER_ID'):
        raise RuntimeError('Numerical reporting runs on Vast only')
    main()
