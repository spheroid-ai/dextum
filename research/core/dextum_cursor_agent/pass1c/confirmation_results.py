"""Seed replication report from frozen outcomes; no policy re-evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    import numpy as np
    from scipy.stats import t
    from pass0.review_gate import exact_mcnemar

    p = argparse.ArgumentParser()
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    reg = json.loads(a.registration.read_text())
    assert reg['authority'] == 'review_section16_confirmation'
    digest = hashlib.sha256(a.registration.read_bytes()).hexdigest()
    evidence, rows, paired = {}, {}, {}

    def read(path):
        path = Path(path)
        evidence[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    def outcomes(path):
        path = Path(path)
        evidence[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as f:
            return {key: f[key].copy() for key in f.files}

    def checks(root, accepted_history=False):
        records = [read(path) for path in sorted((root/'monitor').glob('*.json'),
                                                key=lambda x: int(x.stem))]
        assert records and records[-1]['steps'] == 20316160
        if not accepted_history:
            assert all(row['state'] == 'PASS' for row in records)
        return [{key: row[key] for key in ('steps', 'internal_success',
                 'faithful_success', 'exact_mcnemar_p', 'state')} for row in records[-2:]]

    legacy = reg['seed1_matched_evidence']
    legacy_path = Path(legacy['path'])
    assert hashlib.sha256(legacy_path.read_bytes()).hexdigest() == legacy['sha256']
    accepted_set = read(legacy_path)
    assert accepted_set['decision'] == 'MASKING_AUTHORIZED_ONE_SEED_EFFECT'

    for seed in (1, 2, 3):
        rows[str(seed)] = {}
        arms = {}
        for arm in ('c_touch', 'c_notouch'):
            if seed == 1:
                accepted = reg['seed1_accepted'][arm]
                root = Path(accepted['training'])
                saved = root/'monitor/20316160.npz'
                assert hashlib.sha256(saved.read_bytes()).hexdigest() == accepted['outcomes_sha256']
                authority = accepted['authority']
            else:
                candidates = [job for job in reg['cases'].values()
                              if (job['seed'], job['arm'], job['phase']) == (seed, arm, 'adaptation')]
                assert len(candidates) == 1
                job = candidates[0]
                root = Path(job['training'])
                if not (root/'report.json').exists():
                    rows[str(seed)][arm] = dict(state='PENDING')
                    continue
                report = read(root/'report.json')
                if report['state'] != 'COMPLETE':
                    rows[str(seed)][arm] = dict(state=report['state'], error=report.get('error'))
                    continue
                assert report['registration_sha256'] == digest
                assert (report['job']['seed'], report['job']['arm']) == (seed, arm)
                teacher = read(root/'teacher.json')
                assert (teacher['seed'], teacher['arm']) == (seed, arm)
                teacher_path = Path(job['teacher_run'])/'policy_108134400.pkl'
                assert teacher['sha256'] == hashlib.sha256(teacher_path.read_bytes()).hexdigest()
                saved = root/'monitor/20316160.npz'
                authority = 'review_section16_confirmation'
            values = outcomes(saved)
            assert values['faithful'].shape == (200,)
            drops = (int(values['faithful_dropped'].sum())
                     if 'faithful_dropped' in values else None)
            if seed == 1:
                old = accepted_set['arms'][{'c_touch': 'c_touch_realistic',
                                            'c_notouch': 'c_notouch'}[arm]]
                assert old['outcome_sha256'] == accepted['outcomes_sha256']
                assert old['path'] == str(root) and old['seed'] == seed
                assert old['success_count'] == int(values['faithful'].sum())
                assert old['faithful_drops'] == drops
            else:
                assert drops is not None
            arms[arm] = values
            rows[str(seed)][arm] = dict(state='ACCEPTED', n=200,
                faithful_success=int(values['faithful'].sum()),
                faithful_drops=drops, authority=authority,
                last_two_checks=checks(root, accepted_history=seed == 1))
        if len(arms) == 2:
            x, y = arms['c_touch'], arms['c_notouch']
            if seed != 1:
                for key in ('reset_keys', 'direct_reset_keys', 'dr_keys', 'sign'):
                    assert np.array_equal(x[key], y[key]), key
            b = int(np.sum(x['faithful'] & ~y['faithful']))
            c = int(np.sum(~x['faithful'] & y['faithful']))
            if seed == 1:
                assert (b, c) == (accepted_set['touch_effect']['left_only'],
                                  accepted_set['touch_effect']['right_only'])
            paired[str(seed)] = dict(gain_points=(b-c)/2, touch_only=b, no_touch_only=c,
                exact_mcnemar_p=exact_mcnemar(b, c), bank_provenance=(
                    accepted_set['accepted_bank_provenance'] if seed == 1
                    else 'Saved reset, direct-reset, randomization keys and goal signs match exactly.'))
    result = dict(state='IN_PROGRESS', registration_sha256=digest, seeds=rows, paired=paired,
        evidence_sha256=evidence, pooled_start_p=None,
        limitations=['Three independent training seeds, shared evaluation starts.',
                     'Intervals describe variation across training seeds; n=3 is small.',
                     'A pooled 600-trial McNemar test would treat repeated starts as independent and is not reported.',
                     'Seed 1 retains the explicit review acceptance of its historical guard stop.',
                     'Seed-1 C-touch saved outcomes only: bank provenance comes from the hash-pinned accepted matched-set record; its drop count remains unavailable.'])
    if len(paired) == 3:
        def summary(values):
            values = np.asarray(values, dtype=float)
            half = float(t.ppf(.975, 2)*values.std(ddof=1)/np.sqrt(3))
            mean = float(values.mean())
            return dict(mean_points=mean, seed_values=values.tolist(),
                        confidence_interval_95=[mean-half, mean+half],
                        method='Student t across three training seeds, df=2; unclipped')
        result['seed_summary'] = {arm: summary([rows[str(seed)][arm]['faithful_success']/2
                                              for seed in (1, 2, 3)])
                                  for arm in ('c_touch', 'c_notouch')}
        result['touch_effect_summary'] = summary([paired[str(seed)]['gain_points'] for seed in (1, 2, 3)])
        result['state'] = 'CONFIRMATION_MEASURED'
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    if not os.environ.get('CONTAINER_ID'):
        raise RuntimeError('Vast only')
    main()
