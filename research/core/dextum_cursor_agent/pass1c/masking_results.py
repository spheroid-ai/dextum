"""Derive the provisional E3 signal table from saved paired outcomes only."""
import argparse
import hashlib
import json
import os
from pathlib import Path


GROUPS = ('shear', 'location', 'slip', 'pvdf', 'palm', 'proximal', 'normal_only')


def classification(loss):
    return 'required' if loss > 10 else 'helpful' if loss >= 3 else 'unneeded on this task'


def main():
    import numpy as np
    from pass0.review_gate import exact_mcnemar
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    root = a.run
    reg = json.loads((root / 'MATCHED_REGISTRATION.json').read_text())
    assert reg['authority'] == 'review_section15_masking'
    evidence = {}

    def read(path):
        evidence[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    def final(group):
        case = root / 'cases' / group / 'c_touch'
        if not (case / 'report.json').exists():
            return None
        report = read(case / 'report.json')
        if report['state'] != 'COMPLETE':
            return None
        config = read(case / 'config.json')
        assert config['intervention'] == {'mask': group}
        assert config['configuration']['seed'] == 1
        assert config['configuration']['num_timesteps'] == 20000000
        check = read(case / 'monitor/20316160.json')
        assert check['state'] == 'PASS'
        path = case / 'monitor/20316160.npz'
        evidence[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as saved:
            data = {key: saved[key].copy() for key in saved.files}
        assert int(data['faithful'].sum()) == check['faithful_success']
        reached = [int(path.stem) for path in (case / 'monitor').glob('*.json')
                   if json.loads(path.read_text())['faithful_success'] >= 160]
        return dict(check=check, data=data, steps_to_80=min(reached) if reached else None)

    def contrast(baseline, variant):
        b = int(np.sum(baseline & ~variant))
        c = int(np.sum(~baseline & variant))
        return dict(baseline_success=int(baseline.sum()), variant_success=int(variant.sum()),
                    baseline_only=b, variant_only=c, exact_mcnemar_p=exact_mcnemar(b, c),
                    loss_points=(int(baseline.sum()) - int(variant.sum())) / 2)

    reference = final('none')
    result = dict(state='IN_PROGRESS', task='E3 cube rotation', seeds=1,
                  training='20,316,160 extra steps from the accepted Realistic checkpoint in every arm',
                  reference='Equal-budget unmasked continuation; not the 55% checkpoint before continuation',
                  absent_signals={'patch size': 'not implemented, not masked',
                                  'curvature': 'not implemented, not masked'},
                  limitations=['One training seed; classifications are provisional and specific to E3.',
                               'No hold/regrasp or real-hardware requirement claim.',
                               'Dense-skin 4a and permutation attribution are reported separately.',
                               'Fresh evaluations can differ across compilations; accepted matched results remain frozen.'],
                  signals={}, parked={})
    disposition = root / 'INDEPENDENT_REGISTRATION.json'
    if disposition.exists():
        decision = read(disposition)
        assert decision['authority'] == 'user_park_proximal_continue_independent'
        check = read(root / 'cases/proximal/c_touch/monitor/20316160.json')
        assert check['state'] == 'EXECUTION_FAILURE'
        result['parked']['proximal'] = dict(reason='Registered per-check guard stop; user keeps this row parked',
            steps=check['steps'], internal_success=check['internal_success'],
            faithful_success_diagnostic=check['faithful_success'], exact_mcnemar_p=check['exact_mcnemar_p'],
            classification=None, accepted=False)
    if reference:
        result['reference_success'] = reference['check']['faithful_success']
        result['reference_steps_to_80'] = reference['steps_to_80']
        for group in GROUPS:
            arm = final(group)
            if arm is None:
                continue
            for key in ('reset_keys', 'direct_reset_keys', 'dr_keys', 'sign'):
                assert np.array_equal(reference['data'][key], arm['data'][key]), key
            row = contrast(reference['data']['faithful'], arm['data']['faithful'])
            row['classification'] = ('combined ablation' if group == 'normal_only'
                                     else classification(row['loss_points']))
            row['steps_to_80'] = arm['steps_to_80']
            ref_path = root / 'evaluations/none/result.json'
            arm_path = root / 'evaluations' / group / 'result.json'
            if ref_path.exists() and arm_path.exists():
                ref_eval, arm_eval = read(ref_path), read(arm_path)
                if ref_eval['state'] == arm_eval['state'] == 'COMPLETE':
                    per_condition = {}
                    files = [ref_path.with_name('per_start.npz'), arm_path.with_name('per_start.npz')]
                    for path in files:
                        evidence[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
                    with np.load(files[0]) as x, np.load(files[1]) as y:
                        assert np.array_equal(x['reset_keys'], y['reset_keys'])
                        for name in ref_eval['conditions']:
                            if name != 'train':
                                assert np.array_equal(x[name + '/sign'], y[name + '/sign'])
                                per_condition[name] = contrast(x[name + '/success'], y[name + '/success'])
                    loss = sum(v['loss_points'] for v in per_condition.values()) / len(per_condition)
                    row['held_out'] = dict(loss_points=loss, conditions=per_condition,
                        classification='combined ablation' if group == 'normal_only' else classification(loss),
                        scope='Registered cube physical-variation subset; not full T1, F1 or F2',
                        pooled_p_value=None, reason='Starts repeat across conditions; do not treat them as independent trials')
            result['signals'][group] = row
    if len(result['signals']) == len(GROUPS) and all('held_out' in r for r in result['signals'].values()):
        result['state'] = 'MASKING_MEASURED_ATTRIBUTION_AND_DENSE_SKIN_PENDING'
    elif len(result['signals']) + len(result['parked']) == len(GROUPS) and all('held_out' in r for r in result['signals'].values()):
        result['state'] = 'MASKING_ACCOUNTED_WITH_PARKED_ROW_DENSE_SKIN_PENDING'
    result['required_set'] = [group for group, row in result['signals'].items()
        if row['classification'] == 'required' or row.get('held_out', {}).get('classification') == 'required']
    result['attribution'] = ('Pending permutation check on the measured required set' if result['required_set'] else
        'No measured required group to check on this task; parked and unimplemented channels carry no attribution conclusion')
    result['evidence_sha256'] = evidence
    (root / 'SIGNAL_TABLE.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# E3 rotation signal table', '',
             'One seed; equal-budget 20.3M continuations from the accepted Realistic checkpoint. '
             'The unmasked continuation is the reference. This table is specific to the cube rotation task.', '',
             '| Removed group | Faithful success | Loss vs unmasked | Classification | Held-out loss |',
             '|---|---:|---:|---|---:|']
    for group, row in result['signals'].items():
        held = row.get('held_out', {}).get('loss_points')
        lines.append(f"| {group} | {row['variant_success']}/200 | {row['loss_points']:+.1f} pp | "
                     f"{row['classification']} | {held:+.1f} pp |" if held is not None else
                     f"| {group} | {row['variant_success']}/200 | {row['loss_points']:+.1f} pp | "
                     f"{row['classification']} | pending |")
    for group in result['parked']:
        lines.append(f'| {group} | parked after guard failure | — | not classified | not run |')
    lines += ['', result['attribution'], '', 'Patch size and curvature: not implemented, not masked. '
              'Hold/regrasp follow their task implementation. Dense-skin 4a and permutation attribution remain separate phase items.',
              '', 'Mapping audit: conventional PPO and physical sensor interfaces only; no Metriplector compliance claim.']
    (root / 'SIGNAL_TABLE.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(state=result['state'], measured_groups=list(result['signals']))))


if __name__ == '__main__':
    if not os.environ.get('CONTAINER_ID'):
        raise RuntimeError('Analyze numerical results on Vast only')
    main()
