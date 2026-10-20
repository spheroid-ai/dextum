"""Report the frozen section-14 matched batch; no training or policy selection."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def exact_mcnemar(b, c):
    n = b + c
    return min(1., 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n) if n else 1.


def main():
    if not os.environ.get('CONTAINER_ID'):
        raise RuntimeError('Numerical analysis runs on Vast only')
    import numpy as np
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    a = p.parse_args()
    reg = json.loads((a.run / 'MATCHED_REGISTRATION.json').read_text())
    cases = ['c_touch_realistic', 'c_notouch', 'c_touch_ideal', 'a_task_ideal', 'a_task_realistic']
    rows, outcomes, banks, signs = {}, {}, [], []
    for name in cases:
        root = a.run / 'cases' / name
        evaluation = root / 'evaluation'
        report = json.loads((evaluation / 'paired_evaluators.json').read_text())
        assert report['finite'] and report['model_sha256'] == reg['model_sha256']
        assert report['n'] == 200 and report['steps'] == 20316160
        with np.load(evaluation / 'paired_evaluators.npz') as f:
            x = np.array(f['post_training_success'], dtype=bool)
            sign = np.array(f['post_training_sign'])
        if signs:
            assert np.array_equal(sign, signs[0])
        signs.append(sign)
        with np.load(evaluation / 'start_bank.npz') as f:
            bank = {k: np.array(f[k]) for k in f.files}
        if banks:
            assert bank.keys() == banks[0].keys()
            assert all(np.array_equal(bank[k], banks[0][k]) for k in bank)
        banks.append(bank)
        train = json.loads((root / report['arm'] / 'report.json').read_text())
        assert train['state'] == 'COMPLETE' and train['configuration']['seed'] == 1
        assert train['configuration']['num_timesteps'] == 20000000
        assert int(x.sum()) == report['success_counts']['post_training']
        count = int(x.sum()); rate = count / 200; z = 1.959963984540054
        center = (rate + z*z/400) / (1 + z*z/200)
        half = z * math.sqrt(rate*(1-rate)/200 + z*z/(4*200**2)) / (1+z*z/200)
        rows[name] = dict(arm=report['arm'], sensor_level=report['sensor_level'],
            hand_model='Corrected conformal-pad Realistic hand', seed=1, steps=report['steps'],
            faithful_success_count=count, n=200, faithful_success=rate,
            binomial_wilson_95=[center-half, center+half],
            faithful_drops=report['results']['post_training']['drops'],
            training_success_count=report['success_counts']['training'],
            evaluator_gap=report['absolute_success_gap'],
            training_wall_s=train['wall_s'], actor_inputs=train['actor_inputs'],
            checkpoint_sha256=report['checkpoint_sha256'],
            evaluation_sha256=hashlib.sha256((evaluation/'paired_evaluators.json').read_bytes()).hexdigest())
        outcomes[name] = x
    def paired(left, right):
        x, y = outcomes[left], outcomes[right]
        b, c = int((x & ~y).sum()), int((~x & y).sum())
        return dict(left=left, right=right, difference_count=b-c, difference_points=(b-c)/2,
                    both_success=int((x & y).sum()), left_only=b, right_only=c,
                    both_fail=int((~x & ~y).sum()), exact_mcnemar_p=exact_mcnemar(b,c))
    contrasts = dict(touch_effect=paired('c_touch_realistic','c_notouch'),
                     c_touch_sensor_cost=paired('c_touch_ideal','c_touch_realistic'),
                     plain_sensor_level_difference=paired('a_task_ideal','a_task_realistic'),
                     c_touch_vs_plain_realistic=paired('c_touch_realistic','a_task_realistic'))
    effect = contrasts['touch_effect']
    established = effect['difference_count'] >= 20 and effect['exact_mcnemar_p'] < .05
    baseline_path = a.run.parent / 'pass1_audit_01/c_touch_reference_evaluation/paired_evaluators.json'
    baseline = json.loads(baseline_path.read_text())
    assert baseline['finite'] and baseline['n'] == 200
    baseline_count = baseline['success_counts']['post_training']
    hand_effect = dict(bare_hand_ideal_sensor_success_count=baseline_count,
        corrected_hand_ideal_sensor_success_count=rows['c_touch_ideal']['faithful_success_count'],
        difference_points=(baseline_count-rows['c_touch_ideal']['faithful_success_count'])/2,
        baseline_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        scope='Registered aggregate hand contrast after 20M adaptation; no paired p-value inferred from aggregates.')
    result = dict(state='MEASURED', authority='review_section14', model_sha256=reg['model_sha256'],
        shared_bank_verified=True, goal_signs_verified=True, rows=rows, contrasts=contrasts,
        hand_effect=hand_effect,
        one_seed_touch_effect_established=established,
        next_step='MASKING_AND_SENSOR_SWEEPS' if established else 'C_TOUCH_AND_C_NOTOUCH_SEEDS_2_3',
        attempts_consumed=0,
        limitations=['One training seed; binomial intervals describe this evaluation bank, not training-seed variance.',
                    'Plain PPO a_task has no tactile actor inputs; its sensor-level contrast is not a direct tactile effect.',
                    'All effects apply to the recorded corrected hand, including its reduced thumb footprint.'])
    (a.run/'MATCHED_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
