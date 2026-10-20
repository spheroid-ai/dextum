"""C9 paired-bank report; GPU reductions, no pooling starts across conditions."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--registration-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert sha(a.registration) == a.registration_sha256
    reg = json.loads(a.registration.read_text())
    boot = dict(x.split(b'=', 1) for x in Path('/proc/1/environ').read_bytes().split(b'\0') if b'=' in x)
    assert boot[b'CONTAINER_ID'].decode() == str(reg['instance_id'])
    os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
    import jax
    import jax.numpy as j
    import numpy as np
    from pass0.review_gate import exact_mcnemar
    assert jax.default_backend() == 'gpu'
    source_root = Path(__file__).resolve().parents[2]
    for relative, digest in reg['source_sha256'].items():
        assert sha(source_root / relative) == digest, relative
    roots = {arm: Path(reg['output']) / arm / 'RESULT.json' for arm in ('c_touch', 'c_notouch')}
    records = {arm: json.loads(path.read_text()) for arm, path in roots.items()}
    for row in records.values():
        assert row['state'] == 'COMPLETE' and row['registration_sha256'] == a.registration_sha256
    comparisons = {}
    seed_gains = []
    all_heldout_counts = {}
    total_integer_gain = 0
    for seed in (1, 2, 3):
        counts = {'c_touch': 0, 'c_notouch': 0}
        seed_rows = {}
        for condition in reg['conditions']:
            arrays = {}
            condition_rows = {}
            for arm in ('c_touch', 'c_notouch'):
                case = records[arm]['cases'][f'seed{seed}_{arm}']
                assert case['state'] == 'COMPLETE'
                registered = next(x for x in reg['cases'] if x['case'] == case['case'])
                assert case['checkpoint_sha256'] == registered['checkpoint_sha256']
                row = case['conditions'][condition]
                assert row['repeat']['state'] == 'PASS'
                assert row['repeat']['first_sha256'] == row['repeat']['second_sha256']
                assert sha(row['per_start']) == row['per_start_sha256']
                assert sha(row['trace']) == row['trace_sha256']
                with np.load(row['per_start']) as archive:
                    arrays[arm] = {key: j.asarray(archive[key]) for key in archive.files}
                success = int(j.sum(arrays[arm]['success']))
                drops = int(j.sum(arrays[arm]['dropped']))
                assert success == row['success'] and drops == row['drops']
                condition_rows[arm] = dict(success=success, drops=drops, n=200)
                if condition in reg['held_out']:
                    counts[arm] += success
            for key in ('reset_keys', 'dr_keys', 'direct_reset_keys', 'sign'):
                assert bool(j.array_equal(arrays['c_touch'][key], arrays['c_notouch'][key])), key
            x, y = arrays['c_touch']['success'], arrays['c_notouch']['success']
            only_touch, only_no_touch = int(j.sum(x & ~y)), int(j.sum(~x & y))
            seed_rows[condition] = dict(**condition_rows,
                difference_points=(only_touch - only_no_touch) / 2,
                touch_only=only_touch, no_touch_only=only_no_touch,
                exact_mcnemar_p=exact_mcnemar(only_touch, only_no_touch))
        difference_count = counts['c_touch'] - counts['c_notouch']
        # 10 conditions x 200 starts; integer comparisons avoid a floating
        # point roundoff changing the existing ten-percentage-point boundary.
        denominator = 200 * len(reg['held_out'])
        gain = 100 * difference_count / denominator
        seed_gains.append(gain)
        total_integer_gain += difference_count
        all_heldout_counts[str(seed)] = counts
        comparisons[str(seed)] = dict(conditions=seed_rows,
            held_out_macro_success={arm: count / denominator for arm, count in counts.items()},
            held_out_gain_points=gain, ten_point_rule_pass=difference_count * 10 >= denominator)
    gains = j.asarray(seed_gains, j.float32)
    mean = float(j.mean(gains))
    half = float(4.302652729911275 * j.std(gains, ddof=1) / j.sqrt(j.asarray(3.)))
    report = dict(state='MEASURED', run_id=reg['run_id'], model_version='v2',
        registration_sha256=a.registration_sha256,
        input_results_sha256={arm: sha(path) for arm, path in roots.items()},
        scope=reg['scope'], n_per_condition=200, seeds=[1, 2, 3], comparisons=comparisons,
        held_out_integer_counts=all_heldout_counts, mean_gain_points=mean,
        seed_level_student_t95_points=[mean - half, mean + half],
        interval_method='Same registered three-seed Student-t convention; df=2, critical4.302652729911275',
        mean_ten_point_rule_pass=total_integer_gain * 10 >= 3 * 200 * len(reg['held_out']),
        all_three_seed_ten_point_rules_pass=all(x['ten_point_rule_pass'] for x in comparisons.values()),
        registered_target_points=10,
        limits=['The same starts repeat across conditions; trials are not pooled as independent.',
                'Per-condition exact McNemar p values are descriptive; no multiple-condition selection.',
                'Cube physical variations only: no new geometry, camera-dropout or hardware claim.',
                'All six actors are accepted v2 20.3M warm starts; no v1 or from-scratch arm is pooled.'])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    assert not a.output.exists()
    a.output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(state='MEASURED', mean_gain_points=mean,
                          mean_ten_point_rule_pass=report['mean_ten_point_rule_pass'])))


if __name__ == '__main__':
    main()
