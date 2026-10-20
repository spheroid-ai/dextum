"""User review section 7 qualification; historical gate preserved in plan archive."""
import math


def exact_mcnemar(training_only, direct_only):
    n = training_only + direct_only
    return 1.0 if n == 0 else min(1.0, 2 * sum(
        math.comb(n, k) for k in range(min(training_only, direct_only) + 1)) / 2 ** n)


def require_review_pass(result):
    if result.get('criterion_version') != 'review_20261010_section7' or result.get('state') != 'PASS':
        raise RuntimeError('A passing section-7 fixed-policy qualification is required')
    if result['n'] != 200 or result['steps'] != 108134400 or not result['finite']:
        raise RuntimeError('Missing registered historical checkpoint/bank or finite evidence')
    counts = result['paired_counts']
    values = [counts[k] for k in ('both_success', 'training_only', 'direct_only', 'both_fail')]
    if any(type(v) is not int or not 0 <= v <= 200 for v in values) or sum(values) != 200:
        raise RuntimeError('Invalid paired outcome counts')
    both, b, c, _ = values
    if result['success_counts'] != dict(training=both+b, post_training=both+c):
        raise RuntimeError('Paired and aggregate counts do not agree')
    median = result['matched_step_median_max_qpos_error']
    if (abs(b-c) > 10 or exact_mcnemar(b, c) <= .05 or
            result['warmup_within_5mm'] < 14 or
            not math.isfinite(median) or not 0 <= median < 1e-5):
        raise RuntimeError('Section-7 qualification failed')
