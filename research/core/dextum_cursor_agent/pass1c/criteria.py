"""Review §15 monitor decision. Pure bookkeeping; no simulator imports."""
from fractions import Fraction
from collections import Counter


def assess_monitor(row, history=()):
    """Use each step once; the current observation replaces that step on resume."""
    rows = {int(r['steps']): r for r in history if int(r['steps']) < int(row['steps'])}
    rows[int(row['steps'])] = row
    ordered = [rows[k] for k in sorted(rows)]
    if any(r['n'] != 200 for r in ordered):
        raise ValueError('Review §15 requires the shared 200-start bank')
    gaps = [Fraction(100 * (r['internal_success'] - r['faithful_success']), r['n']) for r in ordered]
    gap, mean = gaps[-1], sum(gaps) / len(gaps)
    reasons = []
    if row['exact_mcnemar_p'] < .01:
        reasons.append('exact_mcnemar_p_below_0.01')
    if abs(gap) > 15:
        reasons.append('single_absolute_gap_above_15_points')
    if len(gaps) >= 3 and abs(mean) > 3:
        reasons.append('running_mean_absolute_gap_above_3_points')
    if not row['finite']:
        reasons.append('nonfinite_evaluation')
    return dict(criterion_version='review_section15', signed_gap_points=float(gap),
                discordant_count=row['internal_only'] + row['faithful_only'],
                running_mean_signed_gap_points=float(mean), checks_so_far=len(gaps),
                stop_reasons=reasons, state='EXECUTION_FAILURE' if reasons else 'PASS')


def exact_cluster_signflip(differences):
    """Exact two-sided test, one sign per start's entire check history.

    The null requires sign symmetry of independent start histories. Repeated
    checkpoints within a start are never treated as independent observations.
    Integer weights allow full enumeration by dynamic programming, without
    Monte Carlo error or importing numerical/simulator libraries.
    """
    rows = [list(r) for r in differences]
    if not rows or not rows[0] or any(len(r) != len(rows[0]) for r in rows):
        raise ValueError('A rectangular nonempty check-by-start history is required')
    if any(v not in (-1, 0, 1) for r in rows for v in r):
        raise ValueError('Differences must be paired binary outcome differences')
    totals = [sum(int(r[i]) for r in rows) for i in range(len(rows[0]))]
    weights = [abs(v) for v in totals if v]
    distribution = Counter({0: 1})
    for weight in weights:
        updated = Counter()
        for value, count in distribution.items():
            updated[value - weight] += count
            updated[value + weight] += count
        distribution = updated
    observed = abs(sum(totals))
    tail = sum(count for value, count in distribution.items() if abs(value) >= observed)
    denominator = 1 << len(weights)
    assert sum(distribution.values()) == denominator
    return dict(exact_start_cluster_signflip_p=float(Fraction(tail, denominator)),
                cluster_tail_count=tail, cluster_sign_assignments=denominator,
                nonzero_start_clusters=len(weights), cluster_observed_sum=sum(totals))


def assess_clustered_monitor(row, history, paired_outcomes):
    """Delegated ruling: mean >3 points AND clustered p<.01, after 3 checks.

    `paired_outcomes` maps step to (internal, faithful) binary vectors in the
    same 200-start order. Original §15 remains available above for history.
    """
    result = assess_monitor(row, history)
    rows = {int(r['steps']): r for r in history if int(r['steps']) < int(row['steps'])}
    rows[int(row['steps'])] = row
    differences = []
    for step in sorted(rows):
        x, y = [list(v) for v in paired_outcomes[step]]
        if len(x) != 200 or len(y) != 200 or any(v not in (0, 1) for v in x + y):
            raise ValueError('The paired 200-start binary outcomes are required')
        r = rows[step]
        if (sum(x), sum(y), sum(bool(a) and not bool(b) for a, b in zip(x, y)),
            sum(not bool(a) and bool(b) for a, b in zip(x, y))) != (
                r['internal_success'], r['faithful_success'], r['internal_only'], r['faithful_only']):
            raise ValueError('Saved counts differ from paired outcomes')
        differences.append([int(a) - int(b) for a, b in zip(x, y)])
    cluster = exact_cluster_signflip(differences)
    reasons = [s for s in result['stop_reasons'] if s != 'running_mean_absolute_gap_above_3_points']
    if (result['checks_so_far'] >= 3 and
        'running_mean_absolute_gap_above_3_points' in result['stop_reasons'] and
        100 * cluster['cluster_tail_count'] < cluster['cluster_sign_assignments']):
        reasons.append('running_mean_above_3_points_and_clustered_p_below_0.01')
    return dict(result, **cluster, criterion_version='review_section15_clustered',
                stop_reasons=reasons, state='EXECUTION_FAILURE' if reasons else 'PASS')
