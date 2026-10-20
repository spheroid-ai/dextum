"""Pass 1 training-path criteria from user review section 8; stdlib only."""
def require_training_pass(record):
    version = record.get('criterion_version')
    powered = version in ('review_section13', 'review_section15')
    detector_limit = 1e-4 if version == 'review_section15' else 1e-5
    if not (record.get('state')=='PASS' and record.get('finite') is True and
            (powered or record.get('within_5mm',0)>=14) and
            record.get('matched_step_median',float('inf'))<detector_limit and
            record.get('full_reset_exercised') is True and
            record.get('full_reset_preserves_physics') is True and
            record.get('actual_ppo_training_path') is True):
        raise RuntimeError('Pass 1 actual-PPO training-path qualification is required')
