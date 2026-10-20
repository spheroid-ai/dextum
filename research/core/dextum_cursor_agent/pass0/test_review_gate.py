import unittest
from pass0.review_gate import require_review_pass, exact_mcnemar


class ReviewedGate(unittest.TestCase):
    def result(self, **changes):
        return dict(dict(criterion_version='review_20261010_section7', state='PASS', n=200,
            warmup_within_5mm=14, finite=True, steps=108134400,
            success_counts=dict(training=175, post_training=175),
            paired_counts=dict(both_success=162, training_only=13, direct_only=13, both_fail=12),
            matched_step_median_max_qpos_error=3.427267074584961e-7), **changes)

    def test_observed_87_percent_agreement_passes(self):
        require_review_pass(self.result())
        self.assertEqual(exact_mcnemar(13, 13), 1.0)

    def test_significant_asymmetry_fails_even_with_small_gap(self):
        with self.assertRaises(RuntimeError):
            require_review_pass(self.result(success_counts=dict(training=170, post_training=164),
                paired_counts=dict(both_success=164, training_only=6, direct_only=0, both_fail=30)))

    def test_aggregate_gap_fails_even_without_significant_asymmetry(self):
        self.assertGreater(exact_mcnemar(51, 40), .05)
        with self.assertRaises(RuntimeError):
            require_review_pass(self.result(success_counts=dict(training=151, post_training=140),
                paired_counts=dict(both_success=100, training_only=51, direct_only=40, both_fail=9)))

    def test_median_boundary_and_invalid_evidence_fail(self):
        for change in (dict(matched_step_median_max_qpos_error=1e-5),
                       dict(matched_step_median_max_qpos_error=float('nan')),
                       dict(warmup_within_5mm=13), dict(finite=False), dict(steps=1),
                       dict(n=16), dict(criterion_version='review_20261010')):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                require_review_pass(self.result(**change))


if __name__ == '__main__': unittest.main()
