"""Boundary and saved-history regression tests for the registered guard."""
import unittest
import itertools
from fractions import Fraction
from pass1c.criteria import assess_monitor, assess_clustered_monitor, exact_cluster_signflip
from pass1.qualification import require_training_pass


def row(step=0, difference=0, p=1., finite=True):
    return dict(steps=step,n=200,internal_success=100+difference,faithful_success=100,
                internal_only=25,faithful_only=25-difference,exact_mcnemar_p=p,finite=finite)


class GuardTests(unittest.TestCase):
    def test_accepted_history_and_resume(self):
        history=[]
        for i,d in enumerate([1,-8,-5,1,-1,5,12]):
            current=row(i,d,p=.07295138851623051 if i==6 else 1.)
            result=assess_monitor(current,history)
            self.assertEqual(result['state'],'PASS')
            history.append(current)
            self.assertEqual(result,assess_monitor(current,history))
        self.assertAlmostEqual(result['running_mean_signed_gap_points'],5/14)
        self.assertEqual(result['checks_so_far'],7)

    def test_strict_significance_boundary(self):
        self.assertEqual(assess_monitor(row(p=.01))['state'],'PASS')
        self.assertEqual(assess_monitor(row(p=.00999))['state'],'EXECUTION_FAILURE')

    def test_strict_single_gap_boundary_both_signs(self):
        for sign in [-1,1]:
            self.assertEqual(assess_monitor(row(difference=30*sign))['state'],'PASS')
            self.assertEqual(assess_monitor(row(difference=31*sign))['state'],'EXECUTION_FAILURE')

    def test_running_mean_uses_sign_and_three_checks(self):
        for sign in [-1,1]:
            history=[row(0,6*sign),row(1,6*sign)]
            self.assertEqual(assess_monitor(row(2,6*sign),history)['state'],'PASS')
            self.assertEqual(assess_monitor(row(2,7*sign),history)['state'],'EXECUTION_FAILURE')
        self.assertEqual(assess_monitor(row(1,12),[row(0,12)])['state'],'PASS')
        self.assertEqual(assess_monitor(row(2,0),[row(0,24),row(1,-24)])['state'],'PASS')

    def test_nonfinite_and_bank_validation(self):
        self.assertEqual(assess_monitor(row(finite=False))['state'],'EXECUTION_FAILURE')
        with self.assertRaises(ValueError):assess_monitor(dict(row(),n=128))

    def test_versioned_detector(self):
        r=dict(state='PASS',finite=True,full_reset_exercised=True,
               full_reset_preserves_physics=True,actual_ppo_training_path=True,
               criterion_version='review_section15',matched_step_median=9e-5)
        require_training_pass(r)
        with self.assertRaises(RuntimeError):require_training_pass(dict(r,matched_step_median=1e-4))
        with self.assertRaises(RuntimeError):require_training_pass(dict(r,criterion_version='review_section13'))


class ClusterGuardTests(unittest.TestCase):
    def test_exact_distribution_against_enumeration(self):
        for rows in ([[1,-1,0,1]], [[1,0,-1,1],[1,1,0,-1]], [[0,0],[0,0]],
                     [[1,1,1],[-1,1,1],[1,-1,1]]):
            totals=list(map(sum,zip(*rows)))
            samples=[sum(s*v for s,v in zip(signs,totals))
                     for signs in itertools.product([-1,1],repeat=len(totals))]
            expected=Fraction(sum(abs(s)>=abs(sum(totals)) for s in samples),len(samples))
            result=exact_cluster_signflip(rows)
            self.assertEqual(Fraction(result['cluster_tail_count'],result['cluster_sign_assignments']),expected)

    def test_repeating_correlated_checks_does_not_inflate_evidence(self):
        one=[1]*12+[-1]*5+[0]*183
        self.assertEqual(exact_cluster_signflip([one])['exact_start_cluster_signflip_p'],
                         exact_cluster_signflip([one]*7)['exact_start_cluster_signflip_p'])

    @staticmethod
    def history(positive,negative,checks=3):
        x=[1]*positive+[0]*(200-positive)
        y=[0]*positive+[1]*negative+[0]*(200-positive-negative)
        rows=[dict(steps=i,n=200,internal_success=positive,faithful_success=negative,
                   internal_only=positive,faithful_only=negative,finite=True,
                   exact_mcnemar_p=1.) for i in range(checks)]
        return rows,{i:(x,y) for i in range(checks)}

    def test_mean_alone_does_not_stop_but_consistent_bias_does(self):
        rows,paired=self.history(12,5)
        self.assertEqual(assess_monitor(rows[-1],rows)['state'],'EXECUTION_FAILURE')
        self.assertEqual(assess_clustered_monitor(rows[-1],rows,paired)['state'],'PASS')
        rows,paired=self.history(24,0)
        result=assess_clustered_monitor(rows[-1],rows,paired)
        self.assertEqual(result['stop_reasons'],['running_mean_above_3_points_and_clustered_p_below_0.01'])

    def test_existing_individual_guards_remain_and_resume_deduplicates(self):
        rows,paired=self.history(12,5)
        result=assess_clustered_monitor(rows[-1],rows,paired)
        self.assertEqual(result,assess_clustered_monitor(rows[-1],rows[:-1],paired))
        self.assertEqual(result['checks_so_far'],3)
        rows[-1]['exact_mcnemar_p']=.009
        self.assertIn('exact_mcnemar_p_below_0.01',assess_clustered_monitor(rows[-1],rows,paired)['stop_reasons'])
        rows,paired=self.history(31,0,1)
        self.assertIn('single_absolute_gap_above_15_points',assess_clustered_monitor(rows[0],[],paired)['stop_reasons'])

    def test_invalid_data_fails_closed(self):
        with self.assertRaises(ValueError):exact_cluster_signflip([[2]])
        with self.assertRaises(ValueError):exact_cluster_signflip([[1],[1,0]])
        rows,paired=self.history(12,5)
        rows[-1]['internal_success']=13
        with self.assertRaises(ValueError):assess_clustered_monitor(rows[-1],rows,paired)


if __name__=='__main__':unittest.main()
