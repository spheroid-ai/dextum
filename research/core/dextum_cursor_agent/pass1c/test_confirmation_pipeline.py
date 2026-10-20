"""Orchestration controls: a cap can resume, a guard failure cannot."""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from pass1c import confirmation_pipeline as pipeline


class ConfirmationPipelineTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='confirmation_boundary_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.case = self.root/'teacher'
        self.job = dict(phase='teacher', arm='c_touch', seed=2,
                        detector=str(self.case/'detector/result.json'),
                        qualification=str(self.case/'qualification/full_state'),
                        training=str(self.case/'training'), maximum_projected_budget_s=9600)
        self.reg = self.root/'REGISTRATION.json'
        self.reg.write_text(json.dumps(dict(authority='review_section16_confirmation',
                            phases={'seed2':['teacher']}, cases={'teacher':self.job})))
        self.digest = hashlib.sha256(self.reg.read_bytes()).hexdigest()
        self.clock = 100.
        self.calls = []

    def invoke(self, name, mode='success', deadline=20000., previous=None):
        lease = dict(registration_sha256=self.digest, phase='seed2', deadline=deadline,
                     instance_id=123, rss_cap_gib=56)
        if previous:
            lease['previous_run'] = str(previous)
        path = self.root/(name+'_lease.json'); path.write_text(json.dumps(lease))
        out = self.root/name
        read_bytes = Path.read_bytes

        def read(p):
            if str(p) == '/proc/1/environ':
                return b'CONTAINER_ID=123\0CONTAINER_API_KEY=test-only\0'
            return read_bytes(p)

        def run(root, label, args, seconds, rss_gib):
            del root, seconds, rss_gib
            self.calls.append((label, args))
            target = Path(args[args.index('--output')+1])
            target.mkdir(parents=True, exist_ok=True)
            kind = args[args.index('--mode')+1]
            if kind == 'audit':
                (target/'result.json').write_text(json.dumps(dict(state='PASS',
                    registration_sha256=self.digest)))
            elif kind == 'qualify':
                q = target/'full_state'; q.mkdir()
                (q/'resume_test.json').write_text(json.dumps(dict(state='PASS')))
                (q/'throughput.json').write_text(json.dumps(dict(updates=10,
                    projected_rounded_budget_s=2000, metadata=dict(registration_sha256=self.digest))))
            else:
                q = target/'full_state'; q.mkdir(exist_ok=True)
                (q/'latest.pkl').write_bytes(b'opaque test continuation')
                if mode == 'guard':
                    (target/'report.json').write_text(json.dumps(dict(state='EXECUTION_FAILURE',
                                                    error='MonitorGapError')))
                    return dict(exit_code=1, reason='child_exit')
                if mode in ('cap', 'cap_boundary') and '--resume' not in args:
                    if mode == 'cap_boundary':
                        self.clock = deadline-1000
                    return dict(exit_code=-9, reason='time_cap')
                (target/'report.json').write_text(json.dumps(dict(state='COMPLETE',
                    registration_sha256=self.digest, job=self.job)))
            return dict(exit_code=0, reason='completed')

        with patch.object(sys, 'argv', ['pipeline','--registration',str(self.reg),
                '--lease',str(path),'--run',str(out)]), \
             patch.dict(os.environ, {'CONTAINER_ID':'123'}), \
             patch.object(Path, 'read_bytes', read), \
             patch('pass1c.bounded.run', run), \
             patch('pass1c.matched_batch.archive', lambda p: dict(source=str(p), verified=True)), \
             patch.object(pipeline.time, 'time', lambda: self.clock), \
             patch.object(pipeline, 'urlopen', lambda *a, **k: io.BytesIO(b'{"success":true}')):
            pipeline.main()
        return out, json.loads((out/'PIPELINE_STATE.json').read_text())

    def test_completed_phase_has_one_training_child(self):
        _, state = self.invoke('fresh')
        self.assertEqual(state['state'], 'CONFIRMATION_PHASE_COMPLETE')
        self.assertEqual(state['training_children'], {'teacher':1})

    def test_guard_failure_is_not_resumed(self):
        _, state = self.invoke('guarded', 'guard')
        self.assertEqual(state['state'], 'EXECUTION_STOP')
        self.assertEqual(state['training_children'], {'teacher':1})
        self.assertFalse(any('--resume' in args for _, args in self.calls))

    def test_teacher_cap_resumes_same_attempt_once(self):
        _, state = self.invoke('capped', 'cap')
        self.assertEqual(state['state'], 'CONFIRMATION_PHASE_COMPLETE')
        self.assertEqual(state['training_children'], {'teacher':2})
        self.assertEqual(sum('--resume' in args for _, args in self.calls), 1)

    def test_lease_boundary_reuses_qualification_and_restores_child(self):
        old, state = self.invoke('first_lease', 'cap_boundary')
        self.assertEqual(state['state'], 'LEASE_BOUNDARY')
        self.assertEqual(state['next_mode'], 'resume')
        before = len(self.calls)
        _, state = self.invoke('next_lease', deadline=40000., previous=old)
        self.assertEqual(state['state'], 'CONFIRMATION_PHASE_COMPLETE')
        self.assertEqual(state['training_children'], {'teacher':2})
        self.assertEqual(len(self.calls)-before, 1)
        self.assertIn('--resume', self.calls[-1][1])

    def test_failed_report_cannot_hide_behind_lease_boundary(self):
        old, state = self.invoke('first_lease', 'cap_boundary')
        report = self.case/'training/report.json'
        report.write_text(json.dumps(dict(state='EXECUTION_FAILURE', error='MonitorGapError')))
        before = len(self.calls)
        _, state = self.invoke('next_lease', deadline=40000., previous=old)
        self.assertEqual(state['state'], 'EXECUTION_STOP')
        self.assertEqual(len(self.calls), before)


if __name__ == '__main__':
    if not os.environ.get('CONTAINER_ID'):
        raise RuntimeError('Run checks on Vast only')
    unittest.main()
