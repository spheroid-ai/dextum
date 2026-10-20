"""Administrative interruption tests; no simulator or numerical execution."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from pass1c.boundary_monitor import replay_boundary_monitor


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.calls = []
        outer = self
        class Monitor:
            output = Path(outer.temp.name)
            def __call__(self, steps, params):
                outer.calls.append(('monitor', steps, params))
                outer.write(steps, 'PASS')
                return {'steps': steps, 'state': 'PASS'}
        self.monitor = Monitor()

    def write(self, steps, state, json_record=True):
        path = self.monitor.output
        with zipfile.ZipFile(path/f'{steps}.npz', 'w') as archive:
            archive.writestr('internal.npy', b'fixture')
            archive.writestr('faithful.npy', b'fixture')
        if json_record:
            (path/f'{steps}.json').write_text(json.dumps(dict(steps=steps,state=state)))

    def replay(self, steps=80, updates=4, within_epoch=0, budget=100):
        return replay_boundary_monitor(self.monitor, updates=updates, within_epoch=within_epoch,
            steps=steps, budget=budget, params='exact_saved_weights',
            save_policy=lambda s,p:self.calls.append(('save_policy',s,p)))

    def test_interrupted_boundary_replayed_before_next_update(self):
        self.replay()
        self.calls.append(('update',))
        self.assertEqual([x[0] for x in self.calls], ['save_policy','monitor','update'])
        self.assertEqual(self.calls[1][2], 'exact_saved_weights')

    def test_completed_check_is_not_resampled(self):
        self.write(80,'PASS'); self.replay(); self.assertEqual(self.calls,[])

    def test_failed_check_stays_stopped(self):
        self.write(80,'STOP')
        with self.assertRaises(RuntimeError): self.replay()
        self.assertEqual(self.calls,[])

    def test_partial_pair_is_preserved_then_replayed(self):
        self.write(80,'PASS',json_record=False); self.replay()
        self.assertEqual(len(list((self.monitor.output/'interrupted').glob('*/80.npz'))),1)
        self.assertEqual(self.calls[-1][0],'monitor')

    def test_final_budget_still_gets_its_monitor(self):
        self.replay(steps=100,within_epoch=2)
        self.assertEqual(self.calls[-1][1],100)

    def test_nonboundary_does_not_add_check(self):
        self.replay(within_epoch=2);self.assertEqual(self.calls,[])

    def test_initial_checkpoint_replays_zero_check(self):
        self.replay(steps=10,updates=0)
        self.assertEqual(self.calls[-1][1],0)

    def test_completed_without_evidence_stops(self):
        self.write(80,'PASS');(self.monitor.output/'80.npz').unlink()
        with self.assertRaises(RuntimeError):self.replay()
        self.assertEqual(self.calls,[])


if __name__ == '__main__': unittest.main()
