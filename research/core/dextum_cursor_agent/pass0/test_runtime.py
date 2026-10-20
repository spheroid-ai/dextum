"""Remote boundary and task-loop checks, independent of simulator success."""
import ast
from dataclasses import replace
from pathlib import Path
import unittest

from pass0.runtime import Edge, Packet, TaskLoop, reorientation_dag, shortest_path, scripted_action


def packet(tick, yaw=0.0, contact=False):
    n = (0.0,)*10 + (1.0 if contact else 0.0,)
    return Packet(tick, n, (0.0,)*11, (False,)*11, (0.0,)*7, yaw, "cube", True)


class RuntimeAudit(unittest.TestCase):
    def test_blackout_ignores_post_contact_camera(self):
        a, b = TaskLoop(reorientation_dag("turn 30 degrees")), TaskLoop(reorientation_dag("turn 30 degrees"))
        for t in range(55):
            x = packet(t, 0, t >= 2)
            y = replace(x, camera_yaw_deg=30.0) if t > 2 else x
            self.assertEqual(a.update(x), b.update(y))
        self.assertFalse(a.goal_observed)
        self.assertFalse(b.goal_observed)

    def test_visible_goal_requires_full_hold_and_contact(self):
        loop = TaskLoop(reorientation_dag("turn 30 degrees"), blackout=False)
        loop.update(packet(0, 0, True))
        for t in range(1, 40):
            self.assertFalse(loop.update(packet(t, 30, True))["goal_observed"])
        self.assertTrue(loop.update(packet(40, 30, True))["goal_observed"])

    def test_shortest_path_uses_costs_and_excludes_failed_edges(self):
        edges = (Edge("direct", "a", "c", "pivot", 100), Edge("first", "a", "b", "engage", 1),
                 Edge("second", "b", "c", "roll", 2))
        self.assertEqual([e.id for e in shortest_path(edges, "a", "c")], ["first", "second"])
        self.assertEqual([e.id for e in shortest_path(edges, "a", "c", {"first": 3})], ["direct"])

    def test_runtime_imports_only_stdlib(self):
        tree = ast.parse(Path(__file__).with_name("runtime.py").read_text())
        allowed = {"collections", "dataclasses", "heapq", "math", "re"}
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                self.assertTrue(all(a.name in allowed for a in n.names))
            elif isinstance(n, ast.ImportFrom):
                self.assertIn(n.module, allowed)

    def test_unknown_fields_do_not_enter_packet(self):
        with self.assertRaises(TypeError):
            Packet(**{**vars(packet(0)), "phi_truth": .0692})

    def test_unknown_shape_and_prompt_fail_explicitly(self):
        with self.assertRaises(ValueError):
            reorientation_dag("insert the plug")
        result = TaskLoop(reorientation_dag("rotate -30 deg")).update(replace(packet(0), camera_shape=None))
        self.assertEqual(result["quantum"], "wait")

    def test_scripts_are_bounded(self):
        for quantum in ("engage", "hold", "pivot", "regrasp", "roll"):
            for tick in range(40):
                a = scripted_action(quantum, tick, -1, 2)
                self.assertEqual(len(a), 31)
                self.assertTrue(all(-1 <= x <= 1 for x in a))

    def test_history_and_trace_bounded(self):
        loop = TaskLoop(reorientation_dag("rotate 30 deg"))
        for t in range(1000):
            loop.update(packet(t, 0, True))
        self.assertLessEqual(len(loop.history), 5)
        self.assertLessEqual(len(loop.trace), 32)
        self.assertTrue(any(v >= 3 for v in loop.failures.values()))


if __name__ == "__main__":
    unittest.main()
