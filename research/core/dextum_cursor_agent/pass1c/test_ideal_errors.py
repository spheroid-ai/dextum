"""Cloud CPU checks for isolation and faithful packet timing."""
import unittest
import jax
import jax.numpy as j
import numpy as np
from pass1c.ideal_errors import Sensors
from pass1c.sensors import Sensors as Original


def make():
    s = Sensors.__new__(Sensors)
    s.level = 'Ideal'
    s.coupling = np.eye(20, dtype=np.float32)
    s.cell_pos = np.arange(120, dtype=np.float32).reshape(40, 3) * .0001
    return s


class IsolatedErrors(unittest.TestCase):
    def test_neutral_matches_registered_ideal_packet_history(self):
        sensor = make()
        raw = j.sin(j.arange(120, dtype=j.float32).reshape(40, 3)) + .5
        def collect(method, initial):
            def body(state, t):
                s = method(sensor, state, raw * (1 + t / 50), j.full(40, .01), False)
                return s, s['packet']
            return jax.jit(lambda: jax.lax.scan(body, initial, j.arange(64)))()[1]
        old = collect(Original.advance, Original.initial(sensor, jax.random.PRNGKey(4)))
        new = collect(Sensors.advance, sensor.initial(jax.random.PRNGKey(4)))
        np.testing.assert_allclose(np.asarray(old), np.asarray(new), atol=1e-7, rtol=0)

    def packet(self, errors, raw=None):
        s = make(); state = s.initial(jax.random.PRNGKey(5)); state['errors'] = j.array(errors, j.float32)
        if raw is None: raw = j.ones((40, 3))
        return s.advance(state, raw, j.zeros(40), False)['packet']

    def test_optical_floor_keeps_barometers_and_pvdf_ideal(self):
        raw = j.zeros((40, 3)).at[:, 0].set(.1)
        clean, cut = self.packet([0, 0, 0, 0, 0], raw), self.packet([.3, 0, 0, 0, 0], raw)
        np.testing.assert_array_equal(np.asarray(cut[:60]), 0)
        np.testing.assert_array_equal(np.asarray(cut[100:120]), np.asarray(clean[100:120]))
        np.testing.assert_array_equal(np.asarray(cut[135:155]), np.asarray(clean[135:155]))

    def test_noise_does_not_add_barometric_or_pvdf_noise(self):
        clean, noisy = self.packet([0, 0, 0, 0, 0]), self.packet([0, 0, .05, 0, 0])
        self.assertFalse(np.array_equal(clean[:60], noisy[:60]))
        np.testing.assert_array_equal(np.asarray(clean[100:120]), np.asarray(noisy[100:120]))
        np.testing.assert_array_equal(np.asarray(clean[135:155]), np.asarray(noisy[135:155]))

    def test_direction_preserves_normal_and_tangent_magnitude(self):
        clean = np.asarray(self.packet([0, 0, 0, 0, 0]))[:60].reshape(20, 3)
        rotated = np.asarray(self.packet([0, 0, 0, 30, 0]))[:60].reshape(20, 3)
        np.testing.assert_array_equal(clean[:, 0], rotated[:, 0])
        np.testing.assert_allclose(np.linalg.norm(clean[:, 1:], axis=1), np.linalg.norm(rotated[:, 1:], axis=1), atol=2e-7)

    def test_delay_is_exact_in_five_ms_sensor_ticks(self):
        s = make(); state = s.initial(jax.random.PRNGKey(5))
        state['errors'] = j.array([0, 2, 0, 0, 0], j.float32)
        first = None
        for tick in range(3):
            state = s.advance(state, j.ones((40, 3)) * (tick + 1), j.zeros(40), False)
            if tick == 0: first = self.packet([0, 0, 0, 0, 0])
            if tick < 2: np.testing.assert_array_equal(np.asarray(state['packet']), 0)
        np.testing.assert_array_equal(np.asarray(state['packet']), np.asarray(first))

    def test_false_alarms_change_only_no_event_onset_flags(self):
        s = make(); state = s.initial(jax.random.PRNGKey(5))
        state['errors'] = j.array([0, 0, 0, 0, 1], j.float32)
        events = []
        for _ in range(80):
            state = s.advance(state, j.zeros((40, 3)), j.zeros(40), False)
            packet = np.asarray(state['packet']); events.append(packet[138:155:4])
            keep = np.ones(200, bool); keep[138:155:4] = False
            np.testing.assert_array_equal(packet[keep], 0)
        self.assertTrue(np.asarray(events).any())


if __name__ == '__main__': unittest.main()
