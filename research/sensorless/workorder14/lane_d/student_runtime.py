"""Bare-Aero deployment adapter for a D2 student. No simulator or G dependency.

The runtime consumes a calibrated camera feature packet and measured motor
values. It cannot load or request tactile features, contact state or a teacher.
"""
import numpy as np

HOME = np.array([.09, .09, .09, .09, .75, .035, .1], np.float32)
SCALE = np.array([.02, .02, .02, .02, .7, .003, .012], np.float32)
SENSOR_ORDER = np.array([0, 1, 2, 3, 5, 6, 4])
FRAME = 31
HISTORY = 3


class Student:
    def __init__(self, checkpoint, ctrl_range):
        with np.load(checkpoint, allow_pickle=False) as record:
            assert str(record['interface']) == 'dextum_d2_motor_v1'
            self.layers = [(record[f'w{i}'], record[f'b{i}']) for i in range(int(record['layers']))]
        assert self.layers[0][0].shape[0] == HISTORY * FRAME
        assert self.layers[-1][1].shape == (7,)
        limits = np.asarray(ctrl_range, np.float32)
        assert limits.shape == (7, 2)
        self.lo = np.maximum(limits[:, 0], HOME - SCALE)
        self.hi = np.minimum(limits[:, 1], HOME + SCALE)
        self.reset()

    def reset(self):
        self.history = None
        self.command = HOME.copy()

    def frame(self, measured_motor_values, camera_features, goal_face_rad, sign, tick):
        measured = np.asarray(measured_motor_values, np.float32)
        camera = np.asarray(camera_features, np.float32)
        if measured.shape != (7,) or camera.shape != (13,):
            raise ValueError('Expected 7 measured motor values and 13 camera features')
        proprio = (measured - HOME[SENSOR_ORDER]) / SCALE[SENSOR_ORDER]
        task = np.array([np.cos(4 * goal_face_rad), np.sin(4 * goal_face_rad), sign]) if tick >= 20 else np.zeros(3)
        values = np.r_[proprio, (self.command - HOME) / SCALE, (tick - 20) / 280, task, camera]
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite sensorless inputs')
        return np.clip(values, -5, 5).astype(np.float32)

    def action(self, observation):
        x = np.asarray(observation, np.float32)
        if x.shape != (HISTORY * FRAME,) or not np.isfinite(x).all():
            raise ValueError('Invalid student observation')
        x = np.clip(x, -5, 5)
        for index, (weight, bias) in enumerate(self.layers):
            x = x @ weight + bias
            if index + 1 < len(self.layers):
                x = x / (1 + np.exp(-np.clip(x, -80, 80)))
        return np.tanh(x)

    def __call__(self, measured_motor_values, camera_features, goal_face_rad, sign, tick):
        frame = self.frame(measured_motor_values, camera_features, goal_face_rad, sign, tick)
        self.history = np.tile(frame, HISTORY) if self.history is None else np.r_[frame, self.history[:2 * FRAME]]
        normalized = self.action(self.history)
        self.command = HOME.copy() if tick < 20 else np.clip(HOME + SCALE * normalized, self.lo, self.hi)
        return self.command.copy()
