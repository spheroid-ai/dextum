"""Export only the D2 student motor policy, with GPU/JAX-to-NumPy parity."""
import os
if not os.environ.get('CONTAINER_ID'):
    raise RuntimeError('GPU cloud only')
os.environ['JAX_PLATFORMS'] = 'cuda,cpu'  # CPU device for host callbacks; model execution must remain on GPU.
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from brax.io import model
from workorder14.lane_d import learner
from workorder14.lane_d.student_runtime import Student


def export(source, output, ctrl_range):
    assert jax.default_backend() == 'gpu'
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    params = model.load_params(source)
    arrays = dict(interface=np.array('dextum_d2_motor_v1'), layers=np.array(len(params)))
    for index, layer in enumerate(params):
        arrays[f'w{index}'] = np.asarray(layer['kernel'])
        arrays[f'b{index}'] = np.asarray(layer['bias'])
    checkpoint = output / 'student_motor.npz'
    np.savez(checkpoint, **arrays)
    runtime = Student(checkpoint, ctrl_range)
    x = jax.random.normal(jax.random.PRNGKey(11014), (64, 93)) * 3
    expected = jax.jit(learner.apply)(params, x)
    actual = np.stack([runtime.action(np.asarray(row)) for row in x])
    error = float(np.max(np.abs(np.asarray(expected) - actual)))
    result = dict(state='PASS' if error < 1e-5 else 'FAIL', actor_error=error, maximum_error=1e-5,
        observation_width=93, action_width=7, touch_inputs=False, G_controller=False,
        simulator_at_runtime=False, runtime_dependencies=['numpy'], activation='swish', output='tanh',
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        source_checkpoint_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        deployment_status='Packaged candidate; bare-hand success not measured by sensorised-hand evaluation')
    (output / 'EXPORT_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'student_runtime.py').write_bytes(Path(__file__).with_name('student_runtime.py').read_bytes())
    assert result['state'] == 'PASS'
    return result
