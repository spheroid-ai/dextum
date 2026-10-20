"""GPU imitation primitives; all learned arrays remain anonymous MLP weights."""
import os
if not os.environ.get('CONTAINER_ID'):
    raise RuntimeError('GPU cloud only')
os.environ['JAX_PLATFORMS'] = 'cuda,cpu'  # CPU device for host callbacks; model execution must remain on GPU.
import jax
import jax.numpy as j
import optax

HIDDEN = (512, 256, 128)


def initialize(key, inputs, outputs, *, zero_output=False):
    dimensions = (inputs, *HIDDEN, outputs)
    keys = jax.random.split(key, len(dimensions) - 1)
    layers = []
    for index, (a, b) in enumerate(zip(dimensions[:-1], dimensions[1:])):
        weight = jax.random.normal(keys[index], (a, b)) * j.sqrt(2. / (a + b))
        if zero_output and index == len(keys) - 1:
            weight = j.zeros_like(weight)
        layers.append(dict(kernel=weight, bias=j.zeros(b)))
    return tuple(layers)


def apply(params, x, *, tanh=True):
    x = j.clip(x, -5, 5)
    for index, layer in enumerate(params):
        x = x @ layer['kernel'] + layer['bias']
        if index + 1 < len(params):
            x = jax.nn.swish(x)
    return j.tanh(x) if tanh else x


def policy(params, deterministic=True):
    del deterministic
    def infer(obs, key):
        del key
        return apply(params, obs['state']), {}
    return infer


def optimizer():
    return optax.chain(optax.clip_by_global_norm(1.), optax.adam(3e-4))


def make_fit(batch_size, updates, *, tanh=True, scales=None):
    transform = optimizer()
    def loss(params, x, y):
        delta = apply(params, x, tanh=tanh) - y
        if scales is not None:
            delta = delta / scales
        return j.mean(delta ** 2)
    def fit(params, state, key, dataset, count):
        def one(carry, _):
            params, state, key = carry
            key, sample = jax.random.split(key)
            indices = jax.random.randint(sample, (batch_size,), 0, count)
            rows = dataset[indices]
            value, gradient = jax.value_and_grad(loss)(params, rows[:, :93], rows[:, 93:])
            changes, state = transform.update(gradient, state, params)
            return (optax.apply_updates(params, changes), state, key), value
        (params, state, key), values = jax.lax.scan(one, (params, state, key), None, length=updates)
        return params, state, key, dict(loss=j.mean(values), last_loss=values[-1])
    return jax.jit(fit)
