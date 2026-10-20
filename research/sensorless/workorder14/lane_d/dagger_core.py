"""DAgger aggregation with the accepted faithful simulator wrapper.

20M means collected environment transitions, not optimizer updates. Every
visited observation and expert motor label is retained once in a fixed-size
aggregate, and sampled uniformly by the learner. Checkpoints contain that
aggregate plus the complete opaque simulator state, RNG and optimizer.
"""
import os
if not os.environ.get('CONTAINER_ID'):
    raise RuntimeError('GPU cloud only')
os.environ['JAX_PLATFORMS'] = 'cuda,cpu'  # CPU device for host callbacks; model execution must remain on GPU.
import functools
import jax
import jax.numpy as j
from pass1 import faithful
from pass1c.resume_loop import canonical
from workorder14.lane_d import learner


def wrapped_environment(env, worlds, seed):
    dr = jax.random.split(jax.random.PRNGKey(seed + 71000), worlds)
    return faithful.wrap_for_training(env, episode_length=300, action_repeat=1,
        full_reset=True, randomization_fn=functools.partial(env.randomize, rng=dr))


def make_collect(wrapped, worlds, unroll=40, *, taps=False):
    def collect(params, state, key, beta):
        def one(carry, _):
            state, key = carry
            key, sample = jax.random.split(key)
            proposed = learner.apply(params, state.obs['state'])
            expert = state.obs['expert_action']
            choose = jax.random.bernoulli(sample, beta, (worlds, 1))
            action = j.where(choose, expert, proposed)
            next_state = wrapped.step(state, action)
            rows = j.concatenate((state.obs['state'], expert), axis=-1)
            if taps:
                from pass0.faithful import MODEL_KEY
                before = state.replace(info={k: v for k, v in state.info.items() if k != MODEL_KEY})
                record = dict(rows=rows, state=before, action=action,
                              qpos=next_state.data.qpos, ctrl=next_state.data.ctrl,
                              done=next_state.done)
            else:
                record = rows
            return (next_state, key), record
        (state, key), rows = jax.lax.scan(one, (state, key), None, length=unroll)
        if not taps:
            rows = rows.reshape(worlds * unroll, 100)
        return state, key, rows
    return jax.jit(collect)


class Dagger:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.worlds = config['worlds']
        self.block = self.worlds * config['unroll']
        self.capacity = ((config['environment_steps'] + self.block - 1) // self.block) * self.block
        self.wrapped = wrapped_environment(env, self.worlds, config['seed'])
        self.collect = make_collect(self.wrapped, self.worlds, config['unroll'])
        self.fit = learner.make_fit(config['minibatch'], config['gradient_updates_per_collection'])
        self.transform = learner.optimizer()
        self.insert = jax.jit(lambda dataset, rows, count: jax.lax.dynamic_update_slice(
            dataset, rows, (count, j.array(0, j.int32))))

    def initial(self):
        key, reset_key, weight_key = jax.random.split(jax.random.PRNGKey(self.config['seed']), 3)
        states = jax.jit(self.wrapped.reset)(jax.random.split(reset_key, self.worlds))
        params = learner.initialize(weight_key, 93, 7, zero_output=True)
        return dict(params=params, optimizer=self.transform.init(params), key=key, env=states,
                    dataset=j.zeros((self.capacity, 100), j.float32), count=0, updates=0)

    def prepare(self, payload):
        payload = canonical(payload)
        if 'pending' in payload:
            return payload
        beta = max(0., 1. - payload['count'] / self.config['teacher_mix_decay_steps'])
        env, key, rows = self.collect(payload['params'], payload['env'], payload['key'], j.array(beta))
        jax.block_until_ready(rows)
        return {**payload, 'env': env, 'key': key, 'pending': rows}

    def advance(self, payload):
        payload = self.prepare(payload)
        dataset = self.insert(payload['dataset'], payload['pending'], j.array(payload['count'], j.int32))
        count = payload['count'] + self.block
        params, optimizer, key, loss = self.fit(payload['params'], payload['optimizer'], payload['key'],
                                              dataset, j.array(count, j.int32))
        result = dict(params=params, optimizer=optimizer, key=key, env=payload['env'],
                      dataset=dataset, count=count, updates=payload['updates'] + 1)
        jax.block_until_ready((result, loss))
        return canonical(result), loss
