"""D3 pure-teacher rollouts and disjoint-world touch-feature regression."""
import os
if not os.environ.get('CONTAINER_ID'):
    raise RuntimeError('GPU cloud only')
os.environ['JAX_PLATFORMS'] = 'cuda,cpu'  # CPU device for host callbacks; model execution must remain on GPU.
import jax
import jax.numpy as j
from pass1c.resume_loop import canonical
from workorder14.lane_d.dagger_core import wrapped_environment
from workorder14.lane_d import learner


def touch_scales():
    return j.concatenate((j.full(60, 5.), j.full(40, 10.), j.full(20, 20.),
                          j.ones(15 + 20 + 5 + 40)))


def make_teacher_collect(wrapped, worlds, unroll=40, *, taps=False):
    def collect(unused_params, state, key, unused_beta):
        del unused_params, unused_beta
        def one(carry, _):
            state, key = carry
            action = state.obs['expert_action']
            next_state = wrapped.step(state, action)
            rows = j.concatenate((state.obs['state'], state.info['sensor_window']), axis=-1)
            if taps:
                from pass0.faithful import MODEL_KEY
                before = state.replace(info={k: v for k, v in state.info.items() if k != MODEL_KEY})
                record = dict(rows=rows, state=before, action=action,
                              qpos=next_state.data.qpos, ctrl=next_state.data.ctrl, done=next_state.done)
            else:
                record = rows
            return (next_state, key), record
        (state, key), rows = jax.lax.scan(one, (state, key), None, length=unroll)
        return state, key, rows
    return jax.jit(collect)


class Encoder:
    def __init__(self, env, config):
        self.config = config
        self.worlds = config['worlds']
        self.block = self.worlds * config['unroll']
        self.capacity = ((config['environment_steps'] + self.block - 1) // self.block) * self.block
        self.wrapped = wrapped_environment(env, self.worlds, config['seed'])
        self.collect = make_teacher_collect(self.wrapped, self.worlds, config['unroll'])
        self.fit = learner.make_fit(config['minibatch'], config['gradient_updates_per_collection'],
                                   tanh=False, scales=touch_scales())
        self.transform = learner.optimizer()
        train = [index for index in range(self.worlds) if index % 10 != 9]
        valid = [index for index in range(self.worlds) if index % 10 == 9]
        self.train_indices, self.valid_indices = j.array(train), j.array(valid)
        self.retained_ticks = len(range(0, config['unroll'], config['retain_every_control_steps']))
        self.train_block, self.valid_block = len(train) * self.retained_ticks, len(valid) * self.retained_ticks
        blocks = self.capacity // self.block
        self.train_capacity, self.valid_capacity = blocks * self.train_block, blocks * self.valid_block
        self.insert = jax.jit(lambda dataset, rows, count: jax.lax.dynamic_update_slice(dataset, rows,
                                               (count, j.array(0, j.int32))))
        self.no_policy = learner.initialize(jax.random.PRNGKey(0), 93, 7, zero_output=True)

    def initial(self):
        key, reset_key, weight_key = jax.random.split(jax.random.PRNGKey(self.config['seed']), 3)
        states = jax.jit(self.wrapped.reset)(jax.random.split(reset_key, self.worlds))
        params = learner.initialize(weight_key, 93, 200)
        return dict(params=params, optimizer=self.transform.init(params), key=key, env=states,
            dataset=j.zeros((self.train_capacity, 293), j.float32),
            validation=j.zeros((self.valid_capacity, 293), j.float32),
            count=0, train_count=0, valid_count=0, updates=0)

    def prepare(self, payload):
        payload = canonical(payload)
        if 'pending' in payload:
            return payload
        env, key, rows = self.collect(self.no_policy, payload['env'], payload['key'], j.array(1.))
        rows = rows[::self.config['retain_every_control_steps']]
        training = rows[:, self.train_indices, :].reshape(self.train_block, 293)
        validation = rows[:, self.valid_indices, :].reshape(self.valid_block, 293)
        jax.block_until_ready((training, validation))
        return {**payload, 'env': env, 'key': key, 'pending': (training, validation)}

    def advance(self, payload):
        payload = self.prepare(payload)
        training, validation = payload['pending']
        dataset = self.insert(payload['dataset'], training, j.array(payload['train_count'], j.int32))
        heldout = self.insert(payload['validation'], validation, j.array(payload['valid_count'], j.int32))
        train_count, valid_count = payload['train_count'] + self.train_block, payload['valid_count'] + self.valid_block
        params, optimizer, key, loss = self.fit(payload['params'], payload['optimizer'], payload['key'],
                                              dataset, j.array(train_count, j.int32))
        result = dict(params=params, optimizer=optimizer, key=key, env=payload['env'], dataset=dataset,
            validation=heldout, count=payload['count'] + self.block,
            train_count=train_count, valid_count=valid_count, updates=payload['updates'] + 1)
        jax.block_until_ready((result, loss))
        return canonical(result), loss

    def validation_errors(self, payload):
        from workorder14.lane_d.inferred_touch import constrain_packet
        import numpy as np
        apply = jax.jit(lambda x: jax.vmap(constrain_packet)(learner.apply(payload['params'], x, tanh=False)))
        total, square, absolute = 0, j.zeros(200), j.zeros(200)
        for start in range(0, payload['valid_count'], 8192):
            rows = payload['validation'][start:min(start + 8192, payload['valid_count'])]
            error = apply(rows[:, :93]) - rows[:, 93:]
            square += j.sum(error ** 2, axis=0)
            absolute += j.sum(j.abs(error), axis=0)
            total += rows.shape[0]
        return dict(n=total, rmse=np.asarray(j.sqrt(square / total)).tolist(),
                    mae=np.asarray(absolute / total).tolist(),
                    target='Realistic 200-value teacher sensor window, including simulated observation noise',
                    split='Disjoint environment ids/physical draws: index modulo10==9, never fitted')
