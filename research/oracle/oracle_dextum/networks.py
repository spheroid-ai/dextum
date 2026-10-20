"""Conventional PPO encoders for O1; no Metriplector implementation.

The structured arm groups explicit interface records by element and patch,
with two message passes along shared-element and adjacent-patch relations.
The flat control gets every same number without these grouping operations.
Hidden widths are selected from parameter counts before any learning, never
from success. Report counts and widths with every registration.
"""
import jax
import jax.numpy as j
from flax import linen as nn
from brax.training import distribution, networks as base_networks
from brax.training.agents.ppo import networks as ppo_networks
from .observations_mjx import O1_ARMS


STRUCTURED_ARMS = ('oracle_structured_full', 'oracle_contact_only', 'patch_identity_removed_full')


def group_mean(h, membership):
    sums = j.einsum('...ng,...nf->...gf', membership, h)
    counts = j.sum(membership, axis=-2)
    return sums/j.maximum(counts[..., None], 1.)


class Actor(nn.Module):
    arm: str
    outputs: int
    flat_width: int = 128

    @nn.compact
    def __call__(self, obs):
        records, global_state = obs['records'], obs['state']
        leading = records.shape[:-3]
        if self.arm in STRUCTURED_ARMS:
            # Preserve temporal order within each record's history.
            temporal = j.swapaxes(records, -3, -2).reshape((*leading, 72, -1))
            h = nn.swish(nn.Dense(64, name='record_encoder')(temporal))
            current = records[..., 0, :, :]
            classes, instances, patches = current[..., 11:16], current[..., 16:21], current[..., 21:27]
            elements = (classes[..., :, :, None]*instances[..., :, None, :]).reshape((*leading, 72, 25))
            axes = j.arange(6)//2
            adjacency = (axes[:, None] != axes[None, :]).astype(h.dtype)
            for k in range(2):
                e = group_mean(h, elements)
                p = group_mean(h, patches)
                adjacent = j.einsum('pq,...qf->...pf', adjacency/4., p)
                eh = j.einsum('...ng,...gf->...nf', elements, e)
                ph = j.einsum('...np,...pf->...nf', patches, adjacent)
                h = nn.swish(nn.Dense(64, name=f'message_{k}')(j.concatenate((h, eh, ph), -1)))
            groups = j.concatenate((group_mean(h, elements), group_mean(h, patches)), axis=-2)
            global_h = nn.swish(nn.Dense(128, name='global_encoder')(global_state))
            encoded = j.concatenate((groups.reshape((*leading, -1)), global_h), -1)
        else:
            fields = [records.reshape((*leading, -1)), global_state]
            if self.arm == 'privileged_ceiling':
                # The raw state includes solver accelerations and actuator
                # coefficients with very different units. A fixed, invertible
                # transform bounds their dynamic range without clipping away
                # information or fitting statistics on evaluation outcomes.
                fields.append(j.arcsinh(obs['ceiling']))
            encoded = nn.swish(nn.Dense(self.flat_width, name='flat_encoder')(j.concatenate(fields, -1)))
        for i, width in enumerate((512, 256, 128)):
            encoded = nn.swish(nn.Dense(width, name=f'head_{i}')(encoded))
        return nn.Dense(self.outputs, name='action_distribution')(encoded)


def parameter_count(params):
    return sum(value.size for value in jax.tree.leaves(params))


def dummy_observation(shapes, leading=(1,)):
    return {key: j.zeros((*leading, *((shape,) if isinstance(shape, int) else shape)))
            for key, shape in shapes.items()}


def capacity_contract(shapes, action_size):
    """Find flat widths by architecture arithmetic, before any task results.

    Only the first flat layer varies. Both structured field ablations retain
    exactly the main structured architecture and parameter count.
    """
    dummy = dummy_observation(shapes)
    target = parameter_count(Actor('oracle_structured_full', 2*action_size).init(
        jax.random.PRNGKey(0), dummy))
    report = {'structured_parameters': target, 'flat': {}}
    for arm in ('flat_same_information_full', 'privileged_ceiling'):
        # Parameter count is affine in this single layer's width. Two shape
        # traces determine it without initializing hundreds of candidate nets.
        def count(width):
            abstract = jax.eval_shape(lambda key: Actor(arm, 2*action_size, width).init(key, dummy),
                                      jax.random.PRNGKey(0))
            return parameter_count(abstract)
        one, two = count(1), count(2)
        width = max(1, round((target-(2*one-two))/(two-one)))
        actual = count(width)
        relative = abs(actual-target)/target
        if relative > .01:
            raise ValueError('Flat/structured parameter counts cannot be matched within 1%')
        report['flat'][arm] = dict(width=width, parameters=actual,
                                  relative_parameter_difference=relative)
    return report


def network_factory(arm, shapes, capacity):
    if arm not in O1_ARMS:
        raise ValueError(arm)
    flat_width = capacity['flat'].get(arm, {}).get('width', 128)

    def factory(observation_size, action_size, preprocess_observations_fn=None, **unused):
        del preprocess_observations_fn, unused
        for key, shape in shapes.items():
            reported = observation_size[key]
            actual = (reported,) if isinstance(reported, int) else tuple(reported)
            expected = (shape,) if isinstance(shape, int) else tuple(shape)
            if actual != expected:
                raise ValueError(f'Observation contract changed: {key}: {actual} != {expected}')
        dist = distribution.NormalTanhDistribution(event_size=action_size)
        actor = Actor(arm, dist.param_size, flat_width)
        dummy = dummy_observation(shapes)
        policy = base_networks.FeedForwardNetwork(
            init=lambda key: actor.init(key, dummy),
            apply=lambda normalizer, params, obs: actor.apply(params, obs))

        def normalize_value(x, stats):
            return j.clip((x-stats.mean)/j.maximum(stats.std, .01), -5., 5.)

        value = base_networks.make_value_network(
            observation_size, preprocess_observations_fn=normalize_value,
            hidden_layer_sizes=(512, 256, 128), activation=nn.swish,
            obs_key='privileged_state')
        return ppo_networks.PPONetworks(policy_network=policy, value_network=value,
                                      parametric_action_distribution=dist)
    return factory
