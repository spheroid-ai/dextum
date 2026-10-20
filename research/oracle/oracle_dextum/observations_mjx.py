"""Cube-only continuous DEXTUM interface for the privileged O1 experiment.

The main interface includes the 13-value palm-frame object state in spec 4.1.
Contact-only removes that block explicitly. Neither interface exposes raw
solver/contact arrays. These are physical ports for conventional PPO, not
named coordinates of a Metriplector.
"""
import numpy as np
import jax
import jax.numpy as j
from mujoco.mjx._src import support


# normal load, object-frame force(3), object-frame centroid(3), slip,
# conservative effective friction, loaded, normalized contact count.
CONTACT_FIELDS = ('normal_n', 'fx_obj', 'fy_obj', 'fz_obj', 'cx_obj', 'cy_obj',
                  'cz_obj', 'slip', 'effective_mu', 'engaged', 'point_count')
N_PATCHES = 6
OBJECT_STATE_FIELDS = ('x_m', 'y_m', 'z_m', 'qw', 'qx', 'qy', 'qz',
                       'vx_m_s', 'vy_m_s', 'vz_m_s', 'wx_rad_s', 'wy_rad_s', 'wz_rad_s')
O1_ARMS = ('privileged_ceiling', 'oracle_structured_full',
           'flat_same_information_full', 'oracle_contact_only',
           'patch_identity_removed_full')


def quaternion_product(a, b):
    scalar = a[0]*b[0]-j.dot(a[1:], b[1:])
    vector = a[0]*b[1:]+b[0]*a[1:]+j.cross(a[1:], b[1:])
    return j.concatenate((scalar[None], vector))


def palm_frame_object_state(data, object_body, palm_body, body_roots):
    """Position/quaternion and relative twist in a possibly moving palm frame.

MuJoCo cvel is angular:linear at the root subtree COM. Shift each spatial
velocity to its body's origin, then differentiate position in the rotating
palm frame. Merely subtracting linear cvel would be wrong for moving palms.
"""
    def origin_velocity(body):
        spatial = data.cvel[body]
        offset = data.xpos[body]-data.subtree_com[body_roots[body]]
        return spatial[3:]+j.cross(spatial[:3], offset)

    palm_rotation = data.xmat[palm_body]
    displacement = data.xpos[object_body]-data.xpos[palm_body]
    palm_angular = data.cvel[palm_body, :3]
    position = displacement@palm_rotation
    linear = (origin_velocity(object_body)-origin_velocity(palm_body)-
              j.cross(palm_angular, displacement))@palm_rotation
    angular = (data.cvel[object_body, :3]-palm_angular)@palm_rotation
    inverse_palm = data.xquat[palm_body]*j.array([1., -1., -1., -1.])
    quaternion = quaternion_product(inverse_palm, data.xquat[object_body])
    # q and -q are the same orientation. Canonicalize without assuming qw>0
    # is numerically stable at a half-turn.
    quaternion *= j.where(quaternion[j.argmax(j.abs(quaternion))] < 0., -1., 1.)
    return j.concatenate((position, quaternion, linear, angular))


class CubeContactOracle:
    def __init__(self, model, touch_model):
        self.object_body = touch_model.object_body
        self.object_geom = np.asarray(touch_model.object_geom, bool)
        self.element_of_geom = np.asarray(touch_model.pad_of_geom, int).copy()
        self.n_hand = len(touch_model.frame_bodies)
        self.environment_element = self.n_hand
        environment = (np.asarray(model.geom_bodyid) == 0) & ~self.object_geom
        self.element_of_geom[environment] = self.environment_element
        # IDs are within canonical classes; anatomy stays in this connector.
        if self.n_hand != 11:
            raise ValueError('This first adapter covers the existing 11 Aero contact elements only')
        self.classes = np.array([0]*5+[1]*5+[2]+[4], dtype=int)
        self.instances = np.array(list(range(5))+list(range(5))+[0, 0], dtype=int)
        self.n_elements = self.n_hand+1
        self.n_records = self.n_elements*N_PATCHES
        self.geom_body = np.asarray(model.geom_bodyid)
        self.body_root = np.asarray(model.body_rootid)
        self.palm_body = int(touch_model.frame_bodies[10])
        self.cube_geoms = np.flatnonzero(self.object_geom)
        if len(self.cube_geoms) != 1:
            raise ValueError('Cube oracle requires one validated cube collider')
        # MuJoCo box type = 6; require actual box, not a visual shape name.
        if int(model.geom_type[self.cube_geoms[0]]) != 6 or not np.allclose(
                model.geom_size[self.cube_geoms[0]], .025):
            raise ValueError('Analytic oracle is registered only for the 50 mm cube')

    def read(self, model, data):
        contact = data._impl.contact
        geom = j.maximum(contact.geom, 0)
        # MJX retains padded/candidate pair slots in static arrays. A valid
        # geom ID alone is not an active physical contact: exposing its count
        # or patch would leak broad-phase geometry into the contact-only arm.
        active = j.all(contact.geom >= 0, axis=-1) & (contact.dist < contact.includemargin)
        force = j.zeros((contact.pos.shape[0], 6), dtype=data.qpos.dtype)
        for dim in np.unique(contact.dim):
            wrench, ids = support.contact_force_dim(model, data, int(dim))
            force = force.at[ids].set(wrench)
        elements = j.asarray(self.element_of_geom)
        objects = j.asarray(self.object_geom)
        first = (elements[geom[:, 0]] >= 0) & objects[geom[:, 1]] & active
        second = (elements[geom[:, 1]] >= 0) & objects[geom[:, 0]] & active
        selected = first | second
        any_object = active & (objects[geom[:, 0]] | objects[geom[:, 1]])
        unmapped = j.sum(any_object & ~selected)
        element = j.where(first, elements[geom[:, 0]], elements[geom[:, 1]])
        sign = j.where(first, 1., -1.)
        rotation = data.xmat[self.object_body]
        force_world = j.einsum('ci,cij->cj', force[:, :3], contact.frame)*sign[:, None]
        outward_object = -(contact.frame[:, 0]*sign[:, None])@rotation
        axis = j.argmax(j.abs(outward_object), axis=-1)
        component = j.take_along_axis(outward_object, axis[:, None], axis=1)[:, 0]
        patch = 2*axis+(component >= 0).astype(j.int32)
        segment = j.where(selected, element*N_PATCHES+patch, 0)
        position_object = (contact.pos-data.xpos[self.object_body])@rotation
        f_n = j.maximum(force[:, 0], 0.)*selected
        force_object = force_world@rotation
        roots = j.asarray(self.body_root)
        bodies = j.asarray(self.geom_body)
        other_geom = j.where(first, geom[:, 0], geom[:, 1])

        def point_velocity(body):
            velocity = data.cvel[body]
            offset = contact.pos-data.subtree_com[roots[body]]
            return velocity[:, 3:]+j.cross(velocity[:, :3], offset)

        relative = point_velocity(j.full(segment.shape, self.object_body))-point_velocity(bodies[other_geom])
        normal = contact.frame[:, 0]
        tangential = relative-j.sum(relative*normal, axis=-1, keepdims=True)*normal
        slip = (f_n > .05) & (j.linalg.norm(tangential, axis=-1) > .01) & selected

        def total(value):
            return jax.ops.segment_sum(value, segment, self.n_records)

        normal_sum = total(f_n)
        weighted_point = total(position_object*f_n[:, None])
        centre = weighted_point/j.maximum(normal_sum[:, None], 1e-9)
        force_sum = total(j.where(selected[:, None], force_object, 0.))
        mu = jax.ops.segment_min(j.where(selected, contact.friction[:, 0], j.inf),
                                 segment, self.n_records)
        mu = j.where(j.isfinite(mu), mu, 0.)
        contacts = j.concatenate((normal_sum[:, None], force_sum, centre/.02,
            (total(slip.astype(j.int32)) > 0)[:, None], mu[:, None],
            (normal_sum > .3)[:, None], (total(selected.astype(j.int32))/8.)[:, None]), axis=-1)
        # Explicit IDs allow record permutation without changing information.
        element_index = j.repeat(j.arange(self.n_elements), N_PATCHES)
        patch_index = j.tile(j.arange(N_PATCHES), self.n_elements)
        identities = j.concatenate((jax.nn.one_hot(j.asarray(self.classes)[element_index], 5),
            jax.nn.one_hot(j.asarray(self.instances)[element_index], 5),
            jax.nn.one_hot(patch_index, N_PATCHES)), axis=-1)
        return dict(records=j.concatenate((contacts, identities), axis=-1),
                    unmapped_contacts=unmapped)

    def object_state(self, data):
        return palm_frame_object_state(data, self.object_body, self.palm_body,
                                       j.asarray(self.body_root))

    def read_full(self, model, data):
        contact = self.read(model, data)
        return dict(**contact, object_state=self.object_state(data),
                    physical_parameters=self.parameters(model))

    def parameters(self, model):
        """Actual randomized physical fields, without an invented stiffness fit."""
        body, geom = self.object_body, self.cube_geoms[0]
        return j.concatenate((model.body_mass[body:body+1]/.1,
            model.body_ipos[body]/.025, model.body_inertia[body]/1e-4,
            model.body_iquat[body], model.geom_friction[geom],
            model.geom_solref[geom], model.geom_solimp[geom]))


def records_for_arm(records, permutation, arm):
    """Episode-fixed permutation is shared across oracle/flat/ID-ablation arms."""
    if arm not in O1_ARMS:
        raise ValueError('Unknown oracle arm')
    values = records[permutation]
    if arm == 'patch_identity_removed_full':
        values = values.at[:, -N_PATCHES:].set(0.)
    return values


def interface_for_arm(interface, permutation, arm):
    """All arms retain shape; ablations zero only their registered fields.

The ceiling receives these fields plus its raw state in the environment
adapter. This function does not claim to construct the ceiling by itself.
The flat arm retains both values and identity labels, in the same order as
the structured arm. Architecture, not hidden information loss, is its change.
"""
    records = records_for_arm(interface['records'], permutation, arm)
    state = interface['object_state']
    if arm == 'oracle_contact_only':
        state = j.zeros_like(state)
    return dict(interface, records=records, object_state=state)
