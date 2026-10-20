"""Read-only native MuJoCo oracle adapter and SI power ledger.

This is simulator mechanics, not Metriplector energy. No force is inferred from
penetration or a pad centroid. Contact wrenches come from mj_contactForce.
"""
from dataclasses import dataclass
from .schema import ContactKey, ContactRecord, OracleFrame


def point_velocity(model, data, body, point):
    import numpy as np
    import mujoco
    linear = np.zeros((3, model.nv)); angular = np.zeros_like(linear)
    mujoco.mj_jac(model, data, linear, angular, point, body)
    return linear@data.qvel, angular@data.qvel


@dataclass(frozen=True)
class ObjectSurface:
    object_id: str
    body: int
    geometry: tuple[int, ...]
    patches: object


class ContactOracle:
    def __init__(self, model, objects, element_of_geom):
        self.model = model
        self.objects = tuple(objects)
        self.element_of_geom = dict(element_of_geom)  # geom -> (class, within-class ID)
        self.object_of_geom = {}
        for obj in self.objects:
            for geom in obj.geometry:
                if geom in self.object_of_geom or geom in self.element_of_geom:
                    raise ValueError('Overlapping object/element geometry mapping')
                self.object_of_geom[geom] = obj

    def read(self, data, *, task_id, quantum='unknown'):
        import numpy as np
        import mujoco
        groups = {}
        unrepresented = []
        for index in range(data.ncon):
            contact = data.contact[index]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 not in self.object_of_geom and g2 not in self.object_of_geom:
                continue
            # Object-object endpoints need two patch identities; never silently
            # reinterpret an object as a canonical hand element.
            if g1 in self.object_of_geom and g2 in self.object_of_geom:
                raise NotImplementedError('Two-object endpoint schema is not qualified')
            obj_second = g2 in self.object_of_geom
            og, other = (g2, g1) if obj_second else (g1, g2)
            if other not in self.element_of_geom:
                unrepresented.append(other); continue
            obj = self.object_of_geom[og]
            rotation = np.asarray(data.xmat[obj.body]).reshape(3, 3)
            point = rotation.T@(contact.pos-data.xpos[obj.body])
            normal = np.asarray(contact.frame).reshape(3, 3)[0]
            assignment = obj.patches.assign(point, outward_normal=
                (-1 if obj_second else 1)*(rotation.T@normal))
            element_class, element_id = self.element_of_geom[other]
            key = ContactKey(obj.object_id, assignment.patch_id, element_class, element_id)
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, index, wrench)
            force_world = np.asarray(contact.frame).reshape(3, 3).T@wrench[:3]
            force_world *= 1 if obj_second else -1
            vo, _ = point_velocity(self.model, data, obj.body, contact.pos)
            vh, _ = point_velocity(self.model, data, int(self.model.geom_bodyid[other]), contact.pos)
            relative = vo-vh
            speed = np.linalg.norm(relative-normal*np.dot(relative, normal))
            row = groups.setdefault(key, dict(normal=0., force=np.zeros(3), point=np.zeros(3),
                unweighted=np.zeros(3), count=0, speed=0., friction=float(contact.friction[0]),
                distance=0., ambiguous=False))
            # A merged element/patch pair can have multiple effective frictions.
            # Retain the conservative minimum and label the raw count explicitly.
            load = max(float(wrench[0]), 0.)
            row['normal'] += load; row['force'] += rotation.T@force_world
            row['point'] += load*point; row['unweighted'] += point; row['count'] += 1
            row['speed'] = max(row['speed'], float(speed))
            row['friction'] = min(row['friction'], float(contact.friction[0]))
            row['distance'] = max(row['distance'], assignment.distance_m)
            row['ambiguous'] |= assignment.boundary_ambiguous
        if unrepresented:
            raise ValueError(f'Object contact has unmapped counterpart geoms: {sorted(set(unrepresented))}')
        contacts = []
        for key, r in sorted(groups.items()):
            point = r['point']/r['normal'] if r['normal'] > 0 else r['unweighted']/r['count']
            contacts.append(ContactRecord(key, r['normal'], tuple(r['force']), tuple(point),
                r['speed'], r['friction'], r['count'], r['distance'], r['ambiguous']))
        return OracleFrame(float(data.time), tuple(contacts), task_id, quantum)


def power_sample(model, data):
    """Instantaneous powers after forward/solve, before integration.

    Signed friction power is retained: positive values are a diagnostic, not
    silently clipped into a pass. Soft normal contact storage/impact loss is
    unaccounted; the energy residual must not be labelled dissipation.
    """
    import numpy as np
    import mujoco
    friction_power = 0.
    for index in range(data.ncon):
        c = data.contact[index]; wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, wrench)
        b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        v1, w1 = point_velocity(model, data, b1, c.pos)
        v2, w2 = point_velocity(model, data, b2, c.pos)
        frame = np.asarray(c.frame).reshape(3, 3)
        relative_v, relative_w = frame@(v2-v1), frame@(w2-w1)
        friction_power += float(wrench[1:3]@relative_v[1:3] + wrench[3:]@relative_w)
    external = float(np.dot(data.qfrc_applied, data.qvel))
    for body in range(1, model.nbody):
        linear, angular = point_velocity(model, data, body, data.xipos[body])
        external += float(data.xfrc_applied[body, :3]@linear +
                          data.xfrc_applied[body, 3:]@angular)
    return dict(actuator_w=float(np.dot(data.qfrc_actuator, data.qvel)),
                external_w=external, friction_signed_w=friction_power)


class EnergyLedger:
    """Per-physics-step rectangular work integral with a declared convention."""
    def __init__(self):
        self.duration_s = 0.
        self.actuator_signed_j = self.actuator_positive_j = 0.
        self.external_signed_j = self.friction_signed_j = 0.
        self.friction_positive_j = 0.
        self.samples = 0

    def add(self, sample, dt):
        import math
        if not math.isfinite(dt) or dt <= 0 or not all(math.isfinite(x) for x in sample.values()):
            raise ValueError('Invalid physical power sample')
        self.duration_s += dt; self.samples += 1
        self.actuator_signed_j += sample['actuator_w']*dt
        self.actuator_positive_j += max(sample['actuator_w'], 0)*dt
        self.external_signed_j += sample['external_w']*dt
        self.friction_signed_j += sample['friction_signed_w']*dt
        self.friction_positive_j += max(sample['friction_signed_w'], 0)*dt

    def report(self, mechanical_energy_change_j=None):
        values = vars(self).copy()
        values.update(integration='Pre-integration solved power, every physics step, rectangular dt',
            mechanical_energy_change_j=mechanical_energy_change_j,
            energy_residual_j=None if mechanical_energy_change_j is None else
                self.actuator_signed_j+self.external_signed_j-mechanical_energy_change_j,
            dexterity_score=None, minimum_energy_reference=None,
            limitation='Residual includes unmodelled soft-contact storage, impact and discretization; '
                       'not total dissipation. No physical minimum-energy reference exists yet.')
        return values
