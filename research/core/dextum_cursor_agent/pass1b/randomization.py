"""Original randomization draws, resolved by Kit B names instead of bare-hand IDs."""
import jax
import numpy as np
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.aero_hand.rotate_z import consts

def make_randomizer(mj_model,cells):
    def domain_randomize(model: mjx.Model, rng: jax.Array):
      cube_geom_id = mj_model.geom("cube").id
      cube_body_id = mj_model.body("cube").id
      hand_qids = mjx_env.get_qpos_ids(mj_model, consts.JOINT_NAMES)
      hand_body_names = [
          "palm",
          "right_index_f_link",
          "right_index_proximal_link",
          "right_index_middle_link",
          "right_index_distal_link",
          "right_middle_f_link",
          "right_middle_proximal_link",
          "right_middle_middle_link",
          "right_middle_distal_link",
          "right_ring_f_link",
          "right_ring_proximal_link",
          "right_ring_middle_link",
          "right_ring_distal_link",
          "right_pinky_f_link",
          "right_pinky_proximal_link",
          "right_pinky_middle_link",
          "right_pinky_distal_link",
          "right_t_link",
          "right_thumb_mcp_link",
          "right_thumb_proximal_link",
          "right_thumb_distal_link",
      ]
      hand_body_ids = np.array([mj_model.body(n).id for n in hand_body_names])
      fingertip_geoms = ["if_tip", "mf_tip", "rf_tip", "pf_tip", "th_tip"]
      fingertip_geom_ids = [mj_model.geom(g).id for g in fingertip_geoms]
      fingertip_geom_ids += [mj_model.geom("kit_contact_"+c["name"]).id for c in cells[:20]]
    
      @jax.vmap
      def rand(rng):
        # Cube friction: =U(0.1, 0.5).
        rng, key = jax.random.split(rng)
        cube_friction = jax.random.uniform(key, (1,), minval=0.1, maxval=0.5)
        geom_friction = model.geom_friction.at[
            cube_geom_id : cube_geom_id + 1, 0
        ].set(cube_friction)
    
        # Fingertip friction: =U(0.5, 1.0).
        fingertip_friction = jax.random.uniform(key, (1,), minval=0.5, maxval=1.0)
        geom_friction = model.geom_friction.at[fingertip_geom_ids, 0].set(
            fingertip_friction
        )
    
        # Scale cube mass: *U(0.8, 1.2).
        rng, key1, key2 = jax.random.split(rng, 3)
        dmass = jax.random.uniform(key1, minval=0.8, maxval=1.2)
        cube_mass = model.body_mass[cube_body_id]
        body_mass = model.body_mass.at[cube_body_id].set(cube_mass * dmass)
        body_inertia = model.body_inertia.at[cube_body_id].set(
            model.body_inertia[cube_body_id] * dmass
        )
        dpos = jax.random.uniform(key2, (3,), minval=-5e-3, maxval=5e-3)
        body_ipos = model.body_ipos.at[cube_body_id].set(
            model.body_ipos[cube_body_id] + dpos
        )
    
        # Jitter qpos0: +U(-0.05, 0.05).
        rng, key = jax.random.split(rng)
        qpos0 = model.qpos0
        qpos0 = qpos0.at[hand_qids].set(
            qpos0[hand_qids]
            + jax.random.uniform(key, shape=(16,), minval=-0.05, maxval=0.05)
        )
    
        # Scale static friction: *U(0.9, 1.1).
        rng, key = jax.random.split(rng)
        frictionloss = model.dof_frictionloss[hand_qids] * jax.random.uniform(
            key, shape=(16,), minval=0.5, maxval=2.0
        )
        dof_frictionloss = model.dof_frictionloss.at[hand_qids].set(frictionloss)
    
        # Scale armature: *U(1.0, 1.05).
        rng, key = jax.random.split(rng)
        armature = model.dof_armature[hand_qids] * jax.random.uniform(
            key, shape=(16,), minval=1.0, maxval=1.05
        )
        dof_armature = model.dof_armature.at[hand_qids].set(armature)
    
        # Scale all link masses: *U(0.9, 1.1).
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(
            key, shape=(len(hand_body_ids),), minval=0.9, maxval=1.1
        )
        body_mass = model.body_mass.at[hand_body_ids].set(
            model.body_mass[hand_body_ids] * dmass
        )
    
        # Joint stiffness: *U(0.8, 1.2).
        rng, key = jax.random.split(rng)
        kp = model.actuator_gainprm[:, 0] * jax.random.uniform(
            key, (model.nu,), minval=0.8, maxval=1.2
        )
        actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp)
        actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-kp)
    
        # Joint damping: *U(0.8, 1.2).
        rng, key = jax.random.split(rng)
        kd = model.dof_damping[hand_qids] * jax.random.uniform(
            key, (16,), minval=0.8, maxval=1.2
        )
        dof_damping = model.dof_damping.at[hand_qids].set(kd)
    
        return (
            geom_friction,
            body_mass,
            body_inertia,
            body_ipos,
            qpos0,
            dof_frictionloss,
            dof_armature,
            dof_damping,
            actuator_gainprm,
            actuator_biasprm,
        )
    
      (
          geom_friction,
          body_mass,
          body_inertia,
          body_ipos,
          qpos0,
          dof_frictionloss,
          dof_armature,
          dof_damping,
          actuator_gainprm,
          actuator_biasprm,
      ) = rand(rng)
    
      in_axes = jax.tree_util.tree_map(lambda x: None, model)
      in_axes = in_axes.tree_replace({
          "geom_friction": 0,
          "body_mass": 0,
          "body_inertia": 0,
          "body_ipos": 0,
          "qpos0": 0,
          "dof_frictionloss": 0,
          "dof_armature": 0,
          "dof_damping": 0,
          "actuator_gainprm": 0,
          "actuator_biasprm": 0,
      })
    
      model = model.tree_replace({
          "geom_friction": geom_friction,
          "body_mass": body_mass,
          "body_inertia": body_inertia,
          "body_ipos": body_ipos,
          "qpos0": qpos0,
          "dof_frictionloss": dof_frictionloss,
          "dof_armature": dof_armature,
          "dof_damping": dof_damping,
          "actuator_gainprm": actuator_gainprm,
          "actuator_biasprm": actuator_biasprm,
      })
    
      return model, in_axes
    
    return domain_randomize
