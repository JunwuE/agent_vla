"""Read robot/controller measurements only; based on the reference ControllerAdapter."""

import numpy as np
from scipy.spatial.transform import Rotation


def read_fingerpad_geometry(robot, sim, step):
    """Known robot collision geometry transformed by measured joints, never task-object state."""
    try:
        gripper = robot.gripper['right']
        site = robot.eef_site_id['right']
        position = np.asarray(sim.data.site_xpos[site])
        rotation = np.asarray(sim.data.site_xmat[site]).reshape(3, 3)
        pads = []
        for role in ('left_fingerpad', 'right_fingerpad'):
            names = gripper.important_geoms[role]
            if len(names) != 1:
                raise ValueError('Expected exactly one pad box per finger')
            index = sim.model.geom_name2id(names[0])
            if int(sim.model.geom_type[index]) != 6:  # MuJoCo mjGEOM_BOX
                raise ValueError('Only declared robot fingerpad boxes are supported')
            center = rotation.T @ (np.asarray(sim.data.geom_xpos[index]) - position)
            orientation = rotation.T @ np.asarray(sim.data.geom_xmat[index]).reshape(3, 3)
            half = np.asarray(sim.model.geom_size[index])[:3]
            if (not np.all(np.isfinite(center)) or np.linalg.norm(center) > .3
                    or not np.all(np.isfinite(half)) or np.any(half <= 0) or np.any(half > .1)
                    or not np.allclose(orientation.T @ orientation, np.eye(3), atol=1e-5)
                    or not np.isclose(np.linalg.det(orientation), 1, atol=1e-5)):
                raise ValueError('Invalid robot pad geometry')
            pads.append(dict(role=role, center_eef=center.tolist(), rotation_eef=orientation.tolist(),
                             half_extents_m=half.tolist()))
        return dict(available=True, step=step, frame='eef_control_site',
                    source='robot_model_and_measured_joint_kinematics', pads=pads,
                    note='Robot geometry only. Not a contact, force, object pose or grasp estimate. '
                         'Use current eef_control_pose to transform into osc_arm_base.')
    except (AttributeError, KeyError, ValueError, IndexError) as exc:
        return dict(available=False, step=step, reason=str(exc))


def read_gripper_kinematics(robot, sim, step):
    """Robot-only calibration; include finger-body rotations relative to grip_site."""
    try:
        gripper = robot.gripper["right"]
        ids = [sim.model.joint_name2id(name) for name in gripper.joints]
        # MuJoCo slide joints are type 2; hinged grippers need another calibration.
        if len(ids) != 2 or any(int(sim.model.jnt_type[i]) != 2 for i in ids):
            raise ValueError("Expected two translational finger joints")
        site_rotation = sim.data.site_xmat[robot.eef_site_id["right"]].reshape(3, 3)
        axes = []
        for joint in ids:
            body = sim.model.jnt_bodyid[joint]
            body_rotation = sim.data.body_xmat[body].reshape(3, 3)
            axis = site_rotation.T @ body_rotation @ sim.model.jnt_axis[joint]
            if not np.all(np.isfinite(axis)) or not np.isclose(np.linalg.norm(axis), 1, atol=1e-5):
                raise ValueError("Invalid finger axis calibration")
            axes.append(axis)
        if abs(float(np.dot(*axes))) < 0.999:
            raise ValueError("Finger axes are not parallel")
        return dict(available=True, step=step, source="robot_joint_kinematics", frame="eef_control_site",
                    joint_names=list(gripper.joints), joint_axes_eef=[a.tolist() for a in axes],
                    jaw_separation_axis_eef=axes[0].tolist(),
                    fingerpad_geometry=read_fingerpad_geometry(robot, sim, step),
                    note="Unoriented jaw separation axis in the measured control-site frame; not an object contact or grasp estimate.")
    except (AttributeError, KeyError, ValueError, IndexError) as exc:
        return dict(available=False, step=step, reason=str(exc))


def read_robot_control(env, step):
    native = env.unwrapped.env
    if len(native.robots) != 1 or type(native.robots[0].robot_model).__name__ != "PandaOmron":
        raise ValueError("OSC tools require one PandaOmron robot")
    robot = native.robots[0]
    composite = robot.composite_controller
    if composite.name != "HYBRID_MOBILE_BASE" or set(composite.part_controllers) != {
        "right",
        "right_gripper",
        "base",
        "torso",
    }:
        raise ValueError("Unsupported composite controller")
    controller = composite.part_controllers["right"]
    contract = dict(
        name=controller.name,
        input_type=controller.input_type,
        input_ref_frame=controller.input_ref_frame,
        impedance_mode=controller.impedance_mode,
        max_position_delta=0.05,
        max_rotation_delta=0.35,
        max_mobile_command=0.25,
    )
    for field in ("input_min", "input_max", "output_min", "output_max"):
        contract[field] = np.broadcast_to(np.asarray(getattr(controller, field), float), (6,)).tolist()
    origin_pos, origin_rot = composite.get_controller_base_pose("right")
    origin_rot = np.asarray(origin_rot)
    site = robot.eef_site_id["right"]
    position = origin_rot.T @ (native.sim.data.site_xpos[site] - origin_pos)
    rotation = origin_rot.T @ native.sim.data.site_xmat[site].reshape(3, 3)
    pose = dict(
        step=step,
        frame="osc_arm_base",
        position=position.tolist(),
        quaternion_xyzw=Rotation.from_matrix(rotation).as_quat().tolist(),
    )
    contract["gripper_kinematics"] = read_gripper_kinematics(robot, native.sim, step)
    return contract, pose
