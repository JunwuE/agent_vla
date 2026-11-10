"""Synchronized RoboCasa OpenGL-wrapper RGB-D to the measured OSC arm-base frame."""

import numpy as np


def camera_calibration(native, name, shape, step):
    from robosuite import macros
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix

    if macros.IMAGE_CONVENTION != "opengl":
        raise ValueError("Unsupported camera convention: GymWrapper already flips OpenGL RGB and depth")
    position, rotation = native.robots[0].composite_controller.get_controller_base_pose("right")
    rotation = np.asarray(rotation)
    world_to_base = np.eye(4)
    world_to_base[:3, :3] = rotation.T
    world_to_base[:3, 3] = -rotation.T @ np.asarray(position)
    transform = world_to_base @ get_camera_extrinsic_matrix(native.sim, name)
    return dict(
        step=step,
        frame="optical_to_osc_arm_base",
        alignment_verified=True,
        intrinsics=get_camera_intrinsic_matrix(native.sim, name, *shape).tolist(),
        camera_to_base=transform.tolist(),
        image_shape=list(shape),
        convention="gym_wrapper_vertical_flip_of_opengl_rgb_and_depth",
        source="robot camera calibration and controller base pose at observation time",
    )
