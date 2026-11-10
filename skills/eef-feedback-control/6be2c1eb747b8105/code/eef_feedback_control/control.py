import numpy as np
from scipy.spatial.transform import Rotation
from rgbd_surface_geometry import decode
from .params import MAX_TRANSLATION_M, MAX_ROTATION_RAD

def pose(ctx):
    obs=decode(ctx.observe())
    obs["eef"]=np.asarray(obs["eef_control_pose"]["position"],float)
    return obs


def move_bounded(ctx, position, quaternion, closed, *, start_step, limit, steps=2):
    obs=pose(ctx)
    if obs["step"]-start_step+steps>limit:
        raise ValueError("Shared revision action bound reached")
    delta=np.asarray(position)-obs["eef"]
    length=float(np.linalg.norm(delta))
    if length>MAX_TRANSLATION_M:
        delta*=MAX_TRANSLATION_M/length
    measured=Rotation.from_quat(obs["eef_control_pose"]["quaternion_xyzw"])
    rv=(Rotation.from_quat(quaternion)*measured.inv()).as_rotvec()
    angle=float(np.linalg.norm(rv))
    if angle>MAX_ROTATION_RAD:
        rv*=MAX_ROTATION_RAD/angle
    target_rotation=Rotation.from_rotvec(rv)*measured
    ctx.move_eef(target={"position":(obs["eef"]+delta).tolist(),
        "quaternion_xyzw":target_rotation.as_quat().tolist(),
        "gripper_closed":closed,"base_motion":[0,0,0,0],"control_mode":"arm"},steps=steps)
