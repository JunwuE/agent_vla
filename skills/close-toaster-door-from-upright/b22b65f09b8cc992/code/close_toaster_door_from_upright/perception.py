import numpy as np
from rgbd_surface_geometry import decode

def read_camera(ctx, camera, step):
    data = ctx.read_observation(camera)
    cal = data.get("metadata",{}).get("calibration",{})
    if cal.get("step") != step or cal.get("alignment_verified") is not True or cal.get("frame") != "optical_to_osc_arm_base":
        raise ValueError("Current aligned calibration is unavailable for "+camera)
    if data.get("depth_m") is None:
        raise ValueError("Metric depth missing for "+camera)
    rgb = np.asarray(data["rgb"],float)
    if np.max(rgb)<=1.:
        rgb=rgb*255.
    depth=np.asarray(data["depth_m"],float)
    if depth.shape!=(256,256) or rgb.shape!=(256,256,3):
        raise ValueError("Unexpected RGB-D shape")
    return rgb,depth,np.asarray(cal["intrinsics"],float),np.asarray(cal["camera_to_base"],float)


def measure_surface(ctx, label, camera, box, step):
    g=decode(ctx.ground_target(target=label,camera=camera,observation_step=step,bbox=list(box),confidence=0.86))
    m=decode(ctx.query_geometry(g["grounding_id"]))
    if m.get("reliable") is not True or m.get("frame")!="osc_arm_base" or m.get("observation_step")!=step or m.get("valid_fraction",0)<0.95:
        raise ValueError("Unreliable or stale surface depth: "+label)
    point=np.asarray(m["centroid_m"],float)
    if point.shape!=(3,) or not np.all(np.isfinite(point)):
        raise ValueError("Invalid measured surface")
    return point,m
