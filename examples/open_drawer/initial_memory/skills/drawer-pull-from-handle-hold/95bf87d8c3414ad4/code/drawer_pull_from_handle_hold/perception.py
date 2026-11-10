import numpy as np
from rgbd_surface_geometry import decode
from .params import MAX_PLANE_RESIDUAL_M, MIN_PLANE_SPREAD_M

def surface(ctx, camera, box, step):
    value = decode(ctx.geometry(camera=camera, bbox=box))
    g = value.get("geometry", value)
    if g.get("reliable") is not True or g.get("frame") != "osc_arm_base":
        raise ValueError("Unreliable or incompatible metric surface")
    current_step = decode(ctx.observe())["step"]
    if current_step != step or g.get("observation_step", step) != step or g.get("valid_fraction", 0) < 0.95:
        raise ValueError("Stale or sparse selected surface")
    p = np.asarray(g["centroid_m"], float)
    if p.shape != (3,) or not np.all(np.isfinite(p)):
        raise ValueError("Invalid surface centroid")
    return p, g

def front(ctx, camera, boxes, step, outward=None, eef=None):
    points = []
    for box in boxes:
        p, g = surface(ctx, camera, box, step)
        if g.get("depth_iqr_m", 1) > 0.03:
            raise ValueError("Selected front patch depth spread changed")
        points.append(p)
    points = np.asarray(points)
    center = points.mean(axis=0)
    _, s, vh = np.linalg.svd(points-center, full_matrices=False)
    if s[1] < MIN_PLANE_SPREAD_M:
        raise ValueError("Front patches are geometrically degenerate")
    normal = vh[-1]
    reference = outward if outward is not None else eef-center
    if np.dot(normal, reference) < 0:
        normal = -normal
    residual = float(np.max(np.abs((points-center)@normal)))
    if residual > MAX_PLANE_RESIDUAL_M:
        raise ValueError("Selected drawer surface no longer planar")
    return center, normal, residual
