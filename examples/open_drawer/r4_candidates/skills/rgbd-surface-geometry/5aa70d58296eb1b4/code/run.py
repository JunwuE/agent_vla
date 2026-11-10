from rgbd_surface_geometry.geometry import project
import numpy as np
def main(ctx, point, intrinsics, camera_to_base):
    pixel = project(np.asarray(point, float), np.asarray(intrinsics, float), np.asarray(camera_to_base, float))
    return {'ok': True, 'measured': {'pixel': pixel.tolist()}, 'why': 'Projection only; no target/contact claim'}
