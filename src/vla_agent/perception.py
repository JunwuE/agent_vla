"""Causal windows and metric geometry; no simulator object-state access."""

import base64
import io

import numpy as np
from PIL import Image


def image_url(rgb, max_edge=384):
    img = Image.fromarray(rgb)
    img.thumbnail((max_edge, max_edge))
    data = io.BytesIO()
    img.save(data, format="PNG")
    return "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()


def window(memory, count):
    history = list(memory.observations)
    indexes = sorted(set(np.linspace(0, len(history) - 1, min(count, len(history)), dtype=int)))
    selected = [history[i] for i in indexes]
    if any(obs.timestamp > history[-1].timestamp for obs in selected):
        raise ValueError("Future observation in online window")
    return selected


def decision_window(memory, config):
    if config.sampling == 'recent':
        history = list(memory.observations)[-32:]
        indices = sorted(set(np.linspace(0, len(history)-1, min(config.window_frames, len(history)), dtype=int)))
        return [history[i] for i in indices]
    history = [o for o in memory.observations if o.step >= memory.attempt_started_step]
    if not history:
        raise ValueError('No current-attempt Decision observations')
    if any(not np.isfinite(t) or not 0 <= t <= 15 for t in config.history_seconds):
        raise ValueError('Decision offsets must be finite and within 0..15 seconds')
    end = history[-1].sim_time
    chosen = {history[-1].step: history[-1]}
    for offset in [t for t in config.history_seconds if t > 0][:config.window_frames-1]:
        target = end - offset
        candidates = [o for o in history if o.sim_time <= target]
        obs = candidates[-1] if candidates else history[0]
        chosen[obs.step] = obs
    return [chosen[k] for k in sorted(chosen)]


def visual_input(observations, cameras=None):
    images, frames = [], []
    for obs in observations:
        selected = sorted(obs.rgb) if cameras is None else cameras
        for camera in selected:
            if camera not in obs.rgb:
                raise ValueError(f"Requested camera {camera!r} unavailable at step {obs.step}")
            rgb = obs.rgb[camera]
            frames.append(dict(image_index=len(images), camera=camera, step=obs.step, sim_time=obs.sim_time))
            images.append(image_url(rgb))
    return images, frames


def decision_visual_input(observations, config):
    cameras = [config.primary_camera] if config.primary_camera is not None else config.history_cameras
    if len(set(cameras)) != len(cameras):
        raise ValueError('Decision historical cameras must be unique')
    images, frames = visual_input(observations, cameras=cameras)
    if config.primary_camera is not None:
        for camera in dict.fromkeys(config.latest_cameras):
            if camera in cameras:
                continue
            extra, mapping = visual_input([observations[-1]], cameras=[camera])
            for frame in mapping:
                frame['image_index'] += len(images)
            images.extend(extra)
            frames.extend(mapping)
    return images, frames


def local_geometry(obs, camera, bbox):
    if camera not in obs.depth_m or camera not in obs.calibration:
        return dict(reliable=False, reason="Depth or camera calibration unavailable")
    depth = np.asarray(obs.depth_m[camera], dtype=float)
    calibration = obs.calibration[camera]
    if (
        calibration.get("step") != obs.step
        or calibration.get("frame") != "optical_to_osc_arm_base"
        or calibration.get("alignment_verified") is not True
    ):
        return dict(reliable=False, reason="Calibration frame or timestamp mismatch")
    if len(bbox) != 4 or any(type(x) is not int for x in bbox):
        raise ValueError("bbox requires four integer pixel coordinates")
    x0, y0, x1, y1 = bbox
    if not 0 <= x0 < x1 <= depth.shape[1] or not 0 <= y0 < y1 <= depth.shape[0]:
        raise ValueError("bbox outside image")
    k = np.asarray(calibration["intrinsics"], dtype=float)
    transform = np.asarray(calibration["camera_to_base"], dtype=float)
    if (
        k.shape != (3, 3)
        or transform.shape != (4, 4)
        or not np.isfinite(k).all()
        or not np.isfinite(transform).all()
        or k[0, 0] <= 0
        or k[1, 1] <= 0
        or abs(np.linalg.det(k)) < 1e-10
        or not np.allclose(transform[3], [0, 0, 0, 1])
        or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-4)
    ):
        return dict(reliable=False, reason="Invalid calibration")
    y, x = np.mgrid[y0:y1, x0:x1]
    z = depth[y0:y1, x0:x1]
    valid = np.isfinite(z) & (z > 0) & (z < 10)
    if np.count_nonzero(valid) < 4:
        return dict(reliable=False, reason="Insufficient valid metric depth", count=int(valid.sum()),
                    minimum_count=4, observation_step=obs.step)
    pixels = np.stack((x[valid], y[valid], np.ones(np.count_nonzero(valid))))
    xyz = (np.linalg.inv(k) @ pixels) * z[valid]
    base = (transform[:3, :3] @ xyz + transform[:3, 3:4]).T
    return dict(
        reliable=True,
        observation_step=obs.step,
        frame="osc_arm_base",
        count=len(base),
        valid_fraction=float(valid.mean()),
        depth_iqr_m=float(np.percentile(z[valid], 75)-np.percentile(z[valid], 25)),
        eef_to_centroid_m=(np.median(base, axis=0)-np.asarray(obs.eef_control_pose['position'])).tolist()
            if obs.eef_control_pose.get('step') == obs.step and obs.eef_control_pose.get('frame') == 'osc_arm_base'
            else None,
        centroid_m=np.median(base, axis=0).tolist(),
        min_m=base.min(axis=0).tolist(),
        max_m=base.max(axis=0).tolist(),
        note="Visible surface only; no collision-free or graspability guarantee",
    )
