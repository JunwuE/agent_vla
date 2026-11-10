"""Offline calibration check using native renders, never exposed as an agent tool."""

import argparse
import json
from pathlib import Path

import numpy as np

from vla_agent.adapters.robocasa_worker import CAMERAS, snapshot
from vla_agent.perception import local_geometry


def main(output):
    import gymnasium as gym
    import robocasa  # noqa: F401
    from robosuite.utils.camera_utils import get_real_depth_map

    output.mkdir(parents=True, exist_ok=False)
    env = gym.make(
        "robocasa/CloseToasterOvenDoor",
        split="target",
        seed=7,
        horizon=450,
        camera_widths=256,
        camera_heights=256,
        camera_depths=True,
    )
    try:
        raw, info = env.reset(seed=7)
        native = env.unwrapped.env
        checks, transforms = [], []
        for step in range(4):
            obs = snapshot(env, raw, "calibration-smoke", step, False, False, True)
            current = {}
            for camera, name in CAMERAS.items():
                rgb, depth = native.sim.render(camera_name=name, width=256, height=256, depth=True)
                assert np.array_equal(obs.rgb[camera], rgb[::-1])
                np.testing.assert_allclose(
                    obs.depth_m[camera], get_real_depth_map(native.sim, depth[::-1]), rtol=1e-5
                )
                k = np.array(obs.calibration[camera]["intrinsics"])
                t = np.array(obs.calibration[camera]["camera_to_base"])
                uv = np.array([128.0, 128.0, 1.0])
                xyz = np.linalg.inv(k) @ uv * obs.depth_m[camera][128, 128]
                base = t[:3, :3] @ xyz + t[:3, 3]
                projection = k @ (t[:3, :3].T @ (base - t[:3, 3]))
                residual = float(np.linalg.norm(projection[:2] / projection[2] - uv[:2]))
                assert residual < 1e-6
                geometry = local_geometry(obs, camera, [120, 120, 136, 136])
                assert geometry["reliable"]
                checks.append(
                    dict(
                        step=step,
                        camera=camera,
                        rgb_matches_native=True,
                        depth_matches_native=True,
                        reprojection_error_px=residual,
                        geometry=geometry,
                    )
                )
                current[camera] = t
            transforms.append(current)
            np.savez_compressed(
                output / f"{step}.npz",
                **{f"{cam}_rgb": obs.rgb[cam] for cam in CAMERAS},
                **{f"{cam}_depth_m": obs.depth_m[cam] for cam in CAMERAS},
            )
            (output / f"{step}.json").write_text(json.dumps(obs.calibration))
            if step < 3:
                raw, _, _, _, info = env.step(
                    {
                        "action.end_effector_position": np.array([0.05, 0, 0]),
                        "action.end_effector_rotation": np.zeros(3),
                        "action.gripper_close": np.zeros(1),
                        "action.base_motion": np.zeros(4),
                        "action.control_mode": np.zeros(1),
                    }
                )
        movement = float(np.linalg.norm(transforms[-1]["wrist"] - transforms[0]["wrist"]))
        assert movement > 1e-8
        (output / "result.json").write_text(
            json.dumps(
                dict(
                    checks=checks,
                    wrist_transform_change=movement,
                    limitation="Renderer alignment and numerical roundtrip only; not target detection, cross-view feature matching or contact accuracy.",
                ),
                indent=2,
            )
        )
        print("PASS:", len(checks), "camera/frame checks; moving wrist calibration updated")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
