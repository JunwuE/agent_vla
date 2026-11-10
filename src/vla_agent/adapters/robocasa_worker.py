"""RoboCasa worker; public task description, RGB/depth, proprioception and native success."""

import json
import sys
import time
import uuid

import numpy as np

from vla_agent.adapters.robocasa import decode, encode, observation_wire
from vla_agent.adapters.robocasa_control import read_robot_control
from vla_agent.adapters.robocasa_geometry import camera_calibration
from vla_agent.contracts import Observation

CAMERAS = {"left": "robot0_agentview_left", "right": "robot0_agentview_right", "wrist": "robot0_eye_in_hand"}
STATE = [
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.base_position",
    "state.base_rotation",
    "state.gripper_qpos",
]


def snapshot(env, obs, episode_id, step, terminal, success, depth_enabled, subtask_checks=None):
    native = env.unwrapped.env
    depths, calibration = {}, {}
    facts = {"task_instruction": str(obs["annotation.human.task_description"])}
    if subtask_checks:
        from vla_agent.adapters.subtask_catalog import native_checks

        facts['subtask_checks'] = native_checks(native, subtask_checks)
        facts['subtask_check_source'] = 'simulator_state_boolean_verifiers'
    try:
        controller, pose = read_robot_control(env, step)
    except (AttributeError, KeyError, ValueError) as error:
        controller, pose = {}, {}
        facts["controller_unavailable"] = str(error)
    if depth_enabled:
        from robosuite.utils.camera_utils import get_real_depth_map

        for camera, name in CAMERAS.items():
            # Do not substitute object ground truth or guessed depth for unavailable observations.
            key = f"video.{name}_depth"
            if key in obs:
                raw = np.asarray(obs[key]).squeeze()
                depths[camera] = get_real_depth_map(native.sim, raw)
                try:
                    if raw.shape != np.asarray(obs[f"video.{name}"]).shape[:2] or not pose:
                        raise ValueError('RGB-D shape or controller frame unavailable')
                    calibration[camera] = camera_calibration(native, name, raw.shape, step)
                except (ValueError, AttributeError, KeyError) as error:
                    calibration[camera] = dict(step=step, alignment_verified=False, reason=str(error))
    return Observation(
        episode_id,
        step,
        time.monotonic(),
        step / float(native.control_freq),
        np.concatenate([obs[k] for k in STATE]),
        {camera: np.asarray(obs[f"video.{name}"]) for camera, name in CAMERAS.items()},
        depth_m=depths,
        calibration=calibration,
        controller=controller,
        eef_control_pose=pose,
        facts=facts,
        terminal=terminal,
        success=success,
    )


def main():
    output = sys.stdout
    sys.stdout = sys.stderr  # Simulator import/startup diagnostics cannot corrupt the protocol.
    config = json.loads(sys.argv[1])
    import gymnasium as gym
    import robocasa  # noqa: F401
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    if config["task_group"] and config["task_name"] not in TASK_SET_REGISTRY[config["task_group"]]:
        raise ValueError("Task is not part of the configured benchmark task group")

    np.random.seed(config["seed"])
    horizon = int(get_task_horizon(config["task_name"]))
    ignore_horizon = config.get("ignore_native_horizon", False)
    env = gym.make(
        f"robocasa/{config['task_name']}",
        split=config["scene_split"],
        seed=config["seed"],
        horizon=horizon,
        camera_widths=config["camera_size"],
        camera_heights=config["camera_size"],
        camera_depths=config["enable_depth"],
    )
    # RoboCasa create_env already supplies ignore_done; passing it through gym.make
    # duplicates that keyword in the shared installation. Configure only our instance.
    if ignore_horizon:
        env.unwrapped.env.ignore_done = True
    episode_id, step, obs = uuid.uuid4().hex, 0, None
    terminal, success = False, False
    try:
        for line in sys.stdin:
            request = json.loads(line, object_hook=decode)
            response = {"id": request["id"]}
            try:
                operation_started = time.perf_counter()
                if request["op"] == "reset" and obs is None:
                    obs, info = env.reset(seed=config["seed"])
                    success = bool(info.get("success", False))
                    terminal = success
                elif (
                    request["op"] == "step"
                    and obs is not None
                    and not terminal
                    and request["episode_id"] == episode_id
                    and request["step"] == step
                ):
                    a = request["action"]
                    native_action = {
                        "action.end_effector_position": a[:3],
                        "action.end_effector_rotation": a[3:6],
                        "action.gripper_close": a[6:7],
                        "action.base_motion": a[7:11],
                        "action.control_mode": a[11:12],
                    }
                    obs, reward, terminated, truncated, info = env.step(native_action)
                    step += 1
                    success = bool(info.get("success", False))
                    terminal = bool(success or terminated or truncated or (not ignore_horizon and step >= horizon))
                else:
                    raise ValueError("Invalid or stale simulator request")
                operation_seconds = time.perf_counter() - operation_started
                snapshot_started = time.perf_counter()
                value = snapshot(env, obs, episode_id, step, terminal, success, config["enable_depth"],
                                 config['task_name'] if config.get('native_subtask_checks') else None)
                value.facts["native_horizon"] = horizon
                value.facts["native_horizon_ignored"] = ignore_horizon
                if terminal:
                    value.facts["termination_reason"] = "success" if success else (
                        "horizon" if step >= horizon and not ignore_horizon else "environment_done"
                    )
                response["result"] = observation_wire(value)
                # Diagnostics stay outside Observation, so prompts and policy inputs are unchanged.
                response["timing"] = dict(
                    simulator_seconds=operation_seconds,
                    snapshot_seconds=time.perf_counter() - snapshot_started,
                )
            except Exception as error:
                response["error"] = f"{type(error).__name__}: {error}"
            output.write(json.dumps(response, default=encode, allow_nan=False) + "\n")
            output.flush()
            if "error" in response:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
