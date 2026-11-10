"""Causal numeric evidence for review scheduling. No failure labels or action authority."""

import numpy as np


def rotation_span(quaternions):
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms < 1e-6):
        return None
    normalized = quaternions / norms[:, None]
    dots = np.abs(normalized @ normalized[0])  # q and -q denote the same orientation.
    return float(np.max(2 * np.arccos(np.clip(dots, 0, 1))))


def low_motion_signals(memory, config):
    if not config.stagnation_enabled or memory.executor != "vla":
        return []
    first = max(memory.attempt_started_step, memory.phases.started_step)
    history = [o for o in memory.observations if o.step >= first]
    if len(history) < config.stagnation_steps:
        return []
    times = np.array([o.sim_time for o in history])
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        return []
    boundary = np.flatnonzero(times <= times[-1] - config.stagnation_seconds + 1e-9)
    if not len(boundary):
        return []
    history = history[int(boundary[-1]) :]
    if len(history) < config.stagnation_steps or any(
        b.step != a.step + 1 for a, b in zip(history, history[1:])
    ):
        return []
    states = np.asarray([o.state for o in history])
    if states.shape != (len(history), 16) or not np.isfinite(states).all():
        return []
    eef_rotation = rotation_span(states[:, 3:7])
    base_rotation = rotation_span(states[:, 10:14])
    if eef_rotation is None or base_rotation is None:
        return []
    measurements = dict(
        eef_position_span_m=float(np.linalg.norm(np.ptp(states[:, :3], axis=0))),
        base_position_span_m=float(np.linalg.norm(np.ptp(states[:, 7:10], axis=0))),
        eef_rotation_span_rad=eef_rotation,
        base_rotation_span_rad=base_rotation,
        gripper_qpos_span_m=float(np.max(np.ptp(states[:, 14:16], axis=0))),
    )
    if (
        measurements["eef_position_span_m"] > config.stagnation_position_m
        or measurements["base_position_span_m"] > config.stagnation_position_m
        or base_rotation > config.stagnation_rotation_rad
    ):
        return []
    if config.stagnation_scope == "whole_robot" and (
        eef_rotation > config.stagnation_rotation_rad
        or measurements["gripper_qpos_span_m"] > config.stagnation_gripper_m
    ):
        return []
    start, end = history[0], history[-1]
    rows = [r for r in memory.action_history if start.step < r["step"] <= end.step]
    if [r["step"] for r in rows] != list(range(start.step + 1, end.step + 1)) or any(
        r["source"] != "vla" for r in rows
    ):
        return []  # Never interpret code control, a pause, or a missing command history as VLA stagnation.
    commands = np.asarray([r["command"] for r in rows])
    if commands.shape != (len(rows), 12) or not np.isfinite(commands).all():
        return []
    motion = commands[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10]]
    fraction = float(np.mean(np.max(np.abs(motion), axis=1) > config.stagnation_command_threshold))
    return [
        dict(
            rule=("sustained_low_eef_translation" if config.stagnation_scope == "eef_translation"
                  else "sustained_low_motion"),
            scope=config.stagnation_scope,
            source="measured_history",
            episode_id=memory.episode_id,
            subtask_version=memory.subtask_version,
            attempt_id=memory.attempt_id,
            start_step=start.step,
            end_step=end.step,
            observed_seconds=end.sim_time - start.sim_time,
            sample_count=len(history),
            **measurements,
            eef_rotation_active=eef_rotation > config.stagnation_rotation_rad,
            gripper_motion_active=measurements["gripper_qpos_span_m"] > config.stagnation_gripper_m,
            active_motion_command_fraction=fraction,
            motion_command_threshold=config.stagnation_command_threshold,
            thresholds=dict(
                position_m=config.stagnation_position_m,
                rotation_rad=config.stagnation_rotation_rad,
                gripper_m=config.stagnation_gripper_m,
                duration_seconds=config.stagnation_seconds,
            ),
            role="review_trigger_only",
            failure_confirmed=False,
            note="Low translation is not task failure. Wrist rotation or gripper changes may be normal alignment; "
            "in eef_translation mode they are reported, not used to veto review. "
            "Holding, settling, contact or scene motion may explain it; "
            "commanded motion is not a proven expected physical displacement. Review visual phase evidence.",
        )
    ]
