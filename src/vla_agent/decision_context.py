"""Lossy, causal model context. Raw commands and measurements remain in Memory/Recorder."""

import math
from collections import Counter

import numpy as np


def rounded(value):
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Nonfinite value in Decision context")
        return round(value, 3)
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(v) for v in value]
    return value


def select(record, keys):
    return {key: record[key] for key in keys if key in record}


def brief(value):
    """Bound auxiliary model opinions; task instructions and physical summaries are not truncated."""
    if isinstance(value, str):
        return value if len(value) <= 400 else value[:400] + " [truncated]"
    if isinstance(value, dict):
        return {k: brief(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = [brief(v) for v in value[:4]]
        return items if len(value) <= 4 else items + ["[additional entries omitted]"]
    return rounded(value)


def execution_summary(history, end_step, attempt_start):
    steps = [row["step"] for row in history]
    if any(s > end_step for s in steps) or any(a >= b for a, b in zip(steps, steps[1:])):
        raise ValueError("Decision action history is future or out of order")
    rows = [r for r in history if r["step"] > attempt_start][-30:]
    result = dict(sample_count=len(rows), omitted_before_attempt=len(history) - len(rows))
    if not rows:
        return result
    commands = np.asarray([r["command"] for r in rows], dtype=float)
    measured = np.asarray([r["measured_state"] for r in rows], dtype=float)
    if commands.shape != (len(rows), 12) or measured.shape != (len(rows), 16):
        raise ValueError("Unexpected action/state dimensions in Decision history")
    if not np.isfinite(commands).all() or not np.isfinite(measured).all():
        raise ValueError("Nonfinite action/state in Decision history")
    result.update(
        from_step=rows[0]["step"], to_step=rows[-1]["step"],
        contiguous=all(b["step"] == a["step"] + 1 for a, b in zip(rows, rows[1:])),
        source_counts=dict(Counter(r["source"] for r in rows)),
        command_summary=dict(mean_abs=commands.__abs__().mean(axis=0).tolist(),
                             peak_abs=float(np.abs(commands).max()),
                             gripper_transitions=int(np.count_nonzero(np.diff(commands[:, 6] >= 0.5))),
                             control_mode_transitions=int(np.count_nonzero(np.diff(commands[:, 11] >= 0.5)))),
        measured_summary=dict(eef_net_m=(measured[-1, :3] - measured[0, :3]).tolist(),
                              eef_path_m=float(np.linalg.norm(np.diff(measured[:, :3], axis=0), axis=1).sum()),
                              base_net_m=(measured[-1, 7:10] - measured[0, 7:10]).tolist(),
                              gripper_qpos_change_m=(measured[-1, 14:16] - measured[0, 14:16]).tolist()),
        recent_samples=[dict(step=r["step"], source=r["source"], command=r["command"],
                             measured_eef_position_m=r["measured_state"][:3],
                             measured_gripper_qpos_m=r["measured_state"][14:16]) for r in rows[-6:]],
    )
    return rounded(result)


def compact_state(state):
    """Build a new mapping without mutating full-precision Memory, observations or image ordering."""
    memory, end = state["memory"], state["current_step"]
    observations = state["observations"]
    if any(o["step"] > end for o in observations) or any(f["step"] > end for f in state["frame_map"]):
        raise ValueError("Future observation in Decision context")
    task = memory.get("task", {})
    subtasks = task.get("subtasks", [])
    index = memory.get("subtask_index", 0)
    attempt = memory.get("attempt", {})
    context = select(memory, ("instruction", "pending_instruction", "subtask_index",
                              "subtask_version", "instruction_version"))
    context.update(
        task=select(task, ("instruction", "robot")),
        current_subtask=subtasks[index] if index < len(subtasks) else None,
        remaining_subtask_names=[s["name"] for s in subtasks[index:]],
        attempt=brief(attempt),
        notes=[brief(select(n, ("observation_step", "source", "facts", "hypotheses", "next_check")))
               for n in memory.get("notes", [])[-3:]],
        recent_reviews=[brief(select(r, ("observation_step", "action", "assessment", "phase", "reason")))
                        for r in memory.get("reviews", [])[-2:]],
        interventions=[brief(select(r, ("kind", "verified", "reason")))
                       for r in memory.get("interventions", [])[-2:]],
        reward_assessments=[dict(score=r["score"], observation_step=r["binding"]["observation_step"],
                                rationale=brief(r.get('rationale')), scope=r.get('scope', 'subtask'),
                                source="model_assessment") for r in memory.get("reward_assessments", [])[-2:]],
        phase_memory=brief(memory.get('phase_memory', {})),
        rule_signals=brief(memory.get('rule_signals', [])),
        executor=memory.get('executor', 'vla'),
        code_assignment=brief(memory.get('code_assignment')),
        motion_windows=rounded(memory.get('motion_windows', {})),
        groundings=brief(memory.get('groundings', [])[-2:]),
        retrieved_experiences=brief(memory.get('retrieved_experiences', [])[:2]),
    )
    latest = observations[-1] if observations else {}
    current = select(latest, ("step", "sim_time", "state", "terminal", "success", "depth_available"))
    current["facts"] = brief({k: v for k, v in latest.get("facts", {}).items() if k != "task_instruction"})
    return dict(
        schema="vla_agent.decision_context.v2", current_step=end,
        memory=rounded(context), current_observation=rounded(current),
        observation_events=[dict(step=o["step"], events=o.get("events", [])) for o in observations if o.get("events")],
        frame_map=rounded(state["frame_map"]),
        execution_window=execution_summary(memory.get("recent_actions", []), end, attempt.get("started_step", -1)),
        representation=dict(
            numeric_decimals=3, raw_history_in_recorder=True,
            command_layout="eef_delta3,rotation_delta3,gripper_close,base_torso4,control_mode; normalized commands",
            state_layout="eef_pos3,eef_quat_xyzw4,base_pos3,base_quat_xyzw4,gripper_qpos2; positions in metres",
            note="Measured EEF positions use the moving policy reference frame. Motion statistics are not task "
                 "progress or failure labels. Commands are not achieved motion; gripper closure is not grasp proof.",
        ),
    )
