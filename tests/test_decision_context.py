import copy
import json

import numpy as np
import pytest

from vla_agent.decision_context import compact_state


def context():
    history = []
    for step in range(1, 31):
        state = np.zeros(16)
        state[0] = (step % 2) * 0.123456789123
        state[7] = step * 0.01
        state[14:16] = [0.02012345678, -0.02012345678]
        history.append(dict(step=step, source="vla", command=np.zeros(12).tolist(), measured_state=state.tolist()))
    return dict(current_step=30, memory=dict(
        task=dict(instruction="Open the drawer.", robot="PandaOmron", subtasks=[
            dict(name="drawer", instruction="Open the drawer.", completion_condition="task_success")]),
        instruction="Open the drawer.", subtask_index=0,
        attempt=dict(id="attempt-a", first_attempt=True, number=1, started_step=0), recent_actions=history),
        observations=[dict(step=30, sim_time=1.5, state=history[-1]["measured_state"], events=[],
                           facts=dict(native_horizon=750), terminal=False, success=False)],
        frame_map=[dict(step=30, camera="left", image_index=0, sim_time=1.5)])


def test_compaction_preserves_raw_data_and_separates_commands_from_actual_motion():
    original = context()
    before = copy.deepcopy(original)
    result = compact_state(original)
    assert original == before
    assert result["memory"]["attempt"]["first_attempt"] is True
    assert result["memory"]["current_subtask"]["completion_condition"] == "task_success"
    assert result["frame_map"] == original["frame_map"]
    window = result["execution_window"]
    assert window["sample_count"] == 30 and len(window["recent_samples"]) == 6
    assert window["command_summary"]["peak_abs"] == 0
    assert window["measured_summary"]["eef_path_m"] > 3  # Actual motion despite zero commands.
    assert window["measured_summary"]["base_net_m"] == [0.29, 0.0, 0.0]
    assert "recent_actions" not in result["memory"]
    assert "stagnated" not in json.dumps(result)  # No manufactured progress/failure judgment.
    assert original["observations"][0]["state"][14] == 0.02012345678
    assert result["current_observation"]["state"][14] == 0.02


def test_retry_history_is_separated_and_small_samples_are_not_padded():
    original = context()
    original["memory"]["attempt"].update(started_step=28, first_attempt=False, number=2)
    result = compact_state(original)
    assert result["execution_window"]["sample_count"] == 2
    assert result["execution_window"]["from_step"] == 29
    assert result["execution_window"]["omitted_before_attempt"] == 28
    assert result["memory"]["attempt"]["number"] == 2


@pytest.mark.parametrize("bad", ["future_action", "future_image", "nonfinite", "out_of_order"])
def test_invalid_history_does_not_produce_a_plausible_summary(bad):
    original = context()
    if bad == "future_action":
        original["memory"]["recent_actions"][-1]["step"] = 31
    elif bad == "future_image":
        original["frame_map"][0]["step"] = 31
    elif bad == "nonfinite":
        original["memory"]["recent_actions"][-1]["command"][0] = float("nan")
    else:
        original["memory"]["recent_actions"].reverse()
    with pytest.raises(ValueError):
        compact_state(original)


def test_context_keeps_success_hypotheses_and_assessments_distinct():
    original = context()
    original["memory"].update(
        notes=[dict(observation_step=30, source="astra", facts=["gripper closed"],
                    hypotheses=["may hold the object"], next_check="verify lifting")],
        reward_assessments=[dict(score=5, binding=dict(observation_step=25))])
    result = compact_state(original)
    assert result["current_observation"]["success"] is False
    assert result["memory"]["notes"][0]["hypotheses"] == ["may hold the object"]
    assert result["memory"]["reward_assessments"][0]["source"] == "model_assessment"
    assert result["memory"]["reward_assessments"][0]["observation_step"] == 25
