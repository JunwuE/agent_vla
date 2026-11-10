from dataclasses import replace

import numpy as np
import pytest
from test_models import Log, response
from test_runtime import config

from vla_agent.adapters.mock import MockEnvironment, MockPlanner, MockVLA
from vla_agent.contracts import Chunk, Decision, DecisionLabel
from vla_agent.data import Recorder, read_events
from vla_agent.decision import build_request, classify
from vla_agent.memory import WorkingMemory
from vla_agent.runtime import AgentRuntime
from vla_agent.supervision_rules import low_motion_signals


async def history(steps=80, change=None, command=0.2):
    c = config()
    obs = await MockEnvironment().reset()
    memory = WorkingMemory(c.task, obs.episode_id)
    for step in range(steps + 1):
        state = obs.state.copy()
        if change:
            change(state, step)
        value = replace(obs, step=step, sim_time=step / 20, state=state)
        memory.add(value)
        if step:
            memory.action_history.append(
                dict(
                    step=step,
                    sim_time=step / 20,
                    source="vla",
                    command=[command] + [0] * 11,
                    measured_state=state.tolist(),
                )
            )
    return c, memory


async def test_signal_requires_sustained_time_and_does_not_override_keep():
    c, memory = await history(79)
    assert low_motion_signals(memory, c.runtime) == []
    c, memory = await history(80)
    result = low_motion_signals(memory, c.runtime)
    assert result[0]["observed_seconds"] == 4 and result[0]["active_motion_command_fraction"] == 1
    assert result[0]["failure_confirmed"] is False
    memory.rule_signals = result
    frozen = memory.summary()
    memory.rule_signals.clear()
    assert frozen["rule_signals"][0]["failure_confirmed"] is False
    memory.rule_signals = frozen["rule_signals"]
    _, state = build_request(c.decision, list(memory.observations)[-5:], memory.summary(), memory.binding(1))
    assert state["memory"]["rule_signals"][0]["role"] == "review_trigger_only"
    assert (
        classify(response("KEEP"), c.decision, memory.binding(1), "possible_stagnation").label
        == DecisionLabel.KEEP
    )


@pytest.mark.parametrize(
    "kind", ["rotation", "base_translation", "base_rotation", "gripper", "back_and_forth"]
)
async def test_other_motion_is_not_reported_as_stationary(kind):
    def change(state, step):
        if kind == "rotation":
            state[3:7] = [0, 0, np.sin(step * 0.002), np.cos(step * 0.002)]
        elif kind == "base_rotation":
            state[10:14] = [0, 0, np.sin(step * 0.002), np.cos(step * 0.002)]
        elif kind == "base_translation":
            state[7] = step * 0.001
        elif kind == "gripper":
            state[14] = step * 0.0001
        else:
            state[0] = 0.02 * np.sin(step / 80 * np.pi * 2)

    c, memory = await history(change=change)
    assert low_motion_signals(memory, c.runtime) == []


@pytest.mark.parametrize(
    "case", ["new_attempt", "new_phase", "gap", "code", "missing_command", "invalid_quaternion", "disabled"]
)
async def test_insufficient_or_incompatible_evidence_does_not_trigger(case):
    c, memory = await history()
    if case == "disabled":
        c.runtime.stagnation_enabled = False
    elif case == "new_attempt":
        memory.attempt_started_step = 40
    elif case == "new_phase":
        memory.phases.started_step = 40
    elif case == "gap":
        del memory.observations[40]
    elif case == "code":
        memory.action_history[-1]["source"] = "code_policy"
    elif case == "missing_command":
        memory.action_history.pop()
    else:
        memory.observations[-1].state[3:7] = 0
    assert low_motion_signals(memory, c.runtime) == []


async def test_quaternion_sign_flip_and_intentional_hold_are_measurements_only():
    def sign_flip(state, step):
        if step % 2:
            state[3:7] *= -1

    c, memory = await history(change=sign_flip, command=0)
    signal = low_motion_signals(memory, c.runtime)[0]
    assert signal["eef_rotation_span_rad"] == 0 and signal["active_motion_command_fraction"] == 0
    assert signal["failure_confirmed"] is False


async def test_short_eef_window_reports_wrist_and_gripper_activity_without_claiming_failure():
    def change(state, step):
        state[3:7] = [0, 0, np.sin(step * 0.002), np.cos(step * 0.002)]
        state[14] = step * 0.0001

    c, memory = await history(39, change=change)
    c.runtime.stagnation_seconds = 2.0
    c.runtime.stagnation_scope = 'eef_translation'
    assert low_motion_signals(memory, c.runtime) == []
    _, memory = await history(40, change=change)
    signal = low_motion_signals(memory, c.runtime)[0]
    assert signal['sample_count'] == 41 and signal['observed_seconds'] == 2
    assert signal['eef_rotation_active'] and signal['gripper_motion_active']
    assert signal['rule'] == 'sustained_low_eef_translation' and not signal['failure_confirmed']
    c.runtime.stagnation_scope = 'whole_robot'
    assert low_motion_signals(memory, c.runtime) == []


@pytest.mark.parametrize('base_motion', ['translation', 'rotation'])
async def test_eef_translation_rule_does_not_ignore_moving_robot_base(base_motion):
    def change(state, step):
        if base_motion == 'translation':
            state[7] = step * 0.001
        else:
            state[10:14] = [0, 0, np.sin(step * 0.002), np.cos(step * 0.002)]

    c, memory = await history(change=change)
    c.runtime.stagnation_scope = 'eef_translation'
    assert low_motion_signals(memory, c.runtime) == []


async def test_low_motion_trigger_has_cooldown():
    c, memory = await history()
    c.decision.interval_steps = 1000
    log = Log()
    rt = AgentRuntime(c, None, None, None, None, log)
    rt.memory = memory
    rt.harness.obs = memory.observations[-1]
    assert rt.trigger() == "possible_stagnation"
    rt.last_decision_step = 79
    rt.last_low_motion_check_time = memory.observations[-1].sim_time
    assert rt.trigger() is None
    await rt.flora.close()


async def test_low_motion_rule_does_not_interrupt_first_vla_attempt(tmp_path):
    class VLA(MockVLA):
        async def infer(self, obs, instruction, binding):
            return Chunk(binding, np.zeros((5, 12)))

    class Keep:
        async def decide(self, observations, summary, binding, trigger):
            return Decision(binding=binding, label=DecisionLabel.KEEP, trigger=trigger)

        async def close(self):
            pass

    class Planner(MockPlanner):
        async def plan(self, *args):
            raise AssertionError("No failure or intervention was requested")

    c = config()
    c.runtime.max_steps = 100
    c.runtime.audit_interval_steps = 0
    c.decision.interval_steps = 1000
    log = Recorder(tmp_path / "episode", c)
    try:
        result = await AgentRuntime(c, MockEnvironment(), VLA(), Keep(), Planner(), log).run()
        rows = read_events(log.root)
        assert result["error"] is None and result["control_steps"] == 100 and result["interventions"] == 0
        signals = [r for r in rows if r["kind"] == "supervision_rule"]
        assert len(signals) == 1 and signals[0]["binding"]["observation_step"] == 80
        assert all(r["owner"] == "vla" for r in rows if r["kind"] == "control_claim")
        assert result["decision_checks"] == 2  # Initial check and one debounced low-motion check.
    finally:
        log.close()
