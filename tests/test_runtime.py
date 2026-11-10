import asyncio
from pathlib import Path

import numpy as np
import pytest

from vla_agent.adapters.mock import MockDecision, MockEnvironment, MockPlanner, MockVLA
from vla_agent.cli import run
from vla_agent.config import load_config
from vla_agent.contracts import Chunk, Decision, DecisionLabel, Source
from vla_agent.data import Recorder, export_candidates, read_events
from vla_agent.harness import Harness, StaleControl
from vla_agent.memory import WorkingMemory
from vla_agent.recovery import RecoveryRunner
from vla_agent.runtime import AgentRuntime

ROOT = Path(__file__).parents[1]


async def test_initial_astra_plan_precedes_any_vla_inference(tmp_path):
    from vla_agent.contracts import Intervention

    order = []

    class Planner:
        async def plan(self, decision, context, toolkit):
            assert decision.trigger == "initialization"
            assert context["intent"] == "initialize_plan"
            assert context["allowed_interventions"] == ["replan", "stop"]
            assert toolkit.harness.obs.step == 0
            order.append("planner")
            return Intervention(kind="replan", reason="initial plan", remaining_plan=c.task.subtasks)

        async def close(self):
            pass

    class VLA(MockVLA):
        async def infer(self, *args):
            order.append("vla")
            return await super().infer(*args)

    class Keep:
        async def decide(self, observations, summary, binding, trigger):
            return Decision(binding=binding, label=DecisionLabel.KEEP)

        async def close(self):
            pass

    c = config()
    c.runtime.initial_planning = True
    c.runtime.max_steps = 1
    log = Recorder(tmp_path / "episode", c)
    try:
        result = await AgentRuntime(c, MockEnvironment(), VLA(), Keep(), Planner(), log).run()
        assert not result["error"] and order == ["planner", "vla"]
    finally:
        log.close()


async def test_environment_language_is_authoritative_before_planning(tmp_path):
    from vla_agent.contracts import Intervention

    class Environment(MockEnvironment):
        async def observe(self):
            obs = await super().observe()
            obs.facts["task_instruction"] = "Place the toasted item on a plate."
            return obs

    class Planner:
        async def plan(self, decision, context, toolkit):
            task = context["memory"]["task"]
            assert task["instruction"] == "Place the toasted item on a plate."
            assert task["subtasks"][0]["instruction"] == task["instruction"]
            return Intervention(kind="stop", reason="verified instruction propagation")

        async def close(self):
            pass

    c = config()
    c.environment.instruction_source = "environment"
    c.runtime.initial_planning = True
    log = Recorder(tmp_path / "episode", c)
    try:
        result = await AgentRuntime(c, Environment(), MockVLA(), MockDecision(), Planner(), log).run()
        assert result["reason"] == "stopped" and result["control_steps"] == 0 and not result["error"]
    finally:
        log.close()


async def test_execution_between_reviews_resets_no_progress_budget(tmp_path):
    from vla_agent.contracts import Intervention

    class Planner:
        async def plan(self, *args):
            return Intervention(kind="continue", reason="reviewed")

    c = config()
    c.runtime.max_no_progress_interventions = 1
    log = Recorder(tmp_path / "episode", c)
    rt = AgentRuntime(c, MockEnvironment(), MockVLA(), MockDecision(), Planner(), log)
    rt.lease = await rt.harness.start()
    rt.memory = WorkingMemory(c.task, rt.harness.obs.episode_id)
    rt.memory.add(rt.harness.obs)
    try:
        for _ in range(3):
            await rt.handle(Decision(binding=rt.memory.binding(rt.harness.epoch), label=DecisionLabel.ESCALATE))
            await rt.harness.execute(np.zeros(12), rt.lease, rt.memory)
        assert rt.harness.obs.step == 3
        await rt.handle(Decision(binding=rt.memory.binding(rt.harness.epoch), label=DecisionLabel.ESCALATE))
        with pytest.raises(RuntimeError, match="no execution progress"):
            await rt.handle(Decision(binding=rt.memory.binding(rt.harness.epoch), label=DecisionLabel.ESCALATE))
    finally:
        log.close()


def config():
    c = load_config(ROOT / "configs/demo.yaml")
    c.runtime.skill_library = str(ROOT / "skills")
    return c


@pytest.mark.parametrize("wall_seconds", [None, 10])
async def test_component_timeout_is_not_misreported_as_wall_budget(tmp_path, wall_seconds):
    class TimedOutVLA(MockVLA):
        async def infer(self, *args):
            raise TimeoutError("policy request deadline")

    c = config()
    c.runtime.initial_planning = False
    c.runtime.supervision_enabled = False
    c.runtime.max_wall_seconds = wall_seconds
    log = Recorder(tmp_path / "episode", c)
    try:
        result = await AgentRuntime(c, MockEnvironment(), TimedOutVLA(), MockDecision(), MockPlanner(), log).run()
        assert result["reason"] == "component_timeout"
        assert result["control_steps"] == 0
        assert "policy request deadline" in result["error"]
    finally:
        log.close()


async def test_actual_wall_budget_expiration_is_identified(tmp_path):
    class SlowVLA(MockVLA):
        async def infer(self, *args):
            await asyncio.sleep(10)

    c = config()
    c.runtime.initial_planning = False
    c.runtime.supervision_enabled = False
    c.runtime.max_wall_seconds = 0.05
    log = Recorder(tmp_path / "episode", c)
    try:
        result = await AgentRuntime(c, MockEnvironment(), SlowVLA(), MockDecision(), MockPlanner(), log).run()
        assert result["reason"] == "wall_time_budget"
        assert result["control_steps"] == 0
    finally:
        log.close()


async def test_complete_recovery_and_verified_subtask_transition(tmp_path):
    episode = tmp_path / "episode"
    result = await run(config(), episode)
    assert result["success"] and result["interventions"] == 2 and result["astra_calls"] == 0
    rows = read_events(episode)
    decisions = [r["decision"]["label"] for r in rows if r["kind"] == "decision"]
    assert "REFINE" in decisions
    assert any(r["kind"] == "code_subtask_complete" for r in rows)
    actions = [r for r in rows if r["kind"] == "action_executed"]
    assert any(r["source"] == "code_policy" for r in actions)
    assert any("Place" in r["instruction"] for r in actions)
    assert any(r["kind"] == "control_claim" and r["owner"] == "code_policy" for r in rows)
    count = export_candidates(episode, tmp_path / "candidates.jsonl")
    assert 0 < count < len(actions)
    for row in actions:
        before = np.load(episode / row["before"])
        after = np.load(episode / row["after"])
        assert before["state"].shape == after["state"].shape == (16,)


async def test_astra_budget_stops_without_bypassing_non_keep(tmp_path):
    c = config()
    c.runtime.max_interventions = 1
    episode = tmp_path / "budget"
    result = await run(c, episode)
    assert result["reason"] == "astra_budget" and result["error"] is None
    rows = read_events(episode)
    budget = next(r for r in rows if r["kind"] == "budget_exhausted")
    assert not any(r["kind"] == "action_command" and r["id"] > budget["id"] for r in rows)


async def make_harness(tmp_path, *, limit=100):
    c = config()
    log = Recorder(tmp_path / "episode", c)
    h = Harness(MockEnvironment(), log, limit)
    lease = await h.start()
    memory = WorkingMemory(c.task, h.obs.episode_id)
    memory.add(h.obs)
    return c, log, h, lease, memory


async def test_takeover_fences_old_chunk_and_action(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    old = Chunk(memory.binding(h.epoch), np.zeros((3, 12)))
    h.enqueue(old, lease, memory, 3)
    correction = await h.claim(Source.CODE)
    assert not h.queue
    with pytest.raises(StaleControl):
        await h.execute(np.zeros(12), lease, memory)
    with pytest.raises(StaleControl):
        h.enqueue(old, correction, memory, 3)
    assert h.obs.step == 0
    log.close()


async def test_language_applies_only_at_inference_boundary(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    old = memory.binding(h.epoch)
    memory.pending_instruction = "Align before grasping"
    assert memory.instruction == "Pick up the cup."
    memory.inference_boundary()
    assert memory.instruction == "Align before grasping" and not memory.matches(old, h.epoch)
    log.close()


async def test_recovery_claim_cannot_assert_success(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    result = await RecoveryRunner(h, memory, c.runtime, log).run(
        'def run(ctx):\n    return "RECOVERED"\n', "object_lifted", "false-success"
    )
    assert result["result"] == "UNRESOLVED" and result["verified"] is False and h.obs.step == 0
    log.close()


async def test_recovery_timeout_revokes_authority(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    c.runtime.recovery_seconds = 0.1
    result = await RecoveryRunner(h, memory, c.runtime, log).run(
        "def run(ctx):\n    while True:\n        pass\n", "object_lifted", "timeout"
    )
    assert result["result"] == "UNRESOLVED" and "TimeoutError" in result["error"]
    with pytest.raises(StaleControl):
        await h.execute(np.zeros(12), lease, memory)
    assert h.obs.step == 0
    log.close()


async def test_recovery_respects_global_episode_budget(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path, limit=2)
    result = await RecoveryRunner(h, memory, c.runtime, log).run(
        "def run(ctx):\n    for _ in range(20):\n        ctx.act([0]*12)\n    return 'RECOVERED'\n",
        "object_lifted",
        "budget",
    )
    assert h.obs.step == 2 and result["result"] == "UNRESOLVED"
    log.close()


async def test_uncertain_execution_never_retries(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    attempts = 0

    async def lost_ack(action):
        nonlocal attempts
        attempts += 1
        raise ConnectionError("acknowledgment lost")

    h.env.step = lost_ack
    with pytest.raises(ConnectionError):
        await h.execute(np.zeros(12), lease, memory)
    with pytest.raises(StaleControl):
        await h.execute(np.zeros(12), lease, memory)
    assert attempts == 1 and h.uncertain
    log.close()


async def test_stale_keep_is_escalated(tmp_path):
    c = config()
    log = Recorder(tmp_path / "episode", c)
    rt = AgentRuntime(c, MockEnvironment(), MockVLA(), MockDecision(), MockPlanner(), log)
    rt.lease = await rt.harness.start()
    rt.memory = WorkingMemory(c.task, rt.harness.obs.episode_id)
    rt.memory.add(rt.harness.obs)
    binding = rt.memory.binding(rt.harness.epoch)

    async def answer():
        return Decision(binding=binding, label=DecisionLabel.KEEP)

    rt.pending = asyncio.create_task(answer())
    await rt.harness.claim(Source.ASTRA)
    result = await rt.receive_decision(wait=True)
    assert result.label == DecisionLabel.ESCALATE and result.trigger == "stale_response"
    log.close()


async def test_baseline_does_not_call_supervisor(tmp_path):
    c = config()
    c.runtime.supervision_enabled = False
    c.runtime.max_steps = 6
    result = await run(c, tmp_path / "baseline")
    assert result["control_steps"] == 6 and result["interventions"] == 0
    assert not any(r["kind"] == "decision" for r in read_events(tmp_path / "baseline"))


async def test_missing_simulator_fails_without_model_calls(tmp_path):
    c = load_config(ROOT / "configs/robocasa.yaml")
    c.environment.source_path = None
    c.environment.python = None
    with pytest.raises(ValueError, match="not configured"):
        await run(c, tmp_path / "real")
    assert not (tmp_path / "real").exists()
