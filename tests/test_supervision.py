import asyncio
import time

import numpy as np
import pytest
from test_runtime import config, make_harness

from vla_agent.adapters.mock import MockEnvironment, MockVLA
from vla_agent.contracts import Decision, DecisionLabel, ExecutionReview, Intervention, Source, Subtask
from vla_agent.data import Recorder, read_events
from vla_agent.memory import SkillLibrary
from vla_agent.runtime import AgentRuntime
from vla_agent.supervision import RunningReviewer
from vla_agent.toolkit import PlannerToolkit, planner_tools


def test_stalled_review_requires_distinct_observed_times():
    kwargs = dict(action="request_intervention", assessment="stalled_progress", phase="contact",
                  reason="Repeated pressing produces no observed door movement")
    with pytest.raises(ValueError, match="two distinct"):
        ExecutionReview(**kwargs, evidence=[dict(observation_step=10, fact="Door remains open")])
    review = ExecutionReview(**kwargs, evidence=[dict(observation_step=10, fact="Door remains open"),
                                                 dict(observation_step=40, fact="Repeated pressing; door remains open")])
    assert review.action == "request_intervention"


class Environment(MockEnvironment):
    def __init__(self, milestone=8, success_at=None):
        super().__init__()
        self.milestone, self.success_at = milestone, success_at
        self.reached = asyncio.Event()

    async def step(self, action):
        await asyncio.sleep(0.001)
        obs = await super().step(action)
        if self.step_id >= self.milestone:
            self.reached.set()
        if self.success_at is not None and self.step_id >= self.success_at:
            self.placed = True
            obs = await self.observe()
        return obs


class DecisionModel:
    async def decide(self, observations, summary, binding, trigger):
        return Decision(binding=binding, label=DecisionLabel.REFINE, trigger=trigger)

    async def close(self):
        pass


class Planner:
    async def close(self):
        pass


async def test_review_merge_preserves_new_monitor_scores_without_readding_old_snapshot(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    for i in range(8):
        memory.reward_assessments.append(dict(request_id=f'old-{i}'))

    class Review(Planner):
        async def plan(self, decision, context, toolkit):
            toolkit.memory.reward_assessments.append(dict(request_id='new-diagnostic'))
            return assessment(context['binding'].observation_step)

    reviewer = RunningReviewer(c, Review(), log, SkillLibrary(c.runtime.skill_library))
    try:
        reviewer.submit(Decision(binding=memory.binding(h.epoch), label=DecisionLabel.ESCALATE), h, memory)
        await reviewer.pending
        # Frequent background scoring has evicted the frozen review's old entries.
        for i in range(8):
            memory.reward_assessments.append(dict(request_id=f'monitor-{i}'))
        assert reviewer.poll(h, memory) is not None
        ids = [r['request_id'] for r in memory.reward_assessments]
        assert ids == [f'monitor-{i}' for i in range(1, 8)] + ['new-diagnostic']
    finally:
        await reviewer.close()
        log.close()


def assessment(step=0, *, intervene=False):
    return ExecutionReview(action="request_intervention" if intervene else "continue",
                           assessment="failed_attempt" if intervene else "progressing",
                           phase="approach", reason="fixture assessment",
                           evidence=[dict(observation_step=step, fact="fixture measurement")])


async def execute(path, planner, *, env=None, c=None, decision=None):
    c = c or config()
    c.runtime.nonblocking_review = True
    c.runtime.max_steps = 18
    c.decision.interval_steps = 100
    c.decision.max_unreviewed_steps = 0
    c.runtime.stagnation_steps = 100
    env = env or Environment()
    log = Recorder(path, c)
    rt = AgentRuntime(c, env, MockVLA(), decision or DecisionModel(), planner, log)
    try:
        result = await rt.run()
        return result, read_events(path), rt
    finally:
        log.close()


async def test_slow_review_continue_preserves_first_attempt_actions_and_control(tmp_path):
    env = Environment()

    class Review(Planner):
        async def plan(self, decision, context, toolkit):
            assert context["intent"] == "review_running"
            assert context["memory"]["attempt"]["first_attempt"]
            snapshot_step = context["binding"].observation_step
            await env.reached.wait()  # Progress would deadlock if the runtime paused here.
            assert env.step_id > snapshot_step
            assert (await toolkit.call("observe", {}))["step"] == snapshot_step
            return assessment(snapshot_step)

    result, rows, rt = await execute(tmp_path / "review", Review(), env=env)
    baseline = config()
    baseline.runtime.supervision_enabled = False
    _, base_rows, _ = await execute(tmp_path / "baseline", Review(), c=baseline)
    assert result["error"] is None and result["control_steps"] == 18
    assert result["astra_reviews"] == 1 and result["interventions"] == 0
    assert [r["owner"] for r in rows if r["kind"] == "control_claim"] == ["vla"]
    assert [r["action"] for r in rows if r["kind"] == "action_command"] == [
        r["action"] for r in base_rows if r["kind"] == "action_command"]
    assert rt.memory.attempt_number == 1 and rt.memory.instruction_version == 0
    assert rt.memory.reviews[-1]["action"] == "continue"


async def test_slow_decision_does_not_wait_at_old_unreviewed_step_limit(tmp_path):
    env = Environment()

    class SlowDecision(DecisionModel):
        async def decide(self, observations, summary, binding, trigger):
            await env.reached.wait()
            return await super().decide(observations, summary, binding, trigger)

    class Continue(Planner):
        async def plan(self, decision, context, toolkit):
            return assessment(context["binding"].observation_step)

    result, rows, _ = await execute(tmp_path / "slow", Continue(), env=env, decision=SlowDecision())
    assert result["error"] is None and result["control_steps"] == 18
    assert not any(r["kind"] == "control_claim" and r["owner"] != "vla" for r in rows)


async def test_expired_problem_requests_fresh_readonly_review_without_pausing(tmp_path):
    env = Environment()
    c = config()
    c.runtime.review_max_age_steps = 2

    class Review(Planner):
        calls = 0

        async def plan(self, decision, context, toolkit):
            self.calls += 1
            step = context["binding"].observation_step
            if self.calls == 1:
                await env.reached.wait()
                return assessment(step, intervene=True)
            assert decision.trigger == "expired_review_recheck"
            assert step >= 8
            return assessment(step)

    planner = Review()
    result, rows, _ = await execute(tmp_path / "refresh", planner, c=c, env=env)
    assert result["error"] is None and planner.calls == 2
    assert result["interventions"] == 0
    assert any(r["kind"] == "review_refresh" for r in rows)
    assert [r["owner"] for r in rows if r["kind"] == "control_claim"] == ["vla"]


@pytest.mark.parametrize("confirm", [False, True])
async def test_fresh_confirmation_cannot_reset_attempt_with_identical_refine(tmp_path, confirm):
    env = Environment()
    intents = []

    class Review(Planner):
        async def plan(self, decision, context, toolkit):
            intents.append(context["intent"])
            if context["intent"] == "review_running":
                await env.reached.wait()
                return assessment(context["binding"].observation_step, intervene=True)
            assert context["intent"] == "confirm_intervention"
            assert context["binding"].observation_step >= 8
            assert context["previous_review"]["binding"]["observation_step"] < 8
            assert context["memory"]["attempt"]["first_attempt"]
            assert toolkit.harness.owner == Source.ASTRA
            return Intervention(kind="refine", instruction_id="pick", reason="Retry the registered task") if confirm else (
                Intervention(kind="continue", reason="Latest observation no longer warrants a change"))

    result, rows, rt = await execute(tmp_path / "confirmed", Review(), env=env)
    assert result["error"] is None and intents == ["review_running", "confirm_intervention"]
    assert result["interventions"] == 1
    assert rt.memory.attempt_number == 1
    assert rt.memory.instruction == "Pick up the cup."
    assert rt.memory.instruction_version == 0 and not rt.memory.attempts
    assert not any(r['kind'] == 'attempt_interrupted' for r in rows)
    assert any(r['kind'] == 'intervention_normalized' for r in rows) == confirm
    request = next(r for r in rows if r["kind"] == "review_end")
    claim = next(r for r in rows if r["kind"] == "control_claim" and r["owner"] == "astra_local")
    assert claim["id"] > request["id"]


async def test_snapshot_tools_reject_future_frames_previews_and_program_edits(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    rt = AgentRuntime(c, h.env, MockVLA(), DecisionModel(), Planner(), log)
    tools = PlannerToolkit.for_review(h, memory, rt.skills, c.runtime.allowed_interventions)
    try:
        await h.execute(np.zeros(12), lease, memory)
        assert (await tools.call("observe", {}))["step"] == 0
        assert tools.memory.observations[-1].step == 0 and memory.observations[-1].step == 1
        with pytest.raises(ValueError):
            await tools.call("view_image", dict(camera="left", step=1))
        for name in ("preview_vla", "preview_edit", "write_program", "patch_program", "check_skill"):
            with pytest.raises(ValueError, match="read-only"):
                await tools.call(name, {})
        with pytest.raises(ValueError, match="Read-only"):
            await tools.validate_intervention(Intervention(kind="continue", reason="bypass"), ["continue"])
        await tools.call("record_note", dict(facts=["snapshot only"]))
        assert not memory.notes and tools.memory.notes[-1]["observation_step"] == 0
        with pytest.raises(ValueError, match="future"):
            tools.validate_review(assessment(1, intervene=True))
    finally:
        log.close()


@pytest.mark.parametrize("invalidated_by", ["steps", "time", "version", "attempt", "terminal"])
async def test_expired_review_never_requests_control_or_updates_memory(tmp_path, invalidated_by):
    class Review(Planner):
        async def plan(self, decision, context, toolkit):
            await toolkit.call("record_note", dict(facts=["historical note"]))
            return assessment(context["binding"].observation_step, intervene=True)

    c, log, h, lease, memory = await make_harness(tmp_path)
    c.runtime.review_max_age_steps = 1
    rt = AgentRuntime(c, h.env, MockVLA(), DecisionModel(), Review(), log)
    reviewer = RunningReviewer(c, rt.planner, log, rt.skills)
    try:
        reviewer.submit(Decision(binding=memory.binding(h.epoch), label=DecisionLabel.REFINE), h, memory)
        await reviewer.pending
        if invalidated_by == "steps":
            for _ in range(2):
                await h.execute(np.zeros(12), lease, memory)
        elif invalidated_by == "time":
            reviewer.binding.observation_time = time.monotonic() - 121
        elif invalidated_by == "version":
            memory.subtask_version += 1
        elif invalidated_by == "attempt":
            memory.attempt_id = "new-attempt"
        else:
            h.obs.terminal = True
        epoch = h.epoch
        assert reviewer.poll(h, memory) is None
        assert h.epoch == epoch and h.owner == Source.VLA
        assert not memory.reviews and not memory.notes
        assert any(r["kind"] == "review_discarded" for r in read_events(log.root))
    finally:
        await reviewer.close()
        log.close()


async def test_terminal_success_cancels_review_without_executing_late_suggestions(tmp_path):
    cancelled = asyncio.Event()

    class Review(Planner):
        async def plan(self, *args):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    result, rows, _ = await execute(tmp_path / "success", Review(), env=Environment(success_at=8))
    assert result["success"] and cancelled.is_set() and result["interventions"] == 0
    stop = next(r for r in rows if r["kind"] == "stop")
    assert not any(r["kind"] == "action_command" and r["id"] > stop["id"] for r in rows)


async def test_review_timeout_does_not_become_a_forced_recovery(tmp_path):
    class Review(Planner):
        async def plan(self, *args):
            await asyncio.Event().wait()

    c = config()
    c.planner.timeout_seconds = 0.004
    result, rows, _ = await execute(tmp_path / "timeout", Review(), c=c)
    assert result["error"] is None and result["control_steps"] == 18
    assert any(r["kind"] == "review_failed" for r in rows)
    assert not any(r["kind"] == "attempt_interrupted" for r in rows)


async def test_refine_and_replan_cannot_inject_unregistered_motion_instructions(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    tools = PlannerToolkit(h, memory, None)
    try:
        for text in ("Open the gripper, move right, lower, close, then lift.", "Pick up the cup. Then move right."):
            with pytest.raises(ValueError, match="language_catalog"):
                await tools.validate_intervention(Intervention(kind="refine", reason="test", instruction=text), ["refine"])
            with pytest.raises(ValueError, match="language_catalog"):
                await tools.validate_intervention(Intervention(kind="replan", reason="test", remaining_plan=[
                    Subtask(name="bypass", instruction=text, completion_condition="object_lifted")]), ["replan"])
        plan = Intervention(kind="refine", instruction_id="pick", reason="Restore known command")
        # An actually incorrect active instruction may still be restored.
        memory.instruction = 'Incorrect conditioning from a legacy record'
        await tools.validate_intervention(plan, ["refine"])
        assert plan.instruction == memory.current.instruction
        with pytest.raises(ValueError, match="goal/verifier"):
            await tools.validate_intervention(Intervention(kind="refine", instruction_id="place", reason="wrong goal"), ["refine"])
        memory.begin_retry("confirmed empty grasp")
        memory.resume_retry()
        assert memory.attempt_number == 2 and not memory.attempt_summary()["first_attempt"]
        assert memory.advance()
        assert memory.attempt_number == 1 and memory.attempt_summary()["first_attempt"]
    finally:
        log.close()


def test_review_tool_contract_excludes_actions_and_requires_problem_evidence():
    names = {t["name"] for t in planner_tools(review_only=True)}
    assert "submit_review" in names and "submit_intervention" not in names
    assert not names & {"preview_vla", "preview_edit", "write_program", "patch_program"}
    with pytest.raises(ValueError, match="evidence"):
        ExecutionReview(action="request_intervention", assessment="uncertain", phase="approach", reason="not done yet")
