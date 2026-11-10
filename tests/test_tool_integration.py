import json

import httpx
import pytest
from test_runtime import config

from vla_agent.adapters.mock import MockDecision, MockEnvironment, MockVLA
from vla_agent.config import PlannerConfig
from vla_agent.contracts import Decision, DecisionLabel, ExecutionReview, Intervention, Source
from vla_agent.data import Recorder, read_events
from vla_agent.memory import WorkingMemory
from vla_agent.planner import AstraPlanner
from vla_agent.runtime import AgentRuntime
from vla_agent.toolkit import PlannerToolkit


async def test_responses_review_exposes_only_snapshot_tools(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_ASTRA_KEY", "fixture-only")
    c = config()
    log = Recorder(tmp_path / "episode", c)
    rt = AgentRuntime(c, MockEnvironment(), MockVLA(), MockDecision(), None, log)
    await rt.harness.start()
    memory = WorkingMemory(c.task, rt.harness.obs.episode_id)
    memory.add(rt.harness.obs)
    toolkit = PlannerToolkit.for_review(rt.harness, memory, rt.skills, c.runtime.allowed_interventions)

    def respond(request):
        payload = json.loads(request.content)
        names = {t["name"] for t in payload["tools"]}
        assert "submit_review" in names and "submit_intervention" not in names and "preview_vla" not in names
        return httpx.Response(200, json=dict(output=[dict(type="function_call", name="submit_review",
            call_id="review", arguments=json.dumps(dict(action="continue", assessment="uncertain",
                phase="approach", reason="No observed failure on the first attempt")))]))

    planner = AstraPlanner(PlannerConfig(api_key_env="TEST_ASTRA_KEY"), log, httpx.MockTransport(respond))
    context = dict(intent="review_running", binding=memory.binding(rt.harness.epoch),
                   observations=[rt.harness.obs], memory=memory.summary(),
                   allowed_interventions=c.runtime.allowed_interventions)
    try:
        result = await planner.plan(Decision(binding=context["binding"], label=DecisionLabel.REFINE), context, toolkit)
        assert isinstance(result, ExecutionReview) and result.action == "continue"
        assert rt.harness.owner == Source.VLA and rt.harness.obs.step == 0
    finally:
        await planner.close()
        log.close()


@pytest.mark.parametrize("kind", ["eef", "edit", "code"])
async def test_runtime_routes_interventions_and_holds_after_failed_verification(tmp_path, kind):
    class Planner:
        async def plan(self, decision, context, toolkit):
            if kind == "code":
                return Intervention(
                    kind="code",
                    reason="Template trial",
                    skill_id="eef_offset",
                    skill_version="v1",
                    parameters=dict(delta_position=[0.01, 0, 0], gripper_closed=False, steps=2),
                    completion_condition="object_lifted",
                )
            if kind == "eef":
                return Intervention(
                    kind="eef",
                    reason="Measured alignment correction",
                    steps=2,
                    target=dict(position=[0.01, 0, 0.1], quaternion_xyzw=[0, 0, 0, 1], gripper_closed=False),
                    completion_condition="object_lifted",
                )
            preview = await toolkit.call("preview_vla", dict(steps=2))
            return Intervention(
                kind="edit",
                reason="Correct prefix alignment",
                proposal_id=preview["proposal_id"],
                steps=2,
                edit=dict(delta_position=[0.01, 0, 0], delta_rotation_vector=[0, 0, 0], gripper="open"),
                completion_condition="object_lifted",
            )

    c = config()
    c.task.robot = "PandaOmron"  # Exercise the real embodiment contract using mock physics only.
    c.runtime.allowed_interventions += ["eef", "edit"]  # Explicit direct-control ablation.
    c.runtime.recovery_attempts = 1
    log = Recorder(tmp_path / "episode", c)
    rt = AgentRuntime(c, MockEnvironment(), MockVLA(), MockDecision(), Planner(), log)
    rt.lease = await rt.harness.start()
    rt.memory = WorkingMemory(c.task, rt.harness.obs.episode_id)
    rt.memory.add(rt.harness.obs)
    epoch = rt.harness.epoch
    await rt.handle(Decision(binding=rt.memory.binding(epoch), label=DecisionLabel.REFINE))
    assert rt.harness.obs.step == 2 and rt.harness.epoch > epoch
    assert rt.harness.owner != Source.VLA and not rt.harness.queue
    assert rt.forced_reason and not rt.harness.obs.success
    rows = read_events(log.root)
    assert any(r["kind"] == "action_executed" for r in rows)
    if kind == "edit":
        assert any(r["kind"] == "action_edit" for r in rows)
    if kind == "code":
        invocation = next(r for r in rows if r["kind"] == "skill_invocation")
        assert invocation["status"] == "template" and not invocation["verified_reuse"]
    log.close()


async def test_astra_revises_invalid_target_then_inspects_recorded_image(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_ASTRA_KEY", "fixture-only")
    c = config()
    log = Recorder(tmp_path / "episode", c)
    rt = AgentRuntime(c, MockEnvironment(), MockVLA(), MockDecision(), None, log)
    await rt.harness.start()
    await rt.harness.claim(Source.ASTRA)
    memory = WorkingMemory(c.task, rt.harness.obs.episode_id)
    memory.add(rt.harness.obs)
    toolkit = PlannerToolkit(rt.harness, memory, rt.skills, rt.vla)
    seen = []

    def handle(request):
        payload = json.loads(request.content)
        seen.append(payload)
        if len(seen) == 1:
            name, args = (
                "submit_intervention",
                dict(
                    kind="eef",
                    reason="test rejection",
                    target=dict(position=[1, 0, 0.1], quaternion_xyzw=[0, 0, 0, 1], gripper_closed=False),
                    completion_condition="object_lifted",
                ),
            )
        elif len(seen) == 2:
            name, args = "view_image", dict(camera="left", bbox=[20, 20, 35, 35])
        else:
            name, args = (
                "submit_intervention",
                dict(kind="continue", reason="Insufficient evidence for a correction"),
            )
        return httpx.Response(
            200,
            json=dict(
                status="completed",
                output=[
                    dict(
                        type="function_call",
                        name=name,
                        call_id=f"call-{len(seen)}",
                        arguments=json.dumps(args),
                    )
                ],
            ),
        )

    planner = AstraPlanner(PlannerConfig(api_key_env="TEST_ASTRA_KEY"), log, httpx.MockTransport(handle))
    binding = memory.binding(rt.harness.epoch)
    plan = await planner.plan(
        Decision(binding=binding, label=DecisionLabel.ESCALATE),
        dict(
            binding=binding,
            observations=[rt.harness.obs],
            allowed_interventions=c.runtime.allowed_interventions,
        ),
        toolkit,
    )
    assert plan.kind == "continue" and rt.harness.obs.step == 0
    error = next(x for x in seen[1]["input"] if x.get("type") == "function_call_output")
    assert json.loads(error["output"])["no_execution"] is True
    assert any(
        x.get("type") == "input_image"
        for item in seen[2]["input"]
        for x in item.get("content", [])
        if isinstance(x, dict)
    )
    assert len(list((log.root / "image_previews").glob("*.png"))) == 1
    await planner.close()
    log.close()
