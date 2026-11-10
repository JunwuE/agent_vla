import asyncio
import copy
import json
from dataclasses import replace

import httpx
import pytest
from pydantic import ValidationError
from test_exploration import runtime_fixture
from test_flora import advance, answer
from test_runtime import config, make_harness

from vla_agent.config import DecisionConfig, FloraConfig
from vla_agent.contracts import Decision, DecisionLabel, Intervention, Source
from vla_agent.experience import ExperienceStore
from vla_agent.flora import FloraEvaluator
from vla_agent.grounding import GroundTarget, ground_target, grounded_geometry
from vla_agent.memory import SkillLibrary
from vla_agent.perception import decision_window
from vla_agent.phases import PhaseAssessment, PhaseMemory, PhasePlan
from vla_agent.policy_candidates import PolicyPatch, PolicyProposal, apply_experimental_patch, save_candidate
from vla_agent.recovery import RecoveryRunner
from vla_agent.toolkit import PlannerToolkit


def phase_plan():
    return PhasePlan(
        expected_revision=0,
        active_phase="approach",
        reason="fixture",
        phases=[
            dict(
                id="approach",
                goal="Approach the handle",
                progress_signals=["gap decreases"],
                completion_signals=["aligned with visible handle"],
            ),
            dict(
                id="close",
                goal="Close the door",
                progress_signals=["door moves"],
                completion_signals=["door remains closed"],
            ),
        ],
    )


def test_phase_evidence_is_causal_versioned_and_not_a_verifier():
    phases = PhaseMemory()
    phases.set_plan(phase_plan(), 5)
    invalid = PhaseAssessment(
        revision=1,
        phase_id="approach",
        status="complete",
        evidence=[dict(observation_step=9, fact="aligned")],
        next_phase="close",
    )
    with pytest.raises(ValueError, match="recorded"):
        phases.assess(invalid, 8, {5, 6, 7, 8})
    valid = invalid.model_copy(
        update={"evidence": [invalid.evidence[0].model_copy(update={"observation_step": 8})]}
    )
    result = phases.assess(valid, 8, {5, 6, 7, 8})
    assert not result["independently_verified"] and phases.active == "close"
    assert phases.started_step == 8 and phases.revision == 2
    with pytest.raises(ValueError, match="Stale"):
        phases.assess(valid, 9, {8, 9})


async def test_multiscale_window_spans_ten_seconds_and_respects_attempt(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    obs = h.obs
    for i in range(301):
        memory.add(replace(obs, step=i, sim_time=i / 20, timestamp=obs.timestamp + i / 20))
    cfg = DecisionConfig()
    assert [o.step for o in decision_window(memory, cfg)] == [100, 200, 260, 290, 300]
    cfg.window_frames = 1
    assert [o.step for o in decision_window(memory, cfg)] == [300]
    memory.attempt_started_step = 295
    cfg.window_frames = 5
    assert [o.step for o in decision_window(memory, cfg)] == [295, 300]
    log.close()


async def test_grounding_identity_and_measured_geometry(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    selection = ground_target(
        h.obs,
        GroundTarget(
            target="fixture surface", camera="left", observation_step=0, bbox=[20, 20, 30, 30], confidence=0.7
        ),
        "test",
    )
    result = grounded_geometry(h.obs, selection)
    assert result["reliable"] and result["valid_fraction"] == 1
    assert result["observation_step"] == 0
    changed = replace(h.obs, rgb={**h.obs.rgb, "left": h.obs.rgb["left"].copy()})
    changed.rgb["left"][0, 0, 0] ^= 1
    with pytest.raises(ValueError, match="identity"):
        grounded_geometry(changed, selection)
    await advance(h, lease, memory, 1)
    with pytest.raises(ValueError, match="Stale"):
        grounded_geometry(h.obs, selection)
    log.close()


async def test_batched_geometry_uses_one_snapshot_and_validates_before_recording(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    toolkit = PlannerToolkit(h, memory, SkillLibrary(tmp_path / 'skills'))
    targets = [dict(target='rim', camera='left', observation_step=0, bbox=[20, 20, 30, 30], confidence=0.7),
               dict(target='hinge', camera='right', observation_step=0, bbox=[10, 10, 20, 20], confidence=0.6)]
    try:
        result = await toolkit.call('measure_targets', dict(targets=targets))
        assert not result['executed'] and h.obs.step == 0
        for entry in result['measurements']:
            assert entry['geometry'] == grounded_geometry(h.obs, entry['grounding'])
        before = copy.deepcopy(memory.groundings)
        with pytest.raises(ValueError, match='matching observation'):
            await toolkit.call('measure_targets', dict(targets=[targets[0], dict(targets[1], observation_step=1)]))
        assert memory.groundings == before and h.obs.step == 0
        reviewer = PlannerToolkit.for_review(h, memory, toolkit.skills, c.runtime.allowed_interventions)
        await advance(h, lease, memory, 1)
        frozen = await reviewer.call('measure_targets', dict(targets=targets))
        assert frozen['observation_step'] == 0 and h.obs.step == 1
    finally:
        log.close()


def test_bbox_schema_has_integer_items_and_rejects_coercion():
    schema = GroundTarget.model_json_schema()['properties']['bbox']
    assert schema['items'] == {'type': 'integer'} and schema['minItems'] == schema['maxItems'] == 4
    for bbox in [[1, 2, 3], [1, 2, 3, 4, 5], ['1', '2', '3', '4'], [True, 2, 3, 4], [1.5, 2, 3, 4]]:
        with pytest.raises(ValidationError):
            GroundTarget(target='fixture', camera='left', observation_step=0, bbox=bbox, confidence=0.5)


async def test_landlock_worker_reads_exports_but_cannot_modify_host_or_network(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    protected = tmp_path / "host-secret"
    protected.write_text("unchanged")
    code = f"""def run(ctx):
    import os, socket, subprocess, resource
    from pathlib import Path
    raw = ctx.read_observation('left')
    assert raw['depth_m'].shape == (64,64)
    attempts = [lambda: Path({str(protected)!r}).read_text(),
                lambda: Path({str(protected)!r}).write_text('changed'),
                lambda: os.truncate({str(protected)!r}, 0),
                lambda: os.chmod({str(protected)!r}, 0),
                lambda: socket.socket(), lambda: subprocess.run(['/bin/true']),
                lambda: resource.prlimit(os.getppid(), resource.RLIMIT_CPU, (0,0))]
    for action in attempts:
        try:
            action()
        except (PermissionError, OSError):
            pass
        else:
            raise AssertionError('isolation bypass')
    ctx.note(facts=['isolation checks passed'], hypotheses=[])
    return 'UNRESOLVED'
"""
    report = await RecoveryRunner(h, memory, c.runtime, log).run(code, "object_lifted", "isolation")
    assert report["error"] is None, report
    assert protected.read_text() == "unchanged" and h.obs.step == 0
    assert memory.notes[-1]["facts"] == ["isolation checks passed"]
    log.close()


async def test_unfinished_code_owns_subtask_across_outer_turns_and_cannot_rename_budget(tmp_path):
    class Planner:
        calls = 0

        async def plan(self, decision, context, toolkit):
            self.calls += 1
            if self.calls == 1:
                return Intervention(
                    kind="code",
                    reason="trial",
                    completion_condition="fixture_aligned",
                    code="def run(ctx):\n    ctx.act([0]*12)\n    return 'RECOVERED'\n",
                )
            assert "code" not in context["allowed_interventions"]
            return Intervention(kind="stop", reason="budget exhausted")

    c, log, rt = await runtime_fixture(tmp_path, Planner())
    c.runtime.recovery_attempts = 1
    decision = Decision(binding=rt.memory.binding(rt.harness.epoch), label=DecisionLabel.ESCALATE)
    await rt.handle(decision)
    assert rt.memory.executor == "code" and rt.harness.owner != Source.VLA
    assert rt.code_session.attempts == 1 and rt.harness.obs.step == 1
    tools = PlannerToolkit(rt.harness, rt.memory, rt.skills)
    alias = rt.memory.current.model_copy(update={"name": "renamed"})
    with pytest.raises(ValueError, match="same unfinished"):
        await tools.validate_intervention(
            Intervention(kind="replan", reason="alias", remaining_plan=[alias]), ["replan"]
        )
    with pytest.raises(ValueError, match="cannot resume"):
        await tools.validate_intervention(Intervention(kind="continue", reason="early handoff"), ["continue"])
    await rt.handle(decision)
    assert rt.harness.halted and rt.harness.obs.step == 1
    log.close()


async def test_flora_phase_rationale_has_separate_cache_and_memory_scope(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    await advance(h, lease, memory, 3)
    memory.phases.set_plan(phase_plan(), 3)
    await advance(h, lease, memory, 7)
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200, json=answer(payload) | {"with_rationale": True, "rationale": "The handle remains distant."}
        )

    client = FloraEvaluator(FloraConfig(enabled=True), log, httpx.MockTransport(handler))
    tools = PlannerToolkit.for_review(h, memory, SkillLibrary(c.runtime.skill_library), [], flora=client)
    try:
        a = await tools.call("evaluate_subtask", dict(phase_id="approach", with_rationale=True))
        b = await tools.call("evaluate_subtask", dict(with_rationale=True))
        assert a["status"] == b["status"] == "scored"
        assert a["request_id"] != b["request_id"]
        assert requests[0]["segment"]["start_step"] == 3 and requests[1]["segment"]["start_step"] == 0
        assert tools.memory.reward_assessments[0]["scope"]["kind"] == "phase"
        assert tools.memory.reward_assessments[0]["rationale"] == "The handle remains distant."
        assert h.owner == Source.VLA and memory.subtask_index == 0
    finally:
        await client.close()
        log.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("recovery_total_steps", 100000),
        ("completion_condition", "always_true"),
        ("source_path", "/tmp"),
        ("decision_history_seconds", [float("nan")]),
        ("audit_interval_steps", 1),
    ],
)
def test_policy_mutation_surface_rejects_protected_or_invalid_fields(field, value):
    with pytest.raises(ValidationError):
        PolicyPatch.model_validate({field: value})


def test_policy_proposals_are_candidates_with_no_live_effect(tmp_path):
    c = config()
    before = copy.deepcopy(c.model_dump())
    proposal = PolicyProposal(
        patch=PolicyPatch(decision_interval_steps=20),
        hypothesis="more temporal coverage",
        evidence_event_ids=[7],
        expected_effect="fewer missed reviews",
    )
    with pytest.raises(ValueError, match="existing"):
        save_candidate(tmp_path, proposal, "episode", {1})
    receipt = save_candidate(tmp_path, proposal, "episode", {7})
    assert not receipt["applied"] and c.model_dump() == before
    experiment = apply_experimental_patch(c, proposal.patch)
    assert experiment.decision.interval_steps == 20 and experiment.runtime == c.runtime


async def test_experience_keeps_negative_outcomes_and_frozen_snapshot(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    store = ExperienceStore(tmp_path / "experiences", "accumulate")
    result = dict(success=False, reason="horizon", control_steps=0, astra_calls=1)
    log.event("fixture_evidence")
    receipt = store.record(memory, result, log, [])
    assert receipt["persisted"] and not store.index
    reloaded = ExperienceStore(tmp_path / "experiences")
    rows = reloaded.search(memory.task.robot, memory.task.instruction)
    assert rows and not rows[0]["result"]["success"] and rows[0]["status"] == "observed_episode"
    record = next((tmp_path / "experiences").glob("*.json"))
    record.write_text(record.read_text().replace("horizon", "edited"))
    with pytest.raises(ValueError, match="checksum"):
        ExperienceStore(tmp_path / "experiences")
    log.close()


def test_retrieval_prioritizes_task_and_executed_recovery_without_hiding_failures(tmp_path):
    store = ExperienceStore(tmp_path)
    common = dict(robot='PandaOmron', task='Close toaster door', phase_memory={}, tools={},
                  evidence_root='fixture', evidence_sha256='fixture')
    store.index = [
        dict(common, episode_id='verbose-failure', notes=['Close toaster door stalled contact geometry'],
             code_versions=[{'verified': False}], result={'success': False}),
        dict(common, episode_id='vla-success', notes=[], code_versions=[], result={'success': True}),
        dict(common, episode_id='code-success', notes=[], code_versions=[{'verified': True}],
             result={'success': True}),
        dict(common, task='Carry cup', episode_id='unrelated', notes=['Close toaster door'],
             code_versions=[{'verified': True}], result={'success': True}),
    ]
    records = store.search('PandaOmron', 'Close toaster door')
    assert records[0]['episode_id'] == 'code-success'
    assert 'verbose-failure' in {r['episode_id'] for r in records}
    assert 'unrelated' not in {r['episode_id'] for r in records}
    # Accumulating many code successes must not crowd normal VLA success out of online review.
    store.index.extend([dict(common, episode_id=f'code-extra-{i}', notes=['Close toaster door geometry'],
                            code_versions=[{'verified': True}], result={'success': True}) for i in range(5)])
    records = store.search('PandaOmron', 'Close toaster door')
    assert len(records) == 3
    assert {'vla-success', 'verbose-failure'} <= {r['episode_id'] for r in records}


def test_experience_resolves_exact_code_to_current_robot_snapshot_without_mutating_history(tmp_path):
    store = ExperienceStore(tmp_path)
    store.index = [dict(robot='PandaOmron', task='Close toaster door', phase_memory={}, tools={},
                        evidence_root='fixture', evidence_sha256='fixture', notes=[], episode_id='old',
                        code_versions=[dict(sha256='matching', verified=True), dict(sha256='missing', verified=False)],
                        result={'success': True})]
    skills = [dict(robot='PandaOmron', sha256='matching', id='one', version='v1', status='candidate'),
              dict(robot='PandaOmron', sha256='matching', id='two', version='v2', status='candidate'),
              dict(robot='other', sha256='matching', id='wrong_robot', version='v1', status='candidate')]
    codes = store.search('PandaOmron', 'Close toaster door', skill_index=skills)[0]['code_versions']
    assert [s['id'] for s in codes[0]['available_skills']] == ['one', 'two']
    assert all(s['status'] == 'candidate' for s in codes[0]['available_skills'])
    assert codes[1]['available_skills'] == []
    assert 'available_skills' not in store.index[0]['code_versions'][0]


async def test_read_only_audit_runs_during_keep_without_claiming_control(tmp_path):
    from vla_agent.adapters.mock import MockEnvironment, MockVLA
    from vla_agent.contracts import ExecutionReview
    from vla_agent.data import Recorder, read_events
    from vla_agent.runtime import AgentRuntime

    class Keep:
        async def decide(self, observations, summary, binding, trigger):
            return Decision(binding=binding, label=DecisionLabel.KEEP)

        async def close(self):
            pass

    class Audit:
        async def plan(self, decision, context, toolkit):
            assert context["intent"] == "review_running"
            return ExecutionReview(
                action="continue", assessment="progressing", phase="approach", reason="normal first attempt"
            )

        async def close(self):
            pass

    c = config()
    c.runtime.audit_interval_steps = 3
    c.runtime.max_steps = 12
    c.runtime.max_reviews = 5
    log = Recorder(tmp_path / "episode", c)
    try:
        result = await AgentRuntime(c, MockEnvironment(), MockVLA(), Keep(), Audit(), log).run()
        rows = read_events(log.root)
        assert result["astra_reviews"] > 0 and result["interventions"] == 0
        assert any(r["kind"] == "supervision_audit" for r in rows)
        assert not any(r["kind"] == "control_claim" and r["owner"] != "vla" for r in rows)
        assert all(r["instruction"] == "Pick up the cup." for r in rows if r["kind"] == "action_executed")
    finally:
        log.close()


async def test_stale_phase_review_cannot_overwrite_new_live_phase_state(tmp_path):
    from vla_agent.contracts import ExecutionReview
    from vla_agent.supervision import RunningReviewer

    c, log, h, lease, memory = await make_harness(tmp_path)
    memory.phases.set_plan(phase_plan(), 0)
    entered, release = asyncio.Event(), asyncio.Event()

    class Planner:
        async def plan(self, decision, context, toolkit):
            entered.set()
            await release.wait()
            return ExecutionReview(
                action="continue", assessment="progressing", phase="approach", reason="older state"
            )

    reviewer = RunningReviewer(c, Planner(), log, SkillLibrary(c.runtime.skill_library))
    reviewer.submit(Decision(binding=memory.binding(h.epoch), label=DecisionLabel.ESCALATE), h, memory)
    await entered.wait()
    memory.phases.assess(
        PhaseAssessment(
            revision=1,
            phase_id="approach",
            status="complete",
            evidence=[dict(observation_step=0, fact="fixture")],
            next_phase="close",
        ),
        0,
        {0},
    )
    release.set()
    await reviewer.pending
    assert reviewer.poll(h, memory) is None and memory.phases.active == "close"
    await reviewer.close()
    log.close()


async def test_code_for_next_subtask_records_that_subtasks_language(tmp_path):
    from vla_agent.adapters.mock import MockDecision, MockEnvironment, MockPlanner, MockVLA
    from vla_agent.data import Recorder, read_events
    from vla_agent.runtime import AgentRuntime

    class Planner(MockPlanner):
        async def plan(self, decision, context, toolkit):
            if context["intent"] == "select_next_executor":
                return Intervention(
                    kind="code",
                    reason="next task code",
                    completion_condition="object_placed",
                    code="def run(ctx):\n    ctx.act([0]*12)\n    return 'UNRESOLVED'\n",
                )
            if context["memory"]["subtask_index"] == 1:
                return Intervention(kind="stop", reason="fixture completed")
            return await super().plan(decision, context, toolkit)

    c = config()
    c.runtime.recovery_attempts = 1
    log = Recorder(tmp_path / "episode", c)
    try:
        result = await AgentRuntime(c, MockEnvironment(), MockVLA(), MockDecision(), Planner(), log).run()
        assert result["error"] is None
        actions = [
            r
            for r in read_events(log.root)
            if r["kind"] == "action_executed" and r["source"] == "code_policy"
        ]
        assert actions[-1]["instruction"] == "Place the cup to the right."
    finally:
        log.close()
