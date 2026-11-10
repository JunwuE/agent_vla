import asyncio
import json
import time

import numpy as np
import pytest
from test_runtime import config, make_harness

from vla_agent.adapters.mock import MockDecision, MockEnvironment, MockVLA
from vla_agent.contracts import Decision, DecisionLabel, Intervention, Verification
from vla_agent.data import Recorder, read_events
from vla_agent.memory import WorkingMemory
from vla_agent.recovery import RecoveryRunner
from vla_agent.runtime import AgentRuntime
from vla_agent.skills import SkillLibrary
from vla_agent.toolkit import PlannerToolkit, planner_tools
from vla_agent.workspace import RecoveryWorkspace


def test_default_schema_routes_motion_through_code():
    c = config()
    assert not {"local", "eef", "edit"} & set(c.runtime.allowed_interventions)
    tools = planner_tools()
    assert not {"preview_vla", "preview_edit"} & {t["name"] for t in tools}
    submit = next(t for t in tools if t["name"] == "submit_intervention")
    assert not {"local", "eef", "edit"} & set(submit["parameters"]["properties"]["kind"]["enum"])
    assert "preview_edit" in {t["name"] for t in planner_tools(["code", "edit"])}


def test_workspace_keeps_lineage_and_checks_without_execution(tmp_path):
    workspace = RecoveryWorkspace(tmp_path / "workspace", {"goal": "fixture"})
    sentinel = tmp_path / "must-not-exist"
    source = f"open({str(sentinel)!r}, 'w').write('bad')\ndef run(ctx):\n    return 'UNRESOLVED'\n"
    v1 = workspace.write(source, "first hypothesis", "test effect")
    assert workspace.check(v1["version"])["ok"] and not sentinel.exists()
    v2 = workspace.patch(v1["version"], "UNRESOLVED", "RECOVERED", "revised hypothesis", "new effect")
    assert v2["parent_version"] == v1["version"]
    assert workspace.read(v1["version"])["source"] == source
    assert not sentinel.exists()
    with pytest.raises(ValueError, match="exactly one"):
        workspace.patch(v2["version"], "missing", "new", "why", "effect")
    with pytest.raises(ValueError, match="Invalid"):
        workspace.read("../../policy")
    (workspace.root / "versions" / v1["version"] / "policy.py").write_text("changed")
    with pytest.raises(ValueError, match="modified"):
        workspace.read(v1["version"])


def test_invalid_program_is_saved_for_repair(tmp_path):
    workspace = RecoveryWorkspace(tmp_path / "workspace", {})
    bad = workspace.write("def run(ctx):\n  syntax !", "test", "test")
    assert not bad["checks"]["ok"]
    fixed = workspace.patch(bad["version"], "syntax !", "return 'UNRESOLVED'", "syntax fix", "program loads")
    assert workspace.check(fixed["version"])["ok"]


async def test_batch_patch_is_atomic_preserves_contract_and_is_unavailable_to_reviews(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    workspace = RecoveryWorkspace(tmp_path / 'workspace', {})
    source = "def run(ctx):\n    first = 1\n    second = 2\n    return 'UNRESOLVED'\n"
    schema = dict(type='object', properties={}, additionalProperties=False)
    original = workspace.write(source, 'fixture', 'fixture', parameter_schema=schema,
                               origin_skill={'id': 'prior'})
    toolkit = PlannerToolkit(h, memory, SkillLibrary(tmp_path / 'skills'), workspace=workspace)
    request = dict(base_version=original['version'], hypothesis='two related edits', expected_effect='fixture',
                   replacements=[dict(old_text='first = 1', new_text='first = 10'),
                                 dict(old_text='second = 2', new_text='second = 20')])
    try:
        for edits in ([dict(old_text='first = 1', new_text='a'), dict(old_text='missing', new_text='b')],
                      [dict(old_text='first = 1', new_text='a'), dict(old_text='first', new_text='b')]):
            with pytest.raises(ValueError):
                await toolkit.call('patch_program_batch', dict(request, replacements=edits))
            assert len(workspace.versions) == 1
        item = await toolkit.call('patch_program_batch', request)
        assert len(workspace.versions) == 2 and item['parent_version'] == original['version']
        contract = await toolkit.call('read_program', dict(version=item['version'], view='contract'))
        assert item['origin_skill'] == {'id': 'prior'} and contract['contract']['parameter_schema'] == schema
        assert workspace.read(original['version'])['source'] == source
        assert 'first = 10' in workspace.read()['source'] and 'second = 20' in workspace.read()['source']
        assert h.obs.step == 0
        reviewer = PlannerToolkit.for_review(h, memory, toolkit.skills, c.runtime.allowed_interventions)
        with pytest.raises(ValueError, match='read-only'):
            await reviewer.call('patch_program_batch', request)
        assert 'patch_program_batch' not in {t['name'] for t in planner_tools(review_only=True)}
    finally:
        log.close()


def test_skill_reuse_reads_later_run_evidence_without_changing_frozen_snapshot(tmp_path):
    library = SkillLibrary(tmp_path / 'skills', 'accumulate')
    source = "def run(ctx): return 'UNRESOLVED'"
    first = library.record_candidate(source, 'mock_panda', 'task_success',
                                     {'report': {'verified': False}, 'parameters': {'trial': 0}}, 'first')
    frozen = SkillLibrary(tmp_path / 'skills')
    old_snapshot = json.loads(json.dumps(frozen.snapshot))
    second = library.record_candidate(source, 'mock_panda', 'task_success',
                                      {'report': {'verified': True}, 'parameters': {'trial': 1}}, 'second')
    for trial in range(2, 12):
        library.record_candidate(source, 'mock_panda', 'task_success',
                                 {'report': {'verified': False}, 'parameters': {'trial': trial}}, str(trial))
    library.record_candidate(source, 'mock_panda', 'task_success',
                             {'report': None, 'parameters': {'trial': 12}}, 'unverified')
    assert (first['id'], first['version']) == (second['id'], second['version'])
    loaded = SkillLibrary(tmp_path / 'skills')
    skill = loaded.get(first['id'], first['version'], 'mock_panda')
    assert skill['status'] == 'candidate'  # New evidence does not promote the program.
    assert len(skill['evidence']) <= 7 and skill['run_summary']['recorded_runs'] == 13
    assert skill['run_summary']['verified_runs'] == 1
    assert any(r['report']['verified'] is True for r in skill['evidence'])
    assert skill['evidence'][-1]['parameters'] == {'trial': 12}
    assert frozen.snapshot == old_snapshot and loaded.snapshot != old_snapshot
    assert not any(r['report']['verified'] for r in frozen.get(first['id'], first['version'], 'mock_panda')['evidence'])


async def test_imported_skill_is_exact_snapshot_copy_with_patch_lineage_and_no_execution(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    source = "def run(ctx):\n    return 'UNRESOLVED'\n"
    library = SkillLibrary(tmp_path / 'skills', 'accumulate')
    skill = library.record_candidate(source, memory.task.robot, 'task_success',
                                     {'report': {'verified': True}}, 'old-layout')
    library = SkillLibrary(tmp_path / 'skills')
    workspace = RecoveryWorkspace(tmp_path / 'workspace', {})
    toolkit = PlannerToolkit(h, memory, library, workspace=workspace)
    request = dict(skill_id=skill['id'], version=skill['version'], hypothesis='Remeasure new scene',
                   expected_effect='Reuse program structure without retyping it')
    try:
        imported = await toolkit.call('import_skill_program', request)
        assert workspace.read(imported['version'])['source'] == source
        assert imported['origin_skill']['sha256'] == skill['sha256']
        assert imported['origin_skill']['status'] == 'candidate' and not imported['executed']
        patched = await toolkit.call('patch_program', dict(base_version=imported['version'],
            old_text="'UNRESOLVED'", new_text="'SUBTASK_DONE'", hypothesis='Fixture only', expected_effect='Test lineage'))
        assert patched['origin_skill'] == imported['origin_skill']
        assert library.get(skill['id'], skill['version'], memory.task.robot)['code'] == source
        assert workspace.read(imported['version'])['source'] == source
        assert h.obs.step == 0
        reviewer = PlannerToolkit.for_review(h, memory, library, c.runtime.allowed_interventions)
        with pytest.raises(ValueError, match='read-only'):
            await reviewer.call('import_skill_program', request)
        with pytest.raises(ValueError, match='snapshot'):
            await toolkit.call('import_skill_program', dict(request, version='nonexistent'))
        assert len(workspace.versions) == 2
        assert 'import_skill_program' not in {t['name'] for t in planner_tools(review_only=True)}
    finally:
        log.close()


def test_program_parameter_contract_survives_patches_and_rejects_invalid_values(tmp_path):
    workspace = RecoveryWorkspace(tmp_path / 'workspace', {})
    schema = dict(type='object', properties={'dx': dict(type='number', minimum=0.001, maximum=0.05)},
                  required=['dx'], additionalProperties=False)
    item = workspace.write("def run(ctx): return 'UNRESOLVED'", 'parameterized', 'fixture',
                           parameter_schema=schema, parameter_sources=['Current observed target offset, meters'])
    patched = workspace.patch(item['version'], 'UNRESOLVED', 'SUBTASK_DONE', 'fixture', 'fixture')
    assert patched['parameter_schema'] == schema
    assert workspace.validate_parameters(patched['version'], {'dx': 0.02}) == {'dx': 0.02}
    for invalid in [{}, {'dx': True}, {'dx': 1}, {'dx': 0.02, 'extra': 1}]:
        with pytest.raises(ValueError, match='parameters'):
            workspace.validate_parameters(item['version'], invalid)
    with pytest.raises(ValueError, match='local references'):
        workspace.write("def run(ctx): pass", 'bad schema', 'fixture',
                        parameter_schema={'type': 'object', '$ref': 'https://invalid/schema'})
    assert len(workspace.versions) == 2


async def test_parameterized_program_executes_and_retains_contract_for_next_episode(tmp_path):
    schema = dict(type='object', properties={'dx': dict(type='number', minimum=0.001, maximum=0.05)},
                  required=['dx'], additionalProperties=False)

    class Planner:
        calls = 0

        async def plan(self, decision, context, toolkit):
            self.calls += 1
            program = await toolkit.call('write_program', dict(
                source="def run(ctx):\n    ctx.move_relative([ctx.parameters['dx'],0,0], False, steps=2)\n    return 'SUBTASK_DONE'\n",
                hypothesis='Fixture measured target', expected_effect='Reach fixture alignment',
                skill_name='align-fixture', when_to_use='Measured residual offset', reuse_reason='No fixture skill exists',
                parameter_schema=schema, parameter_sources=['Measured offset in meters']))
            return Intervention(kind='code', reason='Test parameterized execution', program_version=program['version'],
                                parameters={'dx': 0.04}, completion_condition='fixture_aligned')

    c, log, rt = await runtime_fixture(tmp_path, Planner())
    rt.skills = SkillLibrary(tmp_path / 'skills', 'accumulate')
    rt.reviewer.skills = rt.skills
    try:
        await rt.handle(Decision(binding=rt.memory.binding(rt.harness.epoch), label=DecisionLabel.ESCALATE))
        assert rt.harness.obs.step == 2 and rt.harness.obs.state[0] == pytest.approx(0.04)
        next_library = SkillLibrary(tmp_path / 'skills')
        skills = next_library.search(rt.memory.task.robot, include_candidates=True)
        assert len(skills) == 1 and skills[0]['parameter_schema'] == schema
        assert skills[0]['status'] == 'candidate'
        assert skills[0]['evidence'][0]['parameters'] == {'dx': 0.04}
    finally:
        await rt.flora.close()
        log.close()


async def test_checked_stage_failure_stops_later_motion_and_keeps_measurements(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    code = """def run(ctx):
    with ctx.stage("grasp", completion_condition="object_lifted") as stage:
        ctx.act([0]*12)
        stage.record(claimed_success=True)
    ctx.act([1,0,0,0,0,0,0,0,0,0,0,0])
    return "RECOVERED"
"""
    report = await RecoveryRunner(h, memory, c.runtime, log).run(code, "object_lifted", "stage")
    assert report["result"] == "UNRESOLVED" and h.obs.step == 1
    assert report["failed_stage"] == "grasp" and report["traceback"]
    stage = report["stages"][0]
    assert stage["status"] == "failed" and stage["program_measurements"] == [{"claimed_success": True}]
    assert stage["checks"][-1]["passed"] is False
    log.close()


async def test_direct_parameterized_skill_can_be_revised_without_losing_contract(tmp_path):
    schema = dict(type='object', properties={'dx': dict(type='number', minimum=0.001, maximum=0.05)},
                  required=['dx'], additionalProperties=False)
    source = "def run(ctx):\n    ctx.move_relative([ctx.parameters['dx'],0,0], False, steps=2)\n    return 'UNRESOLVED'\n"
    library = SkillLibrary(tmp_path / 'skills', 'accumulate')
    skill = library.record_candidate(source, 'mock_panda', 'fixture_aligned', {}, 'fixture',
        parent_skill=dict(parameter_schema=schema, parameter_sources=['Measured offset'],
                          preconditions=[], required_capabilities=['native_action12'], known_failures=[]))

    class Planner:
        calls = 0

        async def plan(self, decision, context, toolkit):
            self.calls += 1
            if self.calls == 1:
                return Intervention(kind='code', reason='Invoke reusable skill', skill_id=skill['id'],
                                    skill_version=skill['version'], parameters={'dx': 0.01},
                                    completion_condition='fixture_aligned')
            trace = await toolkit.call('read_recovery_trace', {})
            program = await toolkit.call('patch_program', dict(
                base_version=trace['version'], old_text="'UNRESOLVED'", new_text="'SUBTASK_DONE'",
                hypothesis='Measured residual requires another movement', expected_effect='Reach fixture goal'))
            contract = await toolkit.call('read_program', dict(version=program['version'], view='contract'))
            assert contract['contract']['parameter_schema'] == schema
            assert program['origin_skill']['id'] == skill['id']
            return Intervention(kind='code', reason='Revised parameterized skill', program_version=program['version'],
                                parameters={'dx': 0.03}, completion_condition='fixture_aligned')

    planner = Planner()
    c, log, rt = await runtime_fixture(tmp_path, planner)
    rt.skills = SkillLibrary(tmp_path / 'skills', 'accumulate')
    rt.reviewer.skills = rt.skills
    try:
        await rt.handle(Decision(binding=rt.memory.binding(rt.harness.epoch), label=DecisionLabel.ESCALATE))
        assert planner.calls == 2 and rt.harness.obs.step == 4
        assert rt.harness.obs.state[0] == pytest.approx(0.04)
        rows = read_events(log.root)
        assert sum(r['kind'] == 'skill_invocation' for r in rows) == 1
        assert [r['verified'] for r in rows if r['kind'] == 'recovery_result'] == [False, True]
        learned = SkillLibrary(tmp_path / 'skills')
        assert {x['id'] for x in learned.index} == {skill['id']}
        assert len(learned.versions(skill['id'], rt.memory.task.robot)) == 2
    finally:
        await rt.flora.close()
        log.close()


@pytest.mark.parametrize("condition", ["object_lifted", "uninstalled_condition"])
async def test_failed_or_unknown_stage_precondition_prevents_action(tmp_path, condition):
    c, log, h, lease, memory = await make_harness(tmp_path)
    code = f"""def run(ctx):
    with ctx.stage("lift", preconditions=[{condition!r}]):
        ctx.act([0]*12)
    return "RECOVERED"
"""
    report = await RecoveryRunner(h, memory, c.runtime, log).run(code, "object_lifted", "precondition")
    assert report["result"] == "UNRESOLVED" and h.obs.step == 0 and report["failed_stage"] == "lift"
    log.close()


async def test_program_reads_lossless_causal_rgb_depth_and_robot_state(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    code = """def run(ctx):
    data = ctx.read_observation("left", step=0)
    assert data["rgb"].shape == (64,64,3)
    assert data["rgb"].dtype.name == "uint8"
    assert data["depth_m"].dtype.kind == "f"
    assert float(data["depth_m"][0,0]) == 0.5
    assert data["metadata"]["calibration"]["step"] == 0
    assert data["metadata"]["depth_units"] == "meters"
    ctx.note(facts=["raw depth checked"], hypotheses=[])
    return "UNRESOLVED"
"""
    report = await RecoveryRunner(h, memory, c.runtime, log).run(code, "object_lifted", "raw")
    assert report["error"] is None and h.obs.step == 0
    artifacts = list((log.root / "recovery/raw/observations").glob("*.npz"))
    with np.load(artifacts[0], allow_pickle=False) as raw:
        np.testing.assert_array_equal(raw["rgb"], h.obs.rgb["left"])
        np.testing.assert_array_equal(raw["depth_m"], h.obs.depth_m["left"])
    assert memory.notes[-1]["facts"] == ["raw depth checked"]
    log.close()


async def test_program_observation_limit_is_configured_and_never_authorizes_motion(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    c.runtime.recovery_observation_bytes = 1024
    try:
        report = await RecoveryRunner(h, memory, c.runtime, log).run(
            "def run(ctx):\n    ctx.read_observation('left')\n    ctx.act([0]*12)\n    return 'UNRESOLVED'\n",
            'object_lifted', 'observation-budget')
        assert 'observation byte budget' in report['error']
        assert report['executed_steps'] == 0 and h.obs.step == 0
    finally:
        log.close()


async def test_program_cannot_read_future_observation(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    report = await RecoveryRunner(h, memory, c.runtime, log).run(
        "def run(ctx):\n    ctx.read_observation('left', step=10)\n", "object_lifted", "future"
    )
    assert report["result"] == "UNRESOLVED" and "causal" in report["error"]
    assert h.obs.step == 0
    log.close()


async def test_code_capabilities_expose_fresh_controller_and_pose_after_motion(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    code = """def run(ctx):
    for step in range(2):
        obs = ctx.observe()
        caps = ctx.capabilities()
        assert caps['observation_step'] == obs['step'] == step
        assert caps['controller'] == obs['controller']
        assert caps['eef_control_pose'] == obs['eef_control_pose']
        assert caps['eef_control_pose']['step'] == step
        if step == 0:
            ctx.act([0]*12)
    return 'UNRESOLVED'
"""
    try:
        report = await RecoveryRunner(h, memory, c.runtime, log).run(code, 'task_success', 'capabilities')
        assert report['error'] is None and report['executed_steps'] == 1
        planner = PlannerToolkit(h, memory, SkillLibrary(tmp_path / 'skills'))
        caps = planner.capabilities()
        assert caps['observation_step'] == 1 and caps['eef_control_pose']['step'] == 1
    finally:
        log.close()


class AlignmentEnvironment(MockEnvironment):
    verifier_conditions = MockEnvironment.verifier_conditions + ["fixture_aligned"]

    async def verify(self, condition):
        if condition == "fixture_aligned":
            return Verification(
                condition=condition,
                passed=bool(self.pos[0] >= 0.03),
                observation_step=self.step_id,
                evidence=[f"fixture measured x={self.pos[0]}"],
            )
        return await super().verify(condition)


async def runtime_fixture(tmp_path, planner):
    c = config()
    c.task.subtasks[0].completion_condition = "fixture_aligned"
    log = Recorder(tmp_path / "episode", c)
    rt = AgentRuntime(c, AlignmentEnvironment(), MockVLA(), MockDecision(), planner, log)
    rt.lease = await rt.harness.start()
    rt.memory = WorkingMemory(c.task, rt.harness.obs.episode_id)
    rt.memory.add(rt.harness.obs)
    return c, log, rt


async def test_one_recovery_session_revises_program_from_measured_stage_failure(tmp_path):
    class Planner:
        def __init__(self):
            self.calls = 0

        async def plan(self, decision, context, toolkit):
            self.calls += 1
            if self.calls == 1:
                source = """def run(ctx):
    with ctx.stage("align", completion_condition="fixture_aligned"):
        ctx.move_relative([0.01,0,0], False, steps=2)
    return "RECOVERED"
"""
                result = await toolkit.call(
                    "write_program",
                    dict(source=source, hypothesis="small offset", expected_effect="fixture alignment",
                         skill_name='align-fixture', when_to_use='Unaligned fixture', reuse_reason='New fixture'),
                )
            else:
                trace = await toolkit.call("read_recovery_trace", {})
                assert trace["report"]["failed_stage"] == "align"
                assert trace["report"]["executed_steps"] == 2
                assert context["observations"][-1].step == 2
                assert context['intent'] == 'revise_code'
                assert context['eef_control_pose'] == toolkit.harness.obs.eef_control_pose
                assert context['eef_control_pose']['step'] == 2
                assert context['controller'] == toolkit.harness.obs.controller
                assert context['binding'].observation_step == 2
                result = await toolkit.call(
                    "patch_program",
                    dict(
                        base_version=trace["version"],
                        old_text="[0.01,0,0]",
                        new_text="[0.03,0,0]",
                        hypothesis="previous offset insufficient",
                        expected_effect="reach goal from the current measured state",
                    ),
                )
            return Intervention(
                kind="code",
                reason="closed-loop program trial",
                program_version=result["version"],
                completion_condition="fixture_aligned",
            )

    planner = Planner()
    c, log, rt = await runtime_fixture(tmp_path, planner)
    await rt.handle(Decision(binding=rt.memory.binding(rt.harness.epoch), label=DecisionLabel.REFINE))
    assert planner.calls == 2 and rt.harness.obs.step == 4
    assert rt.harness.env.pos[0] == pytest.approx(0.04)
    assert rt.memory.interventions[-1]["verified"]
    workspaces = list((log.root / "workspaces").iterdir())
    manifest = json.loads((workspaces[0] / "index.json").read_text())
    assert len(manifest["versions"]) == len(manifest["attempts"]) == 2
    assert manifest["versions"][1]["parent_version"] == manifest["versions"][0]["version"]
    assert manifest["attempts"][0]["result"] == "UNRESOLVED"
    assert manifest["attempts"][1]["result"] == "SUBTASK_DONE"
    rows = read_events(log.root)
    assert {r["source"] for r in rows if r["kind"] == "action_executed"} == {"code_policy"}
    log.close()


async def test_recovery_session_action_budget_is_not_reset_between_attempts(tmp_path):
    class Planner:
        calls = 0

        async def plan(self, *args):
            self.calls += 1
            return Intervention(
                kind="code",
                reason="budget fixture",
                code="def run(ctx):\n    for _ in range(8):\n        ctx.act([0]*12)\n    return 'UNRESOLVED'\n",
                completion_condition="fixture_aligned",
            )

    planner = Planner()
    c, log, rt = await runtime_fixture(tmp_path, planner)
    c.runtime.recovery_total_steps = 2
    c.runtime.recovery_steps = 5
    await rt.handle(Decision(binding=rt.memory.binding(rt.harness.epoch), label=DecisionLabel.ESCALATE))
    assert rt.harness.obs.step == 2 and planner.calls == 1 and rt.forced_reason
    log.close()


async def test_recovery_session_deadline_includes_model_revision_wait(tmp_path):
    class Planner:
        calls = 0

        async def plan(self, *args):
            self.calls += 1
            if self.calls > 1:
                await asyncio.sleep(10)
            return Intervention(
                kind="code",
                reason="deadline fixture",
                code="def run(ctx):\n    return 'UNRESOLVED'\n",
                completion_condition="fixture_aligned",
            )

    planner = Planner()
    c, log, rt = await runtime_fixture(tmp_path, planner)
    c.runtime.recovery_total_seconds = 0.5
    c.planner.timeout_seconds = 10
    started = time.monotonic()
    await rt.handle(Decision(binding=rt.memory.binding(rt.harness.epoch), label=DecisionLabel.ESCALATE))
    assert time.monotonic() - started < 2
    assert rt.harness.obs.step == 0 and rt.forced_reason
    assert any(
        row["kind"] == "recovery_session_budget" and row.get("reason") == "revision_deadline"
        for row in read_events(log.root)
    )
    log.close()
