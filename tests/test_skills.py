import json
import shutil

import pytest
from test_runtime import ROOT, make_harness

from vla_agent.control import available_capabilities
from vla_agent.recovery import RecoveryRunner
from vla_agent.skills import SkillLibrary


def test_templates_are_explicit_and_capability_filtered():
    library = SkillLibrary(ROOT / "skills")
    assert library.search("PandaOmron") == []
    assert len(library.search("PandaOmron", include_templates=True)) == 3
    assert library.search("PandaOmron", capabilities=["native_action12"], include_templates=True) == []
    assert library.search("other_robot", include_templates=True) == []
    item = library.get("eef_offset", "v1", "PandaOmron")
    item["parameter_schema"].clear()
    assert library.get("eef_offset", "v1", "PandaOmron")["parameter_schema"]


async def test_skill_parameter_capability_and_verifier_contracts(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    library = SkillLibrary(ROOT / "skills")
    skill = library.get("eef_offset", "v1", "PandaOmron")
    params = dict(delta_position=[0.01, 0, 0], gripper_closed=False, steps=2)
    capabilities = available_capabilities(h.obs)
    assert library.validate_invocation(
        skill, params, capabilities, "object_lifted", h.env.verifier_conditions
    )
    for invalid in [
        dict(params, steps=6),
        dict(params, steps=True),
        dict(params, arbitrary=True),
        dict(params, delta_position=[1, 2]),
        dict(params, gripper_closed="true"),
    ]:
        with pytest.raises(ValueError, match="parameters"):
            library.validate_invocation(
                skill, invalid, capabilities, "object_lifted", h.env.verifier_conditions
            )
    with pytest.raises(ValueError, match="capabilities"):
        library.validate_invocation(skill, params, [], "object_lifted", h.env.verifier_conditions)
    with pytest.raises(ValueError, match="verifier"):
        library.validate_invocation(
            skill, params, capabilities, "invented_grasp_success", h.env.verifier_conditions
        )
    log.close()


async def test_template_runs_via_real_worker_but_does_not_claim_object_success(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    library = SkillLibrary(ROOT / "skills")
    skill = library.get("eef_offset", "v1", "PandaOmron")
    params = dict(delta_position=[0.02, 0, 0], gripper_closed=False, steps=2)
    report = await RecoveryRunner(h, memory, c.runtime, log).run(
        skill["code"], "object_lifted", "template", parameters=params, skill=skill
    )
    assert report["executed_steps"] == 2 and h.env.pos[0] == pytest.approx(0.02)
    assert report["result"] == "UNRESOLVED" and not report["verified"]
    log.close()


def test_candidates_keep_parameter_contract_and_frozen_snapshot(tmp_path):
    library = SkillLibrary(tmp_path, mode="accumulate")
    template = SkillLibrary(ROOT / "skills").get("eef_offset", "v1", "PandaOmron")
    item = library.record_candidate(
        template["code"],
        "PandaOmron",
        "object_lifted",
        dict(episode_id="fixture", parameters={"delta_position": [0.01, 0, 0]}),
        "layout1",
        parent_skill=template,
    )
    assert library.index == []
    next_episode = SkillLibrary(tmp_path)
    learned = next_episode.get(item["id"], item["version"], "PandaOmron")
    assert learned["status"] == "candidate" and learned["parameter_schema"] == template["parameter_schema"]
    assert next_episode.search("PandaOmron", include_templates=True) == []
    frozen = SkillLibrary(tmp_path / "frozen")
    frozen.record_candidate("def run(ctx): pass", "PandaOmron", "test", {}, "layout")
    assert not (tmp_path / "frozen").exists()


def test_skill_integrity_includes_code_and_manifest(tmp_path):
    shutil.copytree(ROOT / "skills/eef_offset", tmp_path / "eef_offset")
    original = SkillLibrary(tmp_path).snapshot
    path = tmp_path / "eef_offset/v1/manifest.json"
    item = json.loads(path.read_text())
    item["required_capabilities"].append("nonexistent")
    path.write_text(json.dumps(item))
    assert SkillLibrary(tmp_path).snapshot != original
    (tmp_path / "eef_offset/v1/policy.py").write_text("def run(ctx): pass")
    with pytest.raises(ValueError, match="hash mismatch"):
        SkillLibrary(tmp_path)


def test_failed_candidate_is_retrievable_only_with_explicit_opt_in(tmp_path):
    library = SkillLibrary(tmp_path, mode="accumulate")
    item = library.record_candidate(
        "def run(ctx): return 'UNRESOLVED'", "PandaOmron", "task_success",
        {"report": {"verified": False, "error": "lost contact"}}, "seed7",
        description="Close toaster door: contact exploration",
    )
    next_episode = SkillLibrary(tmp_path)
    assert next_episode.search("PandaOmron", "toaster", include_templates=True) == []
    results = next_episode.search("PandaOmron", "toaster", include_candidates=True)
    assert len(results) == 1 and results[0]["status"] == "candidate"
    assert results[0]["evidence"][0]["report"]["verified"] is False
    assert next_episode.get(item["id"], item["version"], "PandaOmron")["code"]


def test_successful_trial_breaks_candidate_retrieval_ties_without_promotion(tmp_path):
    library = SkillLibrary(tmp_path, "accumulate")
    for i in range(7):
        library.record_candidate(f"def run(ctx):\n    return {i}\n", "PandaOmron", "task_success",
                                 {"report": {"verified": False}}, "0", description="close toaster")
    successful = library.record_candidate("def run(ctx):\n    return 'SUBTASK_DONE'\n", "PandaOmron", "task_success",
                                          {"report": {"verified": True}}, "0", description="close toaster")
    snapshot = SkillLibrary(tmp_path)
    assert snapshot.search("PandaOmron", "close toaster") == []
    found = snapshot.search("PandaOmron", "close toaster", include_candidates=True)
    assert found[0]["id"] == successful["id"]
    assert len(found) == 1 and found[0]['version'] == successful['version']
    assert found[0]['versions']['count'] == 8
    assert all(x["status"] == "candidate" for x in found)


def test_verbose_failed_variants_do_not_hide_same_task_success_or_grant_authority(tmp_path):
    library = SkillLibrary(tmp_path, 'accumulate')
    for i in range(7):
        library.record_candidate(f'def run(ctx): return {i}', 'PandaOmron', 'task_success',
            {'report': {'verified': False}}, '0',
            description='Close the toaster oven door. | lowered rim grasp lift alignment whole task')
    success = library.record_candidate('def run(ctx): return 10', 'PandaOmron', 'task_success',
        {'report': {'verified': True}}, '1', description='Close the toaster oven door. | release')
    unrelated = library.record_candidate('def run(ctx): return 11', 'PandaOmron', 'task_success',
        {'report': {'verified': True}}, '2', description='Open the cabinet. | lowered rim grasp lift alignment')
    snapshot = SkillLibrary(tmp_path)
    query = 'Close toaster oven door lowered rim grasp lift alignment whole task'
    found = snapshot.search('PandaOmron', query, include_candidates=True)
    assert len(found) == 2 and found[0]['id'] == success['id']
    assert found[0]['version'] == success['version'] and found[0]['versions']['count'] == 8
    assert found[1]['id'] == unrelated['id']
    assert all(x['status'] == 'candidate' for x in found)
    assert snapshot.search('PandaOmron', query) == []
    assert snapshot.search('PandaOmron', query, capabilities=[], include_candidates=True) == []
    found[0]['evidence'].clear()
    assert snapshot.get(success['id'], success['version'], 'PandaOmron')['evidence']
    # Without any successful example in the matched task, do not substitute an unrelated one.
    snapshot.index = [x for x in snapshot.index if x['version'] != success['version']]
    assert snapshot.search('PandaOmron', query, include_candidates=True)[0]['id'] == success['id']


def test_search_summarizes_dense_history_but_keeps_outcomes_parameters_and_expandable_evidence(tmp_path):
    library = SkillLibrary(tmp_path, 'accumulate')
    report = dict(verified=False, result='UNRESOLVED', executed_steps=10, seconds=3.,
                  error='No door progress', failed_stage='lift',
                  evidence={'passed': False, 'observation_step': 100},
                  stages=[dict(name='lift', status='failed', start_step=90, end_step=100,
                               checks=[{'passed': False}], program_measurements=[{'values':list(range(100))}]*40)],
                  program_notes=[{'facts': [str(i)]} for i in range(5)])
    record = dict(report=report, parameters={'box':[1,2,3,4]}, episode_id='prior', evidence_root='/prior/run')
    skill = library.record_candidate('def run(ctx): pass', 'PandaOmron', 'task_success', record, 'layout')
    snapshot = SkillLibrary(tmp_path)
    result = snapshot.search('PandaOmron', include_candidates=True)[0]
    brief = result['evidence'][0]
    assert brief['parameters'] == record['parameters'] and brief['evidence_root'] == record['evidence_root']
    assert brief['report']['verified'] is False and brief['report']['evidence'] == report['evidence']
    assert brief['report']['error'] == report['error'] and brief['report']['failed_stage'] == 'lift'
    assert brief['report']['stages'][0]['measurement_count'] == 40
    assert len(json.dumps(brief)) < len(json.dumps(record)) / 5
    assert result['evidence_detail']['expand_tool'] == 'read_skill'
    full = snapshot.get(skill['id'], skill['version'], 'PandaOmron')
    assert full['evidence'][0] == record and full['code'] == 'def run(ctx): pass'
    brief['report']['stages'][0]['status'] = 'success'
    assert snapshot.get(skill['id'], skill['version'], 'PandaOmron')['evidence'][0] == record
