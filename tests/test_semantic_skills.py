import hashlib
import json

import pytest
from test_exploration import runtime_fixture
from test_runtime import ROOT, make_harness

from vla_agent.contracts import Decision, DecisionLabel, Intervention
from vla_agent.data import read_events
from vla_agent.skills import SkillLibrary
from vla_agent.toolkit import PlannerToolkit
from vla_agent.workspace import RecoveryWorkspace

SOURCE = ("def run(ctx):\n    ctx.move_relative([ctx.parameters['dx'],0,0], False, steps=2)\n"
          "    return 'SUBTASK_DONE' if ctx.check(ctx.completion_condition)['passed'] is True else 'UNRESOLVED'\n")
CONTRACT = dict(preconditions=[], required_capabilities=['native_action12'], known_failures=[],
                parameter_schema=dict(type='object', properties={'dx': dict(type='number', minimum=0.001, maximum=0.05)},
                                      required=['dx'], additionalProperties=False),
                parameter_sources=['Current measured residual offset in meters'])


def seed_library(root):
    return SkillLibrary(root, 'accumulate').record_candidate(SOURCE, 'mock_panda', 'fixture_aligned',
        {'report': {'verified': True}}, 'prior', parent_skill=CONTRACT,
        skill_name='align-fixture', when_to_use='Unaligned fixture with measured residual offset')


def test_runs_parameters_revisions_and_procedures_have_separate_identity(tmp_path):
    original = seed_library(tmp_path)
    library = SkillLibrary(tmp_path, 'accumulate')
    parent = library.get(original['id'], original['version'], 'mock_panda')
    before = library.snapshot.copy()
    for dx in (0.01, 0.02, 0.03):
        result = library.record_candidate(SOURCE, 'mock_panda', 'fixture_aligned',
            {'parameters': {'dx': dx}, 'report': {'verified': False}}, 'new-seed', parent_skill=parent)
        assert (result['id'], result['version']) == (original['id'], original['version'])
    assert library.snapshot == before and len(list(tmp_path.glob('*/*/manifest.json'))) == 1
    changed = library.record_candidate(SOURCE.replace('steps=2', 'steps=1'), 'mock_panda', 'fixture_aligned',
        {'report': {'verified': False}}, 'new-seed', parent_skill=parent)
    assert changed['id'] == original['id'] and changed['version'] != original['version']
    assert changed['revision_of'] == dict(id=original['id'], version=original['version'])
    next_library = SkillLibrary(tmp_path)
    found = next_library.search('mock_panda', include_candidates=True)
    assert len(found) == 1 and found[0]['version'] == original['version']
    assert found[0]['versions']['count'] == 2
    assert next_library.get(original['id'], original['version'], 'mock_panda')['run_summary']['recorded_runs'] == 4
    assert len(next_library.versions('align-fixture', 'mock_panda')) == 2
    assert all(x['status'] == 'candidate' for x in next_library.index)


def test_catalog_aliases_preserve_old_addresses_and_source_without_promotion():
    library = SkillLibrary(ROOT / 'skills')
    old = library.get('recovery-6e0569a16adc', '6e0569a16adc2f86', 'PandaOmron')
    named = library.get('close-toaster-door-from-upright', old['version'], 'PandaOmron')
    assert old == named and named['status'] == 'candidate'
    assert named['sha256'] == hashlib.sha256(named['code'].encode()).hexdigest()
    results = library.search('PandaOmron', 'close toaster door', include_candidates=True)
    assert {x.id for x in library.catalog.skills} <= {x['id'] for x in results}
    assert not any(x['id'].startswith('recovery-') for x in results)
    assert sum(x['archived'] for x in library.index) == 149
    versions = library.versions('close-toaster-door-from-upright', 'PandaOmron')
    assert sum(x['package'] is None for x in versions) == 3
    assert library.search('PandaOmron') == []
    archived = next(x for x in library.index if x['archived'])
    assert library.get(archived['id'], archived['version'], archived['robot'])['code']
    visible = library.search('PandaOmron', archived['description'], include_candidates=True, include_archived=True)
    assert any(x['archived'] for x in visible)


def test_missing_or_ambiguous_catalog_mapping_is_rejected(tmp_path):
    item = seed_library(tmp_path)
    catalog = dict(skills=[dict(id='align-measured-fixture', robot='mock_panda', description='Fixture alignment',
        versions=[dict(storage_id=item['id'], version='missing', when_to_use='Fixture')])])
    (tmp_path / 'catalog.json').write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match='missing'):
        SkillLibrary(tmp_path)
    catalog['skills'][0]['versions'][0]['version'] = item['version']
    catalog['skills'].append(dict(catalog['skills'][0], id='another-name'))
    (tmp_path / 'catalog.json').write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match='assigned twice'):
        SkillLibrary(tmp_path)


def test_alias_reuse_appends_to_original_storage_and_revision_keeps_semantic_name(tmp_path):
    item = seed_library(tmp_path)
    catalog = dict(skills=[dict(id='align-measured-fixture', robot='mock_panda', description='Fixture alignment',
        versions=[dict(storage_id=item['id'], version=item['version'], when_to_use='Measured fixture offset')])])
    (tmp_path / 'catalog.json').write_text(json.dumps(catalog))
    library = SkillLibrary(tmp_path, 'accumulate')
    parent = library.get('align-measured-fixture', item['version'], 'mock_panda')
    original_path = tmp_path / item['id'] / item['version']
    manifest = (original_path / 'manifest.json').read_bytes()
    library.record_candidate(SOURCE, 'mock_panda', 'fixture_aligned', {'report': {'verified': False}},
                             'another-layout', parent_skill=parent)
    assert (original_path / 'manifest.json').read_bytes() == manifest
    assert len((original_path / 'runs.jsonl').read_text().splitlines()) == 2
    assert not (tmp_path / 'align-measured-fixture').exists()
    revision = library.record_candidate(SOURCE.replace('steps=2', 'steps=1'), 'mock_panda', 'fixture_aligned',
        {'report': {'verified': False}}, 'another-layout', parent_skill=parent)
    assert revision['id'] == 'align-measured-fixture'
    reloaded = SkillLibrary(tmp_path)
    assert len(reloaded.versions('align-measured-fixture', 'mock_panda')) == 2
    assert reloaded.get(item['id'], item['version'], 'mock_panda')['code'] == SOURCE


async def test_parameter_only_retry_reuses_one_worker_program_and_library_version(tmp_path):
    original = seed_library(tmp_path / 'skills')

    class Planner:
        calls = 0

        async def plan(self, decision, context, toolkit):
            self.calls += 1
            assert self.calls <= 2
            return Intervention(kind='code', reason='Use measured residual without changing code',
                skill_id=original['id'], skill_version=original['version'],
                parameters={'dx': 0.01 if self.calls == 1 else 0.03}, completion_condition='fixture_aligned')

    planner = Planner()
    _, log, rt = await runtime_fixture(tmp_path, planner)
    rt.skills = SkillLibrary(tmp_path / 'skills', 'accumulate')
    rt.reviewer.skills = rt.skills
    try:
        await rt.handle(Decision(binding=rt.memory.binding(rt.harness.epoch), label=DecisionLabel.ESCALATE))
        assert rt.harness.obs.state[0] == pytest.approx(0.04)
        events = read_events(log.root)
        attempts = [e for e in events if e['kind'] == 'recovery_result']
        assert [e['verified'] for e in attempts] == [False, True]
        code = [e for e in events if e['kind'] == 'code_version']
        assert len(code) == 2 and len({e['sha256'] for e in code}) == 1
        assert len({e['program_version'] for e in code}) == 1
        assert not any(e['kind'] == 'new_skill_proposal' for e in events)
        assert len(list((tmp_path / 'skills').glob('*/*/manifest.json'))) == 1
        saved = SkillLibrary(tmp_path / 'skills').get(original['id'], original['version'], 'mock_panda')
        assert saved['run_summary']['recorded_runs'] == 3 and saved['status'] == 'candidate'
    finally:
        await rt.flora.close()
        log.close()


async def test_existing_procedure_cannot_be_rewritten_with_write_program(tmp_path):
    _, log, h, _, memory = await make_harness(tmp_path)
    seed_library(tmp_path / 'skills')
    toolkit = PlannerToolkit(h, memory, SkillLibrary(tmp_path / 'skills'),
                             workspace=RecoveryWorkspace(tmp_path / 'workspace', {}))
    try:
        with pytest.raises(ValueError, match='already exists'):
            await toolkit.call('write_program', dict(source=SOURCE, hypothesis='new layout', expected_effect='align',
                skill_name='align-fixture', when_to_use='Unaligned fixture', reuse_reason='Only the scene changed'))
        assert not toolkit.workspace.versions
    finally:
        log.close()


async def test_imported_revisions_preserve_preconditions_and_reject_unverified_entry(tmp_path):
    _, log, h, _, memory = await make_harness(tmp_path)
    library = SkillLibrary(tmp_path / 'skills', 'accumulate')
    skill = library.record_candidate(SOURCE, memory.task.robot, memory.current.completion_condition, {}, 'prior',
        parent_skill=dict(CONTRACT, preconditions=['object_lifted']), skill_name='move-held-object')
    library = SkillLibrary(tmp_path / 'skills')
    workspace = RecoveryWorkspace(tmp_path / 'workspace', {})
    toolkit = PlannerToolkit(h, memory, library, workspace=workspace)
    try:
        imported = await toolkit.call('import_skill_program', dict(skill_id=skill['id'], version=skill['version'],
            hypothesis='Correct controller logic', expected_effect='Maintain held-object motion'))
        patch = await toolkit.call('patch_program', dict(base_version=imported['version'], old_text='steps=2',
            new_text='steps=1', hypothesis='Measured feedback frequency', expected_effect='Smaller segment'))
        assert patch['id'] == 'move-held-object'
        contract = await toolkit.call('read_program', dict(version=patch['version'], view='contract'))
        assert contract['contract']['preconditions'] == ['object_lifted']
        plan = Intervention(kind='code', reason='Try revised skill', program_version=patch['version'],
                            parameters={'dx': 0.01}, completion_condition=memory.current.completion_condition)
        with pytest.raises(ValueError, match='preconditions'):
            await toolkit.validate_intervention(plan, ['code'])
        assert h.obs.step == 0
    finally:
        log.close()
