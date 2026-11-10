"""Catalog aliases must execute and edit the original pinned Python package."""

import json

import pytest
from test_exploration import runtime_fixture
from test_skill_packages import dependency, parent_files, retain

from vla_agent.contracts import Decision, DecisionLabel, Intervention
from vla_agent.data import read_events
from vla_agent.skills import SkillLibrary


@pytest.mark.parametrize('mode', ['direct', 'import_and_edit'])
async def test_catalog_family_preserves_package_namespace_through_real_session(tmp_path, mode):
    root = tmp_path / 'skills'
    dep = dependency(root)
    schema = dict(type='object', properties={'dx': dict(type='number', minimum=0.001, maximum=0.05)},
                  required=['dx'], additionalProperties=False)
    original = retain(root, 'align-with-library', parent_files(dep), [dep], schema=schema,
                      condition='fixture_aligned')
    family = 'fixture-alignment-family'
    (root / 'catalog.json').write_text(json.dumps(dict(schema_version=1, skills=[dict(
        id=family, robot='mock_panda', description='One operation with pinned implementations',
        versions=[dict(storage_id=original['id'], version=original['version'], when_to_use='Mock fixture')])])))

    class Planner:
        calls = 0
        version = None

        async def plan(self, decision, context, toolkit):
            self.calls += 1
            if mode == 'import_and_edit' and self.version is None:
                imported = await toolkit.call('import_skill_program', dict(skill_id=family,
                    version=original['version'], hypothesis='Reuse exact package under family alias',
                    expected_effect='Fixture alignment'))
                source = toolkit.workspace.read(imported['version'])
                assert source['skill_name'] == family
                assert source['package']['name'] == 'align-with-library'
                assert source['package']['sha256'] == original['package']['sha256']
                edited = await toolkit.call('edit_program', dict(base_version=imported['version'],
                    edits=[dict(path='code/run.py', operation='replace', old_text='Primitive finished;',
                                new_text='Reused primitive finished;')], hypothesis='Clarify result provenance',
                    expected_effect='Same measured primitive with revised report'))
                self.version = edited['version']
                child = toolkit.workspace.read(self.version)
                assert child['package']['name'] == 'align-with-library'
                assert child['package']['uses'] == [dep]
            selection = dict(program_version=self.version) if self.version else dict(
                skill_id=family, skill_version=original['version'])
            return Intervention(kind='code', reason='Alias-selected bounded correction',
                completion_condition='fixture_aligned', parameters={'dx': 0.01 if self.calls == 1 else 0.03},
                **selection)

    _, log, runtime = await runtime_fixture(tmp_path, Planner())
    runtime.skills = SkillLibrary(root, 'accumulate')
    runtime.reviewer.skills = runtime.skills
    original_runs = runtime.skills.get(family, original['version'], 'mock_panda')['run_summary']['recorded_runs']
    try:
        await runtime.handle(Decision(binding=runtime.memory.binding(runtime.harness.epoch),
                                      label=DecisionLabel.ESCALATE))
        events = read_events(log.root)
        reports = [e for e in events if e['kind'] == 'recovery_result']
        assert [e['verified'] for e in reports] == [False, True]
        assert runtime.harness.obs.state[0] == pytest.approx(0.04)
        assert len({e['package_sha256'] for e in reports}) == 1
        assert all(len(e['stages']) == 1 and e['stages'][0]['name'] == 'reused offset' for e in reports)
        if mode == 'direct':
            assert all(e['package_sha256'] == original['package']['sha256'] for e in reports)
            kept = SkillLibrary(root).get(family, original['version'], 'mock_panda')
            assert kept['run_summary']['recorded_runs'] == original_runs + 2
            assert len(SkillLibrary(root).versions(family, 'mock_panda')) == 1
    finally:
        await runtime.flora.close()
        log.close()
