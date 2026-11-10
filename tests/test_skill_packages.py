import json

import pytest
from test_exploration import runtime_fixture
from test_runtime import make_harness

from vla_agent.contracts import Decision, DecisionLabel, Intervention
from vla_agent.data import read_events
from vla_agent.recovery import RecoveryRunner
from vla_agent.skill_packages import build_package, check_package, execution_bundle
from vla_agent.skills import SkillLibrary
from vla_agent.toolkit import PlannerToolkit, planner_tools
from vla_agent.workspace import RecoveryWorkspace


def package_files(name, body, *, uses=(), robot='mock_panda', helper='', params='LIMIT = 2\n'):
    module = name.replace('-', '_')
    return {
        'SKILL.md': f'''---
name: {name}
description: Test fixture only
status: draft
robot: {robot}
entry: code/run.py
version: 1
uses: {json.dumps([x['skill_id'] for x in uses])}
---
# Test fixture
## When to use
Mock fixture only.
## Preconditions
Current measured offset.
## Procedure
Run a bounded correction through ctx.
## Code
main(ctx, **params).
## Checks
Fixed host verifier; program reports are claims.
### Perception checkpoints
None: mock fixture.
## Failure modes
Insufficient correction; retry measured parameters.
## Evidence
Unit tests only; no physical validation.
''',
        'code/run.py': body,
        f'code/{module}/__init__.py': helper,
        f'code/{module}/params.py': params,
    }


def retain(root, name, files, uses=(), *, robot='mock_panda', schema=None, condition=None):
    package = build_package(name, files, uses)
    contract = dict(preconditions=[], required_capabilities=['native_action12'], known_failures=[],
                    parameter_sources=['Current measured offset'],
                    parameter_schema=schema or dict(type='object', properties={}, additionalProperties=False))
    return SkillLibrary(root, 'accumulate').record_candidate(files['code/run.py'], robot, condition, {}, 'fixture',
        skill_name=name, description='Fixture package', parent_skill=contract,
        package=package, package_files=files)


def dependency(root):
    files = package_files('fixture-offset', "def main(ctx, **params):\n    return {'ok': True, 'measured': {}, 'why': ''}\n",
        helper="from .params import LIMIT\ndef move(ctx, dx):\n    with ctx.stage('reused offset'):\n        ctx.move_relative([dx, 0, 0], False, steps=LIMIT)\n")
    skill = retain(root, 'fixture-offset', files)
    return dict(skill_id=skill['id'], version=skill['version'])


def parent_files(dep, body=None):
    return package_files('align-with-library', body or '''from fixture_offset import move
def main(ctx, dx):
    move(ctx, dx)
    return {'ok': True, 'measured': {'dx': dx}, 'why': 'Primitive finished; host verifies whole subtask'}
''', uses=[dep])


def test_package_layout_static_checks_and_dependency_binding(tmp_path):
    dep = dependency(tmp_path / 'skills')
    files = parent_files(dep)
    package = build_package('align-with-library', files, [dep])
    assert check_package(package, files)['ok']
    bundle = execution_bundle(package, files, SkillLibrary(tmp_path / 'skills'), 'mock_panda')
    assert set(bundle['manifest']['packages']) == {'fixture-offset', 'align-with-library'}
    assert bundle['manifest']['packages']['fixture-offset']['reference']['version'] == dep['version']
    with pytest.raises(ValueError, match='capabilities'):
        execution_bundle(package, files, SkillLibrary(tmp_path / 'skills'), 'mock_panda', capabilities=[])
    bad = dict(files, **{'../outside.py': 'pass'})
    with pytest.raises(ValueError, match='path'):
        build_package('align-with-library', bad, [dep])
    with pytest.raises(ValueError, match='uses'):
        build_package('align-with-library', files, [])
    files['code/align_with_library/params.py'] = 'syntax !'
    package = build_package('align-with-library', files, [dep])
    assert not check_package(package, files)['ok']


def test_dependency_closure_rejects_missing_conflicting_and_cyclic_versions(tmp_path):
    dep = dependency(tmp_path / 'skills')
    files = parent_files(dep)
    package = build_package('align-with-library', files, [dep])
    with pytest.raises(ValueError, match='snapshot'):
        execution_bundle(package, files, SkillLibrary(tmp_path / 'empty'), 'mock_panda')
    # root imports its own previous version: Python cannot provide two versions of one namespace.
    old = retain(tmp_path / 'skills', 'align-with-library', files, [dep])
    self_ref = dict(skill_id=old['id'], version=old['version'])
    recursive = parent_files(self_ref)
    recursive_package = build_package('align-with-library', recursive, [self_ref])
    with pytest.raises(ValueError, match='Cyclic'):
        execution_bundle(recursive_package, recursive, SkillLibrary(tmp_path / 'skills'), 'mock_panda')
    changed_files = package_files('fixture-offset', "def main(ctx):\n    return {'ok': False, 'measured': {}, 'why': 'v2'}\n")
    changed = retain(tmp_path / 'skills', 'fixture-offset', changed_files)
    new_dep = dict(skill_id=changed['id'], version=changed['version'])
    chain = dict(skill_id=old['id'], version=old['version'])
    top = package_files('compose-fixture', 'def main(ctx): return {}', uses=[chain, new_dep])
    with pytest.raises(ValueError, match='Conflicting'):
        execution_bundle(build_package('compose-fixture', top, [chain, new_dep]), top,
                         SkillLibrary(tmp_path / 'skills'), 'mock_panda')


def test_workspace_patches_only_selected_module_and_detects_tampering(tmp_path):
    workspace = RecoveryWorkspace(tmp_path / 'workspace', {})
    files = package_files('fixture-control', 'def main(ctx): return {}')
    first = workspace.write_package(files, 'fixture', 'fixture', skill_name='fixture-control')
    second = workspace.patch_package(first['version'], [dict(path='code/fixture_control/params.py',
        old_text='LIMIT = 2', new_text='LIMIT = 3')], 'parameter contract experiment', 'fixture')
    assert first['sha256'] == second['sha256']  # Entry wrapper unchanged, full package hash changed.
    assert first['package']['sha256'] != second['package']['sha256']
    assert second['parent_version'] == first['version']
    assert workspace.read(first['version'])['package_files'] == files
    with pytest.raises(ValueError, match='multi-file'):
        workspace.patch(first['version'], 'return {}', 'return None', 'wrong tool', 'fixture')
    with pytest.raises(ValueError, match='exactly one'):
        workspace.patch_package(second['version'], [dict(path='code/fixture_control/params.py',
            old_text='missing', new_text='bad')], 'bad', 'fixture')
    assert len(workspace.versions) == 2
    path = workspace.root / 'versions' / second['version'] / 'code/fixture_control/params.py'
    path.write_text('LIMIT = 99\n')
    with pytest.raises(ValueError, match='modified'):
        workspace.read(second['version'])


async def test_imported_function_runs_in_same_worker_and_preserves_whole_task_verification(tmp_path):
    dep = dependency(tmp_path / 'skills')
    files = parent_files(dep)
    package = build_package('align-with-library', files, [dep])
    bundle = execution_bundle(package, files, SkillLibrary(tmp_path / 'skills'), 'mock_panda')
    c, log, h, _, memory = await make_harness(tmp_path)
    try:
        report = await RecoveryRunner(h, memory, c.runtime, log).run(files['code/run.py'], 'object_lifted', 'composed',
            parameters={'dx': 0.02}, package_bundle=bundle)
        assert report['executed_steps'] == 2 and h.env.pos[0] == pytest.approx(0.02)
        assert report['program_result']['ok'] is True and report['verified'] is False
        assert report['result'] == 'UNRESOLVED'
        assert report['stages'][0]['name'] == 'reused offset'
        assert report['bundle_sha256'] == bundle['sha256']
        assert memory.current_code['bundle_sha256'] == bundle['sha256']
        assert memory.current_code['package_sha256'] == package['sha256']
        assert (log.root / report['artifacts']['bundle']).is_file()
    finally:
        log.close()


async def test_composed_functions_share_action_budget_and_cannot_reset_it(tmp_path):
    dep = dependency(tmp_path / 'skills')
    files = parent_files(dep, '''from fixture_offset import move
def main(ctx, dx):
    move(ctx, dx)
    move(ctx, dx)
    return {'ok': True, 'measured': {}, 'why': ''}
''')
    bundle = execution_bundle(build_package('align-with-library', files, [dep]), files,
                              SkillLibrary(tmp_path / 'skills'), 'mock_panda')
    c, log, h, _, memory = await make_harness(tmp_path)
    try:
        report = await RecoveryRunner(h, memory, c.runtime, log).run(files['code/run.py'], 'object_lifted', 'budget',
            parameters={'dx': 0.01}, remaining_steps=3, package_bundle=bundle)
        assert report['executed_steps'] == 3 and not report['verified']
        assert 'budget' in report['error']
    finally:
        log.close()


async def test_package_tools_execute_reuse_and_retain_named_package_for_next_episode(tmp_path):
    dep = dependency(tmp_path / 'skills')
    schema = dict(type='object', properties={'dx': dict(type='number', minimum=0.001, maximum=0.05)},
                  required=['dx'], additionalProperties=False)

    class Planner:
        calls = 0

        async def plan(self, decision, context, toolkit):
            self.calls += 1
            if self.calls == 1:
                item = await toolkit.call('write_skill_package', dict(files=parent_files(dep), uses=[dep],
                    skill_name='align-with-library', when_to_use='Measured fixture offset', reuse_reason='Compose offset primitive',
                    hypothesis='Reuse measured correction', expected_effect='Fixture alignment', parameter_schema=schema))
                self.version = item['version']
                checked = await toolkit.call('check_program', dict(version=self.version))
                assert checked['ok'] and len(checked['dependency_lock']) == 2
                brief = await toolkit.call('read_program', dict(version=self.version))
                assert 'package_files' not in brief and 'SKILL.md' in brief['file_paths']
            return Intervention(kind='code', reason='Measured correction with identical package', program_version=self.version,
                completion_condition='fixture_aligned', parameters={'dx': 0.01 if self.calls == 1 else 0.03})

    planner = Planner()
    _, log, rt = await runtime_fixture(tmp_path, planner)
    rt.skills = SkillLibrary(tmp_path / 'skills', 'accumulate')
    rt.reviewer.skills = rt.skills
    try:
        await rt.handle(Decision(binding=rt.memory.binding(rt.harness.epoch), label=DecisionLabel.ESCALATE))
        events = read_events(log.root)
        reports = [e for e in events if e['kind'] == 'recovery_result']
        assert [e['verified'] for e in reports] == [False, True]
        assert len({e['package_sha256'] for e in reports}) == 1
        assert rt.harness.obs.state[0] == pytest.approx(0.04)
        later = SkillLibrary(tmp_path / 'skills')
        versions = later.versions('align-with-library', 'mock_panda')
        assert len(versions) == 1 and versions[0]['package']['uses'] == [dep]
        kept = later.get(versions[0]['id'], versions[0]['version'], 'mock_panda')
        assert kept['run_summary']['recorded_runs'] == 2 and kept['status'] == 'candidate'
        assert kept['package_files'] == parent_files(dep)
    finally:
        await rt.flora.close()
        log.close()
    # The following episode invokes the persisted package directly: no authoring or patch tool.
    class ReusePlanner:
        async def plan(self, decision, context, toolkit):
            return Intervention(kind='code', reason='Reuse persisted package with current parameters',
                skill_id=kept['id'], skill_version=kept['version'], completion_condition='fixture_aligned',
                parameters={'dx': 0.04})

    _, log2, rt2 = await runtime_fixture(tmp_path / 'next-episode', ReusePlanner())
    rt2.skills = SkillLibrary(tmp_path / 'skills', 'accumulate')
    rt2.reviewer.skills = rt2.skills
    try:
        await rt2.handle(Decision(binding=rt2.memory.binding(rt2.harness.epoch), label=DecisionLabel.ESCALATE))
        events = read_events(log2.root)
        assert sum(e['kind'] == 'skill_invocation' for e in events) == 1
        assert not any(e['kind'] in {'skill_package_proposal', 'new_skill_proposal'} for e in events)
        assert next(e for e in events if e['kind'] == 'recovery_result')['verified'] is True
        assert len(SkillLibrary(tmp_path / 'skills').versions('align-with-library', 'mock_panda')) == 1
    finally:
        await rt2.flora.close()
        log2.close()


async def test_package_tools_are_unavailable_in_read_only_review(tmp_path):
    c, log, h, _, memory = await make_harness(tmp_path)
    toolkit = PlannerToolkit.for_review(h, memory, SkillLibrary(tmp_path / 'skills'), c.runtime.allowed_interventions)
    try:
        assert not {'write_skill_package', 'patch_skill_package'} & {x['name'] for x in planner_tools(review_only=True)}
        for name in ('write_skill_package', 'patch_skill_package'):
            with pytest.raises(ValueError, match='read-only'):
                await toolkit.call(name, {})
    finally:
        log.close()


def test_full_package_identity_changes_when_helper_changes_and_rejects_runtime_drift(tmp_path):
    files = package_files('measured-helper', "def main(ctx): return {'ok': True, 'measured': {}, 'why': ''}")
    first = retain(tmp_path / 'skills', 'measured-helper', files)
    changed = dict(files)
    changed['code/measured_helper/params.py'] = 'LIMIT = 3\n'
    second = retain(tmp_path / 'skills', 'measured-helper', changed)
    assert first['version'] != second['version']
    assert first['sha256'] == second['sha256']
    library = SkillLibrary(tmp_path / 'skills')
    item = library.get(first['id'], first['version'], 'mock_panda')
    drifted = build_package('measured-helper', files, sdk_digest='0' * 64)
    with pytest.raises(ValueError, match='Runtime API changed'):
        execution_bundle(drifted, files, library, 'mock_panda')
    with pytest.raises(ValueError, match='robot'):
        execution_bundle(item['package'], item['package_files'], library, 'PandaOmron')
    module = tmp_path / 'skills' / first['id'] / first['version'] / 'code/measured_helper/params.py'
    module.write_text('LIMIT = 999\n')
    with pytest.raises(ValueError, match='modified'):
        SkillLibrary(tmp_path / 'skills')
    # An already-created episode snapshot owns the previous bytes.
    assert library.get(first['id'], first['version'], 'mock_panda')['package_files'] == files


async def test_native_prints_do_not_corrupt_protocol_and_structured_failure_is_preserved(tmp_path):
    source = '''import os
def main(ctx):
    os.write(1, b'native diagnostic\\n')
    return {'ok': False, 'measured': {'observation_step': ctx.observe()['step']}, 'why': 'Missing observed prerequisite'}
'''
    files = package_files('report-failure', source)
    package = build_package('report-failure', files)
    bundle = execution_bundle(package, files, SkillLibrary(tmp_path / 'skills'), 'mock_panda')
    c, log, h, _, memory = await make_harness(tmp_path)
    try:
        report = await RecoveryRunner(h, memory, c.runtime, log).run(source, 'object_lifted', 'structured-failure',
                                                                   package_bundle=bundle)
        assert report['executed_steps'] == 0 and not report['verified']
        assert report['error'] == 'Missing observed prerequisite'
        assert report['program_result']['measured']['observation_step'] == 0
        assert 'native diagnostic' in (log.root / report['artifacts']['stderr']).read_text()
    finally:
        log.close()
