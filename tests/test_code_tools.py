import json

import pytest
from test_runtime import make_harness
from test_skill_packages import dependency, package_files, parent_files, retain

from vla_agent.code_tools import code_view, sha
from vla_agent.skills import SkillLibrary
from vla_agent.toolkit import PlannerToolkit, planner_tools
from vla_agent.workspace import RecoveryWorkspace


def test_source_is_lossless_paged_and_does_not_repeat_contract_or_documentation(tmp_path):
    files = package_files('measured-control', 'def main(ctx): return {}\n',
                          helper='def move(ctx):\n    return 1\n\n' + '# ' + '注释' * 8000 + '\n')
    workspace = RecoveryWorkspace(tmp_path / 'workspace', {})
    item = workspace.write_package(files, 'test', 'test', skill_name='measured-control',
        parameter_schema={'type': 'object', 'description': 'schema' * 4000})
    value = workspace.read(item['version'])
    index = code_view(value)
    assert len(json.dumps(index)) < 2500
    assert 'documentation' not in index and 'parameter_schema' not in index
    path = 'code/measured_control/__init__.py'
    chunks, offset = [], 0
    while offset is not None:
        response = code_view(value, path, offset=offset, limit=500)
        assert 'contract' not in response and 'file_paths' not in response
        chunk = response['file']
        assert chunk['sha256'] == sha(files[path]) and len(chunk['source']) <= 500
        chunks.append(chunk['source'])
        offset = chunk['next_offset']
    assert ''.join(chunks) == files[path]
    symbol = code_view(value, path, symbol='move')['file']
    assert symbol['source'] == 'def move(ctx):\n    return 1\n' and symbol['first_line'] == 1
    contract = code_view(value, view='contract')['contract']
    assert contract['parameter_schema'] == value['parameter_schema']
    assert code_view(value, view='documentation', limit=128)['page']['truncated']
    with pytest.raises(ValueError, match='exactly one'):
        code_view(value, path, symbol='missing')
    with pytest.raises(ValueError, match='frozen package'):
        code_view(value, '../outside')


def test_edits_validate_entire_transaction_and_keep_unchanged_modules(tmp_path):
    workspace = RecoveryWorkspace(tmp_path / 'workspace', {})
    files = package_files('measured-control', 'def main(ctx): return {}\n')
    name = 'code/measured_control/params.py'
    first = workspace.write_package(files, 'test', 'test', skill_name='measured-control',
        parameter_schema={'type': 'object'}, skill_contract={'preconditions': ['task_success']})
    stable = workspace.read(first['version'])
    good = dict(path=name, old_text='LIMIT = 2', new_text='LIMIT = 3')
    bad_cases = [
        [good, dict(path='code/run.py', old_text='missing', new_text='bad')],
        [good, dict(path=name, old_text='LIMIT', new_text='OTHER')],
        [dict(path=name, operation='write', source='LIMIT=4')],
        [dict(path=name, operation='write', source='LIMIT=4', expected_sha256='0' * 64)],
        [dict(path='SKILL.md', operation='delete', expected_sha256=sha(files['SKILL.md']))],
        [dict(path='../escape.py', operation='write', source='pass')],
    ]
    for edits in bad_cases:
        with pytest.raises(ValueError):
            workspace.edit(first['version'], edits, 'invalid fixture', 'test')
        assert len(workspace.versions) == 1
        assert len(list((workspace.root / 'versions').iterdir())) == 1
        assert workspace.read(first['version']) == stable
    second = workspace.edit(first['version'], [good,
        dict(path='code/measured_control/perception.py', operation='write', source='def locate(): return None\n')],
        'related changes', 'test')
    assert len(workspace.versions) == 2
    updated = workspace.read(second['version'])
    assert updated['parameter_schema'] == stable['parameter_schema']
    assert updated['skill_contract'] == stable['skill_contract']
    assert updated['package_files']['code/run.py'] == files['code/run.py']
    assert updated['sha256'] == first['sha256']
    assert updated['package']['sha256'] != first['package']['sha256']
    third = workspace.edit(second['version'], [dict(path=name, operation='write', source='LIMIT = 4\n',
        expected_sha256=sha(updated['package_files'][name])), dict(path='code/measured_control/perception.py',
        operation='delete', expected_sha256=sha(updated['package_files']['code/measured_control/perception.py']))],
        'calibration and cleanup', 'test')
    final = workspace.read(third['version'])
    assert name in workspace.diff(third['version'])
    assert final['package_files'][name] == 'LIMIT = 4\n'
    assert 'code/measured_control/perception.py' not in final['package_files']
    assert 'perception.py' in workspace.diff(third['version'])
    with pytest.raises(ValueError, match='No code or contract change'):
        workspace.edit(third['version'], [dict(path=name, old_text='LIMIT = 4', new_text='LIMIT = 4')], 'same', 'same')


def test_interface_resolves_public_function_signature_without_executing_code(tmp_path):
    files = package_files('reusable-control', 'def main(ctx): return {}\n',
                          helper='from .controller import move as approach\n')
    files['code/reusable_control/controller.py'] = '''raise RuntimeError("must not import")
def move(ctx, target, *, tolerance=0.01):
    """Move to a fresh measured target."""
    return {}
'''
    workspace = RecoveryWorkspace(tmp_path / 'workspace', {})
    item = workspace.write_package(files, 'test', 'test', skill_name='reusable-control')
    interface = code_view(workspace.read(item['version']))
    exported = interface['public_exports'][0]
    assert exported['name'] == 'approach'
    assert exported['signature'] == 'approach(ctx, target, *, tolerance=0.01)'
    assert exported['definition_path'] == 'code/reusable_control/controller.py'
    assert exported['definition_symbol'] == 'move'


async def test_write_receipt_checks_pinned_dependencies_once_without_moving_robot(tmp_path):
    dep = dependency(tmp_path / 'skills')
    _, log, harness, _, memory = await make_harness(tmp_path)
    workspace = RecoveryWorkspace(tmp_path / 'workspace', {})
    toolkit = PlannerToolkit(harness, memory, SkillLibrary(tmp_path / 'skills'), workspace=workspace)
    try:
        args = dict(files=parent_files(dep), uses=[dep], skill_name='align-with-library',
                    when_to_use='Fixture only', reuse_reason='Compose existing fixture primitive',
                    hypothesis='Test reuse', expected_effect='Test fixture', parameter_schema={'type': 'object'})
        written = await toolkit.call('write_skill_package', args)
        assert written['saved'] and written['checks']['ok'] and not written['executed']
        assert len(written['checks']['dependencies']) == 2
        assert 'source' not in written and 'package' not in written and 'parameter_schema' not in written
        assert len(json.dumps(written)) < 4000
        name = 'code/align_with_library/params.py'
        edited = await toolkit.call('edit_program', dict(base_version=written['version'],
            edits=[dict(path=name, old_text='LIMIT = 2', new_text='LIMIT = 3')],
            hypothesis='Fixture calibration', expected_effect='Keep dependency unchanged'))
        assert edited['checks']['ok']
        assert [f['path'] for f in edited['changed_files']] == [name]
        assert len(edited['checks']['dependencies']) == 2 and harness.obs.step == 0
        broken = await toolkit.call('edit_program', dict(base_version=edited['version'],
            edits=[dict(path=name, old_text='LIMIT = 3', new_text='syntax !')],
            hypothesis='Invalid syntax fixture', expected_effect='Return diagnostics, never execute'))
        assert broken['saved'] and not broken['checks']['ok'] and not broken['executed']
        assert len(workspace.versions) == 3 and harness.obs.step == 0
        # Missing dependency is reported against the saved draft; never a misleading failed save.
        missing = dict(skill_id='uninstalled-helper', version='unknown')
        args.update(files=parent_files(missing), uses=[missing])
        missing_receipt = await toolkit.call('write_skill_package', args)
        assert missing_receipt['saved'] and not missing_receipt['checks']['ok']
        assert 'snapshot' in missing_receipt['checks']['error']
    finally:
        log.close()


async def test_skill_catalog_pages_then_invocation_contract_without_source_copy(tmp_path):
    for n in range(8):
        files = package_files('reusable-step', f'def main(ctx): return {{"ok": False, "measured": {{}}, "why": "{n}"}}\n')
        retain(tmp_path / 'skills', 'reusable-step', files)
    c, log, harness, _, memory = await make_harness(tmp_path)
    library = SkillLibrary(tmp_path / 'skills')
    toolkit = PlannerToolkit(harness, memory, library)
    try:
        cards = await toolkit.call('search_skills', dict(query='reusable', include_candidates=True))
        assert len(cards) == 1 and len(json.dumps(cards)) < 4000
        assert 'code' not in cards[0] and 'parameter_schema' not in cards[0]
        first = await toolkit.call('list_skill_versions', dict(skill_id='reusable-step', limit=3))
        assert first['total'] == 8 and len(first['versions']) == 3 and first['next_offset'] == 3
        seen, offset = [], 0
        while offset is not None:
            result = await toolkit.call('list_skill_versions', dict(skill_id='reusable-step', offset=offset, limit=3))
            seen += [x['version'] for x in result['versions']]
            offset = result['next_offset']
        assert len(set(seen)) == 8
        card = first['versions'][0]
        interface = await toolkit.call('read_skill', dict(skill_id=card['id'], version=card['version']))
        assert interface['status'] == 'candidate' and 'source' not in interface
        contract = await toolkit.call('read_skill', dict(skill_id=card['id'], version=card['version'], view='contract'))
        assert contract['contract']['required_capabilities'] == ['native_action12']
        reviewer = PlannerToolkit.for_review(harness, memory, library, c.runtime.allowed_interventions)
        with pytest.raises(ValueError, match='read-only'):
            await reviewer.call('edit_program', {})
        names = {t['name'] for t in planner_tools()}
        assert 'edit_program' in names and not {'patch_program', 'patch_program_batch', 'patch_skill_package'} & names
        assert 'patch_program' in {t['name'] for t in planner_tools(include_legacy=True)}
        assert 'edit_program' not in {t['name'] for t in planner_tools(review_only=True)}
    finally:
        log.close()


def test_one_editor_also_handles_legacy_programs_with_exact_contract(tmp_path):
    workspace = RecoveryWorkspace(tmp_path / 'workspace', {})
    first = workspace.write('def run(ctx): return "UNRESOLVED"', 'test', 'test',
                            parameter_schema={'type': 'object'})
    second = workspace.edit(first['version'], [dict(path='policy.py', old_text='"UNRESOLVED"',
        new_text='"RECOVERED"')], 'fixture claim', 'host must still verify')
    assert workspace.read(second['version'])['parameter_schema'] == {'type': 'object'}
    assert workspace.check(second['version'])['ok']
    third = workspace.edit(second['version'], [], 'Clarify measurement provenance', 'Same source and bounds',
                           parameter_sources=['Fresh metric depth in the base frame'])
    assert workspace.read(third['version'])['parameter_schema'] == {'type': 'object'}
    assert workspace.read(third['version'])['parameter_sources'] == ['Fresh metric depth in the base frame']
    with pytest.raises(ValueError, match='migrate'):
        workspace.edit(second['version'], [dict(path='helper.py', operation='write', source='pass')], 'bad', 'test')
