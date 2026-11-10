import copy
import hashlib
import json

import pytest
from test_runtime import config, make_harness

from vla_agent.adapters.mock import MockDecision, MockEnvironment, MockVLA
from vla_agent.codex_planner import CodexPlanner
from vla_agent.contracts import Intervention
from vla_agent.data import Recorder, dumps, read_events
from vla_agent.experience import ExperienceStore
from vla_agent.runtime import AgentRuntime
from vla_agent.skills import SkillLibrary
from vla_agent.task_memory import TaskMemoryStore, TaskProfile
from vla_agent.toolkit import PlannerToolkit


def fixtures(root, scope=None):
    c = config()
    scope = scope or dict(robot=c.task.robot, environment_kind=c.environment.kind, task_name=c.environment.task_name)
    record = dict(robot=scope['robot'], episode_id='historical-success', task='Close toaster door',
                  conditions=dict(environment=dict(kind=scope['environment_kind'], task_name=scope['task_name'])),
                  phase_memory={}, notes=[], tools={}, code_versions=[], result=dict(success=True),
                  evidence_root='fixture', evidence_sha256='a'*64)
    experiences = root/'experiences'
    experiences.mkdir(parents=True)
    (experiences/(hashlib.sha256(dumps(record).encode()).hexdigest()+'.json')).write_text(dumps(record))
    evidence = dict(episode_id=record['episode_id'], evidence_sha256=record['evidence_sha256'])
    profile = TaskProfile(scope=scope, summary='Fixture task guidance', applicability=['Current evidence must match'],
        rules=[dict(id='keep-effective-vla', when=['Measured progress'], recommendation='continue_vla',
                    rationale='This task completed without code', status='observed_pattern', evidence=[evidence])])
    return profile, ExperienceStore(experiences)


def test_profiles_are_task_scoped_versioned_and_frozen_with_checked_evidence(tmp_path):
    profile, experiences = fixtures(tmp_path)
    root = tmp_path/'tasks'
    first = TaskMemoryStore(root).write_profile(profile, experiences)
    store = TaskMemoryStore(root)
    scope = profile.scope.model_dump()
    skills = SkillLibrary(tmp_path/'skills')
    snapshot = store.read(scope, experiences, skills)
    assert snapshot['version'] == first['version'] and snapshot['profile']['rules'][0]['evidence_available']
    assert store.read(dict(scope, task_name='DifferentTask'), experiences, skills)['status'] == 'missing'
    assert store.read(dict(scope, robot='other'), experiences, skills)['status'] == 'missing'
    changed = profile.model_copy(deep=True)
    changed.summary = 'A revised guide for a later episode'
    second = store.write_profile(changed, experiences, expected_version=first['version'])
    assert first['version'] != second['version'] and store.read(scope, experiences, skills) == snapshot
    assert TaskMemoryStore(root).read(scope, experiences, skills)['version'] == second['version']
    assert len(list(root.glob('*/versions/*.json'))) == 2
    with pytest.raises(ValueError, match='revision'):
        store.write_profile(profile, experiences, expected_version=first['version'])
    bad = profile.model_copy(deep=True)
    bad.rules[0].evidence[0].evidence_sha256 = 'b'*64
    with pytest.raises(ValueError, match='checksum'):
        store.write_profile(bad, experiences, expected_version=second['version'])
    path = next(root.glob('*/profile.json'))
    path.write_text(path.read_text().replace('A revised guide', 'Tampered guide'))
    with pytest.raises(ValueError, match='checksum'):
        TaskMemoryStore(root)


def test_missing_evidence_and_skills_are_explicit_without_promoting_a_rule(tmp_path):
    profile, experiences = fixtures(tmp_path)
    raw = profile.model_dump()
    raw['rules'][0]['skills'] = [dict(skill_id='unavailable-procedure', version='v1')]
    profile = TaskProfile.model_validate(raw)
    root = tmp_path/'tasks'
    TaskMemoryStore(root).write_profile(profile, experiences)
    snapshot = TaskMemoryStore(root).read(profile.scope, ExperienceStore(tmp_path/'absent'), SkillLibrary(tmp_path/'skills'))
    rule = snapshot['profile']['rules'][0]
    assert not rule['evidence_available'] and rule['skills'][0]['available'] is False
    assert rule['status'] == 'observed_pattern' and 'Historical guidance' in snapshot['authority']


def test_experience_lookup_enforces_exact_scope_even_with_similar_language(tmp_path):
    profile, store = fixtures(tmp_path)
    other = copy.deepcopy(store.index[0])
    other.update(episode_id='other-task', notes=['Close toaster door']*5)
    other['conditions']['environment']['task_name'] = 'OtherTask'
    unknown = dict(other, episode_id='missing-scope', conditions={})
    store.index += [other, unknown]
    found = store.search(profile.scope.robot, 'Close toaster door', scope=profile.scope)
    assert [x['episode_id'] for x in found] == ['historical-success']
    with pytest.raises(ValueError, match='snapshot'):
        store.get(profile.scope.robot, 'other-task', scope=profile.scope)


async def test_task_memory_proposal_is_evidence_backed_and_has_no_live_effect(tmp_path):
    c, log, h, _, memory = await make_harness(tmp_path)
    profile, experiences = fixtures(tmp_path/'library')
    root = tmp_path/'library/tasks'
    TaskMemoryStore(root).write_profile(profile, experiences)
    skills = SkillLibrary(tmp_path/'skills')
    memory.task_scope = profile.scope.model_dump()
    memory.task_memory = TaskMemoryStore(root).read(profile.scope, experiences, skills)
    visible = log.event('current_observation', observation_step=h.obs.step)
    reviewer = PlannerToolkit.for_review(h, memory, skills, c.runtime.allowed_interventions, experiences=experiences)
    before = memory.summary()
    proposal = dict(parent_version=memory.task_memory['version'], rule=dict(id='new-hypothesis',
        when=['A new observed condition'], recommendation='inspect', rationale='Needs verification'),
        current_event_ids=[visible])
    try:
        loaded = await reviewer.call('read_task_memory', {})
        loaded['profile']['summary'] = 'caller modification'
        assert memory.summary() == before
        receipt = await reviewer.call('propose_task_memory', proposal)
        assert receipt['applied'] is False and memory.summary() == before and h.obs.step == 0
        rows = list((log.root/'task_memory_candidates').glob('*.json'))
        assert len(rows) == 1 and json.loads(rows[0].read_text())['scope'] == memory.task_scope
        invisible = log.event('later_event', observation_step=h.obs.step+1)
        with pytest.raises(ValueError, match='visible events'):
            await reviewer.call('propose_task_memory', dict(proposal, current_event_ids=[invisible]))
        with pytest.raises(ValueError, match='stale'):
            await reviewer.call('propose_task_memory', dict(proposal, parent_version='b'*64))
        with pytest.raises(ValueError, match='requires'):
            await reviewer.call('propose_task_memory', dict(proposal, current_event_ids=[]))
        assert TaskMemoryStore(root).read(profile.scope, experiences, skills)['version'] == memory.task_memory['version']
    finally:
        log.close()


async def test_runtime_loads_task_guide_before_initial_plan_without_executing_it(tmp_path):
    profile, experiences = fixtures(tmp_path/'library')
    root = tmp_path/'library/tasks'
    receipt = TaskMemoryStore(root).write_profile(profile, experiences)
    c = config()
    c.runtime.experience_library = str(experiences.root)
    c.runtime.initial_planning = True
    c.runtime.max_steps = 1
    c.runtime.memory_mode = 'frozen'

    class Planner:
        async def plan(self, decision, context, toolkit):
            assert context['memory']['task_memory']['version'] == receipt['version']
            assert context['memory']['phase_memory']['revision'] == 0
            assert context['memory']['instruction'] == c.task.subtasks[0].instruction
            assert toolkit.harness.obs.step == 0
            assert (await toolkit.call('read_task_memory', {}))['status'] == 'task_guidance'
            return Intervention(kind='stop', reason='Task guide integration checked')

        async def close(self):
            pass

    log = Recorder(tmp_path/'episode', c)
    try:
        result = await AgentRuntime(c, MockEnvironment(), MockVLA(), MockDecision(), Planner(), log).run()
        assert not result['error'] and result['control_steps'] == 0
        rows = [r for r in read_events(log.root) if r['kind'] == 'task_memory_snapshot']
        assert len(rows) == 1 and rows[0]['memory']['version'] == receipt['version']
        assert len(list(experiences.root.glob('*.json'))) == 1  # frozen run
    finally:
        log.close()


def test_codex_deduplicates_unchanged_task_guide_with_retrievable_reference(tmp_path):
    c = config()
    log = Recorder(tmp_path/'episode', c)
    planner = CodexPlanner(c.planner, log)
    memory = dict(task_memory=dict(version='a'*64, status='task_guidance',
        profile=dict(scope={'task_name': 'fixture'}, summary='task evidence')))
    try:
        first = planner.experience_context(memory)
        second = planner.experience_context(memory)
        assert 'profile' in first['task_memory'] and 'context_reference' in second['task_memory']
        assert 'read_task_memory' in second['task_memory']['detail'] and 'profile' in memory['task_memory']
    finally:
        log.close()
