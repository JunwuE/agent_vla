import copy
import json

from test_codex import call, complete, setup

from vla_agent.data import dumps
from vla_agent.memory_context import compact_historical_memory, experience_card


def historical_memory():
    return dict(
        instruction='Current goal', attempt={'number': 2, 'last_problem': 'current contact failure'},
        phase_memory={'active': 'grasp', 'assessments': [{'status': 'stalled'}]},
        retrieved_experiences=[dict(
            episode_id=f'prior-{i}', task='Close fixture', result={'success': success, 'reason': 'done'},
            evidence_sha256=str(i) * 64,
            notes=[dict(facts=['Measured door did not rotate.'], hypotheses=['Possible contact error.'])],
            code_versions=[dict(verified=False, trace=['dense log'] * 2000,
                                available_skills=[dict(id='close-fixture', version='v1', status='candidate')])],
        ) for i, success in enumerate([False, True, False])],
        task_memory=dict(version='a' * 64, status='task_guidance', profile=dict(
            scope=dict(robot='panda', environment_kind='robocasa', task_name='CloseFixture'),
            summary='Guide with conditional recovery.', phase_guide=[],
            rules=[dict(id='recover', recommendation='consider_skill', status='hypothesis',
                        when=['Only after confirmed failed engagement'], avoid=['Never during normal approach'],
                        skills=[], evidence=[{'episode_id': 'prior-0'}], evidence_available=False)],
        )),
    )


def test_bounded_cards_keep_outcomes_uncertainty_and_current_memory_without_mutation():
    original = historical_memory()
    before = copy.deepcopy(original)
    compact = compact_historical_memory(original, 10000)
    assert original == before
    assert compact['historical_context']['sent_chars'] <= 10000
    assert compact_historical_memory(original, 4000)['historical_context']['sent_chars'] <= 4000
    assert compact['historical_context']['sent_chars'] < compact['historical_context']['original_chars'] / 5
    for key in ('instruction', 'attempt', 'phase_memory'):
        assert compact[key] == original[key]
    assert [x['outcome']['success'] for x in compact['retrieved_experiences']] == [False, True, False]
    assert compact['retrieved_experiences'][0]['unconfirmed_hypotheses'] == ['Possible contact error.']
    rule = compact['task_memory']['rule_index'][0]
    assert not rule['evidence_available'] and rule['status'] == 'hypothesis'
    assert 'expand before adopting' in compact['task_memory']['authority']
    assert compact['retrieved_experiences'][0]['skill_references'][0]['status'] == 'candidate'


def test_large_statement_is_omitted_whole_not_truncated_into_misleading_advice():
    record = historical_memory()['retrieved_experiences'][0]
    statement = 'Do this ' + 'when conditions are verified ' * 500 + ' but never otherwise.'
    record['notes'] = [dict(facts=[statement], hypotheses=[])]
    card = experience_card(record)
    assert card['reported_facts'] == [] and card['reported_facts_omitted'] == 1
    assert len(dumps(card)) <= 1800
    assert record['notes'][0]['facts'][0] == statement


async def test_codex_receives_cards_but_explicit_read_returns_full_evidence(tmp_path):
    from vla_agent.experience import ExperienceStore

    events = [call(1, 'read_experience', dict(episode_id='prior-0')),
              call(2, 'submit_intervention', dict(kind='continue', reason='Keep going')), complete()]
    log, rt, toolkit, transport, planner, context, decision = await setup(tmp_path, [events])
    memory = historical_memory()
    record = memory['retrieved_experiences'][0]
    record['robot'] = toolkit.memory.task.robot
    store = ExperienceStore(tmp_path / 'history')
    store.index = [record]
    toolkit.experiences = store
    context['memory'] = memory
    try:
        await planner.plan(decision, context, toolkit)
        sent = next(p for m, p in transport.requests if m == 'turn/start')
        sent_memory = json.loads(sent['input'][0]['text'])['memory']
        assert sent_memory['historical_context']['mode'] == 'extractive_index'
        assert 'code_versions' not in sent_memory['retrieved_experiences'][0]
        expanded = json.loads(transport.replies[0][1]['contentItems'][0]['text'])
        assert expanded == record
        assert store.index[0] == record
    finally:
        await planner.close()
        log.close()
