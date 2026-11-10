import copy
import json

import pytest
from test_codex import call, complete, setup
from test_runtime import config

from vla_agent.data import Recorder, dumps
from vla_agent.working_context import compact_working_context, read_context


def test_dense_notes_and_feedback_are_indexed_with_exact_causal_readback(tmp_path):
    log = Recorder(tmp_path/'episode', config())
    context = dict(binding=dict(episode_id='e1', observation_step=10),
        memory=dict(instruction='Close the door.', executor='code', attempt={'last_problem':'contact uncertain'},
            notes=[dict(event_id=i, facts=['Measured surface: '+str(i)]*1000,
                        hypotheses=['Possible occlusion; not confirmed.']) for i in range(20)]),
        recovery_feedback=dict(result='UNRESOLVED', error='Contact guard', verified=False,
            program_notes=[{'facts':['dense measurements']*2000}], failed_stage='approach', stages=[]),
        workspace=dict(versions=[dict(version=str(i), parameter_schema={'long':['schema']*1000}) for i in range(8)]))
    original = copy.deepcopy(context)
    try:
        indexed = compact_working_context(context, log)
        assert context == original
        assert len(dumps(indexed)) < len(dumps(original))/4
        assert indexed['memory']['attempt'] == context['memory']['attempt']
        assert indexed['workspace']['current_version'] == context['workspace']['versions'][-1]
        assert indexed['recovery_feedback']['error'] == 'Contact guard'
        reference = indexed['working_context']['reference']
        value = read_context(log, reference, ['memory','notes',19,'hypotheses'],0,5,'e1',10)
        assert value['value'] == original['memory']['notes'][19]['hypotheses']
        with pytest.raises(ValueError, match='snapshot'):
            read_context(log,reference,['memory'],0,5,'e1',9)
        with pytest.raises(ValueError, match='snapshot'):
            read_context(log,reference,['memory'],0,5,'another',10)
        path=log.root/'context_archive'/f'{reference}.json'
        path.write_text(path.read_text()+' ')
        with pytest.raises(ValueError, match='hash'):
            read_context(log,reference,['memory'],0,5,'e1',10)
    finally:
        log.close()


async def test_codex_working_index_and_turn_rotation_preserve_authority(tmp_path):
    turns=[[call(1,'submit_intervention',dict(kind='continue',reason='normal')), complete()]]*2
    log, rt, toolkit, transport, planner, context, decision = await setup(tmp_path,[*turns])
    planner.config.working_memory_mode = 'summary'
    planner.config.thread_max_turns = 1
    context['memory'] = toolkit.memory.summary()
    try:
        await planner.plan(decision,context,toolkit)
        await planner.plan(decision,context,toolkit)
        requests=[p for m,p in transport.requests if m=='turn/start']
        assert len(requests)==2 and planner.calls==2
        assert sum(m=='thread/start' for m,p in transport.requests)==2
        sent=json.loads(requests[-1]['input'][0]['text'])
        assert sent['working_context']['mode']=='extractive_index'
        assert sent['binding']['subtask_version']==context['binding'].subtask_version
        expanded=await toolkit.call('read_context',dict(reference=sent['working_context']['reference'],
                                                      path=['memory','instruction']))
        assert expanded['value']==toolkit.memory.instruction
    finally:
        await planner.close()
        log.close()
