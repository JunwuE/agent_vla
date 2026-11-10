from test_runtime import make_harness

from vla_agent.recovery import RecoveryRunner
from vla_agent.stages import StageTracker


async def test_nested_worker_stages_resume_parent_and_keep_measurements(tmp_path):
    c, log, harness, lease, memory = await make_harness(tmp_path)
    code = '''def run(ctx):
    with ctx.stage("outer") as outer:
        with ctx.stage("inner") as inner:
            ctx.move_relative([.01,0,0], False, steps=1)
            inner.record(measured="child")
        ctx.move_relative([.01,0,0], False, steps=1)
        outer.record(measured="parent")
    return "UNRESOLVED"
'''
    report = await RecoveryRunner(harness, memory, c.runtime, log).run(code, 'object_lifted', 'nested')
    assert report['error'] is None and report['executed_steps'] == 2 and not report['verified']
    outer, inner = report['stages']
    assert inner['parent_stage_id'] == outer['stage_id']
    assert outer['end_step'] == 2 and inner['end_step'] == 1
    assert outer['program_measurements'] == [{'measured':'parent'}]
    assert inner['program_measurements'] == [{'measured':'child'}]
    assert outer['status'] == inner['status'] == 'unchecked'
    log.close()


async def test_nested_completion_cannot_bypass_parent_verifier(tmp_path):
    c, log, harness, lease, memory = await make_harness(tmp_path)
    code = '''def run(ctx):
    with ctx.stage("grasp", completion_condition="object_lifted"):
        with ctx.stage("alignment"):
            ctx.move_relative([.01,0,0], False, steps=1)
    ctx.move_relative([.01,0,0], False, steps=1)
    return "SUBTASK_DONE"
'''
    report = await RecoveryRunner(harness, memory, c.runtime, log).run(code, 'object_lifted', 'checked')
    assert report['executed_steps'] == 1 and not report['verified']
    assert report['failed_stage'] == 'grasp' and report['stages'][0]['checks'][-1]['passed'] is False
    assert report['stages'][1]['status'] == 'unchecked'
    log.close()


async def test_nested_budget_cleanup_closes_every_context_without_extra_actions(tmp_path):
    c, log, harness, lease, memory = await make_harness(tmp_path)
    c.runtime.recovery_steps = 1
    code = '''def run(ctx):
    with ctx.stage("outer"):
        with ctx.stage("inner"):
            ctx.move_relative([.01,0,0], False, steps=2)
    return "SUBTASK_DONE"
'''
    report = await RecoveryRunner(harness, memory, c.runtime, log).run(code, 'object_lifted', 'budget')
    assert report['executed_steps'] == 1 and not report['verified']
    assert all(s['end_step'] == 1 and s['status'] == 'failed' for s in report['stages'])
    log.close()


async def test_legacy_stage_markers_and_out_of_order_completion(tmp_path):
    import pytest

    c, log, harness, lease, memory = await make_harness(tmp_path)
    tracker = StageTracker(harness, 'fixture')
    legacy = await tracker.start({'name':'legacy'})
    outer = await tracker.start({'name':'outer'})
    assert tracker.rows[0]['status'] == 'unchecked'
    inner = await tracker.start({'name':'inner', 'parent_stage_id':outer['stage_id']})
    with pytest.raises(ValueError, match='parent'):
        await tracker.start({'name':'invalid child', 'parent_stage_id':'unknown'})
    assert tracker.active['stage_id'] == inner['stage_id']
    with pytest.raises(ValueError, match='active stage'):
        await tracker.finish(outer['stage_id'])
    assert tracker.active['stage_id'] == inner['stage_id']
    await tracker.finish(inner['stage_id'])
    await tracker.finish(outer['stage_id'])
    with pytest.raises(ValueError, match='active stage'):
        await tracker.finish(legacy['stage_id'])
    assert tracker.active is None and not tracker.parents
    log.close()
