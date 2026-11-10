import asyncio
import json

import httpx
import numpy as np
import pytest
from test_flora import advance, answer
from test_runtime import config, make_harness

from vla_agent.adapters.mock import MockEnvironment, MockVLA
from vla_agent.config import FloraMonitorConfig
from vla_agent.contracts import Binding, Chunk, DecisionLabel, ExecutionReview
from vla_agent.data import Recorder, export_decision_windows, read_events
from vla_agent.flora import prepare_segment
from vla_agent.flora_supervision import ScoreRouter
from vla_agent.memory import WorkingMemory
from vla_agent.runtime import AgentRuntime


def binding(step):
    return Binding(episode_id='test', subtask_version=1, instruction_version=1,
                   control_epoch=1, observation_step=step, observation_time=0)


@pytest.mark.parametrize('scores,expected', [
    ([1], ['KEEP']),
    ([2, 2], ['KEEP', 'ESCALATE']),
    ([2, 3], ['KEEP', 'KEEP']),
    ([3, 3, 3], ['KEEP', 'KEEP', 'ESCALATE']),
    ([4], ['ESCALATE']),
    ([5], ['ESCALATE']),
    ([3, 1], ['KEEP', 'ESCALATE']),
])
def test_score_rules_request_review_without_claiming_failure_or_completion(scores, expected):
    router = ScoreRouter(FloraMonitorConfig())
    actual = [router.route(dict(status='scored', score=s), binding(50 * (i + 1)), ('phase',)).label
              for i, s in enumerate(scores)]
    assert actual == expected


def test_new_phase_or_unavailable_score_breaks_low_score_streak():
    router = ScoreRouter(FloraMonitorConfig())
    low = dict(status='scored', score=1)
    assert router.route(low, binding(50), ('approach',)).label == 'KEEP'
    assert router.route(low, binding(100), ('close',)).label == 'KEEP'
    assert router.route(dict(status='unavailable', score=None), binding(150), ('close',)).label == 'ESCALATE'
    assert router.route(low, binding(200), ('close',)).label == 'KEEP'


async def test_explicit_window_respects_attempt_boundary_and_uses_only_recorded_past(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    try:
        await advance(h, lease, memory, 12)
        snapshot = memory.binding(h.epoch)
        await advance(h, lease, memory, 2)
        payload = prepare_segment(log, memory, snapshot, c.flora, with_rationale=False, window_steps=5)
        assert payload['segment']['start_step'] == 7
        assert payload['segment']['end_step'] == 12
        assert max(payload['segment']['sampled_steps']) == 12
        assert payload['with_rationale'] is False
        memory.begin_retry('new attempt')
        memory.resume_retry()
        await advance(h, lease, memory, 3)
        payload = prepare_segment(log, memory, memory.binding(h.epoch), c.flora, window_steps=5)
        assert payload['segment']['start_step'] == 14
    finally:
        log.close()


class ProgressEnvironment(MockEnvironment):
    def __init__(self):
        super().__init__()
        self.past_review = asyncio.Event()

    async def step(self, action):
        await asyncio.sleep(0.003)
        obs = await super().step(action)
        if self.step_id >= 180:
            self.past_review.set()
        return obs


class ReviewingPlanner:
    def __init__(self, env):
        self.env, self.calls = env, 0

    async def plan(self, decision, context, toolkit):
        self.calls += 1
        assert context['intent'] == 'review_running'
        await self.env.past_review.wait()
        return ExecutionReview(action='continue', assessment='uncertain', phase='approach',
                               reason='Low score is not proof that normal approach has failed')

    async def close(self):
        pass


def monitor_config():
    c = config()
    c.decision.kind = 'flora'
    c.decision.interval_steps = 50
    c.decision.max_age_steps = 50
    c.flora.enabled = True
    c.runtime.max_steps = 225
    c.runtime.initial_planning = False
    c.runtime.nonblocking_review = True
    c.runtime.audit_interval_steps = 0
    c.runtime.review_max_age_steps = 150
    c.runtime.stagnation_enabled = False  # Isolate score routing; numeric routing is tested separately.
    return c


async def test_flora_ticks_continue_during_astra_review_without_duplicate_reviews(tmp_path):
    c = monitor_config()
    env = ProgressEnvironment()
    log = Recorder(tmp_path / 'episode', c)
    rt = AgentRuntime(c, env, MockVLA(), None, ReviewingPlanner(env), log)
    requests = []

    async def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload['with_rationale'] is False
        assert payload['segment']['end_step'] <= env.step_id
        return httpx.Response(200, json=answer(payload, score=1))

    await rt.monitor.evaluator.client.aclose()
    rt.monitor.evaluator.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await rt.run()
    finally:
        log.close()
    events = read_events(log.root)
    assert result['error'] is None
    assert [r['binding']['observation_step'] for r in requests] == [50, 100, 150, 200]
    assert result['flora_monitor_calls'] == 4 and result['flora_tool_calls'] == 0
    assert result['astra_reviews'] == 1 and result['interventions'] == 0
    assert any(e['kind'] == 'review_coalesced' and e['reason'] == 'astra_already_reviewing' for e in events)
    assert any(e['kind'] == 'review_coalesced' and e['reason'] == 'score_review_cooldown' for e in events)
    assert all(e['source'] == 'vla' for e in events if e['kind'] == 'action_executed')
    exported = tmp_path / 'windows.jsonl'
    assert export_decision_windows(log.root, exported) == 4
    for row in map(json.loads, exported.read_text().splitlines()):
        assert row['backend'] == 'flora'
        assert (log.root / row['model_request_path']).is_file()
        assert row['segment']['end_step'] == row['binding']['observation_step']


@pytest.mark.parametrize('change', ['attempt', 'phase', 'control'])
async def test_stale_monitor_result_never_updates_live_memory_or_score_streak(tmp_path, change):
    c, log, h, lease, memory = await make_harness(tmp_path)
    c.decision.kind, c.flora.enabled = 'flora', True
    await advance(h, lease, memory, 10)
    rt = AgentRuntime(c, h.env, MockVLA(), None, ReviewingPlanner(None), log)
    rt.harness, rt.memory = h, memory

    async def handler(request):
        return httpx.Response(200, json=answer(json.loads(request.content)))

    await rt.monitor.evaluator.client.aclose()
    rt.monitor.evaluator.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        rt.submit_decision('periodic')
        await rt.pending
        if change == 'attempt':
            memory.attempt_id = 'another-attempt'
        elif change == 'phase':
            memory.phases.active = 'another-phase'
        else:
            h.epoch += 1
        assert await rt.receive_decision() is None
        assert not memory.reward_assessments and not rt.monitor.router.history
        assert log.counts['stale_decision'] == 1
    finally:
        await rt.monitor.close()
        await rt.flora.close()
        log.close()


async def test_monitor_deadline_includes_wait_for_diagnostic_lock(tmp_path):
    c = monitor_config()
    c.decision.timeout_seconds = 0.01
    log = Recorder(tmp_path / 'lock', c)
    rt = AgentRuntime(c, MockEnvironment(), MockVLA(), None, ReviewingPlanner(None), log)
    try:
        async with rt.flora.lock:
            result = await rt.monitor.evaluate(WorkingMemory(c.task, 'test'), binding(50))
        assert result['status'] == 'unavailable'
        assert result['reason'] == 'flora_monitor_timeout'
        assert rt.monitor.router.route(result, binding(50), ('task',)).label == DecisionLabel.ESCALATE
    finally:
        await rt.monitor.close()
        await rt.flora.close()
        log.close()


async def test_slow_flora_skips_ticks_without_stalling_vla_or_backlogging_requests(tmp_path):
    c = monitor_config()
    env = ProgressEnvironment()
    log = Recorder(tmp_path / 'slow', c)
    rt = AgentRuntime(c, env, MockVLA(), None, ReviewingPlanner(env), log)
    requests = []

    async def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            await env.past_review.wait()  # If VLA blocks on Flora, this deadlocks.
        return httpx.Response(200, json=answer(payload, score=3))

    await rt.monitor.evaluator.client.aclose()
    rt.monitor.evaluator.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await rt.run()
    finally:
        log.close()
    events = read_events(log.root)
    assert result['error'] is None and result['control_steps'] == 225
    assert [r['binding']['observation_step'] for r in requests] == [50, 200]
    assert [r['step'] for r in events if r['kind'] == 'monitor_tick_skipped'] == [100, 150]
    assert any(r['kind'] == 'flora_monitor_result' and not r['accepted'] for r in events)
    assert result['astra_reviews'] == 0


async def test_monitor_budget_is_independent_from_astra_diagnostic_budget(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    c.decision.kind, c.flora.enabled = 'flora', True
    c.flora.max_calls = 1
    c.flora.with_rationale = False
    rt = AgentRuntime(c, h.env, MockVLA(), None, ReviewingPlanner(None), log)
    await advance(h, lease, memory, 10)

    async def handler(request):
        return httpx.Response(200, json=answer(json.loads(request.content), score=2))

    for client in [rt.flora, rt.monitor.evaluator]:
        await client.client.aclose()
        client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        assert rt.monitor.evaluator.lock is rt.flora.lock
        assert (await rt.monitor.evaluate(memory, memory.binding(h.epoch)))['status'] == 'scored'
        assert (await rt.flora.evaluate(memory, memory.binding(h.epoch)))['status'] == 'scored'
        await advance(h, lease, memory, 2)
        assert (await rt.flora.evaluate(memory, memory.binding(h.epoch)))['status'] == 'budget_exhausted'
        assert (await rt.monitor.evaluate(memory, memory.binding(h.epoch)))['status'] == 'scored'
        assert rt.flora.calls == 1 and rt.monitor.evaluator.calls == 2
    finally:
        await rt.monitor.close()
        await rt.flora.close()
        log.close()


@pytest.mark.parametrize('seconds,delayed_score', [(2.0, False), (4.0, True)])
async def test_low_motion_requests_astra_independently_without_stopping_vla_or_flora_clock(
        tmp_path, seconds, delayed_score):
    class HoldingVLA(MockVLA):
        async def infer(self, obs, instruction, binding):
            return Chunk(binding, np.zeros((5, 12)))

    class Planner(ReviewingPlanner):
        def __init__(self, env):
            super().__init__(env)
            self.contexts = []

        async def plan(self, decision, context, toolkit):
            self.contexts.append(context)
            return await super().plan(decision, context, toolkit)

    c = monitor_config()
    c.runtime.stagnation_enabled = True
    c.runtime.stagnation_scope = 'eef_translation'
    c.runtime.stagnation_seconds = seconds
    c.runtime.stagnation_position_m = 0.01
    c.runtime.stagnation_cooldown_seconds = 4.0
    env = ProgressEnvironment()
    planner = Planner(env)
    log = Recorder(tmp_path / 'episode', c)
    rt = AgentRuntime(c, env, HoldingVLA(), None, planner, log)
    requests = []

    async def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if delayed_score and len(requests) == 1:
            await env.past_review.wait()
        return httpx.Response(200, json=answer(payload, score=3))

    await rt.monitor.evaluator.client.aclose()
    rt.monitor.evaluator.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await rt.run()
    finally:
        log.close()
    rows = read_events(log.root)
    reviews = [r for r in rows if r['kind'] == 'review_begin']
    assert reviews[0]['binding']['observation_step'] == int(seconds * 20)
    assert reviews[0]['decision']['trigger'] == 'possible_stagnation'
    signal = planner.contexts[0]['memory']['rule_signals'][0]
    assert signal['sample_count'] == int(seconds * 20) + 1 and not signal['failure_confirmed']
    triggers = [r['binding']['observation_step'] for r in rows
                if r['kind'] == 'supervision_rule' and r['action'] == 'request_astra_review']
    assert triggers == ([80, 160] if delayed_score else [40, 120, 200])
    assert any(r['kind'] == 'review_coalesced' and r['reason'] == 'astra_already_reviewing' for r in rows)
    assert [r['binding']['observation_step'] for r in requests] == (
        [50, 200] if delayed_score else [50, 100, 150, 200])
    assert result['error'] is None and result['control_steps'] == 225 and result['interventions'] == 0
    assert all(r['source'] == 'vla' for r in rows if r['kind'] == 'action_executed')
