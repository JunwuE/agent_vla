"""Review-only pauses must not silently resample or truncate VLA chunks."""

import asyncio

import numpy as np
import pytest
from test_runtime import config, make_harness
from test_supervision import DecisionModel, Environment, Planner, assessment

from vla_agent.contracts import Chunk, Decision, DecisionLabel, Intervention
from vla_agent.data import Recorder, read_events
from vla_agent.planner_service import PlannerServiceError
from vla_agent.runtime import AgentRuntime
from vla_agent.supervision import RunningReviewer


@pytest.mark.parametrize('during_inference', [False, True])
async def test_confirm_continue_preserves_full_sample_sequence(tmp_path, during_inference):
    ready = asyncio.Event()

    class Review(Planner):
        async def plan(self, decision, context, toolkit):
            if context['intent'] == 'review_running':
                await ready.wait()
                return assessment(context['binding'].observation_step, intervene=True)
            return Intervention(kind='continue', reason='Fresh observation permits original attempt')

    class SamplingVLA:
        def __init__(self):
            self.calls = 0

        async def infer(self, obs, instruction, binding):
            self.calls += 1
            if during_inference and self.calls == 2:
                ready.set()
                await asyncio.sleep(0.01)
            actions = np.zeros((5, 12))
            actions[:, 0] = np.arange(5) / 100 + self.calls / 10
            return Chunk(binding, actions)

        async def close(self):
            pass

    class Env(Environment):
        async def step(self, action):
            obs = await super().step(action)
            if not during_inference and obs.step == 7:
                ready.set()
                await asyncio.sleep(0.01)
            return obs

    results = []
    for supervised in (True, False):
        c = config()
        c.runtime.initial_planning = False
        c.runtime.nonblocking_review = True
        c.runtime.supervision_enabled = supervised
        c.runtime.chunk_steps = 5
        c.runtime.max_steps = 18
        c.decision.interval_steps = 1000
        c.runtime.audit_interval_steps = 0
        c.runtime.stagnation_steps = 1000
        log = Recorder(tmp_path / str(supervised), c)
        vla = SamplingVLA()
        try:
            result = await AgentRuntime(c, Env(), vla, DecisionModel(), Review(), log).run()
            assert result['error'] is None
            results.append((read_events(log.root), vla.calls))
        finally:
            log.close()
    rows, calls = results[0]
    baseline, baseline_calls = results[1]
    assert calls == baseline_calls == 4
    assert [(r['action'], r['measured_state']) for r in rows if r['kind'] == 'action_executed'] == [
        (r['action'], r['measured_state']) for r in baseline if r['kind'] == 'action_executed']
    request = next(r for r in rows if r['kind'] == 'review_waiting_for_chunk_boundary')
    assert request['current_step'] == (5 if during_inference else 7)
    plan = next(r for r in rows if r['kind'] == 'intervention_plan')
    assert plan['binding']['observation_step'] == 10
    assert all(r['discarded_steps'] == 0 for r in rows if r['kind'] == 'control_claim')


@pytest.mark.parametrize('unsafe', [False, True])
async def test_boundary_deferral_revalidates_expiry_and_never_delays_unsafe(tmp_path, unsafe):
    c, log, h, lease, memory = await make_harness(tmp_path)
    c.runtime.review_max_age_steps = 1

    class Review(Planner):
        async def plan(self, decision, context, toolkit):
            result = assessment(0, intervene=True)
            if unsafe:
                result.assessment = 'unsafe'
            return result

    from vla_agent.skills import SkillLibrary

    reviewer = RunningReviewer(c, Review(), log, SkillLibrary(c.runtime.skill_library))
    try:
        h.enqueue(Chunk(memory.binding(h.epoch), np.zeros((3, 12))), lease, memory, 3)
        reviewer.submit(Decision(binding=memory.binding(h.epoch), label=DecisionLabel.ESCALATE), h, memory)
        await reviewer.pending
        result = reviewer.poll(h, memory)
        if unsafe:
            assert result.review.assessment == 'unsafe' and h.obs.step == 0
        else:
            assert result is None and reviewer.pending is not None
            while h.queue:
                await h.next_step(lease, memory)
            assert reviewer.poll(h, memory) is None
            assert reviewer.refresh_reason and not memory.reviews
            assert any(r['kind'] == 'review_discarded' for r in read_events(log.root))
    finally:
        await reviewer.close()
        log.close()


@pytest.mark.parametrize('code,model_called,refund', [
    ('localAdmissionTimeout', False, True), ('localAdmissionTimeout', True, False),
    ('serverOverloaded', True, False), ('usageLimitExceeded', False, False),
])
async def test_no_slot_does_not_consume_review_budget_or_reuse_ids(tmp_path, code, model_called, refund):
    c, log, h, lease, memory = await make_harness(tmp_path)

    class Review(Planner):
        calls = 0
        async def plan(self, decision, context, toolkit):
            if model_called:
                self.calls += 1
            raise PlannerServiceError(code, 'Simulated service failure')

    from vla_agent.skills import SkillLibrary
    reviewer = RunningReviewer(c, Review(), log, SkillLibrary(c.runtime.skill_library))
    try:
        for i in range(2):
            reviewer.submit(Decision(binding=memory.binding(h.epoch), label=DecisionLabel.ESCALATE), h, memory)
            await asyncio.gather(reviewer.pending, return_exceptions=True)
            assert reviewer.poll(h, memory) is None
            assert reviewer.count == i + 1
            assert reviewer.budget_count == (0 if refund else i + 1)
        assert [r['review'] for r in read_events(log.root) if r['kind'] == 'review_begin'] == [1, 2]
        assert not memory.reviews and h.obs.step == 0
    finally:
        await reviewer.close()
        log.close()
