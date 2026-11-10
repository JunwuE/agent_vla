import numpy as np
import pytest
from test_runtime import make_harness
from test_supervision import Planner

from vla_agent.adapters.mock import MockPlanner, MockVLA
from vla_agent.budgets import PlannerBudget
from vla_agent.contracts import Decision, DecisionLabel, ExecutionReview, Source
from vla_agent.data import read_events
from vla_agent.memory import SkillLibrary
from vla_agent.runtime import AgentRuntime
from vla_agent.supervision import RunningReviewer


@pytest.mark.parametrize('assessment,move,enabled,discard', [
    ('stalled_progress', True, True, True), ('stalled_progress', False, True, False),
    ('stalled_progress', True, False, False), ('wrong_target', True, True, False),
    ('unsafe', True, True, False), ('failed_attempt', True, True, False),
])
async def test_only_contradicted_stall_reviews_are_rechecked(tmp_path, assessment, move, enabled, discard):
    c, log, h, lease, memory = await make_harness(tmp_path)
    c.runtime.review_stall_recheck_displacement_m = .05 if enabled else None
    for _ in range(4):
        await h.execute(np.zeros(12), lease, memory)

    class Review(Planner):
        async def plan(self, decision, context, toolkit):
            return ExecutionReview(action='request_intervention', assessment=assessment,
                phase='engage', reason='Observed fixture evidence',
                evidence=[dict(observation_step=1, fact='Handle unchanged'),
                          dict(observation_step=4, fact='Handle still unchanged')])

    reviewer = RunningReviewer(c, Review(), log, SkillLibrary(c.runtime.skill_library))
    try:
        reviewer.submit(Decision(binding=memory.binding(h.epoch), label=DecisionLabel.ESCALATE), h, memory)
        await reviewer.pending
        for _ in range(10):
            await h.execute(np.array([1 if move else 0] + [0] * 11), lease, memory)
        epoch = h.epoch
        answer = reviewer.poll(h, memory)
        assert (answer is None) == discard
        assert h.epoch == epoch and h.owner == Source.VLA and not h.halted
        if discard:
            assert reviewer.refresh_reason and not memory.reviews
            row = next(r for r in read_events(log.root) if r['kind'] == 'review_discarded')
            assert row['reason'] == 'stagnation_evidence_changed' and not row['task_progress_confirmed']
    finally:
        await reviewer.close()
        log.close()


@pytest.mark.parametrize('limit', ['turns', 'tokens', 'reviews'])
async def test_optional_review_budget_does_not_stop_existing_vla_owner(tmp_path, limit):
    c, log, h, lease, memory = await make_harness(tmp_path)
    c.runtime.continue_vla_on_review_budget = True
    planner = MockPlanner()
    rt = AgentRuntime(c, h.env, MockVLA(), None, planner, log)
    rt.harness, rt.memory = h, memory
    if limit == 'turns':
        planner.calls = c.planner.max_calls
    elif limit == 'tokens':
        c.planner.max_total_tokens = 100
        planner.tokens = {'input_tokens': 100}
    else:
        rt.reviewer.count = c.runtime.max_reviews
    try:
        await rt.route_decision(Decision(binding=memory.binding(h.epoch), label=DecisionLabel.ESCALATE,
                                        trigger='flora_score'))
        assert rt.reviewer.pending is None and h.owner == Source.VLA and not h.halted
        await h.execute(np.zeros(12), lease, memory)
        assert h.obs.step == 1 and log.counts['review_deferred'] == 1
    finally:
        log.close()


async def test_unaccepted_optional_review_can_exhaust_tokens_without_releasing_action(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    c.runtime.continue_vla_on_review_budget = True

    class Exhausted(Planner):
        async def plan(self, *args):
            raise PlannerBudget('astra_token_budget')

    reviewer = RunningReviewer(c, Exhausted(), log, SkillLibrary(c.runtime.skill_library))
    try:
        reviewer.submit(Decision(binding=memory.binding(h.epoch), label=DecisionLabel.ESCALATE), h, memory)
        with pytest.raises(PlannerBudget):
            await reviewer.pending
        assert reviewer.poll(h, memory) is None
        assert h.owner == Source.VLA and h.obs.step == 0 and not memory.reviews
    finally:
        await reviewer.close()
        log.close()
