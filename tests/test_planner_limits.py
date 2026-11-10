import pytest
from test_runtime import config

from vla_agent.adapters.mock import MockDecision, MockEnvironment, MockVLA
from vla_agent.budgets import PlannerBudget
from vla_agent.codex_planner import CodexPlanner
from vla_agent.data import Recorder, read_events
from vla_agent.runtime import AgentRuntime


def test_reported_token_limit_counts_threads_without_double_counting(tmp_path):
    c = config()
    c.planner.max_total_tokens = 100
    log = Recorder(tmp_path / 'episode', c)
    planner = CodexPlanner(c.planner, log)
    try:
        planner.record_tokens('first', {'inputTokens': 40, 'outputTokens': 10})
        planner.record_tokens('first', {'inputTokens': 40, 'outputTokens': 10})
        assert sum(planner.tokens.values()) == 50
        with pytest.raises(PlannerBudget, match='astra_token_budget'):
            planner.record_tokens('second', {'inputTokens': 45, 'outputTokens': 5})
        assert sum(planner.tokens.values()) == 100
        with pytest.raises(PlannerBudget):
            planner.check_token_budget()
    finally:
        log.close()


async def test_planner_budget_is_reported_stop_without_a_robot_action(tmp_path):
    class Exhausted:
        async def plan(self, *args):
            raise PlannerBudget('astra_turn_budget')

        async def close(self):
            pass

    c = config()
    c.runtime.initial_planning = True
    log = Recorder(tmp_path / 'episode', c)
    try:
        result = await AgentRuntime(c, MockEnvironment(), MockVLA(), MockDecision(), Exhausted(), log).run()
        assert result['reason'] == 'astra_turn_budget' and result['error'] is None
        assert result['control_steps'] == 0
        assert not any(e['kind'] == 'action_command' for e in read_events(log.root))
    finally:
        log.close()


def test_coffee_catalog_requires_staged_placement_before_pressing():
    from vla_agent.adapters.subtask_catalog import subtask_catalog

    tasks = subtask_catalog('PrepareCoffee')
    assert [t.completion_condition for t in tasks] == ['coffee_mug_on_counter', 'coffee_mug_ready', 'task_success']
    assert tasks[1].preconditions == ['coffee_mug_on_counter']
    assert tasks[2].preconditions == ['coffee_mug_ready']
    assert len(subtask_catalog('WaffleReheat')) == 4
