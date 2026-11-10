import numpy as np
import pytest
from test_runtime import config

from vla_agent.adapters.mock import MockEnvironment, MockVLA
from vla_agent.contracts import Chunk, Decision, DecisionLabel, Intervention, Source, Subtask, Verification
from vla_agent.data import Recorder, read_events
from vla_agent.memory import WorkingMemory
from vla_agent.runtime import AgentRuntime
from vla_agent.toolkit import PlannerToolkit
from vla_agent.workspace import RecoveryWorkspace


class TwoGoals(MockEnvironment):
    subtask_boundary_checks = True
    verifier_conditions = ['opened', 'task_success']
    language_catalog = [Subtask(name='open', instruction='Open fixture.', completion_condition='opened'),
                        Subtask(name='finish', instruction='Finish fixture.', completion_condition='task_success',
                                preconditions=['opened'])]

    async def observe(self):
        obs = await super().observe()
        obs.facts['task_instruction'] = 'Open and finish the fixture.'
        obs.success = obs.terminal = self.step_id >= 4
        return obs

    async def verify(self, condition):
        return Verification(condition=condition, passed=self.step_id >= (2 if condition == 'opened' else 4),
                            observation_step=self.step_id, evidence=['software fixture counter only'])


class Keep:
    async def decide(self, observations, memory_summary, binding, trigger):
        return Decision(binding=binding, label=DecisionLabel.KEEP)

    async def close(self):
        pass


async def test_registered_plan_changes_actual_vla_language_only_at_chunk_boundary(tmp_path):
    instructions = []

    class VLA(MockVLA):
        async def infer(self, obs, instruction, binding):
            instructions.append((obs.step, instruction))
            return Chunk(binding, np.zeros((5, 12)))

    class Planner:
        async def plan(self, decision, context, toolkit):
            if context['intent'] == 'initialize_plan':
                return Intervention(kind='replan', reason='Two registered goals',
                                    remaining_plan=TwoGoals.language_catalog)
            assert decision.trigger == 'subtask_verifier'
            assert toolkit.harness.obs.step == 2 and not toolkit.harness.obs.success
            return Intervention(kind='switch', reason='Local verifier passed')

        async def close(self):
            pass

    c = config()
    c.environment.instruction_source = 'environment'
    c.runtime.initial_planning = True
    c.runtime.chunk_steps = 2
    c.runtime.max_steps = 8
    log = Recorder(tmp_path/'episode', c)
    try:
        result = await AgentRuntime(c, TwoGoals(), VLA(), Keep(), Planner(), log).run()
        assert result['success'] and result['error'] is None
        assert instructions == [(0, 'Open fixture.'), (2, 'Finish fixture.')]
        actions = [e for e in read_events(log.root) if e['kind'] == 'action_executed']
        assert [a['instruction'] for a in actions] == ['Open fixture.']*2 + ['Finish fixture.']*2
        assert len({a['subtask_version'] for a in actions}) == 2
        assert any(e['kind'] == 'subtask_transition' for e in read_events(log.root))
    finally:
        log.close()


async def test_code_finishes_local_goal_before_native_task_and_next_vla_is_fresh(tmp_path):
    class Planner:
        async def plan(self, decision, context, toolkit):
            if context['intent'] == 'select_next_executor':
                return Intervention(kind='continue', reason='Next registered task gets its first VLA attempt')
            return Intervention(kind='code', reason='Finish the local fixture goal', completion_condition='opened',
                code='def run(ctx):\n    ctx.act([0.0]*12)\n    ctx.act([0.0]*12)\n    return "SUBTASK_DONE"\n')

    c = config()
    c.task = c.task.model_copy(update=dict(subtasks=TwoGoals.language_catalog))
    c.runtime.skill_library = str(tmp_path/'skills')
    log = Recorder(tmp_path/'episode', c)
    rt = AgentRuntime(c, TwoGoals(), MockVLA(), Keep(), Planner(), log)
    rt.lease = await rt.harness.start()
    rt.memory = WorkingMemory(c.task, rt.harness.obs.episode_id)
    rt.memory.add(rt.harness.obs)
    rt.harness.enqueue(Chunk(rt.memory.binding(rt.harness.epoch), np.ones((2,12))), rt.lease, rt.memory, 2)
    try:
        await rt.handle(Decision(binding=rt.memory.binding(rt.harness.epoch), label=DecisionLabel.ESCALATE))
        assert rt.harness.obs.step == 2 and not rt.harness.obs.success
        assert rt.memory.current.name == 'finish' and rt.memory.executor == 'unassigned'
        assert rt.code_session is None and not rt.harness.queue
        rt.forced_reason = None
        await rt.handle(Decision(binding=rt.memory.binding(rt.harness.epoch), label=DecisionLabel.ESCALATE))
        assert rt.harness.owner == Source.VLA
        rt.memory.inference_boundary()
        assert rt.memory.instruction == 'Finish fixture.'
        assert rt.memory.attempt_number == 1
        assert any(e['kind']=='code_subtask_complete' for e in read_events(log.root))
    finally:
        await rt.harness.env.close()
        log.close()


async def test_missing_entry_or_native_final_goal_cannot_install_plan(tmp_path):
    c = config()
    log = Recorder(tmp_path/'episode', c)
    rt = AgentRuntime(c, TwoGoals(), MockVLA(), Keep(), None, log)
    await rt.harness.start()
    memory = WorkingMemory(c.task.model_copy(update=dict(language_catalog=TwoGoals.language_catalog)),
                           rt.harness.obs.episode_id)
    memory.add(rt.harness.obs)
    kit = PlannerToolkit(rt.harness, memory, rt.skills)
    try:
        with pytest.raises(ValueError, match='final goal'):
            await kit.validate_intervention(Intervention(kind='replan', reason='bad',
                remaining_plan=[TwoGoals.language_catalog[0]]), ['replan'])
        # Omitting the registered prerequisite does not bypass it.
        with pytest.raises(ValueError, match='entry condition'):
            await kit.validate_intervention(Intervention(kind='replan', reason='bad', remaining_plan=[
                Subtask(name='finish',instruction='Finish fixture.',completion_condition='task_success')]), ['replan'])
        assert rt.harness.obs.step == 0
    finally:
        log.close()


def test_goal_adaptation_versions_contract_without_rewriting_or_promoting_original(tmp_path):
    workspace = RecoveryWorkspace(tmp_path/'workspace', {})
    source = 'def run(ctx):\n    return "UNRESOLVED"\n'
    contract = dict(completion_condition='task_success', preconditions=['entry'], known_failures=['unverified'])
    original = workspace.write(source, 'import', 'task_success', skill_contract=contract)
    adapted = workspace.adapt_goal(original['version'], 'microwave_open', 'Local goal needs fresh verification')
    assert adapted['parent_version'] == original['version']
    assert workspace.read(adapted['version'])['source'] == source
    assert workspace.read(original['version'])['skill_contract'] == contract
    assert adapted['skill_contract'] == dict(contract, completion_condition='microwave_open')
