import asyncio
import json

import pytest
from test_runtime import config

from vla_agent.adapters.mock import MockDecision, MockEnvironment, MockVLA
from vla_agent.codex_planner import CodexPlanner, compact_numeric_memory
from vla_agent.config import PlannerConfig
from vla_agent.contracts import Decision, DecisionLabel, ExecutionReview, Source
from vla_agent.data import Recorder, read_events
from vla_agent.memory import WorkingMemory
from vla_agent.runtime import AgentRuntime
from vla_agent.toolkit import PlannerToolkit
from vla_agent.workspace import RecoveryWorkspace


def test_usage_survives_thread_restart_without_double_counting_cumulative_updates(tmp_path):
    c = config()
    log = Recorder(tmp_path / "usage", c)
    planner = CodexPlanner(c.planner, log)
    try:
        planner.record_tokens("first", {"inputTokens": 100, "outputTokens": 10})
        planner.record_tokens("first", {"inputTokens": 120, "outputTokens": 15})
        planner.record_tokens("restarted", {"inputTokens": 50, "outputTokens": 3})
        planner.record_tokens("first", {"inputTokens": 100, "outputTokens": 10})
        assert planner.tokens == {"input_tokens": 170, "output_tokens": 18}
    finally:
        log.close()
    assert read_events(log.root)[-1]["episode_total"] == planner.tokens


def test_numeric_compaction_preserves_history_and_semantic_memory():
    memory = dict(recent_actions=[dict(step=i, command=[0.123456789] * 12,
                                      measured_state=[0.987654321] * 16) for i in range(30)],
                  phase_memory={"status": "unknown"}, motion_windows={"10s": {"eef_path_m": 0.0123456}})
    compact = compact_numeric_memory(memory)
    assert len(compact["recent_actions"]) == 6
    assert [compact["recent_actions"][i]["step"] for i in (0, -1)] == [0, 29]
    assert compact["recent_actions"][0]["command"][0] == 0.123
    assert memory["recent_actions"][0]["command"][0] == 0.123456789
    assert compact["motion_windows"] == memory["motion_windows"]
    compact["phase_memory"]["status"] = "modified"
    assert memory["phase_memory"]["status"] == "unknown"


async def test_historical_experience_references_reset_on_new_thread_and_keep_current_memory(tmp_path):
    from vla_agent.experience import ExperienceStore

    c = config()
    log = Recorder(tmp_path / 'experience-context', c)
    planner = CodexPlanner(c.planner, log)
    record = dict(episode_id='prior', robot='mock_panda', task='fixture', result={'success': False},
                  notes=[{'facts': ['long historical evidence'], 'hypotheses': ['not confirmed']}],
                  code_versions=[{'sha256': 'source', 'available_skills': [{'id': 'skill', 'version': 'v1'}]}])
    memory = dict(retrieved_experiences=[record], phase_memory={'status': 'current'}, notes=['new fact'])
    try:
        first = planner.experience_context(memory)
        second = planner.experience_context(memory)
        assert first == memory
        assert 'context_reference' in second['retrieved_experiences'][0]
        assert second['retrieved_experiences'][0]['code_versions'] == record['code_versions']
        assert second['phase_memory'] == memory['phase_memory'] and second['notes'] == memory['notes']
        assert 'notes' not in second['retrieved_experiences'][0] and 'context_reference' not in record
        store = ExperienceStore(tmp_path / 'experiences')
        store.index = [record]
        expanded = store.get('mock_panda', 'prior')
        expanded['notes'].clear()
        assert store.get('mock_panda', 'prior')['notes'] == record['notes']
        with pytest.raises(ValueError, match='snapshot'):
            store.get('other_robot', 'prior')
        await planner.close()
        assert planner.experience_context(memory) == memory
        changed = json.loads(json.dumps(memory))
        changed['retrieved_experiences'][0]['notes'].append('new historical evidence')
        assert planner.experience_context(changed) == changed
    finally:
        log.close()


class ScriptedTransport:
    def __init__(self, turns):
        self.turns = iter(turns)
        self.events = asyncio.Queue()
        self.requests, self.replies = [], []
        self.closed = False

    async def start(self, argv, cwd):
        self.argv = argv

    async def request(self, method, params):
        self.requests.append((method, params))
        if method == "thread/start":
            return dict(thread=dict(id="thread"), model=params["model"],
                        reasoningEffort=params["config"]["model_reasoning_effort"])
        if method == "turn/start":
            for event in next(self.turns):
                self.events.put_nowait(event)
            return dict(turn=dict(id="turn"))
        return {}

    async def notify(self, method, params):
        pass

    async def next_event(self):
        return await self.events.get()

    async def reply(self, identifier, result):
        self.replies.append((identifier, result))

    async def close(self):
        self.closed = True


def call(identifier, name, args, **overrides):
    return dict(id=identifier, method="item/tool/call", params=dict(
        threadId="thread", turnId="turn", callId=str(identifier), tool=name,
        arguments=args, **overrides,
    ))


def complete(status="completed"):
    return dict(method="turn/completed", params=dict(
        threadId="thread", turn=dict(id="turn", status=status),
    ))


async def setup(tmp_path, turns, timeout=1):
    c = config()
    log = Recorder(tmp_path / "episode", c)
    rt = AgentRuntime(c, MockEnvironment(), MockVLA(), MockDecision(), None, log)
    await rt.harness.start()
    await rt.harness.claim(Source.ASTRA)
    memory = WorkingMemory(c.task, rt.harness.obs.episode_id)
    memory.add(rt.harness.obs)
    workspace = RecoveryWorkspace(log.root / "workspace", {})
    toolkit = PlannerToolkit(rt.harness, memory, rt.skills, rt.vla, workspace)
    transport = ScriptedTransport(turns)
    planner = CodexPlanner(PlannerConfig(kind="codex", timeout_seconds=timeout), log,
                           transport_factory=lambda: transport)
    context = dict(binding=memory.binding(rt.harness.epoch), observations=[rt.harness.obs],
                   allowed_interventions=c.runtime.allowed_interventions)
    decision = Decision(binding=context["binding"], label=DecisionLabel.ESCALATE)
    return log, rt, toolkit, transport, planner, context, decision


async def test_bad_tool_and_unavailable_verifier_are_repaired_without_motion(tmp_path):
    invalid = dict(kind="replan", reason="test", remaining_plan=[dict(
        name="invented", instruction="Reach", completion_condition="invented_predicate",
    )])
    valid = dict(kind="refine", reason="corrected", instruction_id="pick")
    events = [call(1, "unknown_tool", {}), call(2, "submit_intervention", invalid),
              call(3, "submit_intervention", json.dumps(valid)), complete()]
    log, rt, toolkit, transport, planner, context, decision = await setup(tmp_path, [events, events])
    try:
        result = await planner.plan(decision, context, toolkit)
        assert result.kind == "continue" and rt.harness.obs.step == 0
        assert [r[1]["success"] for r in transport.replies] == [False, False, True]
        assert json.loads(transport.replies[-1][1]["contentItems"][0]["text"])["executed"] is False
        # A second intervention retains one Codex thread, with fresh authoritative context.
        await planner.plan(decision, context, toolkit)
        assert sum(m == "thread/start" for m, _ in transport.requests) == 1
        params = next(p for m, p in transport.requests if m == "thread/start")
        assert params["environments"] == [] and params["approvalPolicy"] == "never"
        assert all(t["type"] == "function" for t in params["dynamicTools"])
        assert len(next(p for m, p in transport.requests if m == "turn/start")["input"]) == 4
    finally:
        await planner.close()
        log.close()


async def test_robot_image_alias_avoids_builtin_collision_and_preserves_causal_checks(tmp_path):
    events = [call(1, 'view_observation_image', dict(camera='left', step=100)),
              call(2, 'view_observation_image', dict(camera='left', step=0, bbox=[20, 20, 35, 35])),
              call(3, 'submit_intervention', dict(kind='continue', reason='Read-only inspection')), complete()]
    log, rt, toolkit, transport, planner, context, decision = await setup(tmp_path, [events])
    try:
        await planner.plan(decision, context, toolkit)
        params = next(p for m, p in transport.requests if m == 'thread/start')
        names = [t['name'] for t in params['dynamicTools']]
        assert 'view_observation_image' in names and 'view_image' not in names
        assert len(names) == len(set(names))
        assert not transport.replies[0][1]['success']
        assert transport.replies[1][1]['success']
        assert any(c['type'] == 'inputImage' for c in transport.replies[1][1]['contentItems'])
        assert rt.harness.obs.step == 0
    finally:
        await planner.close()
        log.close()


@pytest.mark.parametrize("events,match", [
    ([complete()], "without a completed"),
    ([call(1, "observe", {}), call(1, "observe", {})], "Repeated"),
    ([dict(id=1, method="item/tool/call", params=dict(threadId="other", turnId="turn"))], "Stale"),
    ([dict(id=1, method="item/commandExecution/requestApproval", params={})], "Unexpected"),
    ([call(1, "submit_intervention", dict(kind="continue", reason="test")), complete("failed")],
     "without a completed"),
])
async def test_invalid_turn_never_releases_an_intervention(tmp_path, events, match):
    log, rt, toolkit, transport, planner, context, decision = await setup(tmp_path, [events])
    try:
        with pytest.raises(RuntimeError, match=match):
            await planner.plan(decision, context, toolkit)
        assert transport.closed and rt.harness.obs.step == 0
    finally:
        log.close()


async def test_timeout_cancels_model_even_after_proposal_acceptance(tmp_path):
    events = [call(1, "submit_intervention", dict(kind="continue", reason="test"))]
    log, rt, toolkit, transport, planner, context, decision = await setup(tmp_path, [events], timeout=0.05)
    try:
        with pytest.raises(TimeoutError):
            await planner.plan(decision, context, toolkit)
        assert transport.closed and rt.harness.obs.step == 0
        assert not any(r["kind"] == "action_executed" for r in read_events(log.root))
    finally:
        log.close()


@pytest.mark.parametrize('enabled,valid', [(True, True), (True, False), (False, True)])
async def test_budget_can_release_only_a_host_validated_proposal(tmp_path, enabled, valid):
    from vla_agent.budgets import PlannerBudget

    proposal = dict(kind='continue', reason='Current evidence supports existing execution')
    if not valid:
        proposal['kind'] = 'not_a_kind'
    usage = dict(method='thread/tokenUsage/updated', params=dict(
        threadId='thread', tokenUsage={'total': {'inputTokens': 150, 'outputTokens': 5}}))
    events = [call(1, 'submit_intervention', proposal), usage]
    log, rt, toolkit, transport, planner, context, decision = await setup(tmp_path, [events])
    planner.config.max_total_tokens = 100
    planner.config.release_validated_proposal_on_budget = enabled
    try:
        if enabled and valid:
            result = await planner.plan(decision, context, toolkit)
            assert result.kind == 'continue'
            assert log.counts['codex_budget_proposal_released'] == 1
            # The host, not the model/server, finalized this bounded turn.
            end = next(r for r in read_events(log.root) if r['kind'] == 'codex_turn_end')
            assert end['status'] == 'host_finalized_on_budget' and not end['server_reported_completion']
        else:
            with pytest.raises(PlannerBudget):
                await planner.plan(decision, context, toolkit)
            assert not log.counts['codex_budget_proposal_released']
        assert transport.closed and rt.harness.obs.step == 0
        with pytest.raises(PlannerBudget):
            await planner.plan(decision, context, toolkit)
        assert planner.calls == 1
    finally:
        await planner.close()
        log.close()


async def test_tool_budget_continuation_preserves_workspace_without_executing(tmp_path):
    source = "def run(ctx): return 'UNRESOLVED'"
    first = [call(1, 'write_program', dict(source=source, hypothesis='fixture', expected_effect='fixture',
              skill_name='align-fixture', when_to_use='Unaligned fixture', reuse_reason='New fixture')),
             call(2, 'observe', {}), call(3, 'observe', {})]
    second = [call(1, 'read_program', {}),
              call(2, 'submit_intervention', dict(kind='continue', reason='No action needed after inspection')), complete()]
    log, rt, toolkit, transport, planner, context, decision = await setup(tmp_path, [first, second])
    planner.config.max_tool_calls = 2
    planner.config.max_turns_per_intervention = 2
    try:
        result = await planner.plan(decision, context, toolkit)
        assert result.kind == 'continue' and planner.calls == 2 and rt.harness.obs.step == 0
        assert toolkit.workspace.read()['source'] == source and len(toolkit.workspace.versions) == 1
        assert sum(m == 'thread/start' for m, _ in transport.requests) == 2
        sent = [p for m, p in transport.requests if m == 'turn/start'][-1]
        feedback = json.loads(sent['input'][0]['text'])['planner_feedback']
        assert feedback['continuation'] == 1 and not feedback['executed']
        assert 'tool_budget_remaining' in transport.replies[0][1]['contentItems'][1]['text']
        assert len([r for r in read_events(log.root) if r['kind'] == 'codex_budget_continuation']) == 1
    finally:
        await planner.close()
        log.close()


async def test_tool_budget_continuations_are_bounded_and_readonly_reviews_do_not_restart(tmp_path):
    events = [call(1, 'observe', {}), call(2, 'observe', {})]
    log, rt, toolkit, transport, planner, context, decision = await setup(tmp_path, [events, events])
    planner.config.max_tool_calls = 1
    planner.config.max_turns_per_intervention = 2
    try:
        with pytest.raises(RuntimeError, match='tool budget'):
            await planner.plan(decision, context, toolkit)
        assert planner.calls == 2 and rt.harness.obs.step == 0 and transport.closed
        toolkit.read_only = True
        transport.turns = iter([events])
        with pytest.raises(RuntimeError, match='tool budget'):
            await planner.plan(decision, context, toolkit)
        assert planner.calls == 3 and rt.harness.obs.step == 0
    finally:
        await planner.close()
        log.close()


async def test_nan_tool_arguments_return_error_and_do_not_break_logging(tmp_path):
    events = [call(1, "record_note", '{"value": NaN}'),
              call(2, "submit_intervention", dict(kind="continue", reason="test")), complete()]
    log, rt, toolkit, transport, planner, context, decision = await setup(tmp_path, [events])
    try:
        assert (await planner.plan(decision, context, toolkit)).kind == "continue"
        assert not transport.replies[0][1]["success"]
        assert read_events(log.root)
    finally:
        await planner.close()
        log.close()


async def test_initial_plan_preserves_native_instruction_and_repairs_expansion(tmp_path):
    def plan(instruction):
        return dict(kind="replan", reason="initialize", remaining_plan=[dict(
            name="task", instruction=instruction, completion_condition="task_success")])
    events = [call(1, "submit_intervention", plan("Expanded motion instructions")),
              call(2, "submit_intervention", plan("Open the drawer.")), complete()]
    log, rt, toolkit, transport, planner, context, decision = await setup(tmp_path, [events])
    planner.config.preserve_initial_instruction = True
    context.update(intent="initialize_plan", available_verifiers=["task_success"],
                   memory=dict(task=dict(instruction="Open the drawer.")))
    from vla_agent.contracts import Subtask, Task
    toolkit.memory = WorkingMemory(Task(instruction="Open the drawer.", subtasks=[Subtask(
        name="task", instruction="Open the drawer.", completion_condition="task_success")]),
        rt.harness.obs.episode_id)
    toolkit.memory.add(rt.harness.obs)
    try:
        result = await planner.plan(decision, context, toolkit)
        assert result.remaining_plan[0].instruction == "Open the drawer."
        assert [r[1]["success"] for r in transport.replies] == [False, True]
        assert rt.harness.obs.step == 0
    finally:
        await planner.close()
        log.close()


async def test_compact_context_keeps_motion_summary_without_mutating_memory(tmp_path):
    events = [call(1, "submit_intervention", dict(kind="continue", reason="test")), complete()]
    log, rt, toolkit, transport, planner, context, decision = await setup(tmp_path, [events])
    planner.config.compact_context = True
    history = [dict(step=i, command=[0] * 12, measured_state=[i / 100, 0, 0] + [0] * 13) for i in range(30)]
    context.update(memory=dict(recent_actions=history, notes=[], interventions=[]))
    context["observations"] *= 3
    try:
        await planner.plan(decision, context, toolkit)
        sent = next(p for m, p in transport.requests if m == "turn/start")
        data = json.loads(sent["input"][0]["text"])
        assert len(data["memory"]["recent_actions"]) == 6 and len(history) == 30
        assert data["memory"]["motion_summary"]["measured_eef_delta"] == [0.29, 0, 0]
        assert len(sent["input"]) == 7  # Text + two timestamps with three views each.
    finally:
        await planner.close()
        log.close()


async def test_running_review_repairs_wrong_tools_and_cannot_submit_an_intervention(tmp_path):
    events = [call(1, "preview_vla", {"steps": 1}),
              call(2, "submit_intervention", dict(kind="refine", reason="bypass", instruction_id="pick")),
              call(3, "submit_review", dict(action="continue", assessment="progressing",
                                            phase="approach", reason="First attempt is still progressing")), complete()]
    log, rt, toolkit, transport, planner, context, decision = await setup(tmp_path, [events])
    await rt.harness.claim(Source.VLA)
    frozen = PlannerToolkit.for_review(rt.harness, toolkit.memory, rt.skills, context["allowed_interventions"])
    context.update(intent="review_running", binding=frozen.memory.binding(frozen.harness.epoch),
                   memory=frozen.memory.summary())
    epoch = rt.harness.epoch
    try:
        result = await planner.plan(decision, context, frozen)
        assert isinstance(result, ExecutionReview) and result.action == "continue"
        assert [reply[1]["success"] for reply in transport.replies] == [False, False, True]
        assert rt.harness.epoch == epoch and rt.harness.owner == Source.VLA and rt.harness.obs.step == 0
        assert not any(r["kind"] == "vla_preview" for r in read_events(log.root))
        args = next(p for m, p in transport.requests if m == "thread/start")
        assert "submit_review" in {t["name"] for t in args["dynamicTools"]}
    finally:
        await planner.close()
        log.close()


async def test_quick_review_enforces_narrow_tools_then_restores_paused_profile(tmp_path):
    review = dict(action='continue', assessment='progressing', phase='approach',
                  reason='Observed approach still changing')
    first = [call(1, 'search_skills', {'query': 'drawer'}), call(2, 'measure_targets', {}),
             call(3, 'submit_review', review), complete()]
    log, rt, toolkit, transport, planner, context, decision = await setup(tmp_path, [first])
    second = ScriptedTransport([[call(4, 'submit_intervention', dict(kind='continue', reason='fresh state')), complete()]])
    transports = iter([transport, second])
    planner.transport_factory = lambda: next(transports)
    planner.config.review_tool_limit = 4
    await rt.harness.claim(Source.VLA)
    frozen = PlannerToolkit.for_review(rt.harness, toolkit.memory, rt.skills, context['allowed_interventions'])
    context.update(intent='review_running', binding=frozen.memory.binding(frozen.harness.epoch),
                   memory=frozen.memory.summary())
    try:
        answer = await planner.plan(decision, context, frozen)
        assert answer.action == 'continue' and rt.harness.obs.step == 0
        assert [r[1]['success'] for r in transport.replies] == [False, False, True]
        registered = next(p for m, p in transport.requests if m == 'thread/start')
        names = {t['name'] for t in registered['dynamicTools']}
        assert {'submit_review', 'view_observation_image', 'evaluate_subtask', 'read_execution'} <= names
        assert not names & {'read_skill', 'search_skills', 'measure_targets', 'submit_intervention', 'edit_program'}
        assert (log.root / 'codex/tools-review.json').exists()
        assert not any(e['kind'] == 'skill_search' for e in read_events(log.root))
        await rt.harness.claim(Source.ASTRA)
        context.update(intent='confirm_intervention', binding=toolkit.memory.binding(rt.harness.epoch))
        assert (await planner.plan(decision, context, toolkit)).kind == 'continue'
        full = next(p for m, p in second.requests if m == 'thread/start')
        assert {'read_skill', 'measure_targets', 'submit_intervention'} <= {t['name'] for t in full['dynamicTools']}
        assert planner.calls == 2 and transport.closed and rt.harness.obs.step == 0
        assert (log.root / 'codex/tools-full.json').exists()
        assert any(e['kind'] == 'codex_thread_rotated' and e.get('reason') == 'Tool authority changed'
                   for e in read_events(log.root))
    finally:
        await planner.close()
        log.close()


async def test_quick_review_uses_its_own_tool_limit_without_executing(tmp_path):
    events = [call(i, 'observe', {}) for i in range(1, 4)]
    events += [call(4, 'submit_review', dict(action='continue', assessment='uncertain',
                                          phase='approach', reason='Not enough evidence')), complete()]
    log, rt, toolkit, transport, planner, context, decision = await setup(tmp_path, [events])
    planner.config.review_tool_limit = 3
    await rt.harness.claim(Source.VLA)
    frozen = PlannerToolkit.for_review(rt.harness, toolkit.memory, rt.skills, context['allowed_interventions'])
    context.update(intent='review_running', binding=frozen.memory.binding(frozen.harness.epoch))
    epoch = rt.harness.epoch
    try:
        from vla_agent.codex_planner import ToolBudgetExhausted

        with pytest.raises(ToolBudgetExhausted):
            await planner.plan(decision, context, frozen)
        assert len(transport.replies) == 3 and transport.closed
        assert planner.calls == 1 and rt.harness.epoch == epoch and rt.harness.obs.step == 0
        data = json.loads(next(p for m, p in transport.requests if m == 'turn/start')['input'][0]['text'])
        assert data['tool_budget']['max_calls'] == 3 and data['tool_budget']['reserve_for_proposal'] == 1
    finally:
        await planner.close()
        log.close()


@pytest.mark.parametrize('error_code', ['serverOverloaded', 'usageLimitExceeded', 'contextWindowExceeded'])
async def test_service_error_preserves_category_without_spending_local_budget_or_executing(tmp_path, error_code):
    event = dict(method='error', params=dict(willRetry=False,
        error=dict(codexErrorInfo=error_code, message='Service rejected this request')))
    log, rt, toolkit, transport, planner, context, decision = await setup(tmp_path, [[event]])
    try:
        with pytest.raises(RuntimeError, match=error_code):
            await planner.plan(decision, context, toolkit)
        assert rt.harness.obs.step == 0 and sum(planner.tokens.values()) == 0
        assert planner.calls == 1 and transport.closed
        errors = [e for e in read_events(log.root) if e['kind'] == 'codex_error']
        assert errors[0]['error']['error']['codexErrorInfo'] == error_code
        assert not any(e['kind'] == 'budget_exhausted' for e in read_events(log.root))
    finally:
        await planner.close()
        log.close()


@pytest.mark.parametrize('error_channel', ['notification', 'completion'])
async def test_bounded_service_retry_keeps_notes_and_counts_every_turn(tmp_path, error_channel):
    failure = dict(method='error', params=dict(willRetry=False,
        error=dict(codexErrorInfo='serverOverloaded', message='busy')))
    if error_channel == 'completion':
        failure = complete('failed')
        failure['params']['turn']['error'] = dict(codexErrorInfo='serverOverloaded', message='busy')
    log, rt, toolkit, transport, planner, context, decision = await setup(tmp_path, [
        [call(1, 'record_note', {'facts': ['retained observation']}), failure],
        [call(2, 'submit_intervention', dict(kind='continue', reason='fresh decision')), complete()]])
    planner.config.service_retry_limit = 1
    planner.config.service_backoff_seconds = 0
    try:
        assert (await planner.plan(decision, context, toolkit)).kind == 'continue'
        assert planner.calls == 2 and rt.harness.obs.step == 0
        assert len(toolkit.memory.notes) == 1
        sent = [p for m, p in transport.requests if m == 'turn/start'][-1]
        content = json.loads(sent['input'][0]['text'])
        assert 'retained observation' in json.dumps(content['memory'])
        assert not content['planner_feedback']['executed']
        assert log.counts['planner_service_retry'] == 1
    finally:
        await planner.close()
        log.close()


@pytest.mark.parametrize('mode', ['quota', 'turn_limit', 'review'])
async def test_service_retry_cannot_bypass_quota_turn_limit_or_retry_stale_review(tmp_path, mode):
    from vla_agent.budgets import PlannerBudget
    from vla_agent.planner_service import PlannerServiceError

    code = 'usageLimitExceeded' if mode == 'quota' else 'serverOverloaded'
    failure = dict(method='error', params=dict(willRetry=False, error=dict(codexErrorInfo=code, message='busy')))
    log, rt, toolkit, transport, planner, context, decision = await setup(tmp_path, [[failure]])
    planner.config.service_retry_limit = 1
    planner.config.service_backoff_seconds = 0
    if mode == 'turn_limit':
        planner.config.max_calls = 1
    if mode == 'review':
        await rt.harness.claim(Source.VLA)
        toolkit = PlannerToolkit.for_review(rt.harness, toolkit.memory, rt.skills, context['allowed_interventions'])
        context.update(intent='review_running', binding=toolkit.memory.binding(rt.harness.epoch))
    try:
        with pytest.raises(PlannerBudget if mode == 'turn_limit' else PlannerServiceError):
            await planner.plan(decision, context, toolkit)
        assert planner.calls == 1 and rt.harness.obs.step == 0 and transport.closed
        assert not log.counts['planner_service_retry']
    finally:
        await planner.close()
        log.close()
