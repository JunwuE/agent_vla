import pytest
from test_runtime import make_harness

from vla_agent.adapters.mock import MockPlanner, MockVLA
from vla_agent.contracts import Decision, DecisionLabel
from vla_agent.runtime import AgentRuntime


@pytest.mark.parametrize(
    "change,expected",
    [
        ("same_progress", 0),
        ("uncertain", 1),
        ("new_attempt", 1),
        ("new_phase", 1),
        ("new_control", 1),
        ("expired", 1),
        ("unsafe_event", 1),
        ("fresh_recheck", 1),
    ],
)
async def test_progress_review_reuse_is_scoped_and_never_suppresses_new_evidence(tmp_path, change, expected):
    c, log, h, lease, memory = await make_harness(tmp_path)
    c.runtime.progress_review_cooldown_steps = 100
    rt = AgentRuntime(c, h.env, MockVLA(), None, MockPlanner(), log)
    rt.harness, rt.memory = h, memory
    memory.reviews.append(
        dict(
            event_id=1,
            action="continue",
            assessment="progressing",
            attempt_id=memory.attempt_id,
            control_epoch=h.epoch,
            subtask_version=memory.subtask_version,
            phase_revision=memory.phases.revision,
            observation_step=h.obs.step,
        )
    )
    trigger = "possible_stagnation"
    if change == "uncertain":
        memory.reviews[-1]["assessment"] = "uncertain"
    if change == "new_attempt":
        memory.attempt_id = "new-attempt"
    if change == "new_phase":
        memory.phases.revision += 1
    if change == "new_control":
        h.epoch += 1
    if change == "expired":
        memory.reviews[-1]["observation_step"] -= 100
    if change == "unsafe_event":
        trigger = "unsafe"
    if change == "fresh_recheck":
        trigger = "expired_review_recheck"
    calls = []
    rt.reviewer.submit = lambda *args: calls.append(args)
    try:
        await rt.route_decision(
            Decision(binding=memory.binding(h.epoch), label=DecisionLabel.ESCALATE, trigger=trigger)
        )
        assert len(calls) == expected
        assert h.owner == lease.source and not h.halted
        assert log.counts.get("action_command", 0) == 0
    finally:
        log.close()
