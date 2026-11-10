import pytest
from test_runtime import make_harness

from vla_agent.recovery import RecoveryRunner


async def test_syntax_error_is_recorded_for_revision(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    report = await RecoveryRunner(h, memory, c.runtime, log).run(
        "def run(ctx):\n  bad syntax !\n", "object_lifted", "invalid"
    )
    assert report["result"] == "UNRESOLVED" and "SyntaxError" in report["error"]
    assert report["executed_steps"] == 0
    assert (log.root / "recovery/invalid/policy.py").exists()
    log.close()


@pytest.mark.parametrize("result", ["RECOVERED", "SUBTASK_DONE"])
async def test_unknown_verification_cannot_approve_recovery(tmp_path, result):
    c, log, h, lease, memory = await make_harness(tmp_path)
    report = await RecoveryRunner(h, memory, c.runtime, log).run(
        f"def run(ctx):\n    return '{result}'\n", "nonexistent_condition", "unknown"
    )
    assert report["result"] == "UNRESOLVED" and report["verified"] is False
    log.close()
