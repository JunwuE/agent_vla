import json
import shutil

import numpy as np
import pytest
from test_runtime import config

from vla_agent.cli import run
from vla_agent.visualization import build_timeline, canvas, render_episode


def test_same_step_calls_keep_recorded_order_and_do_not_imply_motion():
    rows = [dict(id=1, time_ns=100, kind="observation", step=0, sim_time=0, path="observations/0.npz"),
            dict(id=2, time_ns=200, kind="codex_tool_start", name="write_program"),
            dict(id=3, time_ns=300, kind="codex_tool", name="write_program", result={"executed": False}),
            dict(id=4, time_ns=400, kind="intervention_plan", plan=dict(kind="code", reason="candidate")),
            dict(id=5, time_ns=500, kind="verification", result=dict(passed=False))]
    cards = build_timeline(rows)
    assert [c["event_id"] for c in cards] == [1, 2, 3, 4, 5]
    assert all(c["step"] == 0 and c["action_source"] == "none" for c in cards)
    assert cards[2]["details"]["result"]["executed"] is False
    assert cards[-1]["details"]["result"]["passed"] is False
    assert all(a["video_end"] == b["video_start"] for a, b in zip(cards, cards[1:]))


def test_recovery_replay_shows_actual_operation_and_verification_status():
    rows = [dict(id=1, time_ns=100, kind="observation", step=0, sim_time=0, path="observations/0.npz"),
            dict(id=2, time_ns=200, kind="recovery_tool_request", operation="move_relative", arguments={}),
            dict(id=3, time_ns=300, kind="recovery_tool_result", operation="move_relative", result={"steps": 1}),
            dict(id=4, time_ns=400, kind="stage_end", stage={"name": "align", "status": "failed"})]
    cards = build_timeline(rows)
    assert cards[1]["tool"] == cards[2]["tool"] == "move_relative"
    assert cards[3]["details"]["stage"]["status"] == "failed"


def test_flora_card_renders_and_distinguishes_snapshot_from_live_step(tmp_path):
    np.savez(tmp_path / "observation.npz",
             **{f"rgb/{name}": np.zeros((8, 8, 3), dtype=np.uint8) for name in ("left", "right", "wrist")})
    rows = [dict(id=1, time_ns=100, kind="observation", step=12, sim_time=0.6, path="observation.npz"),
            dict(id=2, time_ns=200, kind="flora_result",
                 result=dict(status="scored", score=5, binding=dict(observation_step=10)))]
    card = build_timeline(rows)[-1]
    assert card["component"] == "Flora" and card["tool"] == "evaluate_subtask"
    assert card["step"] == 12 and "snapshot step=10" in card["description"]
    assert "does not transfer control" in card["description"]
    assert canvas(tmp_path, card, dict(task="test", mode="test", seed=0)).size == (1280, 800)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is a visualization dependency")
async def test_rendered_episode_has_complete_video_and_embedded_safe_evidence(tmp_path):
    c = config()
    c.runtime.supervision_enabled = False
    c.runtime.max_steps = 2
    c.task.instruction = "test </script><script>alert(1)</script>"
    episode, output = tmp_path / "episode", tmp_path / "report"
    await run(c, episode)
    summary = render_episode(episode, output)
    assert summary["tools"] == [] and summary["result"]["control_steps"] == 2
    meta = json.loads((output / "timeline.json").read_text())
    assert {c["step"] for c in meta["timeline"]} == {0, 1, 2}
    assert not any(c["component"] in {"Astra", "Decision", "Code Policy"} for c in meta["timeline"])
    html = (output / "index.html").read_text()
    assert "__REPORT_JSON__" not in html
    assert "</script><script>alert" not in html
    assert "\\u003c/script>" in html
    assert int(json.loads((output / "video-check.json").read_text())["nb_read_frames"]) > 2
