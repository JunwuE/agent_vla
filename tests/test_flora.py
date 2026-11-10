import asyncio
import base64
import io
import json

import httpx
import numpy as np
import pytest
from PIL import Image
from test_runtime import make_harness

from vla_agent.config import FloraConfig
from vla_agent.contracts import Decision, DecisionLabel, ExecutionReview
from vla_agent.flora import FloraEvaluator, prepare_segment
from vla_agent.memory import SkillLibrary
from vla_agent.supervision import RunningReviewer
from vla_agent.toolkit import PlannerToolkit, planner_tools


async def advance(h, lease, memory, steps=10):
    for _ in range(steps):
        await h.execute(np.zeros(12), lease, memory)


async def test_video_preparation_loads_only_requested_rgb_and_preserves_pixels(tmp_path, monkeypatch):
    c, log, h, lease, memory = await make_harness(tmp_path)
    await advance(h, lease, memory, 6)
    loaded = []
    original = np.lib.npyio.NpzFile.__getitem__

    def track(self, key):
        loaded.append(key)
        return original(self, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, '__getitem__', track)
    try:
        payload = prepare_segment(log, memory, memory.binding(h.epoch), c.flora, with_rationale=False)
        assert not any(key.startswith('depth_m/') for key in loaded)
        assert {key for key in loaded if key.startswith('rgb/')} == {'rgb/' + k for k in c.flora.cameras}
        expected = np.concatenate([h.obs.rgb[k] for k in c.flora.cameras], axis=1)
        for encoded in payload['frames']:
            decoded = np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))))
            np.testing.assert_array_equal(decoded, expected)
    finally:
        log.close()


def answer(payload, score=5):
    return {k: payload[k] for k in ("request_id", "model", "revision")} | {"score": score}


async def test_flora_snapshot_is_causal_cached_and_has_no_action_authority(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    await advance(h, lease, memory)
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(200, json=answer(payload))

    client = FloraEvaluator(FloraConfig(with_rationale=False, enabled=True), log, httpx.MockTransport(handler))
    tools = PlannerToolkit.for_review(h, memory, SkillLibrary(c.runtime.skill_library), [], flora=client)
    await advance(h, lease, memory, 2)  # Live VLA proceeds beyond the review's frozen step 10.
    h.queue.extend([np.zeros(12)] * 2)
    before = (h.epoch, h.owner, h.obs.step, len(h.queue), memory.subtask_index)
    try:
        result = await tools.call("evaluate_subtask", {})
        cached = await tools.call("evaluate_subtask", {})
        assert result["status"] == "scored" and result["score"] == 5
        assert cached["cache_hit"] and len(requests) == client.calls == 1
        assert (h.epoch, h.owner, h.obs.step, len(h.queue), memory.subtask_index) == before
        assert not memory.reward_assessments  # Snapshot opinions are not committed to live Memory yet.
        assert len(tools.memory.reward_assessments) == 1
        payload = requests[0]
        assert payload["segment"]["end_step"] == 10 and max(payload["segment"]["sampled_steps"]) == 10
        assert payload["task"] == memory.current.instruction
        assert payload["with_rationale"] is False
        assert not {"actions", "recent_actions", "state", "success"} & payload.keys()
        assert payload["video_metadata"]["fps"] == pytest.approx(20)
        assert payload["video_metadata"]["total_num_frames"] == 11
        assert all(x["step"] <= 10 for x in payload["segment"]["sources"])
        assert "evaluate_subtask" in {x["name"] for x in planner_tools(review_only=True)}
    finally:
        await client.close()
        log.close()


@pytest.mark.parametrize("problem", ["timeout", "invalid_score", "wrong_revision", "wrong_quantization"])
async def test_flora_failure_is_unknown_and_does_not_interrupt(tmp_path, problem):
    c, log, h, lease, memory = await make_harness(tmp_path)
    await advance(h, lease, memory)

    async def handler(request):
        payload = json.loads(request.content)
        if problem == "timeout":
            raise httpx.ReadTimeout("fixture timeout")
        result = answer(payload, score=True if problem == "invalid_score" else 5)
        if problem == "wrong_revision":
            result["revision"] = "different"
        if problem == "wrong_quantization":
            result["quantization"] = "int8"
        return httpx.Response(200, json=result)

    client = FloraEvaluator(FloraConfig(with_rationale=False, enabled=True), log, httpx.MockTransport(handler))
    tools = PlannerToolkit.for_review(h, memory, SkillLibrary(c.runtime.skill_library), [], flora=client)
    before = (h.epoch, h.owner, h.obs.step)
    try:
        result = await tools.call("evaluate_subtask", {})
        assert result["status"] == "unavailable" and result["score"] is None
        assert not tools.memory.reward_assessments
        assert (h.epoch, h.owner, h.obs.step) == before
    finally:
        await client.close()
        log.close()


async def test_flora_budget_and_retry_segment_boundary(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    await advance(h, lease, memory, 6)
    memory.begin_retry("fixture retry")
    memory.resume_retry()
    await advance(h, lease, memory, 4)
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(200, json=answer(payload, score=1))

    client = FloraEvaluator(FloraConfig(with_rationale=False, enabled=True, max_calls=1), log, httpx.MockTransport(handler))
    try:
        result = await client.evaluate(memory, memory.binding(h.epoch))
        assert result["status"] == "scored" and result["score"] == 1
        assert requests[0]["segment"]["start_step"] == 6
        assert min(requests[0]["segment"]["sampled_steps"]) == 6
        await advance(h, lease, memory, 1)
        result = await client.evaluate(memory, memory.binding(h.epoch))
        assert result["status"] == "budget_exhausted" and len(requests) == 1
    finally:
        await client.close()
        log.close()


async def test_expired_review_does_not_commit_flora_assessment(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    await advance(h, lease, memory)
    c.runtime.review_max_age_steps = 1
    entered, release = asyncio.Event(), asyncio.Event()

    async def handler(request):
        entered.set()
        await release.wait()
        return httpx.Response(200, json=answer(json.loads(request.content)))

    client = FloraEvaluator(FloraConfig(with_rationale=False, enabled=True), log, httpx.MockTransport(handler))

    class Planner:
        async def plan(self, decision, context, toolkit):
            assert (await toolkit.call("evaluate_subtask", {}))["score"] == 5
            return ExecutionReview(action="continue", assessment="uncertain", phase="checking", reason="fixture")

    reviewer = RunningReviewer(c, Planner(), log, SkillLibrary(c.runtime.skill_library), flora=client)
    try:
        reviewer.submit(Decision(binding=memory.binding(h.epoch), label=DecisionLabel.SWITCH), h, memory)
        await asyncio.wait_for(entered.wait(), timeout=5)
        await advance(h, lease, memory, 2)
        release.set()
        await reviewer.pending
        assert reviewer.poll(h, memory) is None
        assert not memory.reward_assessments
    finally:
        await reviewer.close()
        await client.close()
        log.close()
