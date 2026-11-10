import asyncio
import json
from dataclasses import replace

import httpx
import numpy as np
import pytest

from vla_agent.adapters.mock import MockEnvironment
from vla_agent.adapters.pi05 import canonical_actions, state_for_policy
from vla_agent.config import DecisionConfig, PlannerConfig
from vla_agent.contracts import Binding, Decision, DecisionLabel, Subtask, Task
from vla_agent.decision import Decision3, checked_decision, classify
from vla_agent.memory import WorkingMemory
from vla_agent.perception import image_url, local_geometry, visual_input, window
from vla_agent.planner import AstraPlanner


class Log:
    def __init__(self):
        self.rows = []

    def event(self, kind, **kwargs):
        self.rows.append(dict(kind=kind, **kwargs))


def binding():
    return Binding(
        episode_id="episode",
        subtask_version=0,
        instruction_version=0,
        control_epoch=1,
        observation_step=0,
        observation_time=1,
    )


def response(label):
    return {
        "answers": {
            "route": {
                "choice": label,
                "probabilities": {c.value: 0.96 if c.value == label else 0.01 for c in DecisionLabel},
            }
        }
    }


@pytest.mark.parametrize("label", list(DecisionLabel))
def test_five_categories(label):
    assert classify(response(label), DecisionConfig(), binding(), "periodic").label == label


def test_ambiguous_keep_is_not_accepted():
    value = response("KEEP")
    value["answers"]["route"]["probabilities"] = dict(
        KEEP=0.4, REFINE=0.3, SWITCH=0.1, REPLAN=0.1, ESCALATE=0.1
    )
    result = classify(value, DecisionConfig(), binding(), "periodic")
    assert result.label == DecisionLabel.ESCALATE


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(KEEP=float("nan")),
        lambda p: p.update(KEEP=2),
        lambda p: p.pop("REFINE"),
    ],
)
def test_invalid_distribution(mutate):
    value = response("KEEP")
    mutate(value["answers"]["route"]["probabilities"])
    with pytest.raises(ValueError):
        classify(value, DecisionConfig(), binding(), "periodic")


async def test_timeout_routes_to_astra():
    class Slow:
        async def decide(self, *args):
            await asyncio.sleep(10)

    log = Log()
    result = await checked_decision(
        Slow(), DecisionConfig(timeout_seconds=0.01), [], {}, binding(), "periodic", log
    )
    assert result.label == DecisionLabel.ESCALATE and "TimeoutError" in result.fallback_reason


async def test_decision_http_and_revision_validation():
    obs = await MockEnvironment().reset()
    seen = []

    def handle(request):
        payload = json.loads(request.content)
        seen.append(payload)
        return httpx.Response(200, json={**response("REPLAN"), "meta": {"revision": "test-revision"}})

    backend = Decision3(DecisionConfig(revision="test-revision", primary_camera=None),
                        Log(), httpx.MockTransport(handle))
    result = await backend.decide([obs], {}, binding(), "periodic")
    assert result.label == DecisionLabel.REPLAN and len(seen[0]["images"]) == 3
    assert set(seen[0]["questions"]["route"]["criteria"]) == set(DecisionLabel)
    await backend.close()


async def test_decision_main_view_preserves_time_and_full_observations():
    obs = await MockEnvironment().reset()
    task = Task(instruction="Lift the object", subtasks=[Subtask(
        name="lift", instruction="Lift the object", completion_condition="object_lifted")])
    memory = WorkingMemory(task, obs.episode_id)
    for step in range(32):
        rgb = dict(obs.rgb)
        rgb["left"] = np.full_like(obs.rgb["left"], step)
        memory.add(replace(obs, step=step, timestamp=100 + step, sim_time=step * 0.05, rgb=rgb))
    seen = []

    def handle(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=response("KEEP"))

    cfg = DecisionConfig(primary_camera="left")
    selected = window(memory, cfg.window_frames)
    backend = Decision3(cfg, Log(), httpx.MockTransport(handle))
    try:
        await backend.decide(selected, memory.summary(), memory.binding(0), "periodic")
    finally:
        await backend.close()
    payload = seen[0]
    mapping = json.loads(payload["state"])["frame_map"]
    steps = [row["step"] for row in mapping]
    assert len(payload["images"]) == 7 and len(set(steps)) == 5
    assert steps == sorted(steps) and steps[0] == 0 and steps[-1] == 31
    assert all(row["camera"] == "left" for row in mapping[:5])
    assert [row["camera"] for row in mapping[5:]] == ["wrist", "right"]
    assert payload["images"][:5] == [image_url(o.rgb["left"]) for o in selected]
    # Extra camera views remain available to Astra/Code Policy and the raw recorder.
    assert len(visual_input(selected)[0]) == 15
    early = WorkingMemory(task, obs.episode_id)
    early.add(obs)
    assert len(window(early, cfg.window_frames)) == 1  # Do not fabricate five frames at startup.


async def test_missing_main_view_escalates_without_substituting_another_camera():
    obs = await MockEnvironment().reset()
    obs = replace(obs, rgb={k: v for k, v in obs.rgb.items() if k != "left"})
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json=response("KEEP"))

    cfg, log = DecisionConfig(), Log()
    backend = Decision3(cfg, log, httpx.MockTransport(handle))
    try:
        result = await checked_decision(backend, cfg, [obs], {}, binding(), "periodic", log)
    finally:
        await backend.close()
    assert result.label == DecisionLabel.ESCALATE and "unavailable" in result.fallback_reason
    assert not calls


def test_action_and_state_permutation_preserves_semantics():
    state = np.arange(16)
    np.testing.assert_array_equal(
        state_for_policy(state, "base_first"), np.r_[np.arange(7, 14), np.arange(7), [14, 15]]
    )
    a = np.arange(12, dtype=float)[None, :] / 12
    np.testing.assert_array_equal(canonical_actions(a, "base_first"), np.c_[a[:, 5:], a[:, :5]])


def test_omnisf_native_signed_channels_are_mapped_before_harness_validation():
    # OmniSF native: base xyz, torso, mode, arm xyz/rot, gripper.
    native = np.array([[0.1, -0.2, 0.3, 0.4, -1, 2, -2, 0.2, 0, 0, 0, 1],
                       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1]], dtype=float)
    canonical = canonical_actions(native, "base_first", "signed", clip=True)
    np.testing.assert_array_equal(canonical[:, [6, 11]], [[1, 0], [0, 1]])
    np.testing.assert_array_equal(canonical[0, :3], [1, -1, 0.2])
    np.testing.assert_array_equal(canonical[0, 7:11], native[0, :4])
    assert native[0, 5] == 2  # Preserve raw commands for attribution.
    native[0, 5] = np.nan
    with pytest.raises(ValueError):
        canonical_actions(native, "base_first", "signed", clip=True)


async def test_depth_has_scale_and_requires_synchronized_calibration():
    obs = await MockEnvironment().reset()
    value = local_geometry(obs, "left", [30, 30, 34, 34])
    assert value["reliable"] and value["centroid_m"][2] == 0.5
    obs.calibration["left"]["step"] = 100
    assert not local_geometry(obs, "left", [30, 30, 34, 34])["reliable"]


async def test_thin_depth_patch_explains_minimum_samples_and_binds_current_step():
    obs = await MockEnvironment().reset()
    sparse = local_geometry(obs, "left", [30, 30, 33, 31])
    assert sparse['reliable'] is False
    assert sparse['count'] == 3 and sparse['minimum_count'] == 4
    valid = local_geometry(obs, "left", [30, 30, 34, 31])
    assert valid['reliable'] is True and valid['count'] == 4
    assert valid['observation_step'] == sparse['observation_step'] == obs.step


async def test_astra_tool_loop_preserves_reasoning_and_counts_usage(monkeypatch):
    monkeypatch.setenv("TEST_ASTRA_KEY", "fixture-only")
    requests = []

    def handle(request):
        data = json.loads(request.content)
        requests.append(data)
        if len(requests) == 1:
            output = [
                {"type": "reasoning", "id": "r", "summary": [], "encrypted_content": "test"},
                {"type": "function_call", "name": "observe", "call_id": "c1", "arguments": "{}"},
            ]
        else:
            output = [
                {
                    "type": "function_call",
                    "name": "submit_intervention",
                    "call_id": "c2",
                    "arguments": json.dumps(
                        dict(kind="refine", reason="alignment", instruction="Align first")
                    ),
                }
            ]
        return httpx.Response(
            200, json=dict(status="completed", output=output, usage=dict(input_tokens=10, output_tokens=5))
        )

    class Tools:
        async def call(self, name, args):
            return {"measured": True}

    cfg = PlannerConfig(api_key_env="TEST_ASTRA_KEY")
    planner = AstraPlanner(cfg, Log(), httpx.MockTransport(handle))
    obs = await MockEnvironment().reset()
    result = await planner.plan(
        Decision(binding=binding(), label=DecisionLabel.REFINE),
        dict(binding=binding(), observations=[obs]),
        Tools(),
    )
    assert result.kind == "refine" and planner.calls == 2 and planner.tokens["input_tokens"] == 20
    assert any(x.get("type") == "reasoning" for x in requests[1]["input"])
    assert any(x.get("type") == "function_call_output" for x in requests[1]["input"])
    await planner.close()


async def test_default_decision_has_five_synchronized_three_view_frames():
    from vla_agent.decision import build_request
    from vla_agent.perception import decision_window

    obs = await MockEnvironment().reset()
    task = Task(instruction="Close the door", subtasks=[Subtask(
        name="door", instruction="Close the door", completion_condition="task_success")])
    memory = WorkingMemory(task, obs.episode_id)
    for step in range(201):
        rgb = {camera: np.full_like(pixels, (step + offset) % 256)
               for offset, (camera, pixels) in enumerate(obs.rgb.items())}
        memory.add(replace(obs, step=step, sim_time=step/20, rgb=rgb))
    cfg = DecisionConfig()
    selected = decision_window(memory, cfg)
    payload, state = build_request(cfg, selected, memory.summary(), memory.binding(0))
    assert len(payload['images']) == 15
    assert [o.step for o in selected] == [0, 100, 160, 190, 200]
    assert state['frame_map'] == [dict(image_index=3*i+j, camera=camera, step=o.step, sim_time=o.sim_time)
                                for i,o in enumerate(selected) for j,camera in enumerate(['left','right','wrist'])]
    assert payload['images'] == [image_url(o.rgb[camera]) for o in selected for camera in ['left','right','wrist']]
    early = WorkingMemory(task, obs.episode_id)
    early.add(obs)
    payload, state = build_request(cfg, decision_window(early,cfg), early.summary(), early.binding(0))
    assert len(payload['images']) == 3  # Never pad missing history with duplicated frames.
    for camera in ['left','right','wrist']:
        incomplete = replace(obs, rgb={k:v for k,v in obs.rgb.items() if k != camera})
        with pytest.raises(ValueError, match='unavailable'):
            build_request(cfg, [incomplete], early.summary(), early.binding(0))
