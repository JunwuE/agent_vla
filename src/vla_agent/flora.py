"""Bounded, causal Outcome scoring. Scores are model assessments, never completion verifiers."""

import asyncio
import base64
import copy
import hashlib
import io
import json
import math
import time

import httpx
import numpy as np
from PIL import Image

from .data import dumps


def prepare_segment(recorder, memory, binding, config, *, phase_id=None, with_rationale=None,
                    window_steps=None):
    """Read only the current attempt, ending at the frozen observation. No action arrays sent."""
    start, end = memory.attempt_started_step, binding.observation_step
    goal = memory.current.instruction
    scope = dict(kind='subtask', subtask_version=memory.subtask_version)
    if phase_id is not None:
        if phase_id != memory.phases.active:
            raise ValueError('Flora can assess only the current phase')
        phase = next(p for p in memory.phases.phases if p['id'] == phase_id)
        start = max(start, memory.phases.started_step)
        goal = phase['goal'] + ' Completion evidence: ' + '; '.join(phase['completion_signals'])
        scope = dict(kind='phase', phase_id=phase_id, phase_revision=memory.phases.revision,
                     subtask_version=memory.subtask_version, source='astra_defined_phase')
    if window_steps is not None:
        if type(window_steps) is not int or window_steps < 2:
            raise ValueError('Explicit Flora window must contain at least two steps')
        start = max(start, end - window_steps)
        scope['window_steps'] = window_steps
    if not 0 <= start < end or end - start > config.max_segment_steps:
        raise ValueError("Insufficient history or attempt exceeds the configured segment budget")
    if len(set(config.cameras)) != len(config.cameras):
        raise ValueError("Flora camera names must be unique")
    times = []
    for step in range(start, end + 1):
        key = f"{memory.episode_id}:{step}"
        if key not in recorder.observations:
            raise ValueError("Current attempt contains an unrecorded observation")
        path = recorder.root / recorder.observations[key]
        metadata = json.loads(path.with_suffix(".json").read_text())
        if metadata["episode_id"] != memory.episode_id or metadata["step"] != step:
            raise ValueError("Recorded episode/step mismatch")
        times.append(float(metadata["sim_time"]))
    intervals = np.diff(times)
    if not np.isfinite(times).all() or np.any(intervals <= 0):
        raise ValueError("Video requires increasing simulation timestamps")
    dt = float(np.median(intervals))
    if not np.allclose(intervals, dt, rtol=0.01, atol=1e-6):
        raise ValueError("Nonuniform observation clock; do not silently reinterpret video timing")
    duration = len(times) * dt
    if duration > config.max_segment_seconds:
        raise ValueError("Attempt exceeds video duration budget; no silent tail cropping")
    # Match the released Outcome sampler: up to 64 uniformly spaced frames at up to 10 FPS,
    # an even frame count, with both temporal endpoints included.
    count = min(64, max(2, math.floor(duration * 10)))
    count = max(2, count - count % 2)
    indices = [round(i * (len(times) - 1) / (count - 1)) for i in range(count)]
    frames, sources = [], []
    for index in indices:
        step = start + index
        obs = recorder.read_observation(step, memory.episode_id, end,
                                        rgb_cameras=config.cameras, include_depth=False)
        views = []
        for camera in config.cameras:
            if camera not in obs.rgb:
                raise ValueError(f"Required Flora camera {camera!r} is missing")
            view = Image.fromarray(obs.rgb[camera]).convert("RGB")
            view.thumbnail((256, 256))
            views.append(view)
        width, height = views[0].size
        if any(view.size != (width, height) for view in views):
            raise ValueError("Synchronized views must have matching image sizes")
        columns = 2 if len(views) == 4 else len(views)
        collage = Image.new("RGB", (columns * width, (2 if len(views) == 4 else 1) * height))
        for j, view in enumerate(views):
            collage.paste(view, ((j % columns) * width, (j // columns) * height))
        buffer = io.BytesIO()
        # Lossless pixels; reduce CPU compression cost for local HTTP transport.
        collage.save(buffer, format="PNG", compress_level=1)
        frames.append(base64.b64encode(buffer.getvalue()).decode())
        path = recorder.root / recorder.observations[obs.key]
        sources.append(dict(step=step, sim_time=obs.sim_time,
                            path=str(path.relative_to(recorder.root)),
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    segment = dict(start_step=start, end_step=end, start_sim_time=times[0], end_sim_time=times[-1],
                   duration_seconds=duration, cameras=config.cameras, sampled_steps=[start + i for i in indices],
                   sources=sources)
    return dict(model=config.model, revision=config.revision, mode="outcome",
                quantization=config.quantization,
                with_rationale=config.with_rationale if with_rationale is None else with_rationale,
                task=goal, scope=scope, completion_condition=memory.current.completion_condition,
                attempt_id=memory.attempt_id, binding=binding.model_dump(exclude={"request_id"}),
                segment=segment, frames=frames,
                video_metadata=dict(fps=1 / dt, total_num_frames=len(times), frames_indices=indices,
                                    duration=duration, video_backend="recorded_rgb"))


class FloraEvaluator:
    def __init__(self, config, recorder, transport=None):
        self.config, self.log = config, recorder
        self.client = httpx.AsyncClient(timeout=config.timeout_seconds, transport=transport)
        self.calls = 0
        self.cache = {}
        self.lock = asyncio.Lock()

    def capabilities(self):
        return dict(enabled=self.config.enabled, mode="outcome", model=self.config.model,
                    revision=self.config.revision, cameras=self.config.cameras,
                    remaining_calls=max(0, self.config.max_calls - self.calls),
                    with_rationale=self.config.with_rationale, phase_scoring=True,
                    quantization=self.config.quantization,
                    authority="assessment_only; not an installed completion verifier")

    async def evaluate(self, memory, binding, *, phase_id=None, with_rationale=None, window_steps=None):
        if not self.config.enabled:
            return dict(status="unavailable", reason="Flora is disabled", score=None)
        async with self.lock:
            started = time.monotonic()
            request_id, payload = None, None
            try:
                async with asyncio.timeout(self.config.timeout_seconds):
                    payload = await asyncio.to_thread(prepare_segment, self.log, memory, binding, self.config,
                                                      phase_id=phase_id, with_rationale=with_rationale,
                                                      window_steps=window_steps)
                    request_id = hashlib.sha256(dumps(payload).encode()).hexdigest()
                    if request_id in self.cache:
                        result = copy.deepcopy(self.cache[request_id])
                        result["cache_hit"] = True
                        self.log.event("flora_cache_hit", request_id=request_id, binding=binding)
                        return result
                    if self.calls >= self.config.max_calls:
                        return dict(status="budget_exhausted", score=None, request_id=request_id)
                    self.calls += 1
                    payload["request_id"] = request_id
                    directory = self.log.root / "flora"
                    directory.mkdir(exist_ok=True)
                    path = directory / (request_id + ".request.json")
                    await asyncio.to_thread(path.write_text, dumps(payload))
                    self.log.event("flora_request", request_id=request_id, binding=binding,
                                   attempt_id=memory.attempt_id, segment=payload["segment"],
                                   task=payload["task"], request_path=str(path.relative_to(self.log.root)))
                    response = await self.client.post(self.config.endpoint, json=payload)
                    response.raise_for_status()
                    answer = response.json()
                    if any(answer.get(k) != payload[k] for k in ("request_id", "model", "revision")):
                        raise ValueError("Flora response identity/revision mismatch")
                    if answer.get("quantization", "none") != self.config.quantization:
                        raise ValueError("Flora response quantization mismatch")
                    if type(answer.get("score")) is not int or not 1 <= answer["score"] <= 5:
                        raise ValueError("Flora returned an invalid Outcome score")
                    rationale = answer.get('rationale')
                    if payload['with_rationale'] and (answer.get('with_rationale') is not True
                            or not isinstance(rationale, str) or not 0 < len(rationale.strip()) <= 2048):
                        raise ValueError('Requested Flora rationale is missing or invalid')
                    result = dict(status="scored", score=answer["score"], scale=[1, 5], cache_hit=False,
                                  request_id=request_id, model=answer["model"], revision=answer["revision"],
                                  quantization=self.config.quantization,
                                  attempt_id=memory.attempt_id, binding=payload["binding"],
                                  segment={k: v for k, v in payload["segment"].items() if k != "sources"},
                                  task=payload["task"],
                                  scope=payload['scope'], rationale=rationale,
                                  explanation_authority='model hypothesis; not an established cause',
                                  completion_condition=payload["completion_condition"],
                                  seconds=time.monotonic() - started,
                                  note="Model assessment of this historical segment, not a calibrated success probability "
                                       "or physical completion proof. Low score can mean unfinished normal execution.")
                    await asyncio.to_thread((directory / (request_id + ".response.json")).write_text, dumps(answer))
                    self.cache[request_id] = copy.deepcopy(result)
                    self.log.event("flora_result", result=result)
                    return result
            except asyncio.CancelledError:
                self.log.event("flora_cancelled", request_id=request_id, binding=binding)
                raise
            except Exception as error:
                result = dict(status="unavailable", score=None, request_id=request_id,
                              reason=f"{type(error).__name__}: {str(error)[:300]}",
                              binding=binding.model_dump(exclude={"request_id"}),
                              seconds=time.monotonic() - started)
                self.log.event("flora_unavailable", result=result)
                return result

    async def close(self):
        await self.client.aclose()
