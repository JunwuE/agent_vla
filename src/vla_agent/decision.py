import asyncio
import math
import time

import httpx

from .contracts import Decision, DecisionLabel
from .data import dumps
from .decision_context import compact_state
from .decision_criteria import CRITERIA, INSTRUCTIONS, VERSION
from .perception import decision_visual_input


def classify(payload, config, binding, trigger):
    answer = payload["answers"]["route"]
    probabilities = answer["probabilities"]
    if set(probabilities) != set(CRITERIA):
        raise ValueError("Expected exactly five category probabilities")
    if (
        any(
            type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
            for p in probabilities.values()
        )
        or abs(sum(probabilities.values()) - 1) > 0.02
    ):
        raise ValueError("Invalid probability distribution")
    ranked = sorted(probabilities, key=probabilities.get, reverse=True)
    label = DecisionLabel(ranked[0])
    if answer.get("choice") != label:
        raise ValueError("Reported choice contradicts probabilities")
    reason = None
    if probabilities[ranked[0]] - probabilities[ranked[1]] < config.min_margin:
        label, reason = DecisionLabel.ESCALATE, "ambiguous_distribution"
    elif label == DecisionLabel.KEEP and probabilities["KEEP"] < config.keep_threshold:
        label, reason = DecisionLabel.ESCALATE, "keep_below_threshold"
    return Decision(
        binding=binding, label=label, probabilities=probabilities, trigger=trigger, fallback_reason=reason
    )


def build_request(config, observations, memory_summary, binding):
    images, frame_map = decision_visual_input(observations, config)
    if not images:
        raise ValueError("Decision requires at least one historical camera image")
    state = dict(
        memory=memory_summary,
        observations=[o.summary() for o in observations],
        frame_map=frame_map,
        current_step=binding.observation_step,
    )
    if config.context_mode == "compact":
        state = compact_state(state)
    request = dict(
        model=config.model,
        state=dumps(state),
        images=images,
        questions={"route": dict(type="choice", criteria=CRITERIA, instructions=INSTRUCTIONS)},
    )
    return request, state


class Decision3:
    def __init__(self, config, recorder, transport=None):
        self.config, self.log = config, recorder
        self.client = httpx.AsyncClient(timeout=config.timeout_seconds, transport=transport)

    async def decide(self, observations, memory_summary, binding, trigger):
        request, state = build_request(self.config, observations, memory_summary, binding)
        self.log.event(
            "decision_request",
            binding=binding,
            state=state,
            model=self.config.model,
            declared_revision=self.config.revision,
            context_mode=self.config.context_mode,
            criteria_version=VERSION,
            image_count=len(request["images"]),
            question=request["questions"]["route"],
            state_characters=len(request["state"]),
            trigger=trigger,
        )
        async with asyncio.timeout(self.config.timeout_seconds):
            response = await self.client.post(self.config.endpoint, json=request)
            response.raise_for_status()
            payload = response.json()
        self.log.event("decision_response", binding=binding, response=payload)
        if (
            self.config.revision is not None
            and payload.get("meta", {}).get("revision") != self.config.revision
        ):
            raise ValueError("Decision model revision differs from the configured revision")
        return classify(payload, self.config, binding, trigger)

    async def close(self):
        await self.client.aclose()


async def checked_decision(backend, config, observations, summary, binding, trigger, recorder):
    started = time.monotonic()
    recorder.event(
        "decision_window",
        binding=binding,
        observations=[o.summary() for o in observations],
        memory=summary,
        trigger=trigger,
    )
    try:
        async with asyncio.timeout(config.timeout_seconds):
            result = await backend.decide(observations, summary, binding, trigger)
        if not isinstance(result, Decision) or result.binding != binding:
            raise ValueError("Decision binding mismatch")
    except Exception as error:
        result = Decision(
            binding=binding,
            label=DecisionLabel.ESCALATE,
            trigger=trigger,
            fallback_reason=f"{type(error).__name__}: {str(error)[:300]}",
        )
    recorder.event("decision", decision=result, seconds=time.monotonic() - started)
    return result
