"""Causal Flora score routing. All thresholds request review, never robot intervention."""

import asyncio
import copy
from collections import deque

from .contracts import Decision, DecisionLabel
from .flora import FloraEvaluator


def score_scope(memory):
    # Phase revisions also change for a new assessment of the SAME goal. Only a
    # changed phase definition/start invalidates comparisons across score windows.
    phase = next((p for p in memory.phases.phases if p['id'] == memory.phases.active), None)
    return (memory.episode_id, memory.subtask_version, memory.instruction_version,
            memory.attempt_id, memory.current.instruction, memory.phases.active,
            memory.phases.started_step, repr(phase))


def frozen_score_memory(memory):
    snapshot = copy.copy(memory)
    snapshot.task = memory.task.model_copy(deep=True)
    snapshot.phases = copy.deepcopy(memory.phases)
    # The scorer reads immutable observations from the recorder. Do not copy
    # hundreds of raw RGB-D arrays just to freeze a goal and attempt boundary.
    snapshot.observations = deque()
    snapshot.action_history = deque()
    return snapshot


class ScoreRouter:
    def __init__(self, config):
        self.config = config
        self.scope = None
        self.history = deque(maxlen=max(config.low_checks, config.plateau_checks, 2))

    def route(self, result, binding, scope):
        if scope != self.scope:
            self.scope, self.history = scope, deque(maxlen=self.history.maxlen)
        label, reason = DecisionLabel.KEEP, 'score_observe'
        score = result.get('score')
        if result.get('status') != 'scored' or type(score) is not int or not 1 <= score <= 5:
            self.history.clear()  # A missing measurement cannot count toward consecutive scores.
            return Decision(binding=binding, label=DecisionLabel.ESCALATE,
                            trigger='flora_score', fallback_reason='flora_unavailable')
        if self.history and binding.observation_step <= self.history[-1][0]:
            raise ValueError('Flora score history must advance in execution steps')
        previous = self.history[-1][1] if self.history else None
        self.history.append((binding.observation_step, score))
        values = [s for _, s in self.history]
        if score >= self.config.high_score:
            # Even a high subtask score asks Astra to check completion; phase
            # completion must not be confused with executable-subtask SWITCH.
            label, reason = DecisionLabel.ESCALATE, 'flora_possible_completion'
        elif previous is not None and previous - score >= self.config.drop_points:
            label, reason = DecisionLabel.ESCALATE, 'flora_score_drop'
        elif len(values) >= self.config.low_checks and all(
                s <= self.config.low_score for s in values[-self.config.low_checks:]):
            label, reason = DecisionLabel.ESCALATE, 'flora_repeated_low_score'
        elif len(values) >= self.config.plateau_checks and len(set(values[-self.config.plateau_checks:])) == 1:
            label, reason = DecisionLabel.ESCALATE, 'flora_unchanged_score_review'
        return Decision(binding=binding, label=label, trigger='flora_score', fallback_reason=reason)


class FloraMonitor:
    def __init__(self, config, recorder, tool_evaluator):
        if not config.flora.enabled:
            raise ValueError('decision.kind=flora requires flora.enabled=true')
        if not config.runtime.nonblocking_review:
            raise ValueError('Flora monitoring requires nonblocking_review=true')
        self.config, self.log = config, recorder
        settings = config.flora.model_copy(update=dict(
            with_rationale=False, max_calls=config.decision.flora.max_calls,
            timeout_seconds=config.decision.timeout_seconds,
        ))
        self.evaluator = FloraEvaluator(settings, recorder)
        # One device request at a time, but separate monitor / diagnostic budgets.
        self.evaluator.lock = tool_evaluator.lock
        self.router = ScoreRouter(config.decision.flora)

    async def evaluate(self, memory, binding):
        try:
            # Include time waiting behind an Astra diagnostic request in the deadline.
            async with asyncio.timeout(self.config.decision.timeout_seconds):
                return await self.evaluator.evaluate(
                    memory, binding, phase_id=memory.phases.active, with_rationale=False,
                    window_steps=self.config.decision.flora.window_steps,
                )
        except TimeoutError:
            return dict(status='unavailable', score=None, reason='flora_monitor_timeout')

    async def close(self):
        await self.evaluator.close()
