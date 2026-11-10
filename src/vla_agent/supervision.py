"""Read-only Astra reviews of frozen evidence while the VLA retains control."""

import asyncio
import time
from dataclasses import dataclass

import numpy as np

from .budgets import PlannerBudget
from .contracts import Binding, Decision, ExecutionReview, Source
from .control import available_capabilities
from .perception import window
from .planner_service import PlannerServiceError
from .toolkit import PlannerToolkit


def planner_context(config, harness, memory, intent):
    return dict(
        binding=memory.binding(harness.epoch), intent=intent, memory=memory.summary(),
        allowed_interventions=config.runtime.allowed_interventions,
        observations=window(memory, config.planner.window_frames),
        available_verifiers=getattr(harness.env, "verifier_conditions", ["task_success"]),
        available_capabilities=available_capabilities(harness.obs),
        controller=harness.obs.controller, eef_control_pose=harness.obs.eef_control_pose,
        recovery_budget=dict(seconds=config.runtime.recovery_seconds, steps=config.runtime.recovery_steps),
        recovery_program_limits=dict(steps=config.runtime.recovery_steps,
                                     seconds=config.runtime.recovery_seconds,
                                     tool_calls=config.runtime.recovery_tool_calls,
                                     observation_bytes=config.runtime.recovery_observation_bytes),
        execution_budget=dict(remaining_episode_steps=max(0, config.runtime.max_steps-harness.obs.step),
                              episode_wall_seconds=config.runtime.max_wall_seconds,
                              code_attempts_per_subtask=config.runtime.recovery_attempts,
                              code_steps_per_subtask=config.runtime.recovery_total_steps),
    )


@dataclass
class ReviewedSnapshot:
    review: ExecutionReview
    binding: Binding
    attempt_id: str
    decision: Decision


class RunningReviewer:
    def __init__(self, config, planner, recorder, skills, flora=None, experiences=None):
        self.config, self.planner, self.log, self.skills = config, planner, recorder, skills
        self.flora = flora
        self.experiences = experiences
        self.pending = None
        self.count = 0
        self.admission_deferrals = 0
        self.refresh_reason = None

    @property
    def budget_count(self):
        # IDs remain monotonic; waiting without a model request is not a review.
        return self.count - self.admission_deferrals

    def submit(self, decision, harness, memory):
        if self.pending is not None:
            raise RuntimeError("Only one Astra turn may run at a time")
        self.toolkit = PlannerToolkit.for_review(harness, memory, self.skills,
                                                self.config.runtime.allowed_interventions, flora=self.flora,
                                                experiences=self.experiences)
        self.phase_revision = memory.phases.revision
        self.initial_reward_ids = {r['request_id'] for r in memory.reward_assessments}
        self.context = planner_context(self.config, self.toolkit.harness, self.toolkit.memory, "review_running")
        self.binding = self.context["binding"]
        self.attempt_id = memory.attempt_id
        self.decision = decision
        self.count += 1
        self.planner_calls_at_submit = getattr(self.planner, 'calls', None)
        self.waiting_for_boundary = False
        self.started = time.monotonic()
        self.log.event("review_begin", review=self.count, binding=self.binding,
                       attempt=memory.attempt_summary(), decision=decision, control_retained="vla")

        async def run():
            async with asyncio.timeout(self.config.planner.timeout_seconds):
                result = await self.planner.plan(decision, self.context, self.toolkit)
                if not isinstance(result, ExecutionReview):
                    raise ValueError("Running review must return ExecutionReview, never an executable intervention")
                self.toolkit.validate_review(result)
                return result

        self.pending = asyncio.create_task(run())

    def poll(self, harness, memory):
        if self.pending is None or not self.pending.done():
            return None
        pending = self.pending
        try:
            review = pending.result()
        except PlannerBudget as error:
            self.pending = None
            if (self.config.runtime.continue_vla_on_review_budget
                    and harness.owner == Source.VLA and memory.executor == 'vla' and not harness.uncertain):
                self.log.event('review_failed', review=self.count, error=str(error),
                    outcome='optional_supervision_budget_exhausted_vla_retained', binding=self.binding)
                return None
            self.log.event('review_failed', review=self.count, error=str(error), outcome='planner_budget_stop')
            raise
        except Exception as error:
            self.pending = None
            deferred = (isinstance(error, PlannerServiceError) and error.code == 'localAdmissionTimeout'
                        and self.planner_calls_at_submit is not None
                        and self.planner_calls_at_submit == getattr(self.planner, 'calls', None))
            if deferred:
                self.admission_deferrals += 1
            self.log.event("review_failed", review=self.count, error=f"{type(error).__name__}: {error}",
                           outcome="deferred_without_model_request" if deferred else "no_intervention",
                           binding=self.binding, review_budget_count=self.budget_count)
            return None
        # An ordinary read-only confirmation must not truncate an executing VLA
        # chunk. Keep the result pending, then validate its age and binding again
        # at the boundary. Unsafe assessments still interrupt immediately.
        if (review.action == "request_intervention" and review.assessment != "unsafe"
                and harness.queue and not harness.halted and not harness.obs.terminal):
            if not self.waiting_for_boundary:
                self.log.event("review_waiting_for_chunk_boundary", review=self.count,
                               binding=self.binding, current_step=harness.obs.step,
                               remaining_steps=len(harness.queue), assessment=review.assessment)
                self.waiting_for_boundary = True
            return None
        self.pending = None
        cfg = self.config.runtime
        same_state = (not harness.halted and not harness.obs.terminal
                 and memory.matches(self.binding, harness.epoch)
                 and memory.attempt_id == self.attempt_id
                 and memory.phases.revision == self.phase_revision)
        fresh = (0 <= harness.obs.step - self.binding.observation_step <= cfg.review_max_age_steps
                 and time.monotonic() - self.binding.observation_time <= cfg.review_max_age_seconds)
        if not (same_state and fresh):
            self.log.event("review_discarded", review=self.count, result=review, binding=self.binding,
                           current_step=harness.obs.step, reason="expired_or_state_changed")
            if same_state and not fresh and review.action == "request_intervention":
                self.refresh_reason = (
                    f"A review of historical step {self.binding.observation_step} expired. "
                    f"It hypothesized: {review.reason} Reassess using this NEW snapshot. "
                    "Old evidence does not authorize pausing or changing state. Focus on whether the "
                    "specific visible problem still exists; reuse historical diagnosis and avoid "
                    "repeating expensive tools unless new evidence requires them."
                )
            return None
        displacement_limit = cfg.review_stall_recheck_displacement_m
        if (displacement_limit is not None and review.action == 'request_intervention'
                and review.assessment == 'stalled_progress'):
            # New movement invalidates an old claim of continued immobility, not a failure
            # diagnosis in general. Do not infer task progress or skip unsafe/wrong-target reviews.
            before = self.toolkit.harness.obs.state
            after = harness.obs.state
            eef_delta = float(np.linalg.norm(after[:3] - before[:3]))
            base_delta = float(np.linalg.norm(after[7:10] - before[7:10]))
            if max(eef_delta, base_delta) > displacement_limit:
                self.log.event('review_discarded', review=self.count, result=review, binding=self.binding,
                    current_step=harness.obs.step, reason='stagnation_evidence_changed',
                    eef_displacement_m=eef_delta, base_displacement_m=base_delta,
                    threshold_m=displacement_limit, control_retained='vla', task_progress_confirmed=False)
                self.refresh_reason = (
                    'Movement after the reviewed snapshot contradicts continued immobility. '
                    'This does not establish useful task progress. Reassess current visual evidence '
                    'without pausing solely on the old stalled-progress claim.'
                )
                return None
        event = self.log.event("review_end", review=self.count, result=review, binding=self.binding,
                               current_step=harness.obs.step, seconds=time.monotonic() - self.started)
        memory.reviews.append(dict(event_id=event, attempt_id=self.attempt_id,
                                   control_epoch=harness.epoch, subtask_version=memory.subtask_version,
                                   phase_revision=self.toolkit.memory.phases.revision,
                                   observation_step=self.binding.observation_step, **review.model_dump()))
        known = {n.get("event_id") for n in memory.notes}
        for note in self.toolkit.memory.notes:
            if note.get("event_id") not in known:
                memory.notes.append(note)
        known_rewards = {r["request_id"] for r in memory.reward_assessments}
        for result in self.toolkit.memory.reward_assessments:
            if result["request_id"] not in known_rewards and result['request_id'] not in self.initial_reward_ids:
                memory.reward_assessments.append(result)
        memory.phases = self.toolkit.memory.phases
        memory.groundings.update(self.toolkit.memory.groundings)
        while len(memory.groundings) > 32:
            memory.groundings.pop(next(iter(memory.groundings)))
        self.log.event("memory", state=memory.summary())
        return ReviewedSnapshot(review, self.binding, self.attempt_id, self.decision)

    async def close(self):
        if self.pending is not None:
            self.pending.cancel()
            await asyncio.gather(self.pending, return_exceptions=True)
            self.pending = None
            self.log.event("review_cancelled", review=self.count, reason="episode_ended_or_control_changed")
