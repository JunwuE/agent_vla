"""Fast action loop with one bounded asynchronous supervisor and on-demand planner."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

from .budgets import PlannerBudget
from .code_session import CodeSession
from .contracts import Decision, DecisionLabel, Source, Subtask, validate_actions
from .control import ActionEdit, EEFRequest, OSCControl, execute_target
from .data import read_events
from .decision import checked_decision
from .experience import ExperienceStore
from .flora import FloraEvaluator
from .flora_supervision import FloraMonitor, frozen_score_memory, score_scope
from .harness import Harness
from .memory import SkillLibrary, WorkingMemory
from .perception import decision_window
from .planner import PlannerToolkit
from .planner_service import PlannerServiceError
from .supervision import RunningReviewer, planner_context
from .supervision_rules import low_motion_signals
from .task_memory import TaskMemoryStore
from .workspace import RecoveryWorkspace


class EpisodeBudget(RuntimeError):
    """A normal bounded stop, never permission to bypass supervision."""


class AgentRuntime:
    def __init__(self, config, environment, vla, decision, planner, recorder):
        self.config, self.env, self.vla = config, environment, vla
        self.decision, self.planner, self.log = decision, planner, recorder
        self.harness = Harness(environment, recorder, config.runtime.max_steps)
        self.skills = SkillLibrary(config.runtime.skill_library, config.runtime.memory_mode)
        self.pending = None
        self.last_decision_step = -config.decision.interval_steps
        self.intervention_count = 0
        self.no_progress = 0
        self.last_intervention_step = 0
        self.active_language = None
        self.forced_reason = None
        self.result = None
        self.started = 0
        self.code_session = None
        self.experiences = ExperienceStore(config.runtime.experience_library, config.runtime.memory_mode)
        self.task_memories = TaskMemoryStore(config.runtime.task_memory_library or
                                            Path(config.runtime.experience_library).parent / 'tasks')
        self.last_audit_step = 0
        self.last_low_motion_check_time = float("-inf")
        self.last_low_motion_review_time = float("-inf")
        self.flora = FloraEvaluator(config.flora, recorder)
        self.monitor = FloraMonitor(config, recorder, self.flora) if config.decision.kind == 'flora' else None
        self.last_monitor_tick = 0
        self.last_score_review_step = None
        self.checked_entry_version = None
        self.reviewer = RunningReviewer(config, planner, recorder, self.skills, flora=self.flora,
                                        experiences=self.experiences)

    def trigger(self):
        obs = self.harness.obs
        if obs.events and obs.step > self.last_decision_step:
            return "event:" + ",".join(obs.events)
        self.memory.rule_signals = low_motion_signals(self.memory, self.config.runtime)
        if (self.memory.rule_signals and obs.step > self.last_decision_step
                and obs.sim_time - self.last_low_motion_check_time >= self.config.runtime.stagnation_cooldown_seconds):
            return "possible_stagnation"
        if obs.step - self.last_decision_step >= self.config.decision.interval_steps:
            return "periodic"
        return None

    def submit_decision(self, trigger):
        self.last_decision_step = self.harness.obs.step
        binding = self.memory.binding(self.harness.epoch)
        self.memory.rule_signals = low_motion_signals(self.memory, self.config.runtime)
        if self.memory.rule_signals:
            if self.monitor is None:
                self.last_low_motion_check_time = self.harness.obs.sim_time
            self.log.event('supervision_rule', binding=binding, signals=self.memory.rule_signals,
                           action='attach_context' if self.monitor else 'schedule_decision_review',
                           control_retained='vla')
        if self.monitor is not None:
            self.pending_binding = binding
            self.pending_scope = score_scope(self.memory)
            self.pending_started = time.monotonic()
            snapshot = frozen_score_memory(self.memory)
            self.log.event('decision_window', binding=binding, observations=[], memory=self.memory.summary(),
                           trigger=trigger, backend='flora', input_record='flora_request',
                           note='Actual sampled video and source hashes are in the associated Flora request.')
            self.pending = asyncio.create_task(self.monitor.evaluate(snapshot, binding))
            return
        # Frozen summary and selected observations: never append future frames to an in-flight request.
        self.pending = asyncio.create_task(
            checked_decision(
                self.decision,
                self.config.decision,
                decision_window(self.memory, self.config.decision),
                self.memory.summary(),
                binding,
                trigger,
                self.log,
            )
        )

    async def receive_decision(self, *, wait=False):
        if self.pending is None or not wait and not self.pending.done():
            return None
        result = await self.pending
        self.pending = None
        cfg = self.config.decision
        if self.monitor is not None:
            binding = self.pending_binding
            fresh = (self.memory.matches(binding, self.harness.epoch)
                     and score_scope(self.memory) == self.pending_scope
                     and 0 <= self.harness.obs.step - binding.observation_step <= cfg.max_age_steps
                     and time.monotonic() - binding.observation_time <= cfg.max_age_seconds)
            self.log.event('flora_monitor_result', binding=binding, result=result, accepted=fresh,
                           current_step=self.harness.obs.step)
            if not fresh:
                self.log.event('stale_decision', backend='flora', binding=binding,
                               reason='observation_attempt_or_phase_changed')
                return None
            if result.get('status') == 'scored':
                self.memory.reward_assessments.append(result)
            decision = self.monitor.router.route(result, binding, self.pending_scope)
            self.log.event('decision', decision=decision, backend='flora', score=result.get('score'),
                           model_request_id=result.get('request_id'),
                           seconds=time.monotonic() - self.pending_started)
            return decision
        fresh = (
            self.memory.matches(result.binding, self.harness.epoch)
            and self.harness.obs.step - result.binding.observation_step <= cfg.max_age_steps
            and time.monotonic() - result.binding.observation_time <= cfg.max_age_seconds
        )
        if not fresh:
            self.log.event("stale_decision", decision=result)
            # An expired KEEP never authorizes more motion. Reassess from the current state.
            return Decision(
                binding=self.memory.binding(self.harness.epoch),
                label=DecisionLabel.ESCALATE,
                trigger="stale_response",
                fallback_reason="observation_or_version_expired",
            )
        return result

    async def close_language_segment(self, reason, *, verify=False):
        if self.active_language is None:
            return
        segment = self.active_language
        proof = await self.harness.verify(segment["condition"]) if verify else None
        self.log.event(
            "intervention_end",
            intervention_id=segment["id"],
            verified=bool(proof and proof.passed is True),
            evidence=proof,
            reason=reason,
        )
        self.active_language = None

    async def handle(self, decision, reviewed=None):
        if decision.label == DecisionLabel.KEEP:
            return
        if self.pending is not None:
            self.pending.cancel()
            await asyncio.gather(self.pending, return_exceptions=True)
            self.pending = None
            self.log.event('decision_cancelled', reason='control_handoff')
        if self.harness.obs.step > self.last_intervention_step:
            self.no_progress = 0
        if self.intervention_count >= self.config.runtime.max_interventions:
            raise EpisodeBudget("astra_budget")
        if self.no_progress >= self.config.runtime.max_no_progress_interventions:
            raise RuntimeError("Repeated interventions made no execution progress")
        await self.close_language_segment("next_intervention", verify=True)
        start_step = self.harness.obs.step
        self.intervention_count += 1
        intervention_id = uuid.uuid4().hex
        self.lease = await self.harness.claim(Source.ASTRA)
        self.memory.add(self.harness.obs)
        binding = self.memory.binding(self.harness.epoch)
        intent = "initialize_plan" if decision.trigger == "initialization" else (
            "confirm_intervention" if reviewed else "review_execution")
        context = planner_context(self.config, self.harness, self.memory, intent)
        if self.memory.executor == 'unassigned':
            context['intent'] = 'select_next_executor'
            context['allowed_interventions'] = ['continue', 'code', 'observe', 'replan', 'stop']
        if self.memory.executor == 'code':
            context['allowed_interventions'] = ['code', 'observe', 'replan', 'stop']
            context['recovery_budget'] = self.code_session.budget(self)
            if not all(v > 0 for v in context['recovery_budget'].values()):
                context['allowed_interventions'] = ['replan', 'stop']
        if reviewed:
            context["previous_review"] = dict(result=reviewed.review.model_dump(),
                                               binding=reviewed.binding.model_dump(),
                                               note="Historical assessment; recheck against the fresh paused observations")
        if decision.trigger == "initialization":
            context["allowed_interventions"] = ["replan", "stop"]
        elif decision.trigger == 'subtask_verifier':
            context['allowed_interventions'] = ['switch', 'replan', 'stop']
        self.log.event(
            "intervention_begin",
            intervention_id=intervention_id,
            decision=decision,
            normal_transition=decision.label == DecisionLabel.SWITCH,
            context_binding=binding,
        )
        workspace = self.code_session.workspace if self.code_session else RecoveryWorkspace(
            self.log.root / "workspaces" / intervention_id,
            dict(
                episode_id=self.memory.episode_id,
                task=self.memory.task,
                subtask=self.memory.current,
                binding=binding,
            ),
            max_versions=self.config.runtime.recovery_program_versions,
        )
        context["workspace"] = workspace.summary()
        toolkit = PlannerToolkit(
            self.harness,
            self.memory,
            self.skills,
            self.vla,
            workspace,
            context["allowed_interventions"],
            flora=self.flora,
            experiences=self.experiences,
        )
        async with asyncio.timeout(self.config.planner.timeout_seconds):
            plan = await self.planner.plan(decision, context, toolkit)
        if (
            not self.memory.matches(binding, self.harness.epoch)
            or self.harness.obs.step != binding.observation_step
        ):
            raise RuntimeError("Stale planner response")
        await toolkit.validate_intervention(plan, context["allowed_interventions"])
        self.log.event("intervention_plan", intervention_id=intervention_id, plan=plan, binding=binding)
        if plan.kind not in context["allowed_interventions"]:
            raise ValueError(f"Intervention {plan.kind} disabled for this ablation")
        verified = False
        report = None
        if decision.trigger != "initialization" and plan.kind in {"refine", "replan", "code", "local", "eef", "edit"}:
            self.memory.begin_retry(plan.reason, reviewed.review.model_dump() if reviewed else None)
            self.log.event("attempt_interrupted", attempt=self.memory.attempt_summary(), plan=plan.kind)
        if plan.kind == "refine":
            if not plan.instruction or not plan.instruction.strip():
                raise ValueError("REFINE requires a nonempty instruction")
            self.memory.pending_instruction = plan.instruction
            self.active_language = dict(
                id=intervention_id, condition=self.memory.current.completion_condition
            )
        elif plan.kind == "switch":
            proof = await self.harness.verify(self.memory.current.completion_condition)
            verified = proof.passed is True
            if verified:
                completed = self.memory.current.model_dump()
                advanced = self.memory.advance(proof.model_dump())
                self.log.event('subtask_transition', completed=completed, verification=proof,
                               next_subtask=self.memory.current if advanced else None,
                               next_executor='vla' if advanced else None)
                if not advanced and not self.harness.obs.success:
                    self.forced_reason = 'Local plan ended without native task success; replan remaining goals'
            else:
                self.forced_reason = "Subtask completion was not verified"
        elif plan.kind == "replan":
            if self.memory.executor == 'code':
                self.log.event('code_subtask_abandoned', assignment=self.memory.code_assignment,
                               reason=plan.reason, remaining_plan=plan.remaining_plan)
                self.code_session = None
            self.memory.replan(plan.remaining_plan or [])
        elif plan.kind in {"local", "eef", "edit"}:
            if not plan.completion_condition:
                raise ValueError("Local correction requires a verification condition")
            if (
                plan.kind in {"eef", "edit"}
                and plan.completion_condition not in context["available_verifiers"]
            ):
                raise ValueError("Correction requires an installed completion verifier")
            if plan.kind == "eef":
                request = EEFRequest.model_validate(dict(target=plan.target, steps=plan.steps))
                report = await execute_target(self.harness, self.memory, self.lease, request, intervention_id)
            else:
                if plan.kind == "edit":
                    original = toolkit.take_proposal(plan.proposal_id, plan.steps)
                    actions, diagnostics = OSCControl(self.harness.obs).edit(
                        original, ActionEdit.model_validate(plan.edit)
                    )
                    self.log.event(
                        "action_edit",
                        before_step=self.harness.obs.step,
                        intervention_id=intervention_id,
                        proposal_id=plan.proposal_id,
                        original=original,
                        edited=actions,
                        diagnostics=diagnostics,
                    )
                    toolkit.proposal = None
                else:
                    actions = validate_actions(plan.actions, maximum=5)
                for action in actions:
                    if self.harness.obs.terminal:
                        break
                    await self.harness.execute(action, self.lease, self.memory, intervention_id)
            proof = await self.harness.verify(plan.completion_condition)
            verified = proof.passed is True
            if not verified:
                self.forced_reason = "Local correction did not pass verification"
        elif plan.kind == "code":
            if self.code_session is None:
                self.code_session = CodeSession(self, workspace)
            plan, report, verified = await self.code_session.run(self, plan, decision, context, toolkit)
        elif plan.kind == "stop":
            await self.harness.stop(plan.reason)
        elif plan.kind == "observe":
            self.harness.obs = await self.env.observe()
            self.forced_reason = "Planner requested reassessment after observation"
        elif plan.kind != "continue":
            raise ValueError(f"Unknown intervention: {plan.kind}")
        if plan.kind in {'continue', 'refine', 'switch', 'replan'}:
            self.memory.executor = 'vla'
        self.memory.interventions.append(
            dict(id=intervention_id, kind=plan.kind, verified=verified, reason=plan.reason, report=report)
        )
        if plan.kind != "refine":
            self.log.event(
                "intervention_end", intervention_id=intervention_id, verified=verified, report=report
            )
        progress = self.harness.obs.step > start_step or verified
        self.no_progress = 0 if progress else self.no_progress + 1
        self.last_intervention_step = self.harness.obs.step
        if (not self.harness.halted and not self.harness.obs.terminal
                and self.memory.executor == 'vla' and not self.forced_reason):
            # New lease and fresh observation; never resume the old VLA chunk.
            self.lease = await self.harness.claim(Source.VLA)
            self.memory.add(self.harness.obs)
            if not self.forced_reason and self.memory.attempt_status == "recovering":
                self.memory.resume_retry()
                self.log.event("attempt_resumed", attempt=self.memory.attempt_summary())
        self.log.event("memory", state=self.memory.summary())
        self.last_decision_step = self.harness.obs.step

    async def check_low_motion_review(self):
        """Independent numeric review trigger; never waits for Flora or takes control."""
        self.memory.rule_signals = low_motion_signals(self.memory, self.config.runtime)
        if not self.memory.rule_signals:
            return
        now = self.harness.obs.sim_time
        if now - self.last_low_motion_review_time < self.config.runtime.stagnation_cooldown_seconds:
            return
        self.last_low_motion_review_time = now
        decision = Decision(binding=self.memory.binding(self.harness.epoch), label=DecisionLabel.ESCALATE,
                            trigger='possible_stagnation',
                            fallback_reason='Measured low motion requests review; not a confirmed failure')
        self.log.event('supervision_rule', binding=decision.binding, signals=self.memory.rule_signals,
                       action='request_astra_review', control_retained='vla')
        await self.route_decision(decision)

    async def route_decision(self, decision):
        if decision.label == DecisionLabel.KEEP:
            return
        if self.reviewer.pending is not None:
            self.log.event('review_coalesced', decision=decision, reason='astra_already_reviewing',
                           control_retained='vla')
            return
        # A recent, accepted progress assessment can cover duplicate low-score /
        # low-motion signals for the same attempt and phase. This never grants
        # action authority, suppresses a new attempt, or handles an unsafe event.
        recent = self.memory.reviews[-1] if self.memory.reviews else {}
        if (self.config.runtime.progress_review_cooldown_steps
                and decision.trigger in {'flora_score', 'possible_stagnation'}
                and recent.get('action') == 'continue' and recent.get('assessment') == 'progressing'
                and recent.get('attempt_id') == self.memory.attempt_id
                and recent.get('control_epoch') == self.harness.epoch
                and recent.get('subtask_version') == self.memory.subtask_version
                and recent.get('phase_revision') == self.memory.phases.revision
                and 0 <= self.harness.obs.step - recent['observation_step']
                < self.config.runtime.progress_review_cooldown_steps):
            self.log.event('review_coalesced', decision=decision, reason='recent_accepted_progress',
                           previous_review=recent.get('event_id'), control_retained='vla')
            return
        if (self.monitor is not None and decision.trigger == 'flora_score'
                and self.last_score_review_step is not None
                and self.harness.obs.step - self.last_score_review_step
                < self.config.decision.flora.review_cooldown_steps):
            self.log.event('review_coalesced', decision=decision, reason='score_review_cooldown',
                           control_retained='vla')
            return
        if not self.config.runtime.nonblocking_review:
            await self.handle(decision)
            return
        if self.config.runtime.continue_vla_on_review_budget:
            reason = None
            if self.reviewer.budget_count >= self.config.runtime.max_reviews:
                reason = 'review_budget'
            elif getattr(self.planner, 'calls', 0) >= self.config.planner.max_calls:
                reason = 'astra_turn_budget'
            elif (self.config.planner.max_total_tokens is not None
                    and sum(getattr(self.planner, 'tokens', {}).values()) >= self.config.planner.max_total_tokens):
                reason = 'astra_token_budget'
            if (reason and decision.trigger in {'flora_score', 'possible_stagnation',
                    'periodic_audit', 'expired_review_recheck'}
                    and self.memory.executor == 'vla' and self.harness.owner == Source.VLA
                    and not self.harness.uncertain):
                self.log.event('review_deferred', decision=decision, reason=reason,
                    control_retained='vla', failure_confirmed=False, model_calls_allowed=False)
                return
        if self.reviewer.budget_count >= self.config.runtime.max_reviews:
            raise EpisodeBudget("review_budget")
        if getattr(self.planner, "calls", 0) >= self.config.planner.max_calls:
            raise EpisodeBudget("astra_budget")
        self.reviewer.submit(decision, self.harness, self.memory)
        self.last_audit_step = self.harness.obs.step
        if self.monitor is not None:
            self.last_score_review_step = self.harness.obs.step

    async def poll_review(self):
        """Only an explicit, still-valid Astra request can pause a running attempt."""
        was_pending = self.reviewer.pending is not None
        reviewed = self.reviewer.poll(self.harness, self.memory)
        if was_pending and self.reviewer.pending is None and self.monitor is None:
            self.last_decision_step = self.harness.obs.step
        refresh, self.reviewer.refresh_reason = self.reviewer.refresh_reason, None
        if reviewed is None and refresh and not self.harness.halted and not self.harness.obs.terminal:
            fresh = Decision(binding=self.memory.binding(self.harness.epoch), label=DecisionLabel.ESCALATE,
                             trigger="expired_review_recheck", fallback_reason=refresh)
            self.log.event("review_refresh", binding=fresh.binding, control_retained="vla")
            await self.route_decision(fresh)
            return False
        if reviewed is None or reviewed.review.action == "continue":
            return False
        if self.harness.obs.terminal or self.harness.halted:
            return False
        if getattr(self.planner, "calls", 0) >= self.config.planner.max_calls:
            raise EpisodeBudget("astra_budget")
        decision = Decision(binding=self.memory.binding(self.harness.epoch),
                            label=reviewed.decision.label, trigger="astra_review_request",
                            fallback_reason=reviewed.review.reason)
        await self.handle(decision, reviewed=reviewed)
        return True

    async def loop(self):
        self.lease = await self.harness.start()
        task = self.config.task
        if self.config.environment.instruction_source == "environment":
            instruction = self.harness.obs.facts.get("task_instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("Environment did not supply the required task instruction")
            task = task.model_copy(update=dict(
                instruction=instruction,
                subtasks=[Subtask(name="task", instruction=instruction, completion_condition="task_success")],
                language_catalog=list(getattr(self.env, 'language_catalog', task.language_catalog)),
            ))
        self.log.event("task_initialized", source=self.config.environment.instruction_source, task=task)
        self.memory = WorkingMemory(task, self.harness.obs.episode_id)
        self.memory.add(self.harness.obs)
        self.memory.task_scope = dict(robot=task.robot, environment_kind=self.config.environment.kind,
                                      task_name=self.config.environment.task_name)
        self.memory.task_memory = self.task_memories.read(self.memory.task_scope, self.experiences, self.skills)
        self.log.event('task_memory_snapshot', task_scope=self.memory.task_scope,
                       memory=self.memory.task_memory)
        self.memory.retrieved_experiences = self.experiences.search(
            task.robot, task.instruction, skill_index=self.skills.index, scope=self.memory.task_scope)
        self.log.event("skill_snapshot", mode=self.config.runtime.memory_mode, skills=self.skills.snapshot)
        if self.config.runtime.supervision_enabled and self.config.runtime.initial_planning:
            await self.handle(Decision(
                binding=self.memory.binding(self.harness.epoch), label=DecisionLabel.ESCALATE,
                trigger="initialization", fallback_reason="Astra must establish the plan before VLA motion",
            ))
        while not self.harness.obs.terminal and not self.harness.halted:
            if self.harness.obs.step >= self.config.runtime.max_steps:
                return "step_budget"
            if self.forced_reason:
                await self.reviewer.close()
                decision = Decision(
                    binding=self.memory.binding(self.harness.epoch),
                    label=DecisionLabel.ESCALATE,
                    trigger="recovery_feedback",
                    fallback_reason=self.forced_reason,
                )
                self.forced_reason = None
                await self.handle(decision)
                continue
            if (self.config.runtime.supervision_enabled and not self.harness.queue
                    and getattr(self.env, 'subtask_boundary_checks', False)):
                # Positive evidence at a chunk boundary goes to Astra. No failure
                # heuristic truncates a first VLA attempt or changes the instruction.
                proof = await self.harness.verify(self.memory.current.completion_condition)
                if proof.passed is True:
                    await self.reviewer.close()
                    self.log.event('subtask_boundary_ready', subtask=self.memory.current, verification=proof)
                    await self.handle(Decision(binding=self.memory.binding(self.harness.epoch),
                        label=DecisionLabel.SWITCH, trigger='subtask_verifier',
                        fallback_reason='Current subtask independently verified; review and switch to the next goal'))
                    continue
                if self.checked_entry_version != self.memory.subtask_version:
                    entry = [await self.harness.verify(c) for c in self.memory.current.preconditions]
                    self.log.event('subtask_entry', subtask=self.memory.current, checks=entry)
                    if any(check.passed is not True for check in entry):
                        await self.reviewer.close()
                        self.forced_reason = 'Subtask entry checks changed; replan before executing'
                        continue
                    self.checked_entry_version = self.memory.subtask_version
            if await self.poll_review():
                continue
            if self.memory.executor != 'vla':
                raise RuntimeError('VLA cannot resume while the subtask is assigned to code or unassigned')
            audit_due = (self.config.runtime.audit_interval_steps > 0
                         and self.harness.obs.step - self.last_audit_step >= self.config.runtime.audit_interval_steps)
            if (self.monitor is None and self.config.runtime.supervision_enabled
                    and self.config.runtime.nonblocking_review and audit_due
                    and self.reviewer.pending is None and self.pending is None
                    and self.reviewer.budget_count < self.config.runtime.max_reviews
                    and getattr(self.planner, 'calls', 0) < self.config.planner.max_calls):
                self.last_audit_step = self.harness.obs.step
                audit = Decision(binding=self.memory.binding(self.harness.epoch), label=DecisionLabel.ESCALATE,
                                 trigger='periodic_audit', fallback_reason='Read-only audit; not a failure label')
                self.log.event('supervision_audit', binding=audit.binding, control_retained='vla')
                await self.route_decision(audit)
            if self.config.runtime.supervision_enabled and self.monitor is not None:
                # Numeric evidence is a separate event source, even while a score is
                # pending or unchanged. It does not reset the 50-step Flora clock.
                await self.check_low_motion_review()
                decision = await self.receive_decision()
                if decision is not None:
                    await self.route_decision(decision)
                tick = self.harness.obs.step // self.config.decision.interval_steps
                if tick > self.last_monitor_tick:
                    self.last_monitor_tick = tick
                    if self.pending is None:
                        self.submit_decision('periodic')
                    else:
                        self.log.event('monitor_tick_skipped', step=self.harness.obs.step,
                                       reason='previous_flora_request_pending', control_retained='vla')
            elif self.config.runtime.supervision_enabled and self.pending is None and self.reviewer.pending is None:
                trigger = self.trigger()
                if trigger:
                    self.submit_decision(trigger)
            await asyncio.sleep(0)
            must_wait = not self.config.runtime.nonblocking_review and self.pending and (
                self.harness.obs.step - self.last_decision_step >= self.config.decision.max_unreviewed_steps
            )
            decision = await self.receive_decision(wait=bool(must_wait))
            if decision and decision.label != DecisionLabel.KEEP:
                await self.route_decision(decision)
                if not self.config.runtime.nonblocking_review:
                    continue
            if not self.harness.queue:
                self.memory.inference_boundary()
                binding = self.memory.binding(self.harness.epoch)
                chunk = await self.vla.infer(self.harness.obs, self.memory.instruction, binding)
                # Account for the completed inference before polling reviews.
                # Otherwise a review arriving during inference can throw away a
                # sampled chunk, changing even a subsequent unchanged continuation.
                remaining = self.config.runtime.max_steps - self.harness.obs.step
                self.harness.enqueue(
                    chunk, self.lease, self.memory, min(self.config.runtime.chunk_steps, remaining)
                )
                if await self.poll_review():
                    continue
                decision = await self.receive_decision()
                if decision and decision.label != DecisionLabel.KEEP:
                    await self.route_decision(decision)
                    if not self.config.runtime.nonblocking_review:
                        continue
            action = self.harness.queue.popleft()
            segment_id = self.active_language["id"] if self.active_language else None
            await self.harness.execute(action, self.lease, self.memory, segment_id)
        return (
            "success"
            if self.harness.obs.success
            else "stopped"
            if self.harness.halted
            else "environment_done"
        )

    async def run(self):
        self.started = time.monotonic()
        reason, error = None, None
        wall_budget = asyncio.timeout(self.config.runtime.max_wall_seconds)
        try:
            async with wall_budget:
                reason = await self.loop()
                await self.close_language_segment("episode_end", verify=True)
        except (EpisodeBudget, PlannerBudget) as exc:
            reason = getattr(exc, 'budget_key', str(exc))
            self.log.event("budget_exhausted", budget=reason, outcome="stop_without_further_actions")
        except PlannerServiceError as exc:
            reason, error = exc.reason, str(exc)
            self.log.event('planner_service_failure', code=exc.code, error=error,
                           retryable=exc.retryable, outcome='stop_without_further_actions')
        except Exception as exc:
            reason = ("wall_time_budget" if wall_budget.expired() else "component_timeout") if isinstance(exc, TimeoutError) else "runtime_error"
            error = f"{type(exc).__name__}: {exc}"
            self.log.event("runtime_error", error=error)
        finally:
            await self.harness.stop(reason or "cancelled")
            await self.reviewer.close()
            if self.pending:
                self.pending.cancel()
                await asyncio.gather(self.pending, return_exceptions=True)
            for obj in (self.vla, self.decision, self.planner, self.monitor, self.flora, self.env):
                if obj is None:
                    continue
                try:
                    await obj.close()
                except Exception as exc:
                    self.log.event("cleanup_error", component=type(obj).__name__, error=str(exc))
        obs = self.harness.obs
        if reason == "environment_done" and obs:
            reason = obs.facts.get("termination_reason", reason)
        result = dict(
            success=bool(obs and obs.success),
            reason=reason,
            error=error,
            control_steps=obs.step if obs else 0,
            interventions=self.intervention_count,
            astra_calls=getattr(self.planner, "calls", 0),
            astra_call_unit=getattr(self.planner, "call_unit", "model_requests"),
            astra_tokens=getattr(self.planner, "tokens", {}),
            decision_checks=self.log.counts["decision"],
            astra_reviews=self.reviewer.count,
            flora_calls=self.flora.calls + (self.monitor.evaluator.calls if self.monitor else 0),
            flora_tool_calls=self.flora.calls,
            flora_monitor_calls=self.monitor.evaluator.calls if self.monitor else 0,
            wall_seconds=time.monotonic() - self.started,
            execution_uncertain=self.harness.uncertain,
            environment=self.config.environment.kind,
        )
        self.log.finish(result)
        if hasattr(self, 'memory'):
            try:
                receipt = self.experiences.record(self.memory, result, self.log, read_events(self.log.root))
                (self.log.root / 'experience-receipt.json').write_text(json.dumps(receipt))
            except Exception as exc:
                (self.log.root / 'experience-error.txt').write_text(f'{type(exc).__name__}: {exc}')
        self.result = result
        return result
