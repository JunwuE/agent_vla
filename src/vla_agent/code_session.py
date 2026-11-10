"""A code executor owns one complete subtask across program revisions."""

import time

from .contracts import Source
from .control import available_capabilities
from .recovery import RecoveryRunner
from .skill_packages import execution_bundle
from .supervision import planner_context


class CodeSession:
    def __init__(self, runtime, workspace):
        self.workspace = workspace
        # VLA's old chunk was revoked before entering this program boundary. Apply the
        # selected subtask language so recorded code actions carry their actual goal.
        runtime.memory.inference_boundary()
        self.goal = runtime.memory.current.model_copy(deep=True)
        self.subtask_version = runtime.memory.subtask_version
        self.start_step = runtime.harness.obs.step
        self.deadline = time.monotonic() + runtime.config.runtime.recovery_total_seconds
        self.attempts = 0
        self.identity = workspace.root.name
        runtime.memory.executor = "code"
        runtime.memory.code_assignment = dict(
            subtask=self.goal.model_dump(),
            subtask_version=self.subtask_version,
            session_id=self.identity,
            start_step=self.start_step,
            attempts=0,
            completion_required=True,
        )
        runtime.log.event("executor_assignment", assignment=runtime.memory.code_assignment, executor="code")

    def budget(self, runtime):
        cfg = runtime.config.runtime
        return dict(
            steps=max(
                0,
                min(
                    cfg.recovery_total_steps - (runtime.harness.obs.step - self.start_step),
                    cfg.max_steps - runtime.harness.obs.step,
                ),
            ),
            seconds=max(0, self.deadline - time.monotonic()),
            attempts_remaining=max(0, cfg.recovery_attempts - self.attempts),
        )

    async def run(self, rt, plan, decision, context, toolkit):
        report = None
        verified = False
        while True:
            budget = self.budget(rt)
            if not all(v > 0 for v in budget.values()):
                rt.log.event("recovery_session_budget", intervention_id=self.identity, **budget)
                break
            if rt.memory.subtask_version != self.subtask_version or rt.memory.current != self.goal:
                raise ValueError("Code subtask changed during its session")
            if plan.kind != "code" or plan.completion_condition != self.goal.completion_condition:
                raise ValueError(
                    "Code must finish its assigned subtask; a local recovery goal cannot release VLA"
                )
            await toolkit.validate_intervention(plan, context["allowed_interventions"])
            skill = None
            version = plan.program_version
            code = plan.code
            if version:
                code = self.workspace.read(version)["source"]
                self.workspace.validate_parameters(version, plan.parameters)
            elif plan.skill_id:
                skill = rt.skills.get(plan.skill_id, plan.skill_version, rt.memory.task.robot)
                rt.skills.validate_invocation(
                    skill,
                    plan.parameters,
                    available_capabilities(rt.harness.obs),
                    plan.completion_condition,
                    context["available_verifiers"],
                )
                code = skill["code"]
                rt.log.event(
                    "skill_invocation",
                    intervention_id=self.identity,
                    skill_id=plan.skill_id,
                    version=plan.skill_version,
                    status=skill["status"],
                    parameters=plan.parameters,
                    verified_reuse=skill["status"] == "verified",
                )
            if not version:
                contract = dict(
                    origin_skill={k: skill[k] for k in ('id', 'version', 'sha256', 'status', 'robot')},
                    parameter_schema=skill['parameter_schema'],
                    parameter_sources=skill['parameter_sources'],
                    skill_name=skill['id'], when_to_use=skill['when_to_use'],
                    skill_contract={k: skill[k] for k in ('preconditions', 'required_capabilities',
                        'parameter_schema', 'parameter_sources', 'known_failures', 'completion_condition')},
                    package_files=skill.get('package_files'),
                    package_name=skill['package']['name'] if skill.get('package') else None,
                    uses=skill['package']['uses'] if skill.get('package') else (),
                ) if skill else {}
                # Parameter rebinding is another run of the same program, not a source revision.
                previous = next((v for v in reversed(self.workspace.versions)
                                 if skill and v.get('origin_skill') == contract['origin_skill']
                                 and v['sha256'] == skill['sha256']
                                 and v.get('package') == skill.get('package')
                                 and v.get('parameter_schema') == skill['parameter_schema']), None)
                version = previous['version'] if previous else self.workspace.write(
                    code,
                    "; ".join(plan.hypotheses) or plan.reason,
                    self.goal.completion_condition,
                    parent_version=self.workspace.versions[-1]["version"]
                    if self.workspace.versions and not skill
                    else None,
                    **contract,
                )["version"]
            attempt_id = f"{self.identity}-{self.attempts}"
            self.attempts += 1
            rt.memory.code_assignment["attempts"] = self.attempts
            metadata = self.workspace.read(version)
            bundle = (execution_bundle(metadata['package'], metadata['package_files'], rt.skills, rt.memory.task.robot,
                                       capabilities=available_capabilities(rt.harness.obs), verifiers=context['available_verifiers'])
                      if metadata.get('package') else None)
            report = await RecoveryRunner(rt.harness, rt.memory, rt.config.runtime, rt.log).run(
                code,
                self.goal.completion_condition,
                attempt_id,
                parameters=plan.parameters,
                skill=skill,
                remaining_steps=budget["steps"],
                remaining_seconds=budget["seconds"],
                program_version=version,
                package_bundle=bundle,
            )
            # Only the fixed subtask verifier can end a code assignment. A return value is a claim.
            verified = report["verified"]
            if verified:
                report["declared_result"] = report["result"]
                report["result"] = "SUBTASK_DONE"
            self.workspace.record_attempt(version, report, attempt_id, parameters=plan.parameters)
            rt.log.event("intervention_end", intervention_id=attempt_id, verified=verified, report=report)
            metadata = self.workspace.read(version)
            candidate_contract = skill
            if candidate_contract is None and metadata.get('origin_skill'):
                origin = metadata['origin_skill']
                candidate_contract = rt.skills.get(origin['id'], origin['version'], rt.memory.task.robot)
                candidate_contract.update(metadata.get('skill_contract') or {})
                candidate_contract.update(parameter_schema=metadata['parameter_schema'],
                                          parameter_sources=metadata['parameter_sources'])
            if candidate_contract is None and metadata.get('parameter_schema') is not None:
                candidate_contract = dict(parameter_schema=metadata['parameter_schema'],
                                          parameter_sources=metadata['parameter_sources'], preconditions=[],
                                          required_capabilities=['native_action12'], known_failures=[])
            candidate = rt.skills.record_candidate(
                code,
                rt.memory.task.robot,
                self.goal.completion_condition,
                dict(episode_id=rt.memory.episode_id, report=report, parameters=plan.parameters,
                     evidence_root=str(rt.log.root.resolve()), program_version=version,
                     origin_skill=metadata.get('origin_skill')),
                str(rt.config.environment.seed),
                parent_skill=candidate_contract,
                description=rt.memory.task.instruction + " | " + plan.reason,
                skill_name=metadata.get('skill_name'), when_to_use=metadata.get('when_to_use', ''),
                package=metadata.get('package'), package_files=metadata.get('package_files'),
            )
            rt.log.event("skill_candidate_recorded", skill_id=candidate["id"],
                         version=candidate["version"], verified=verified,
                         status=candidate['status'], persisted=rt.skills.mode == "accumulate")
            if verified:
                rt.log.event(
                    "code_subtask_complete",
                    subtask=self.goal,
                    subtask_version=self.subtask_version,
                    session_id=self.identity,
                    evidence=report["evidence"],
                )
                rt.memory.code_assignment = None
                rt.memory.executor = "unassigned"
                rt.code_session = None
                completed = self.goal.model_dump()
                advanced = rt.memory.advance(report['evidence'].model_dump() if hasattr(
                    report['evidence'], 'model_dump') else report['evidence'])
                rt.log.event('subtask_transition', completed=completed, verification=report['evidence'],
                             next_subtask=rt.memory.current if advanced else None, next_executor='unassigned')
                if advanced:
                    rt.forced_reason = "Code subtask verified complete; Astra must choose the next executor"
                elif not rt.harness.obs.terminal:
                    await rt.harness.stop("all_subtasks_verified")
                break
            if rt.harness.halted or rt.harness.uncertain or rt.harness.obs.terminal:
                break
            budget = self.budget(rt)
            if not all(v > 0 for v in budget.values()):
                break
            rt.lease = await rt.harness.claim(Source.ASTRA)
            rt.memory.add(rt.harness.obs)
            # Rebuild all authoritative sensor/control fields, not only history and binding.
            context.update(planner_context(rt.config, rt.harness, rt.memory, 'revise_code'))
            context.update(
                recovery_feedback=report,
                workspace=self.workspace.summary(),
                recovery_goal=self.goal.model_dump(),
                recovery_budget=budget,
                recovery_program_limits=dict(steps=min(rt.config.runtime.recovery_steps, budget["steps"]),
                                             seconds=min(rt.config.runtime.recovery_seconds, budget["seconds"]),
                                             tool_calls=rt.config.runtime.recovery_tool_calls,
                                             observation_bytes=rt.config.runtime.recovery_observation_bytes),
                allowed_interventions=["code", "observe", "replan", "stop"],
            )
            toolkit.allowed = context["allowed_interventions"]
            try:
                import asyncio

                async with asyncio.timeout(min(rt.config.planner.timeout_seconds, budget["seconds"])):
                    revision = await rt.planner.plan(decision, context, toolkit)
            except TimeoutError:
                rt.log.event(
                    "recovery_session_budget", intervention_id=self.identity, reason="revision_deadline"
                )
                break
            if not rt.memory.matches(context["binding"], rt.harness.epoch):
                raise ValueError("Stale recovery revision")
            if revision.kind == "stop":
                await rt.harness.stop(revision.reason)
                break
            if revision.kind != "code":
                rt.log.event(
                    "recovery_strategy_change", plan=revision, note="Requires outer Astra reassessment"
                )
                break
            plan = revision
        if not verified:
            rt.forced_reason = "Code subtask unfinished; keep VLA paused. Revise, explicitly replan, or stop."
        return plan, report, verified
