"""Bounded Astra tool loop using Responses API. Model calls never own the robot."""

import asyncio
import json
import os
import time

import httpx

from .contracts import ExecutionReview, Intervention
from .data import dumps
from .perception import visual_input
from .toolkit import ImageResult, planner_tools
from .toolkit import PlannerToolkit as PlannerToolkit

SYSTEM = """You are Astra, the on-demand planner for a pi0.5 robot agent.
Use only the supplied observations and measured state. Commands and model predictions are not facts.
The Decision category is advisory: any category may need any intervention or no action.
SWITCH is normal progression. You cannot declare success; the Harness independently checks conditions.
pi0.5 is a task-conditioned action policy, not a general instruction-following assistant. Its language
must match the registered language_catalog. For refine select instruction_id (or the exact registered
text) to restore/correct an atomic task; never write a motion tutorial, retry narrative, conditional
instructions, or a sequence of gripper/pose commands. Replan only combines registered task commands.
Keep detailed reasoning, failure history and recovery steps in Memory/programs, out of the VLA prompt.
Reissuing the already-active instruction is not a language correction: pi0.5 receives the same text.
Do not request an interruption solely to restart that instruction or label it a fresh attempt.
Identical refine proposals become continue without resetting attempt history. For confirmed ineffective
interaction choose a bounded code recovery, an actually changed registered goal, or continue if uncertain.
Inspect memory.attempt, attempts and reviews before deciding: on the first attempt, give the VLA time
to perform its normal approach/alignment. An unfinished subtask, a periodic check, elapsed steps, or
Decision's category alone is NOT evidence of failure. Do not interrupt merely because a handle is
not yet grasped. Assess the current phase, expected progress, observed problem and prior attempts.
For intent=review_running the VLA is still executing. Use read-only snapshot tools and finish with
submit_review: continue for normal progress or uncertainty; request_intervention only for a concrete
observed failure, wrong target, completed subtask, invalid plan or unsafe state, with evidence steps.
This request does not execute anything. For intent=confirm_intervention the host has paused and sends
fresh state: reassess the earlier review, return continue if it no longer applies, otherwise choose
an allowed intervention. A snapshot is historical; never assume the scene stayed unchanged.
Keep running reviews short: if supplied images, measured motion and prior review already establish
progress or a persistent mismatch, submit_review without repeating geometry and library inspection.
Detailed recovery design belongs to the fresh paused snapshot. Consult extra tools only to resolve
uncertainty that changes the review decision; budget and snapshot age continue to advance meanwhile.
First-attempt protection is not indefinite permission to repeat ineffective interaction. If separated
observations show a persistent task-relevant mismatch despite continued attempts to change it, use
assessment=stalled_progress with at least two evidence times. Name the expected object change and the
repeated ineffective interaction. A dramatic slip, reopening or previously established grasp is not
required to recognize ineffective contact. Tiny robot motion alone is not task progress; normal holding,
settling, occlusion or elapsed time alone is not stalled progress. If the evidence is still ambiguous,
use geometry/Flora/history to distinguish the hypotheses. A stalled-progress review may request a fresh
paused-state assessment, then a bounded Code Policy experiment; it does not prove VLA lacks the skill.
An inactive policy can also stall: it need not issue a large failed pull before intervention is justified.
Compare with previous review evidence. If the phase requires a physical change (e.g. opening a closed
door), repeated separated observations retain the same target and gripper relationship, measured pose
barely changes, and commands remain near-zero or merely maintain closure, report stalled_progress when
no task-relevant settling/holding prerequisite explains it. Name expected progress and the observations;
do not require proving the contact failure's cause first. A held gripper is not evidence of a stable grasp,
and an occluded contact does not establish that continued waiting is productive. After a prior uncertain
review, unchanged evidence of nonprogress can support a bounded recovery experiment even if its physical
cause remains a hypothesis. No fixed elapsed-step cutoff alone authorizes a pause.
Match the expected object change to the observed execution phase: a door need not rotate while open
fingers are still approaching or aligning. Repeated images of an unchanged object do not by themselves
establish failed engagement. Distinguish attempted object manipulation from preparation using actual
gripper/wrist changes, commanded versus measured motion, and contact evidence. If contact has not
been established, diagnose a specific alignment obstruction or wrong target before claiming failed
object manipulation; uncertainty normally continues VLA. Historical code recoveries are not evidence
that the current first attempt needs recovery. Compare retrieved normal VLA successes as well as failures.
Create an execution phase plan with set_phase_plan when initializing a task, even for a single atomic
instruction. Use observed prerequisites/progress/completion/failure signals; assess_phase records causal
phase evidence. Fine phases are beliefs, not new VLA instructions or installed physical verifiers.
When language_catalog contains registered local goals with installed verifiers, initialize a REAL
remaining_plan from those commands using replan. Do not leave a composite task as one whole-task
instruction when suitable atomic commands exist. Astra selects their ordering from dependencies,
current evidence and the native task; the host enforces registered entry/completion contracts.
The last goal must retain task_success. VLA receives only the selected concise task instruction;
the overall goal and rationale stay in Memory. set_phase_plan is optional diagnostic detail within
one executable subtask and never replaces installing the remaining_plan.
For trigger=subtask_verifier, the host observed a positive local completion check at a chunk boundary.
Confirm it and submit switch (or replan if prerequisites changed); this is normal progress, not failure.
Simulator-state boolean checks, when enabled, are verifier assistance, not your visual perception.
Read memory.task_memory first when available: it is an evidence-linked task guide, not current state
or execution authority. Adopt its phase guide only after checking the scene; do not copy historical
coordinates or assume a recommended skill applies. Missing evidence/skills are marked unavailable.
read_task_memory expands the exact episode snapshot. propose_task_memory saves evidence-backed rule
candidates for later consolidation; it cannot change this episode's guidance or controller.
Historical memory may be a bounded index. Full records remain behind read_experience and
read_task_memory. Omitted details are not negative evidence. Index outcomes do not confirm a
cause or current applicability. Expand relevant rule conditions and avoidances before adopting
a recovery rule. Do not expand every history at initialization or during normal VLA progress.
When needed for a concrete diagnosis, search_experiences for positive and negative task/phase evidence.
When exploring code recovery, search_skills with include_candidates=true to inspect prior experimental
programs and their recorded outcomes. Candidates may have failed; reuse only applicable logic and
remeasure scene parameters. A saved program is not a verified skill merely because it is retrievable.
If an imported whole-task program contains reusable local logic but a different completion contract,
use adapt_program_goal for the CURRENT installed local goal, then patch only the relevant source and
documentation (especially termination checks). This creates a candidate version; it does not establish
success, waive original preconditions, change the task or reset the recovery budget.
When memory already supplies relevant retrieved_experiences, read those first; repeat a search only
for a different failure/phase or missing evidence. During revisions, reuse the already-read guide,
skill and geometry conventions unless new observations show they no longer apply.
For an applicable skill, check_skill then submit code with skill_id, skill_version and parameters;
do not regenerate its source. Search returns one named procedure per slot; list_skill_versions exposes
its alternative entry states, contracts and failed revisions. Select a compatible existing version first.
New seed, observation_step, camera selection, image boxes and target poses are measured PARAMETERS,
not reasons to create or patch code when that input is already supported. Retry the same skill_id/version
or program_version with fresh parameters. Such a retry adds run evidence without creating a new skill.
To adapt a prior skill, read_skill once, import_skill_program, and
edit_program only the incompatible logic or add a missing parameter ONCE. Prefer measured parameters and
compact reusable functions over hardcoded episode steps, historical frame numbers or a full rewrite.
New procedures require a semantic skill_name (operation and start state), when_to_use, and reuse_reason
explaining why existing versions cannot solve the observed problem. Same procedure revisions keep its name.
Archived scene-specific experiments remain readable by their old IDs; include_archived=true is for
diagnosis/development, not the default skill shortlist. A successful late-stage finish is not a grasp skill
or evidence that unexecuted branches work. Preserve current-entry checks and whole-subtask verification.
Revisions should address a named observed failure and preserve working stages. Put normal feedback
loops and per-segment checks in the program so they run without another Astra turn.
For new reusable code prefer write_skill_package: SKILL.md, code/run.py main(ctx, **params), and
code/<skill_name_with_underscores>/ modules including params.py. Return {ok, measured, why}; ok is a
program claim, never whole-subtask verification. Split at stable observed states. Put perception,
control and tunable parameters in separate modules; each motion function checks its own entry and exit.
Declare dependencies in uses with exact skill_id/version and import their Python packages instead of
copying functions. Shared imports receive the SAME ctx; they do not get another action lease or budget.
Search results include package_variants; read their docs/evidence before using a migrated variant.
read_skill/read_program default to a short interface index. Read view=contract for exact parameter bounds
and preconditions, view=documentation for usage, and view=outline to locate functions. Source path/symbol
reads return only the selected text; follow next_offset until the required text is complete.
import_skill_program preserves all files and pinned dependencies. Use edit_program for related changes
in one transaction. File overwrite/delete needs expected_sha256; unique-span replacements share one
immutable base. Write/edit/import tools automatically return syntax and dependency checks plus a small
change receipt. Do not repeat check_program or read unchanged code after a successful receipt.
Parameter-only retries never require a patch. check_program remains available for explicit revalidation.
Legacy write_program/patch_program handlers and run(ctx) remain for old records; do not choose that format for
new multi-step procedures. Package inputs are explicit main parameters; legacy code uses ctx.parameters.
Both formats accept parameter_schema and parameter_sources. Successful and failed
candidates retain that schema for later direct skill_id calls. Remeasure current geometry inside
the program; parameters are scene selections/calibration, never permission to assume contact success.
For actual recovery prefer a coherent Code Policy when the existing atomic task cannot recover.
Once assigned to code, the ENTIRE current subtask belongs to code across program revisions. Internal
alignment/grasp phases never hand back to VLA. completion_condition must equal the current SUBTASK
completion condition. RECOVERED is only a claim and cannot release VLA for an unfinished subtask.
For a composite plan this means the installed LOCAL executable goal (e.g. open the microwave), not
every remaining task stage. After its verifier passes, stop the code at that boundary; the host will
advance and ask you to choose the next executor. Prefer VLA for the next registered task's first attempt.
After budget exhaustion choose an actually different registered plan or stop; renaming the same goal
cannot reset budgets. For intent=select_next_executor the previous subtask was verified and the next
subtask is already selected: choose continue for VLA or code for code ownership; do not switch again.
Code revisions always start from the current measured state, sharing the same remaining session budget.
Use the current execution_budget, not historical attempt limits. With useful progress and remaining
budget, continue the appropriate executor or refine the existing program. Stop for specific unsafe or
unrecoverable conditions or exhausted budgets; merely being unfinished or reaching an old 12-attempt
limit is not a stopping reason. Report concrete evidence when further execution cannot help.
Use propose_policy to save bounded Decision history/routing/audit or Flora rationale candidates with
recorded evidence. Candidates do not activate online. Harness, budgets, logs, success criteria,
permissions, model weights, evaluation, promotion and rollback are immutable to the agent.
Direct local/eef/edit interventions are
disabled by default; use them only when explicitly present in allowed_interventions for an experiment.
For code development, use write_skill_package/read_program/edit_program in the persistent
recovery workspace, then submit kind=code with program_version and completion_condition. A version is
immutable. Record the failure hypothesis and expected effect of each revision. Static checks do not
execute code or establish task success. read_recovery_trace provides prior attempt stages and errors.
Keep the recovery goal fixed across revisions. Reobserve and check stage prerequisites from current
state; never assume that restarting code resets the robot. Normal program feedback loops need no LLM.
For replan provide all remaining registered subtasks with their installed completion checks.
Use read_guide(control/recovery/evidence) for detailed operating contracts when needed.
Use capabilities to inspect control limits and installed verifiers. Read execution records and images
when the history summary is insufficient. Use record_note to keep facts, hypotheses and next checks separate.
Working notes/rewards/reports may also be extractive indexes with read_context references. Read only
the relevant record/field, not the entire archive. Keep notes short and record new evidence or changed
hypotheses, not repeated numeric arrays or source code. Current controller/state and the latest program
contract remain explicit; older schemas and dense program measurements can be expanded on demand.
Conversation threads may rotate at a turn boundary; the supplied state and immutable workspace remain
authoritative. Read the current program once when needed, then patch it rather than rebuilding it.
When capabilities.flora.enabled, evaluate_subtask can provide a local Flora Outcome score for the current
subtask attempt's historical video, using only frames up to this turn's snapshot. Specify phase_id
for the active phase and with_rationale=true for a diagnosis hypothesis; it is not a proven cause. Call it selectively when
completion is uncertain, not on every review. A score is a model assessment, not a calibrated probability
or an installed verifier. Low score may mean normal unfinished execution; unavailable/budget/timeout is
unknown, not failure. A high score requires fresh-state confirmation and the existing completion verifier
before switching. Never change a task goal just to obtain a better score. Inspect memory.reward_assessments
and reuse exact-snapshot results. Read-only scoring must not pause the VLA.
If explicitly enabled, for eef supply target in the measured eef_control_pose OSC site frame, with 1..5 feedback steps.
NEVER substitute the policy state quaternion for eef_control_pose. Base motions use normalized units.
For edit call preview_vla, optionally preview_edit, then submit the exact proposal_id and 1..5 steps.
Edit offsets are total meters/rotation vectors distributed across the prefix; clipping is reported.
For local provide at most 5 canonical native actions: [eef_delta(3),rot_delta(3),gripper(1),base(4),mode(1)].
Motion channels are normalized [-1,1]; gripper 0=open/1=close; mode 0=arm/1=base. No absolute pose here.
For code provide Python def run(ctx), a frozen program_version, or a skill id/version. Code uses ctx.observe(),
ctx.act(action12), ctx.check(condition), ctx.stage(name),
ctx.ground_target(target, camera, observation_step, bbox, confidence), ctx.query_geometry(grounding_id),
Use ground_target and query_geometry tools to bind visual selection to current RGB-D. This is your
region selection, not an independent object detector. Geometry reports visible surfaces only; reselect
and remeasure after motion. Missing/unreliable depth must not be replaced with guessed coordinates.
geometry/query_geometry require at least FOUR valid depth pixels: a 3-by-1 or single-pixel region
cannot pass. Use suitable surface patches or calibrated raw RGB-D with the shared point-cloud helpers.
After motion, project measured 3D targets into current calibrated views rather than retaining old pixel
windows. A target leaving one view calls for another observation, not relaxed correspondence checks.
ctx.note(facts=[...], hypotheses=[...]), ctx.capabilities(), ctx.move_eef(target, steps=1),
ctx.move_relative(delta_position, gripper_closed, delta_rotation_vector=[0,0,0], steps=1),
ctx.set_gripper(closed, steps=1), ctx.move_base(motion, gripper_closed, steps=1), ctx.stop(reason).
ctx.check(condition) returns a dictionary with condition, passed, evidence and observation_step.
Use result["passed"] is True to test success. The dictionary itself is not a boolean; passed=None is unknown.
ctx.read_observation(camera, step=None) loads actual rgb/state and optional depth_m arrays plus metadata
with calibration. Missing/unverified calibration is not a geometric estimate. Use numpy for computation.
Use with ctx.stage(name, preconditions=[...], completion_condition=...) as stage for checked stages.
stage.record(**measured) records program-reported values, not independent proof. A failed host check
stops the stage. Legacy ctx.stage(name) without a with block is an unchecked stage marker.
Versioned skills receive ctx.parameters and ctx.completion_condition. Read/check a skill before using it;
include parameters in the code intervention. Templates are unverified starting points, not proven skills.
Every physical helper is bounded to 1..5 steps, also counted against the total recovery budget. It returns RECOVERED, SUBTASK_DONE or UNRESOLVED.
Code must compute targets from current observations, run in short stages and verify execution.
The process has a bounded time and action budget. No direct simulator, network, shell or file access is needed.
completion_condition must equal the complete assigned subtask goal and use an installed verifier.
Separate observed facts from hypotheses. Inspect a retrieved skill and validate its applicability.
End read-only reviews with submit_review and paused planning with submit_intervention.
Do not assume that repeated failure proves the VLA lacks a skill.
"""


TOOLS = planner_tools()


class AstraPlanner:
    def __init__(self, config, recorder, transport=None):
        self.config, self.log = config, recorder
        self.client = httpx.AsyncClient(timeout=config.timeout_seconds, transport=transport)
        self.calls = 0
        self.tokens = {"input_tokens": 0, "output_tokens": 0}

    async def plan(self, decision, context, toolkit):
        key = os.environ.get(self.config.api_key_env)
        if not key:
            raise RuntimeError(f"Missing credential environment variable {self.config.api_key_env}")
        images, frame_map = visual_input(context["observations"])
        data = {k: v for k, v in context.items() if k != "observations"}
        if self.config.historical_memory_mode == 'summary':
            from .memory_context import compact_historical_memory

            data['memory'] = compact_historical_memory(
                data.get('memory', {}), self.config.historical_memory_chars)
        data.update(decision=decision, frames=frame_map)
        if self.config.working_memory_mode == 'summary':
            from .working_context import compact_working_context

            data = compact_working_context(data, self.log, self.config.working_memory_chars)
        content = [{"type": "input_text", "text": dumps(data)}]
        content += [{"type": "input_image", "image_url": url} for url in images]
        conversation = [{"role": "user", "content": content}]
        async with asyncio.timeout(self.config.timeout_seconds):
            for _ in range(self.config.max_turns_per_intervention):
                if self.calls >= self.config.max_calls:
                    raise RuntimeError("Astra episode call budget exhausted")
                self.calls += 1
                started = time.monotonic()
                payload = dict(
                    model=self.config.model,
                    instructions=SYSTEM,
                    input=conversation,
                    tools=planner_tools(context.get("allowed_interventions"),
                                        review_only=context.get("intent") == "review_running"),
                    tool_choice="required",
                    parallel_tool_calls=False,
                    max_output_tokens=self.config.max_output_tokens,
                    reasoning={"effort": self.config.reasoning_effort},
                    store=False,
                    include=["reasoning.encrypted_content"],
                )
                self.log.event(
                    "astra_request",
                    call=self.calls,
                    model=self.config.model,
                    binding=context["binding"],
                    context=data,
                )
                response = await self.client.post(
                    self.config.endpoint, json=payload, headers={"Authorization": f"Bearer {key}"}
                )
                response.raise_for_status()
                result = response.json()
                self.log.event(
                    "astra_response", call=self.calls, seconds=time.monotonic() - started, response=result
                )
                for name in self.tokens:
                    self.tokens[name] += result.get("usage", {}).get(name, 0)
                if result.get("status", "completed") != "completed":
                    raise RuntimeError("Incomplete Astra response")
                output = result.get("output", [])
                conversation.extend(output)  # Preserve reasoning items for the next tool turn.
                calls = [x for x in output if x.get("type") == "function_call"]
                if len(calls) != 1:
                    raise ValueError("Expected one sequential tool call")
                call = calls[0]
                args = None
                try:
                    parsed = json.loads(call["arguments"])
                    dumps(parsed)  # Reject NaN/Inf before storing tool arguments or invoking tools.
                    args = parsed
                    if call["name"] == "submit_review":
                        review = ExecutionReview.model_validate(args)
                        toolkit.validate_review(review)
                        return review
                    if call["name"] == "submit_intervention":
                        plan = Intervention.model_validate(args)
                        if hasattr(toolkit, "validate_intervention"):
                            await toolkit.validate_intervention(plan, context["allowed_interventions"])
                        return plan
                    result = await toolkit.call(call["name"], args)
                except (ValueError, TypeError, KeyError) as error:
                    result = {"error": str(error), "no_execution": True}
                attachment = result if isinstance(result, ImageResult) else None
                if attachment:
                    result = attachment.metadata
                self.log.event("astra_tool", name=call["name"], arguments=args, result=result)
                conversation.append(
                    dict(type="function_call_output", call_id=call["call_id"], output=dumps(result))
                )
                if attachment:
                    conversation.append(
                        dict(
                            role="user",
                            content=[
                                dict(
                                    type="input_text",
                                    text="Recorded image requested by the preceding tool: " + dumps(result),
                                ),
                                dict(type="input_image", image_url=attachment.url),
                            ],
                        )
                    )
        raise RuntimeError("Astra tool turn budget exhausted")

    async def close(self):
        await self.client.aclose()
