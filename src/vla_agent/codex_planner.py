"""Astra through the user's logged-in Codex; host tools retain execution authority.

App-server integration follows the existing gpt_as_policy project. The protocol
is checked against Codex 0.164.0. No credentials are copied or extracted.
"""

import asyncio
import copy
import hashlib
import json

from .budgets import PlannerBudget
from .codex_transport import CodexTransport
from .contracts import ExecutionReview, Intervention
from .data import dumps
from .decision_context import rounded
from .perception import visual_input
from .planner import SYSTEM
from .planner_service import PlannerAdmission, PlannerServiceError
from .toolkit import ImageResult, planner_tools

# Keep the host's existing API, but avoid collisions with built-in Codex tools.
CODEX_TOOL_NAMES = {"view_image": "view_observation_image"}
HOST_TOOL_NAMES = {wire: host for host, wire in CODEX_TOOL_NAMES.items()}
QUICK_REVIEW_TOOLS = {'observe', 'view_image', 'read_execution', 'read_context',
                      'assess_phase', 'record_note', 'evaluate_subtask', 'submit_review'}
QUICK_REVIEW_INSTRUCTIONS = """You are Astra, reviewing a running pi0.5 robot from a frozen historical snapshot.
Use only supplied dynamic tools; no shell, filesystem, network, external apps or subagents.
The VLA keeps executing during this review. Decide whether it can continue or needs a fresh paused
diagnosis. This is routing, not recovery design or proof of the physical root cause. Only the host
can pause, execute actions, change language or verify task success.
First inspect supplied before/after RGB, measured motion, attempt history and previous reviews.
Prefer submit_review directly if they already establish progress or a concrete persistent problem.
Extra tools are only for evidence that would change that decision. At most the supplied tool budget
is available, including submit_review; reserve the final call for it. Defer skill retrieval, geometry
measurement and program work to the paused planner. Use view_observation_image for historical RGB.
Continue for normal preparation, approach, alignment, justified holding or uncertainty. An unopened
drawer, elapsed time or a low Flora Outcome score alone does not establish failure. A phase-level
high score concerns that phase only, not whole-task completion. The first VLA attempt should not be
interrupted solely to retry the same instruction or because the task is unfinished.
First-attempt protection is not indefinite permission to repeat ineffective interaction. When
separated observations show the same task-relevant mismatch despite attempts to change it, request
intervention with assessment=stalled_progress, at least two evidence steps, the expected progress,
and the persistent visible mismatch. Tiny movement is not necessarily useful progress. A known
contact cause, previous successful grasp, dramatic slip or large failed command is not required.
Also request fresh assessment for a concrete failed attempt, wrong target, invalid plan, completed
subtask or unsafe state. Never invent unseen contact or use future observations. Geometry/contact
causes may remain hypotheses; the paused planner will recheck current state before choosing a skill.
Use assess_phase only for evidence-supported phase belief updates; it cannot verify task success.
Finish with submit_review, then end immediately. Requesting intervention executes nothing.
"""


def codex_tool_text(text):
    for host, wire in CODEX_TOOL_NAMES.items():
        text = text.replace(host, wire)
    return text


INSTRUCTIONS = SYSTEM + """
You are hosted inside a robot supervisor, using Codex for reasoning and conversation state.
Use only the supplied dynamic tools. Physical actions are executed by the host after a proposal
is accepted. Native environment access is disabled. Do not use shell, network, external apps,
subagents, or filesystem tools. Program development uses the registered workspace tools.
Use view_observation_image for robot RGB and crops; built-in filesystem image tools cannot access observations.
Each user turn supplies authoritative current task state. Older observations are historical.
If intent is initialize_plan, inspect the scene and submit replan with the remaining task.
Only use listed completion verifiers. If only task_success exists, retain one executable whole-task
subtask and put fine execution phases in set_phase_plan; do not invent verified local predicates.
For a running review use submit_review; submit_intervention is rejected by the host in that context.
After successful submit_review or submit_intervention, end the turn immediately; execution has not yet happened.
Wait for the next host turn for physical results. Never infer success from proposal acceptance.
Use measure_targets to batch fresh image-region depth measurements; keep feedback calculations inside code.
Prefer an applicable existing skill with current measured parameters over rewriting it. When related
source changes are necessary, use edit_program for the affected files in one transaction, including
legacy policy.py. Read interface first, exact contract before invoking; inspect outline or a selected
path/symbol only when editing. Follow next_offset for truncated source. Write/edit/import receipts
include syntax and dependency checks; do not immediately repeat check_program or read unchanged code.
Keep task-specific procedures separate from reusable perception/control functions. Compose declared pinned dependencies via Python
imports instead of copying them into the controller. Use the supplied ctx for every primitive.
Do not change physical constants or relax geometric checks merely to reduce latency. Move scene-specific
inputs into ctx.parameters when justified so subsequent episodes can reuse the same source and contract.
When available, current controller.gripper_kinematics.fingerpad_geometry supplies measured robot pad
boxes; use it for robot geometry instead of repeatedly estimating occluded fingers. It is not object
geometry or contact evidence; current RGB-D and independent object-motion/task checks remain necessary.
Use fit_geometry for numerical plane/circle diagnostics on measured points before repeatedly submitting
zero-action programs just to check a fit. Supplied points are not verified correspondences; a short arc
can have uncertain center/radius despite a small residual. Keep existing physical checks and use new
observation evidence to resolve inconsistent geometry, rather than simply relaxing fit thresholds.
Budget information accompanies tool results. Reserve calls to edit and submit the final proposal;
do not repeatedly read unchanged source. A budget continuation preserves the workspace and executes nothing.
"""


class ToolBudgetExhausted(PlannerBudget):
    budget_key = 'astra_tool_budget'


def compact_numeric_memory(memory):
    """Reduce repeated numeric rows only; preserve visual inputs, task state and control geometry."""
    result = copy.deepcopy(memory)
    history = result.get("recent_actions", [])
    if not history:
        return result
    count = min(6, len(history))
    indices = [round(i * (len(history) - 1) / (count - 1)) for i in range(count)] if count > 1 else list(range(count))
    result["recent_actions"] = [rounded(history[i]) for i in indices]
    result["history_representation"] = dict(
        original_rows=len(history), sampled_rows=count, numeric_decimals=3,
        first_step=history[0]["step"] if history else None,
        last_step=history[-1]["step"] if history else None,
        note="Uniform samples, not contiguous actions. Full-precision history remains available via read_execution; "
             "motion_windows cover longer execution. Use current eef_control_pose for control, not rounded history.",
    )
    return result


class CodexPlanner:
    call_unit = "codex_turns"  # App-server owns internal inference requests and retries.

    def __init__(self, config, recorder, allowed=None, transport_factory=CodexTransport):
        self.config, self.log = config, recorder
        self.allowed = allowed
        self.transport_factory = transport_factory
        self.transport = None
        self.thread_id = None
        self.episode_id = None
        self.calls = 0
        self.tokens = {"input_tokens": 0, "output_tokens": 0}
        self.thread_tokens = {}
        self.sent_experiences = set()
        self.sent_task_memories = set()
        self.thread_turns = 0
        self.tool_profile = None
        self.admission = PlannerAdmission(config, recorder)

    def experience_context(self, memory):
        """Send unchanged historical evidence once per thread; keep exact retrievable references."""
        result = copy.deepcopy(memory)
        records = []
        for record in result.get('retrieved_experiences', []):
            digest = hashlib.sha256(dumps(record).encode()).hexdigest()
            if digest not in self.sent_experiences:
                records.append(record)
                self.sent_experiences.add(digest)
            else:
                records.append({k: record[k] for k in ('episode_id', 'task', 'result', 'code_versions',
                                                       'evidence_sha256', 'evidence_root') if k in record} | dict(
                    context_reference=digest, historical=True,
                    detail='Unchanged historical record sent earlier in this thread. '
                           'Use read_experience(episode_id) for full facts, hypotheses and conditions. '
                           'References do not establish present applicability or task completion.'))
        result['retrieved_experiences'] = records
        task_memory = result.get('task_memory')
        if task_memory and task_memory.get('version'):
            # Include availability annotations in identity: references can differ across snapshots.
            digest = hashlib.sha256(dumps(task_memory).encode()).hexdigest()
            if digest in self.sent_task_memories:
                result['task_memory'] = dict(version=task_memory['version'], status=task_memory['status'],
                    scope=task_memory['profile']['scope'], context_reference=digest,
                    detail='Unchanged task guide sent earlier. Use read_task_memory to expand. Historical guidance only.')
            self.sent_task_memories.add(digest)
        return result

    def record_tokens(self, thread_id, total):
        previous = self.thread_tokens.get(thread_id, {})
        self.thread_tokens[thread_id] = {
            name: max(previous.get(name, 0), total.get(key, 0))
            for name, key in (("input_tokens", "inputTokens"), ("output_tokens", "outputTokens"))
        }
        self.tokens = {name: sum(value[name] for value in self.thread_tokens.values())
                       for name in ("input_tokens", "output_tokens")}
        self.log.event("codex_token_usage", thread_id=thread_id,
                       thread_total=self.thread_tokens[thread_id], episode_total=self.tokens)
        self.check_token_budget()

    def check_token_budget(self):
        if self.config.max_total_tokens is not None and sum(self.tokens.values()) >= self.config.max_total_tokens:
            raise PlannerBudget('astra_token_budget')

    def instructions(self, profile='full'):
        if profile == 'review':
            return QUICK_REVIEW_INSTRUCTIONS
        extra = ""
        if self.config.preserve_initial_instruction:
            extra = """
For initialize_plan with only task_success available, keep exactly one executable subtask.
Its instruction must exactly equal memory.task.instruction. Put the detailed execution phases
in set_phase_plan and evidence in assess_phase/record_note. The native task language matches this VLA's training contract; do not expand
it into motion instructions merely because you are planning. Later refine only when evidence
shows a need, using a concise goal-focused instruction and preserving the original task goal.
"""
        return codex_tool_text(INSTRUCTIONS + extra)

    async def _start(self, context, profile='full'):
        root = self.log.root / "codex"
        root.mkdir(exist_ok=True)
        schemas = planner_tools(self.allowed, review_only=profile == 'review', include_review=True)
        if profile == 'review':
            schemas = [x for x in schemas if x['name'] in QUICK_REVIEW_TOOLS]
        tools = [dict(type="function", name=CODEX_TOOL_NAMES.get(x["name"], x["name"]),
                      description=codex_tool_text(x["description"]),
                      inputSchema=x["parameters"]) for x in schemas]
        (root / "tools.json").write_text(dumps(tools))
        (root / f"tools-{profile}.json").write_text(dumps(tools))
        (root / "instructions.md").write_text(self.instructions(profile))
        (root / f"instructions-{profile}.md").write_text(self.instructions(profile))
        self.transport = self.transport_factory()
        argv = [self.config.codex_command, "app-server", "--stdio",
                "-c", "features.shell_tool=false", "-c", "features.apps=false",
                "-c", "web_search=\"disabled\"",
                "-c", f"sqlite_home={json.dumps(str(root / 'runtime_db'))}",
                "-c", f"log_dir={json.dumps(str(root / 'runtime_logs'))}"]
        await self.transport.start(argv, root)
        await self.transport.request("initialize", dict(
            clientInfo=dict(name="vla-agent", version="0.1"),
            capabilities=dict(experimentalApi=True),
        ))
        await self.transport.notify("initialized", {})
        result = await self.transport.request("thread/start", dict(
            model=self.config.model, cwd=str(root), ephemeral=True,
            config={"model_reasoning_effort": self.config.reasoning_effort},
            developerInstructions=self.instructions(profile), dynamicTools=tools,
            environments=[], approvalPolicy="never", sandbox="read-only",
            allowProviderModelFallback=False,
        ))
        if result.get("model") != self.config.model or result.get("reasoningEffort") != self.config.reasoning_effort:
            raise RuntimeError("Codex did not honor the configured model/reasoning effort")
        self.thread_id = result["thread"]["id"]
        self.thread_turns = 0
        self.tool_profile = profile
        self.episode_id = context["binding"].episode_id
        self.log.event("codex_thread", thread_id=self.thread_id, model=result["model"],
                       reasoning_effort=result["reasoningEffort"], environment_access=False,
                       tool_profile=profile, registered_tools=[x['name'] for x in tools])

    async def plan(self, decision, context, toolkit):
        try:
            async with asyncio.timeout(self.config.timeout_seconds):
                attempts = 1 if toolkit.read_only else self.config.max_turns_per_intervention
                for attempt in range(attempts):
                    try:
                        return await self._with_service_retries(decision, context, toolkit)
                    except ToolBudgetExhausted:
                        await self.close()
                        if attempt + 1 >= attempts:
                            raise
                        # No proposal has been released. Keep paused control and the exact workspace.
                        context = dict(context, memory=toolkit.memory.summary(),
                                       workspace=toolkit.workspace.summary() if toolkit.workspace else None,
                                       planner_feedback=dict(reason='tool_budget_exhausted',
                                           continuation=attempt + 1, executed=False,
                                           instruction='Continue the preserved workspace. Reuse measured evidence and '
                                                       'latest source; finish a bounded proposal instead of restarting exploration.'))
                        self.log.event('codex_budget_continuation', binding=context['binding'],
                                       continuation=attempt + 1, workspace=context['workspace'])
        except BaseException:
            # Killing our app-server cancels any outstanding model turn. No proposal is executed.
            await self.close()
            raise

    async def _with_service_retries(self, decision, context, toolkit):
        retries = 0 if toolkit.read_only else self.config.service_retry_limit
        for retry in range(retries + 1):
            try:
                async with self.admission.lease(read_only=toolkit.read_only):
                    try:
                        value = await self._plan(decision, context, toolkit)
                        self.admission.success()
                        return value
                    except BaseException:
                        # Cancel the owned app-server before freeing its slot.
                        await self.close()
                        raise
            except PlannerServiceError as error:
                self.admission.failure(error)
                if not error.retryable or retry >= retries:
                    raise
                if self.calls >= self.config.max_calls:
                    raise PlannerBudget('astra_turn_budget') from error
                self.check_token_budget()
                context = dict(context, memory=toolkit.memory.summary(),
                    workspace=toolkit.workspace.summary() if toolkit.workspace else None,
                    planner_feedback=dict(reason=error.code, retry=retry + 1, executed=False,
                        instruction='Previous provider request failed. Read current workspace/memory versions; '
                                    'do not replay edits blindly. No physical proposal was released.'))
                self.log.event('planner_service_retry', code=error.code, retry=retry + 1,
                               calls=self.calls, episode_tokens=self.tokens, executed=False)
                await asyncio.sleep(self.config.service_backoff_seconds)

    async def _plan(self, decision, context, toolkit):
        if self.calls >= self.config.max_calls:
            raise PlannerBudget('astra_turn_budget')
        self.check_token_budget()
        profile = 'review' if toolkit.read_only and self.config.review_tool_limit is not None else 'full'
        if self.transport is not None and self.tool_profile != profile:
            self.log.event('codex_thread_rotated', thread_id=self.thread_id, turns=self.thread_turns,
                           reason='Tool authority changed', previous_profile=self.tool_profile, next_profile=profile)
            await self.close()
        if self.config.thread_max_turns and self.thread_turns >= self.config.thread_max_turns:
            self.log.event('codex_thread_rotated', thread_id=self.thread_id, turns=self.thread_turns,
                           reason='Bound accumulated conversation; rebuild from authoritative state and evidence indexes')
            await self.close()
        if self.transport is None:
            await self._start(context, profile)
        if context["binding"].episode_id != self.episode_id:
            raise ValueError("Cannot reuse a Codex thread across episodes")
        observations = context["observations"]
        data = {k: v for k, v in context.items() if k != "observations"}
        tool_limit = min(self.config.max_tool_calls, self.config.review_tool_limit) if profile == 'review' else self.config.max_tool_calls
        data['tool_budget'] = dict(max_calls=tool_limit, reserve_for_proposal=1, profile=profile)
        data['planner_budget'] = dict(remaining_turns=self.config.max_calls-self.calls,
            remaining_reported_tokens=(max(0, self.config.max_total_tokens-sum(self.tokens.values()))
                                       if self.config.max_total_tokens is not None else None),
            note='Every Codex turn and tool continuation consumes the same episode budget. '
                 'Reuse existing interfaces and keep changes focused. Limits do not reset after replanning.')
        if self.config.compact_numeric_history:
            data["memory"] = compact_numeric_memory(data.get("memory", {}))
        if self.config.deduplicate_experiences:
            data['memory'] = self.experience_context(data.get('memory', {}))
        if self.config.historical_memory_mode == 'summary':
            from .memory_context import compact_historical_memory

            data['memory'] = compact_historical_memory(
                data.get('memory', {}), self.config.historical_memory_chars)
        if self.config.compact_context:
            observations = observations if len(observations) < 3 else [observations[0], observations[-1]]
            memory = copy.deepcopy(data.get("memory", {}))
            history = memory.get("recent_actions", [])
            omitted = max(0, len(history) - 6)
            memory["recent_actions"] = history[-6:]
            if len(history) > 1:
                a, b = history[0], history[-1]
                memory["motion_summary"] = dict(
                    from_step=a["step"], to_step=b["step"], frame="policy_state_relative",
                    measured_eef_delta=[round(y - x, 5) for x, y in zip(a["measured_state"][:3], b["measured_state"][:3])],
                    note="Measured state difference, not a planned trajectory or proof of task progress",
                )
            memory["notes"] = memory.get("notes", [])[-6:]
            memory["interventions"] = memory.get("interventions", [])[-4:]
            data["memory"] = memory
            data["context_budget"] = dict(omitted_actions=omitted,
                                           observation_frames=len(observations), full_evidence_via_tools=True)
        images, frame_map = visual_input(observations)
        data.update(decision=decision, frames=frame_map)
        if self.config.working_memory_mode == 'summary':
            from .working_context import compact_working_context

            data = compact_working_context(data, self.log, self.config.working_memory_chars)
        inputs = [dict(type="text", text=dumps(data))]
        inputs += [dict(type="image", url=url) for url in images]
        self.calls += 1
        self.thread_turns += 1
        started = await self.transport.request("turn/start", dict(
            threadId=self.thread_id, model=self.config.model, effort=self.config.reasoning_effort,
            environments=[], input=inputs,
        ))
        turn_id = started["turn"]["id"]
        self.log.event("codex_turn_start", thread_id=self.thread_id, turn_id=turn_id,
                       call=self.calls, binding=context["binding"], context=data)
        proposal = None
        count = 0
        seen_calls = set()
        while True:
            event = await self.transport.next_event()
            method, params = event.get("method"), event.get("params", {})
            if method == "item/tool/call":
                if params.get("threadId") != self.thread_id or params.get("turnId") != turn_id:
                    raise RuntimeError("Stale Codex tool request")
                identity = params["callId"]
                if identity in seen_calls:
                    raise RuntimeError("Repeated Codex tool request; never replay side effects")
                seen_calls.add(identity)
                count += 1
                if count > tool_limit:
                    raise ToolBudgetExhausted("Codex tool budget exhausted")
                name, raw_args = params["tool"], params["arguments"]
                self.log.event("codex_tool_start", turn_id=turn_id, call_id=identity, name=name,
                               binding=context["binding"])
                args = None
                success = True
                try:
                    if proposal is not None:
                        raise ValueError("Proposal already submitted; end the turn and await execution feedback")
                    if profile == 'review' and HOST_TOOL_NAMES.get(name, name) not in QUICK_REVIEW_TOOLS:
                        raise ValueError('Tool unavailable in running review; defer recovery design to a fresh paused planner')
                    parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    dumps(parsed)
                    args = parsed
                    if name == "submit_review":
                        candidate = ExecutionReview.model_validate(args)
                        toolkit.validate_review(candidate)
                        proposal = candidate
                        result = dict(accepted=True, executed=False, review_only=True, next_step="End this turn")
                    elif name == "submit_intervention":
                        candidate = Intervention.model_validate(args)
                        await toolkit.validate_intervention(candidate, context["allowed_interventions"])
                        if (self.config.preserve_initial_instruction
                            and context.get("intent") == "initialize_plan"
                            and context.get("available_verifiers") == ["task_success"]
                            and candidate.kind == "replan"):
                            tasks = candidate.remaining_plan or []
                            if len(tasks) != 1 or tasks[0].instruction != context["memory"]["task"]["instruction"]:
                                raise ValueError("Keep the exact native task instruction in the one whole-task subtask; put finer planning in record_note")
                        proposal = candidate
                        result = dict(accepted=True, executed=False, effective_kind=candidate.kind,
                                      next_step="End this turn")
                    else:
                        result = await toolkit.call(HOST_TOOL_NAMES.get(name, name), args)
                except (ValueError, TypeError, KeyError) as error:
                    result = dict(error=str(error), no_execution=True)
                    success = False
                attachment = result if isinstance(result, ImageResult) else None
                if attachment:
                    result = attachment.metadata
                self.log.event("codex_tool", turn_id=turn_id, call_id=identity,
                               name=name, arguments=args, result=result)
                content = [dict(type="inputText", text=dumps(result))]
                remaining = tool_limit - count
                if remaining <= 8:
                    content.append(dict(type='inputText', text=dumps(dict(tool_budget_remaining=remaining,
                        instruction=('Reserve the final call for submit_review. Decide from supplied temporal evidence; '
                                     'detailed diagnosis belongs to the paused planner.' if profile == 'review' else
                                     'Reserve a call for submit_review/submit_intervention. '
                                     'Reuse existing measurements/source; submit only after required checks.')))))
                if attachment:
                    content.append(dict(type="inputImage", imageUrl=attachment.url))
                await self.transport.reply(event["id"], dict(success=success, contentItems=content))
            elif method == "turn/completed" and params.get("threadId") == self.thread_id:
                turn = params["turn"]
                if turn["id"] != turn_id:
                    raise RuntimeError("Stale Codex turn completion")
                self.log.event("codex_turn_end", turn_id=turn_id, status=turn["status"], error=turn.get("error"))
                if turn['status'] != 'completed' and isinstance(turn.get('error'), dict):
                    error = turn['error']
                    if error.get('codexErrorInfo'):
                        raise PlannerServiceError(error['codexErrorInfo'], error.get('message'))
                if turn["status"] != "completed" or proposal is None:
                    raise RuntimeError("Codex ended without a completed, validated intervention")
                return proposal
            elif method == "thread/tokenUsage/updated" and params.get("threadId") == self.thread_id:
                total = params.get("tokenUsage", {}).get("total", {})
                try:
                    self.record_tokens(self.thread_id, total)
                except PlannerBudget:
                    if not self.config.release_validated_proposal_on_budget or proposal is None:
                        raise
                    # submit_* already passed the host's binding, contract and capability checks.
                    # Cancel further sampling; do not discard an immutable, accepted proposal
                    # merely because the final token notification precedes turn/completed.
                    self.log.event('codex_budget_proposal_released', turn_id=turn_id,
                        proposal=proposal, episode_tokens=self.tokens, executed=False,
                        reason='Validated host proposal retained; further model work cancelled')
                    self.log.event('codex_turn_end', turn_id=turn_id, status='host_finalized_on_budget',
                        error=None, server_reported_completion=False)
                    await self.close()
                    return proposal
            elif method == "item/completed" and params.get("threadId") == self.thread_id:
                item = params.get("item", {})
                if item.get("type") in {"agentMessage", "contextCompaction"}:
                    self.log.event("codex_item", turn_id=params.get("turnId"), item=item)
                    if item.get('type') == 'contextCompaction':
                        self.sent_experiences.clear()
                        self.sent_task_memories.clear()
            elif method == "error":
                self.log.event("codex_error", error=params)
                if not params.get("willRetry"):
                    detail = params.get('error') or {}
                    raise PlannerServiceError(detail.get('codexErrorInfo'), detail.get('message'))
            elif "id" in event and "method" in event:
                raise RuntimeError(f"Unexpected Codex request: {method}")

    async def close(self):
        self.sent_experiences.clear()
        self.sent_task_memories.clear()
        if self.transport:
            await self.transport.close()
            self.transport = None
