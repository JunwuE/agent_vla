"""Validated read/analysis tools for Astra. Execution remains a runtime/Harness decision."""

import base64
import copy
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Literal

import numpy as np
from PIL import Image, ImageDraw
from pydantic import Field, StrictBool, StrictFloat, StrictInt

from .code_tools import CodeRead, EditProgram, check_program, code_view, skill_card, write_receipt
from .contracts import ExecutionReview, Intervention, Model, validate_actions
from .control import ActionEdit, EEFRequest, EEFTarget, OSCControl, available_capabilities, robot_capabilities
from .data import read_events
from .geometry_fitting import fit_geometry
from .grounding import BoundingBox, GroundingQuery, GroundTarget, ground_target, grounded_geometry
from .perception import image_url, local_geometry
from .phases import PhaseAssessment, PhasePlan
from .policy_candidates import PolicyProposal, save_candidate
from .skill_packages import SkillDependency, build_package, execution_bundle
from .task_memory import TaskMemoryProposal, save_proposal


class Empty(Model):
    pass


class EvaluateSubtask(Model):
    phase_id: str | None = None
    with_rationale: StrictBool | None = None


class ExperienceQuery(Model):
    query: str = Field(min_length=1, max_length=500)
    phase: str | None = None


class ExperienceRef(Model):
    episode_id: str = Field(min_length=1, max_length=128)


class ContextRef(Model):
    reference: str = Field(pattern=r'^[0-9a-f]{64}$')
    path: list[str | StrictInt] = Field(default_factory=list, max_length=12)
    start: StrictInt = Field(default=0, ge=0)
    limit: StrictInt = Field(default=5, ge=1, le=20)


class Guide(Model):
    name: Literal["control", "recovery", "evidence"]


class Search(Model):
    query: str
    include_templates: StrictBool = False
    include_candidates: StrictBool = False
    include_archived: StrictBool = False


class SkillVersions(Model):
    skill_id: str
    offset: StrictInt = Field(default=0, ge=0)
    limit: StrictInt = Field(default=5, ge=1, le=10)


class SkillRef(Model):
    skill_id: str
    version: str


class ReadSkill(SkillRef, CodeRead):
    pass


class SkillCheck(SkillRef):
    parameters: dict = Field(default_factory=dict)
    completion_condition: str


class ImportSkillProgram(SkillRef):
    hypothesis: str = Field(min_length=1, max_length=2000)
    expected_effect: str = Field(min_length=1, max_length=2000)


class Geometry(Model):
    camera: str
    bbox: BoundingBox


class MeasureTargets(Model):
    targets: list[GroundTarget] = Field(min_length=1, max_length=8)


class GeometryFit(Model):
    kind: Literal['plane', 'circle']
    points: list[Annotated[list[StrictFloat], Field(min_length=3, max_length=3)]] = Field(min_length=4, max_length=128)


class ImageRequest(Model):
    camera: str
    step: StrictInt | None = Field(default=None, ge=0)
    bbox: BoundingBox | None = None


class History(Model):
    start_step: StrictInt = Field(ge=0)
    end_step: StrictInt = Field(ge=0)


class Preview(Model):
    steps: StrictInt = Field(default=5, ge=1, le=5)


class PreviewEdit(Preview):
    proposal_id: str
    edit: ActionEdit


class Note(Model):
    facts: list[str] = Field(default_factory=list, max_length=12)
    hypotheses: list[str] = Field(default_factory=list, max_length=12)
    next_check: str | None = None


class WriteProgram(Model):
    source: str = Field(min_length=1, max_length=65536)
    hypothesis: str = Field(min_length=1, max_length=2000)
    expected_effect: str = Field(min_length=1, max_length=2000)
    parent_version: str | None = None
    parameter_schema: dict | None = None
    parameter_sources: list[str] = Field(default_factory=list, max_length=32)
    skill_name: str = Field(pattern=r'^[a-z0-9][a-z0-9_-]{0,79}$')
    when_to_use: str = Field(min_length=1, max_length=2000)
    reuse_reason: str = Field(min_length=1, max_length=2000)


class ProgramRef(CodeRead):
    version: str | None = None


class ProgramVersion(Model):
    version: str | None = None


class AdaptProgramGoal(Model):
    version: str
    completion_condition: str
    reason: str = Field(min_length=1, max_length=2000)


class WriteSkillPackage(Model):
    files: dict[str, str]
    uses: list[SkillDependency] = Field(default_factory=list, max_length=16)
    skill_name: str = Field(pattern=r'^[a-z][a-z0-9-]{0,79}$')
    when_to_use: str = Field(min_length=1, max_length=2000)
    reuse_reason: str = Field(min_length=1, max_length=2000)
    hypothesis: str = Field(min_length=1, max_length=2000)
    expected_effect: str = Field(min_length=1, max_length=2000)
    parent_version: str | None = None
    parameter_schema: dict
    parameter_sources: list[str] = Field(default_factory=list, max_length=32)


class PackageReplacement(Model):
    path: str
    old_text: str = Field(min_length=1, max_length=65536)
    new_text: str = Field(max_length=65536)


class PatchSkillPackage(Model):
    base_version: str
    replacements: list[PackageReplacement] = Field(default_factory=list, max_length=16)
    add_files: dict[str, str] = Field(default_factory=dict)
    uses: list[SkillDependency] | None = None
    hypothesis: str = Field(min_length=1, max_length=2000)
    expected_effect: str = Field(min_length=1, max_length=2000)
    parameter_schema: dict | None = None
    parameter_sources: list[str] | None = None


class PatchProgram(Model):
    base_version: str
    old_text: str = Field(min_length=1, max_length=65536)
    new_text: str = Field(max_length=65536)
    hypothesis: str = Field(min_length=1, max_length=2000)
    expected_effect: str = Field(min_length=1, max_length=2000)
    parameter_schema: dict | None = None
    parameter_sources: list[str] | None = Field(default=None, max_length=32)


class Replacement(Model):
    old_text: str = Field(min_length=1, max_length=65536)
    new_text: str = Field(max_length=65536)


class PatchProgramBatch(Model):
    base_version: str
    replacements: list[Replacement] = Field(min_length=1, max_length=16)
    hypothesis: str = Field(min_length=1, max_length=2000)
    expected_effect: str = Field(min_length=1, max_length=2000)
    parameter_schema: dict | None = None
    parameter_sources: list[str] | None = Field(default=None, max_length=32)


class AttemptRef(Model):
    attempt: StrictInt | None = Field(default=None, ge=1)


REGISTRY = {
    'edit_program': (EditProgram, 'Preferred code editor for packages and legacy policy.py. Apply up to 16 '
        'file edits against one immutable base_version in one transaction. replace uses a unique old_text span; '
        'write replaces a whole file with expected_sha256 (or adds a new file without it); delete needs the hash. '
        'Preserve dependency declarations and contracts. Automatically checks syntax and dependency closure; '
        'returns a small change receipt, never executes. Parameter rebinding needs no edit.'),
    'read_task_memory': (Empty, 'Read this episode\'s fixed task-specific phase guide, conditional skill recommendations and evidence. Guidance only; no execution or automatic phase change.'),
    'propose_task_memory': (TaskMemoryProposal, 'Save one evidence-backed task rule candidate against parent_version. '
        'Reference historical episode hashes or current visible event IDs. Does not modify active memory, code or control.'),
    'write_skill_package': (WriteSkillPackage, 'Create a RAX-style multi-file package with SKILL.md, code/run.py '
        'main(ctx, **params), code/<skill_name_as_underscores>/ modules and params.py. Declare pinned uses '
        '(skill_id/version); import dependencies instead of copying their code. Return {ok, measured, why}. '
        'Prefer existing skills; to migrate an imported legacy procedure provide its parent_version. No execution.'),
    'patch_skill_package': (PatchSkillPackage, 'Atomically patch only changed modules of a frozen skill package. '
        'Paths and spans refer to the same base; add_files is for new modules. Parameters alone require no patch. '
        'Update uses and SKILL.md together when changing dependencies; existing versions stay immutable.'),
    "propose_policy": (PolicyProposal, "Save a bounded scheduling/reward policy candidate with evidence. No runtime activation or promotion."),
    'read_context': (ContextRef, 'Expand immutable current-episode notes, reports or older version metadata by supplied reference/path. Lists are paginated; does not change state.'),
    'adapt_program_goal': (AdaptProgramGoal, 'Version an imported skill contract for the current registered subtask goal. Source is unchanged: inspect and patch its termination checks/docs before execution. No verification or budget reset.'),
    'set_phase_plan': (PhasePlan, 'Plan diagnostic phases within the current executable subtask. To install executable subtasks use replan with registered language_catalog commands. No control handoff.'),
    'assess_phase': (PhaseAssessment, 'Record phase evidence and optionally advance the phase belief; not task verification.'),
    'ground_target': (GroundTarget, 'Register your selected target bbox on a specific current image; returns an overlay. No automatic object detector.'),
    'query_geometry': (GroundingQuery, 'Measure aligned metric geometry for a grounding from this exact observation. Reselect after motion.'),
    'measure_targets': (MeasureTargets, 'Select and measure up to 8 image regions in one read-only call, all on the current snapshot. '
                        'Prefer this over repeated ground_target/query_geometry pairs. bbox is [x0,y0,x1,y1] with integer numbers. '
                        'Returns metric geometry and grounding provenance, no overlay or robot motion; use view_image when a crop is needed.'),
    'fit_geometry': (GeometryFit, 'Numerically fit a plane or 3D circle to 4..128 supplied metric points before committing code. '
                     'Returns residuals and degeneracy diagnostics; no robot action, no verified correspondence or success. '
                     'Use points measured from observations in the same frame; for a circle track one physical marker. '
                     'Prefer this to repeated zero-action code attempts merely to test a geometric fit.'),
    'search_experiences': (ExperienceQuery, 'Retrieve cross-episode task/phase/tool experience with evidence and outcomes, including failures.'),
    'read_experience': (ExperienceRef, 'Read the complete historical episode experience from this episode snapshot by exact ID. '
                       'Use to expand a repeated context reference; historical evidence is not current state or execution authority.'),
    "write_program": (
        WriteProgram,
        "Create a NEW named procedure only after searching skills. State why existing procedures cannot be "
        "used or parameterized in reuse_reason. Use an operation/start-state name, never a seed or hash. "
        "For an existing procedure import/patch, or directly invoke with new parameters. "
        "Declare ctx.parameters and their measurement sources; when_to_use describes checked entry conditions.",
    ),
    "import_skill_program": (
        ImportSkillProgram,
        "Copy an exact skill from this episode's library snapshot into the recovery workspace, without "
        "regenerating its source. No execution or promotion. Read the skill, then patch only the incompatible "
        "parts; preserve its evidence and remeasure scene parameters. For directly applicable code use skill_id instead.",
    ),
    "read_program": (
        ProgramRef,
        "Inspect workspace code. Default interface view returns a short index; contract gives exact bounds. "
        "path selects only that file; symbol selects a declaration. Source/docs/outline/evidence/diff are paged "
        "by offset/limit in characters; follow next_offset to finish. No unrelated code or docs are attached.",
    ),
    "patch_program": (
        PatchProgram,
        "Replace exactly one source span in a frozen base version, producing a new version. "
        "Optionally declare/update parameter_schema and parameter_sources while parameterizing that span.",
    ),
    "patch_program_batch": (
        PatchProgramBatch,
        "Apply 1..16 independent non-overlapping replacements against the same frozen base in one call. "
        "All spans must match uniquely in the original source; validation is atomic, with one new version. "
        "Prefer this for related adaptations instead of sequential patch calls. Does not execute code.",
    ),
    "check_program": (
        ProgramVersion,
        "Check legacy run(ctx), or package main(ctx, **params), syntax and pinned dependencies. No code execution or physics.",
    ),
    "read_recovery_trace": (
        AttemptRef,
        "Read one prior attempt's stages, checks, failure, traceback and actual executed steps.",
    ),
    "read_guide": (Guide, "Read the control, recovery or evidence operating guide on demand."),
    "observe": (Empty, "Read current measured state and controller pose; no physics step."),
    "capabilities": (Empty, "Read available robot tools, verifiers and control limits."),
    "evaluate_subtask": (
        EvaluateSubtask,
        "Ask local Flora Outcome to score the current subtask attempt's recorded video up to this snapshot. "
        "Optional phase_id selects the active phase; with_rationale requests an explanation. Assessment only.",
    ),
    "view_image": (
        ImageRequest,
        "View current/past RGB or a pixel crop. bbox uses original image coordinates.",
    ),
    "read_execution": (
        History,
        "Read up to 20 executed steps with commands, measured states and observation links.",
    ),
    "record_note": (Note, "Save facts separately from hypotheses and a proposed verification."),
    "search_skills": (Search, "Search compatible verified skills; opt into unverified templates/candidates explicitly. Candidates may have failed; inspect evidence before reuse."),
    'list_skill_versions': (SkillVersions, 'Page a short version index for one named procedure. '
        'Use exact read_skill references to inspect contracts/evidence; no source is returned.'),
    "read_skill": (
        ReadSkill,
        "Read a skill progressively: default interface gives entry/module/dependencies; contract returns exact "
        "parameter/precondition bounds. documentation/evidence/outline are paged. path reads only that source "
        "file; symbol selects a function/class. Follow next_offset for remaining characters. No execution.",
    ),
    "check_skill": (
        SkillCheck,
        "Check skill parameters, capabilities and observed preconditions without executing.",
    ),
    "geometry": (
        Geometry,
        "Estimate visible surface geometry from aligned metric depth; can return unreliable.",
    ),
    "preview_vla": (
        Preview,
        "Request a fresh short VLA command preview while paused; does not execute or predict object motion.",
    ),
    "preview_edit": (
        PreviewEdit,
        "Preview bounded gradual offsets to a fresh proposal, including clipping diagnostics.",
    ),
}


REVIEW_TOOLS = {"read_guide", "observe", "capabilities", "view_image", "read_execution", 'read_context',
                'read_task_memory', 'propose_task_memory',
                "record_note", "search_skills", "read_skill", "list_skill_versions", "geometry", "evaluate_subtask",
                'set_phase_plan', 'assess_phase', 'ground_target', 'query_geometry', 'measure_targets', 'fit_geometry', 'search_experiences', 'read_experience', 'propose_policy'}


# Compatibility handlers remain callable for existing clients and transcripts.
# New agents see one editor instead of four overlapping patch/write alternatives.
LEGACY_CODE_TOOLS = {'write_program', 'patch_program', 'patch_program_batch', 'patch_skill_package'}
CODE_TOOL_GROUPS = {
    'discover': {'search_skills', 'list_skill_versions'},
    'inspect': {'read_skill', 'read_program', 'read_recovery_trace', 'check_skill', 'check_program'},
    'develop': {'import_skill_program', 'write_skill_package', 'edit_program', 'adapt_program_goal'},
}


def planner_tools(allowed=None, *, review_only=False, include_review=False, include_legacy=False):
    if allowed is None:
        from .config import RuntimeConfig

        allowed = RuntimeConfig().allowed_interventions
    tools = [
        dict(
            type="function",
            name=name,
            description=next((f'[{group}] ' for group, names in CODE_TOOL_GROUPS.items() if name in names), '') + description,
            strict=False,
            parameters=model.model_json_schema(),
        )
        for name, (model, description) in REGISTRY.items()
        if (name not in {"preview_vla", "preview_edit"} or "edit" in allowed)
        and (not review_only or name in REVIEW_TOOLS)
        and (include_legacy or name not in LEGACY_CODE_TOOLS)
    ]
    review_tool = dict(type="function", name="submit_review", strict=False,
                       description="Finish a read-only review. Requesting intervention is not execution; the host rechecks fresh state.",
                       parameters=ExecutionReview.model_json_schema())
    if review_only:
        return tools + [review_tool]
    intervention_schema = Intervention.model_json_schema()
    intervention_schema["properties"]["kind"]["enum"] = list(allowed)
    for name, model, field in (("EEFTarget", EEFTarget, "target"), ("ActionEdit", ActionEdit, "edit")):
        intervention_schema.setdefault("$defs", {})[name] = model.model_json_schema()
        intervention_schema["properties"][field] = {"anyOf": [{"$ref": "#/$defs/" + name}, {"type": "null"}]}
    for kind, fields in {"local": ["actions"], "eef": ["target"], "edit": ["edit", "proposal_id"]}.items():
        if kind not in allowed:
            for field in fields:
                intervention_schema["properties"].pop(field, None)
    if "eef" not in allowed:
        intervention_schema["$defs"].pop("EEFTarget", None)
    if "edit" not in allowed:
        intervention_schema["$defs"].pop("ActionEdit", None)
    if not {"eef", "edit"} & set(allowed):
        intervention_schema["properties"].pop("steps", None)
    result = tools + [
        dict(
            type="function",
            name="submit_intervention",
            strict=False,
            description="Submit while paused. Refine selects a registered language command; never invent detailed motion instructions. Code requires an installed verifier.",
            parameters=intervention_schema,
        )
    ]
    return result + [review_tool] if include_review else result


@dataclass
class ImageResult:
    metadata: dict
    url: str


def package_view(value, path=None, **options):
    """Compatibility import; bounded code presentation lives in code_tools."""
    return code_view(value, path, **options)


class PlannerToolkit:
    def __init__(self, harness, memory, skills, vla=None, workspace=None, allowed=None, *, read_only=False,
                 flora=None, experiences=None):
        self.harness, self.memory, self.skills, self.vla = harness, memory, skills, vla
        self.proposal = None
        self.workspace = workspace
        self.allowed = allowed
        self.read_only = read_only
        self.flora = flora
        self.experiences = experiences
        self.task_evidence_limit = harness.log.sequence if read_only else None

    @classmethod
    def for_review(cls, harness, memory, skills, allowed, *, flora=None, experiences=None):
        """Freeze evidence and memory. No live control object or VLA client reaches the reviewer."""
        obs = copy.deepcopy(harness.obs)

        async def observe():
            return obs

        frozen = SimpleNamespace(obs=obs, log=harness.log, epoch=harness.epoch, owner=harness.owner,
                                 env=SimpleNamespace(observe=observe, verifier_conditions=list(
                                     getattr(harness.env, "verifier_conditions", ["task_success"]))))
        return cls(frozen, copy.deepcopy(memory), skills, allowed=allowed, read_only=True,
                   flora=flora, experiences=experiences)

    def validate_review(self, review):
        if not self.read_only:
            raise ValueError("submit_review is only available for a read-only execution review")
        if any(e.observation_step > self.harness.obs.step for e in review.evidence):
            raise ValueError("Review evidence cannot refer to future observations")

    def capabilities(self):
        return dict(
            **robot_capabilities(self.harness),
            control_owner=self.harness.owner,
            epoch=self.harness.epoch,
            vla_preview=self.vla is not None and (self.allowed is None or "edit" in self.allowed),
            recovery_workspace=self.workspace is not None,
            flora=self.flora.capabilities() if self.flora else dict(enabled=False),
            phase_memory=self.memory.phases.summary(),
            executor=self.memory.executor, code_assignment=self.memory.code_assignment,
            grounding='Astra/program region selection; geometry requires aligned depth',
        )

    def take_proposal(self, proposal_id, steps):
        chunk = self.proposal
        if (
            chunk is None
            or chunk.binding.request_id != proposal_id
            or not self.memory.matches(chunk.binding, self.harness.epoch)
            or chunk.binding.observation_step != self.harness.obs.step
        ):
            raise ValueError("Missing or stale VLA edit proposal; request preview_vla again")
        if not 1 <= steps <= min(5, len(chunk.actions)):
            raise ValueError("Edit exceeds the reviewed proposal prefix")
        return chunk.actions[:steps].copy()

    async def validate_intervention(self, plan, allowed):
        if self.read_only:
            raise ValueError("Read-only review: use submit_review; no intervention can be submitted here")
        if plan.kind not in allowed:
            raise ValueError("Intervention disabled by the configured ablation")
        if self.memory.executor == 'code':
            if plan.kind not in {'code', 'observe', 'stop', 'replan'}:
                raise ValueError('Code owns the unfinished subtask; VLA cannot resume')
            if plan.kind == 'replan' and plan.remaining_plan and (plan.remaining_plan[0].instruction, plan.remaining_plan[0].completion_condition) == (
                    self.memory.current.instruction, self.memory.current.completion_condition):
                raise ValueError('Replanning the same unfinished code goal cannot reset its executor or budget')
        if plan.kind == 'code' and plan.completion_condition != self.memory.current.completion_condition:
            raise ValueError('Code must complete the entire current subtask, not a local recovery predicate')
        if plan.kind == "eef":
            request = EEFRequest.model_validate(dict(target=plan.target, steps=plan.steps))
            OSCControl(self.harness.obs).validate_target(request.target)
        elif plan.kind == "edit":
            OSCControl(self.harness.obs).edit(
                self.take_proposal(plan.proposal_id, plan.steps), ActionEdit.model_validate(plan.edit)
            )
        elif plan.kind == "local":
            validate_actions(plan.actions, maximum=5)
        elif plan.kind == "code":
            if sum(bool(value) for value in (plan.skill_id, plan.code, plan.program_version)) != 1:
                raise ValueError("Choose exactly one skill, source, or frozen program version")
            if plan.skill_id:
                if plan.code:
                    raise ValueError("Choose skill or code, not both")
                result = await self.call(
                    "check_skill",
                    dict(
                        skill_id=plan.skill_id,
                        version=plan.skill_version,
                        parameters=plan.parameters,
                        completion_condition=plan.completion_condition,
                    ),
                )
                if not result["applicable"]:
                    raise ValueError("Skill preconditions are not verified")
            elif plan.program_version:
                if self.workspace is None or not self.workspace.check(plan.program_version)["ok"]:
                    raise ValueError("Program is unavailable or failed static checks")
                self.workspace.validate_parameters(plan.program_version, plan.parameters)
                item = self.workspace.read(plan.program_version)
                if item.get('package'):
                    execution_bundle(item['package'], item['package_files'], self.skills, self.memory.task.robot,
                                     capabilities=available_capabilities(self.harness.obs), verifiers=self.capabilities()['verifiers'])
                contract = self.workspace.read(plan.program_version).get('skill_contract')
                if contract:
                    self.skills.validate_invocation(contract, plan.parameters, available_capabilities(self.harness.obs),
                                                    plan.completion_condition, self.capabilities()['verifiers'])
                    for condition in contract['preconditions']:
                        if (await self.harness.verify(condition)).passed is not True:
                            raise ValueError('Imported skill preconditions are not verified')
            elif not plan.code or plan.parameters:
                raise ValueError("New code requires source; parameters require a versioned skill")
        elif plan.kind == "refine":
            command = self.memory.language_command(plan.instruction, plan.instruction_id)
            if command.completion_condition != self.memory.current.completion_condition:
                raise ValueError("Refine cannot change the subtask goal/verifier; use replan")
            plan.instruction = command.instruction
            effective = self.memory.pending_instruction or self.memory.instruction
            if command.instruction == effective:
                if 'continue' not in allowed:
                    raise ValueError('Instruction already active; refine would not change policy conditioning')
                self.harness.log.event('intervention_normalized', proposed_kind='refine',
                    effective_kind='continue', instruction=effective,
                    reason='Identical instruction cannot reset the attempt or language version')
                plan.kind = 'continue'
                plan.instruction = None
                plan.instruction_id = None
        elif plan.kind == "replan" and not plan.remaining_plan:
            raise ValueError("Replan requires remaining subtasks and completion conditions")
        if plan.kind == "replan":
            available = set(self.capabilities()["verifiers"])
            missing = {task.completion_condition for task in plan.remaining_plan} - available
            if missing:
                raise ValueError("Subtask verifiers are not installed: " + ", ".join(sorted(missing)))
            for task in plan.remaining_plan:
                command = self.memory.language_command(task.instruction)
                if task.completion_condition != command.completion_condition:
                    raise ValueError("Replan must preserve the registered command's completion condition")
                # Entry contracts belong to the catalog, not to an untrusted proposal.
                task.preconditions = list(command.preconditions)
            if getattr(self.harness.env, 'subtask_boundary_checks', False):
                if plan.remaining_plan[-1].completion_condition != 'task_success':
                    raise ValueError('A composite plan must retain native task_success as its final goal')
        if plan.kind == 'replan' or (self.memory.executor == 'unassigned'
                and plan.kind in {'continue', 'code'}):
            selected = plan.remaining_plan[0] if plan.kind == 'replan' else self.memory.current
            for condition in selected.preconditions:
                if (await self.harness.verify(condition)).passed is not True:
                    raise ValueError(f'Subtask entry condition {condition} is not satisfied; replan an entry repair')
        if plan.kind in {"eef", "edit", "local", "code"}:
            if plan.completion_condition not in self.capabilities()["verifiers"]:
                raise ValueError("Recovery requires an installed completion verifier")

    def program_receipt(self, item):
        return write_receipt(self.workspace, item, self.skills, self.memory.task.robot,
                             available_capabilities(self.harness.obs), self.capabilities()['verifiers'])

    async def call(self, name, args):
        if self.read_only and name not in REVIEW_TOOLS:
            raise ValueError("Tool unavailable during read-only review; request intervention before program development or VLA previews")
        if name not in REGISTRY:
            raise ValueError(f"Unsupported planner tool: {name}")
        if (
            name in {"preview_vla", "preview_edit"}
            and self.allowed is not None
            and "edit" not in self.allowed
        ):
            raise ValueError("Direct action editing is disabled; use a recovery program")
        request = REGISTRY[name][0].model_validate(args)
        if name == 'read_task_memory':
            return copy.deepcopy(self.memory.task_memory) or dict(status='missing', scope=self.memory.task_scope)
        if name == 'propose_task_memory':
            # Only events visible at this observation enter an asynchronous review's evidence.
            # Recorder IDs are contiguous and append-only: no full-log scan on the review path.
            limit = self.task_evidence_limit if self.task_evidence_limit is not None else self.harness.log.sequence
            return save_proposal(self.harness.log.root, request, self.memory, self.experiences, limit)
        if name == 'fit_geometry':
            return fit_geometry(request.points, request.kind)
        if name == 'adapt_program_goal':
            if self.workspace is None or request.completion_condition != self.memory.current.completion_condition:
                raise ValueError('Goal adaptation must match the current installed subtask')
            if request.completion_condition not in self.capabilities()['verifiers']:
                raise ValueError('Goal adaptation requires an installed verifier')
            item = self.workspace.adapt_goal(request.version, request.completion_condition, request.reason)
            self.harness.log.event('skill_contract_adapted', parent_version=request.version,
                        program_version=item['version'], completion_condition=request.completion_condition,
                        reason=request.reason, verified=False)
            return dict(**self.program_receipt(item), contract_note='New candidate contract only. Patch task-specific termination/checks/docs as needed; '
                                    'unchanged source is not evidence it completes the local goal.')
        if name == "import_skill_program":
            if self.workspace is None:
                raise ValueError("No recovery development workspace in this context")
            skill = self.skills.get(request.skill_id, request.version, self.memory.task.robot)
            origin = {key: skill[key] for key in ('id', 'version', 'sha256', 'status', 'robot')}
            item = self.workspace.write(skill['code'], request.hypothesis, request.expected_effect,
                                        origin_skill=origin, parameter_schema=skill['parameter_schema'],
                                        parameter_sources=skill['parameter_sources'], skill_name=skill['id'],
                                        when_to_use=skill['when_to_use'], skill_contract={k: skill[k] for k in (
                                            'preconditions', 'required_capabilities', 'parameter_schema',
                                            'parameter_sources', 'known_failures', 'completion_condition')},
                                        package_files=skill.get('package_files'),
                                        package_name=skill['package']['name'] if skill.get('package') else None,
                                        uses=skill['package']['uses'] if skill.get('package') else ())
            self.harness.log.event('skill_program_imported', program_version=item['version'], origin_skill=origin,
                                   executed=False)
            return dict(**self.program_receipt(item), source_contract_ref=dict(
                tool='read_program', version=item['version'], view='contract'),
                import_note='Source copied unchanged. No applicability or success is implied. '
                            'Read the contract, remeasure parameters, edit only incompatible logic.')
        if name in {"write_program", "read_program", "patch_program", "patch_program_batch", "check_program", "read_recovery_trace",
                    'write_skill_package', 'patch_skill_package', 'edit_program'}:
            if self.workspace is None:
                raise ValueError("No recovery development workspace in this context")
            if name == 'write_skill_package':
                values = request.model_dump()
                package = build_package(request.skill_name, request.files, values['uses'])
                if package['robot'] != self.memory.task.robot:
                    raise ValueError('SKILL.md robot must match the current embodiment')
                if any(x['id'] == request.skill_name for x in self.skills.index):
                    parent = self.workspace.read(request.parent_version) if request.parent_version else None
                    if not parent or (parent.get('origin_skill') or {}).get('id') != request.skill_name:
                        raise ValueError('Existing skill: import its exact version before package migration or revision')
                reason = values.pop('reuse_reason')
                result = self.workspace.write_package(**values)
                self.harness.log.event('skill_package_proposal', skill_name=request.skill_name,
                                       reuse_reason=reason, program_version=result['version'])
                return self.program_receipt(result)
            if name == 'edit_program':
                return self.program_receipt(self.workspace.edit(**request.model_dump()))
            if name == 'patch_skill_package':
                return self.program_receipt(self.workspace.patch_package(**request.model_dump()))
            if name == "write_program":
                if request.skill_name.startswith('recovery-'):
                    raise ValueError('Use an operation/start-state name, not a recovery hash')
                if any(x['id'] == request.skill_name for x in self.skills.index):
                    raise ValueError('Skill already exists: invoke with parameters or import_skill_program and patch')
                values = request.model_dump()
                reason = values.pop('reuse_reason')
                self.harness.log.event('new_skill_proposal', skill_name=request.skill_name, reuse_reason=reason)
                return self.program_receipt(self.workspace.write(**values))
            if name == "patch_program":
                return self.program_receipt(self.workspace.patch(**request.model_dump()))
            if name == "patch_program_batch":
                return self.program_receipt(self.workspace.patch_many(**request.model_dump()))
            if name == "read_program":
                return code_view(self.workspace.read(request.version),
                    **request.model_dump(exclude={'version'}),
                    diff=self.workspace.diff(request.version) if request.view == 'diff' else None)
            if name == "check_program":
                return check_program(self.workspace, request.version, self.skills, self.memory.task.robot,
                                     available_capabilities(self.harness.obs), self.capabilities()['verifiers'])
            return self.workspace.trace(request.attempt)
        h = self.harness
        if name == 'propose_policy':
            rows = read_events(h.log.root)
            return save_candidate(h.log.root, request, self.memory.episode_id,
                                  {r['id'] for r in rows})
        if name == 'set_phase_plan':
            value = self.memory.phases.set_plan(request, h.obs.step)
            h.log.event('phase_plan', observation_step=h.obs.step, plan=value, reason=request.reason,
                        staged=self.read_only)
            return value
        if name == 'assess_phase':
            steps = {int(k.rsplit(':', 1)[1]) for k in h.log.observations
                     if k.startswith(self.memory.episode_id + ':')}
            value = self.memory.phases.assess(request, h.obs.step, steps)
            h.log.event('phase_assessment', assessment=value, staged=self.read_only)
            return value
        if name == 'search_experiences':
            if self.experiences is None:
                return dict(records=[], status='unavailable')
            from .memory_context import experience_card

            records = self.experiences.search(self.memory.task.robot, request.query, request.phase,
                                              skill_index=self.skills.index, scope=self.memory.task_scope)
            return dict(records=[experience_card(record) for record in records],
                        format='historical_experience_index', expand_tool='read_experience')
        if name == 'read_experience':
            if self.experiences is None:
                return dict(status='unavailable')
            return self.experiences.get(self.memory.task.robot, request.episode_id, scope=self.memory.task_scope)
        if name == 'measure_targets':
            obs = await h.env.observe()
            # Validate every selection before recording anything; all geometry uses one snapshot.
            selections = [ground_target(obs, target, 'astra_selection') for target in request.targets]
            results = [dict(grounding=value, geometry=grounded_geometry(obs, value)) for value in selections]
            for value in selections:
                self.memory.groundings[value['grounding_id']] = value
                h.log.event('grounding', grounding=value, staged=self.read_only)
            while len(self.memory.groundings) > 32:
                self.memory.groundings.pop(next(iter(self.memory.groundings)))
            return dict(observation_step=obs.step, measurements=results, executed=False,
                        note='Visible surfaces only; these measurements do not verify contact or task completion.')
        if name == 'ground_target':
            obs = await h.env.observe()
            value = ground_target(obs, request, 'astra_selection')
            self.memory.groundings[value['grounding_id']] = value
            while len(self.memory.groundings) > 32:
                self.memory.groundings.pop(next(iter(self.memory.groundings)))
            h.log.event('grounding', grounding=value, staged=self.read_only)
            overlay = Image.fromarray(obs.rgb[request.camera]).copy()
            ImageDraw.Draw(overlay).rectangle(request.bbox, outline=(255, 80, 30), width=2)
            return ImageResult(value, image_url(np.asarray(overlay)))
        if name == 'query_geometry':
            if request.grounding_id not in self.memory.groundings:
                raise ValueError('Unknown grounding')
            return grounded_geometry(await h.env.observe(), self.memory.groundings[request.grounding_id])
        if name == "read_guide":
            path = Path(__file__).with_name("guides") / (request.name + ".md")
            content = path.read_bytes()
            return dict(
                name=request.name, sha256=hashlib.sha256(content).hexdigest(), content=content.decode()
            )
        if name == "observe":
            return (await h.env.observe()).summary()
        if name == "capabilities":
            return self.capabilities()
        if name == "evaluate_subtask":
            if self.flora is None:
                return dict(status="unavailable", score=None, reason="Flora is not configured")
            binding = self.memory.binding(h.epoch)
            result = await self.flora.evaluate(self.memory, binding, phase_id=request.phase_id,
                                               with_rationale=request.with_rationale)
            if not self.memory.matches(binding, h.epoch) or h.obs.step != binding.observation_step:
                return dict(status="stale", score=None, reason="Task/control/observation changed during scoring")
            if result["status"] == "scored":
                record = {k: result[k] for k in ("request_id", "score", "task", "attempt_id", "binding", "segment", "scope", "rationale", "explanation_authority")}
                record["source"] = "flora_assessment"
                if not any(r["request_id"] == record["request_id"] for r in self.memory.reward_assessments):
                    self.memory.reward_assessments.append(record)
            return result
        if name == "search_skills":
            return [skill_card(skill) for skill in self.skills.search(
                self.memory.task.robot,
                request.query,
                capabilities=available_capabilities(h.obs),
                include_templates=request.include_templates,
                include_candidates=request.include_candidates,
                include_archived=request.include_archived,
            )]
        if name == 'list_skill_versions':
            versions = self.skills.versions(request.skill_id, self.memory.task.robot)
            end = min(len(versions), request.offset + request.limit)
            if request.offset > len(versions):
                raise ValueError('Offset is outside the skill version index')
            return dict(total=len(versions), offset=request.offset, next_offset=end if end < len(versions) else None,
                versions=[skill_card(self.skills.get(v['id'], v['version'], self.memory.task.robot))
                          for v in versions[request.offset:end]],
                note='Short index only; inspect exact contracts and evidence with read_skill before reuse.')
        if name in {"read_skill", "check_skill"}:
            skill = self.skills.get(request.skill_id, request.version, self.memory.task.robot)
            if name == "read_skill":
                return code_view(skill, **request.model_dump(exclude={'skill_id', 'version'}))
            if skill.get('package'):
                execution_bundle(skill['package'], skill['package_files'], self.skills, self.memory.task.robot,
                                 capabilities=available_capabilities(h.obs), verifiers=self.capabilities()['verifiers'])
            self.skills.validate_invocation(
                skill,
                request.parameters,
                available_capabilities(h.obs),
                request.completion_condition,
                self.capabilities()["verifiers"],
            )
            checks = [await h.verify(condition) for condition in skill["preconditions"]]
            return dict(
                applicable=all(c.passed is True for c in checks),
                scope='declared_contract_only',
                scene_applicability='requires_program_entry_checks' if not checks else 'declared_preconditions_checked',
                checks=checks,
                status=skill["status"],
                note="This validates declared contracts only. Empty preconditions do not prove scene applicability. "
                     "Remeasure parameters and retain program entry checks; candidates remain experimental.",
            )
        if name == "geometry":
            return local_geometry(await h.env.observe(), request.camera, list(request.bbox))
        if name == "record_note":
            event = h.log.event("planner_note", observation_step=h.obs.step, **request.model_dump())
            self.memory.notes.append(
                dict(event_id=event, observation_step=h.obs.step, source="astra", **request.model_dump())
            )
            return dict(event_id=event, observation_step=h.obs.step)
        if name == 'read_context':
            from .working_context import read_context

            return read_context(h.log, request.reference, request.path, request.start, request.limit,
                                self.memory.episode_id, h.obs.step)
        if name == "view_image":
            step = h.obs.step if request.step is None else request.step
            obs = h.log.read_observation(step, self.memory.episode_id, h.obs.step)
            if request.camera not in obs.rgb:
                raise ValueError("Unknown camera")
            rgb = obs.rgb[request.camera]
            height, width = rgb.shape[:2]
            bbox = request.bbox or (0, 0, width, height)
            x0, y0, x1, y1 = bbox
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError("Crop outside original image")
            url = image_url(rgb[y0:y1, x0:x1], max_edge=768)
            encoded = base64.b64decode(url.split(",", 1)[1], validate=True)
            digest = hashlib.sha256(encoded).hexdigest()
            directory = h.log.root / "image_previews"
            directory.mkdir(exist_ok=True)
            path = directory / (digest + ".png")
            if path.exists():
                if path.read_bytes() != encoded:
                    raise ValueError("Recorded image preview was modified")
            else:
                path.write_bytes(encoded)
            with Image.open(io.BytesIO(encoded)) as preview:
                preview_size = list(preview.size)
            return ImageResult(
                dict(
                    step=step,
                    camera=request.camera,
                    original_size=[width, height],
                    bbox=list(bbox),
                    observation=h.log.observations[obs.key],
                    preview_path=str(path.relative_to(h.log.root)),
                    preview_sha256=digest,
                    preview_size=preview_size,
                    note="Pixels from recorded observation; not a predicted scene",
                ),
                url,
            )
        if name == "read_execution":
            if (
                not 0 <= request.start_step < request.end_step <= h.obs.step
                or request.end_step - request.start_step > 20
            ):
                raise ValueError(
                    "Require a past interval (start_step, end_step] of at most 20 executed steps"
                )
            actions = []
            rows = read_events(h.log.root)
            for row in rows:
                if row["kind"] != "action_executed":
                    continue
                step = int(row["after"].rsplit("/", 1)[-1].split(".")[0])
                if request.start_step < step <= request.end_step:
                    actions.append(row)
            ids = {r["intervention_id"] for r in actions if r.get("intervention_id")}
            diagnostics = [
                row
                for row in rows
                if row["kind"] in {"eef_target", "eef_tracking", "action_edit"}
                and row.get("intervention_id") in ids
                and request.start_step <= row.get("before_step", -1) < request.end_step
            ]
            return dict(
                actions=actions,
                diagnostics=diagnostics,
                note="action is commanded; measured_state/eef_control_pose are measured outcomes",
            )
        if name == "preview_vla":
            if self.vla is None:
                raise ValueError("VLA preview unavailable")
            if h.halted or h.owner != "astra_local":
                raise ValueError("VLA preview requires paused Astra control")
            binding = self.memory.binding(h.epoch)
            OSCControl(h.obs)  # Reject unavailable calibration before paying for inference.
            if self.proposal is not None and (
                not self.memory.matches(self.proposal.binding, h.epoch)
                or self.proposal.binding.observation_step != h.obs.step
            ):
                self.proposal = None
            if self.proposal is None:
                chunk = await self.vla.infer(h.obs, self.memory.instruction, binding)
                if (
                    chunk.binding != binding
                    or not self.memory.matches(chunk.binding, h.epoch)
                    or chunk.binding.observation_step != h.obs.step
                ):
                    raise ValueError("VLA preview returned stale actions")
                # Store only the reviewed prefix; it cannot be expanded silently by a later edit.
                chunk.actions = validate_actions(chunk.actions)[: request.steps]
                self.proposal = chunk
                h.log.event("vla_preview", binding=chunk.binding, actions=chunk.actions)
            actions = self.take_proposal(
                self.proposal.binding.request_id, min(request.steps, len(self.proposal.actions))
            )
            return dict(
                proposal_id=self.proposal.binding.request_id,
                step=h.obs.step,
                actions=actions.tolist(),
                preview=OSCControl(h.obs).preview(actions),
                note="Command-space preview only; no future state or object-motion prediction",
            )
        if name == "preview_edit":
            actions = self.take_proposal(request.proposal_id, request.steps)
            edited, diagnostics = OSCControl(h.obs).edit(actions, request.edit)
            return dict(actions=edited.tolist(), diagnostics=diagnostics, executed=False)
        raise AssertionError(name)
