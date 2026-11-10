# Recovery programs and skills

## Default: reusable Python skill packages

Read memory.task_memory or read_task_memory before reconstructing a task strategy. Its rules link
entry conditions, skill versions, counterexamples and unresolved questions. Recheck current state;
phase suggestions and historical success are not authority to interrupt VLA or execute a skill.
Use propose_task_memory to retain a new evidence-backed lesson as a candidate for later consolidation.

Use the RAX workflow: reuse an existing package, bind fresh measured inputs, run its checked functions,
inspect failures, and patch only the incompatible module. Do not copy another skill's functions into a
new controller. The search result's package_variants identifies modular versions of historical scripts.

`write_skill_package` accepts `files`, exact-version `uses`, `skill_name`, `when_to_use`, `reuse_reason`,
`hypothesis`, `expected_effect`, `parameter_schema` and `parameter_sources`. Layout:

```
SKILL.md                          # name, robot, status=draft, entry=code/run.py, integer version, uses
code/run.py                       # def main(ctx, **params): ...
code/<skill_name_with_underscores>/
    __init__.py                   # public reusable functions
    params.py                     # named bounds, units, tested ranges, CALIBRATE notes
    perception.py                 # fresh observations -> measured geometry
    controller.py                 # checked, resumable manipulation steps
```

SKILL.md has When to use, Preconditions, Procedure, Code, Checks, Failure modes and Evidence sections,
plus a Perception checkpoints subsection (write None explicitly if there are none).
Declare `uses=[{"skill_id":"some-procedure","version":"exact-version"}]` in the tool arguments,
and `uses: [some-procedure]` in SKILL.md. Import the dependency's package normally, e.g.
`from some_procedure import measured_step`. The host resolves and freezes the dependency closure.
Do not import top-level run.py from another skill; its public functions belong in its named package.
Every imported motion primitive receives the SAME supplied ctx and checks its current entry/exit state.
Shared helpers do not obtain another budget, lease, episode or simulator connection. Pure helpers may
return arrays internally; main returns JSON `{ok: bool, measured: {...}, why: str}` with optional checkpoint
information. `ok` is a program claim; only the host verifier ends the assigned complete subtask.

The tools are organized into discovery, inspection, development and invocation:

1. `search_skills` returns short named cards; `list_skill_versions(offset, limit)` pages alternatives.
2. `read_skill` defaults to an interface index: entry, module, exports, dependencies and parameter names.
   Read `view="contract"` for exact parameter bounds/preconditions and `view="documentation"` for usage.
   `view="evidence"` reads outcomes separately. Applicable skills can be checked and invoked directly
   using `skill_id/skill_version/parameters`; reading or copying their source is not required.
3. If logic really differs, `import_skill_program` imports the exact package. `read_program` uses the same
   views. Inspect `view="outline"` to locate functions, then `path` and optionally `symbol` for source.
   Source/docs/outline/evidence/diff use character `offset/limit`; `next_offset` means MORE TEXT REMAINS.
   File source carries its complete file hash even when only a fragment is shown. Do not mistake a page
   for a complete function/file. No unrelated docs/contracts are attached to source reads.
4. `edit_program(base_version, edits, hypothesis, expected_effect)` is the primary editor. Batch up to 16
   edits against one base. `replace` needs unique old_text/new_text spans; `write` adds a file or replaces
   one with its expected_sha256; `delete` also requires the hash. Do not mix whole-file and span edits
   on one file. Required entry/docs/params modules cannot be removed. Update `uses` and SKILL.md together.
5. Write/import/edit returns a compact receipt with changed file hashes and automatic syntax/dependency
   checks. `saved=true, checks.ok=false` is a saved DRAFT with diagnostics, not a failed write; repair it.
   `checks.ok=true` does not prove scene applicability. No immediate duplicate check_program/read_program
   is needed for unchanged code; explicit check_program and paged `view="diff"` remain available.
6. Submit the immutable skill/program reference for execution. Changing measured invocation parameters
   needs no edit or new version. Full completion still requires the independent host verifier.

Checks do not import or execute generated code. Missing dependencies, namespace/version conflicts,
cycles, changed files or a changed runtime API fail before execution. Revalidate after runtime changes.

Dependencies come from this episode's fixed library snapshot. Put newly authored helper modules inside
the current package when composing them in the same episode. Executed packages are saved as candidates
in accumulate mode for later episodes; a frozen run does not mutate the cross-episode library.
Dependency checks establish interface availability, not physical applicability: each function must still
check its actual entry state. A SKILL.md status declaration cannot promote the host's candidate status.

For a legacy procedure that needs modularization, import it first, then use write_skill_package with
that parent_version. Preserve checked boundaries and source provenance. A package refactor or recorded
receipt replay does not automatically transfer physical success to a new version. Read its evidence.

## Compatibility and recovery discipline

Direct Astra local/eef/edit proposals are disabled by default. Keep small motion calls inside a
closed-loop recovery program, and choose language/plan changes when those solve the problem.

First search_skills(include_candidates=true); results are named procedures, not one result per trial.
Use list_skill_versions(skill_id) to compare entry states and contracts. With supported current scene
inputs, directly submit skill_id/skill_version/parameters. Remeasuring boxes, observation_step, camera
or target pose does NOT require importing or patching source. A parameter-only retry uses the same
skill or program_version and only adds run evidence. Candidate does not mean physically applicable.

Legacy run(ctx) programs remain executable and can be imported and edited as path="policy.py" through
edit_program. Old write_program/patch_program/patch_program_batch/patch_skill_package handlers remain
for compatibility but are absent from the default model menu. Prefer write_skill_package for new
procedures; keep the semantic name when revising an existing operation. Do not retype existing skills.
After failure inspect read_recovery_trace, distinguish measured facts from hypotheses, and decide whether
fresh parameters suffice. Preserve the complete recovery goal and reobserve; retries do not reset
physics, execution authority or cumulative budgets. Partial progress can change prerequisites.

During planning, prefer measure_targets(targets=[...]) to select and measure up to eight current RGB-D
regions in one call. Bboxes contain integer pixels, not numeric strings. This returns the same checked
geometry/provenance as ground_target + query_geometry without repeated image overlays. Keep numerical
tracking inside the program. Reuse write/patch checks and source already in context; do not read every
unchanged version again. Reserve a tool call for the final proposal. If the tool budget is exhausted,
the host may start a bounded continuation with the same workspace and paused state; no action is executed.

Code Policy defines run(ctx) and returns RECOVERED, SUBTASK_DONE or UNRESOLVED. Every claimed success is
independently verified against the complete current subtask. RECOVERED cannot end a partial recovery.
The assigned code executor persists between revisions while VLA stays paused. Budget exhaustion
returns to Astra for a genuinely different plan or stop; it never automatically returns to VLA.
After verification, the next subtask is already selected and Astra chooses its executor explicitly.

Read capabilities and the installed verifiers first. Search verified skills for the current robot.
include_templates=true explicitly includes unverified starting programs. include_candidates=true
includes prior experimental programs, including failures; read their evidence and recompute current
geometry before adapting them. They are never automatically considered verified. read_skill exposes
contract, documentation, source and evidence as separate views. check_skill checks a
proposed invocation before execution. Supply exact skill_id, skill_version, parameters and
completion_condition in a code intervention. A skill's fixed completion condition cannot be replaced.
Templates and candidates remain experimental; one success never promotes them to verified.
Uncurated historical recovery versions require include_archived=true; old IDs remain readable.
Empty declared preconditions in check_skill do not establish physical applicability: the program's
current-scene entry checks and fresh measurements remain required. Imported revisions retain the
source capability, prerequisite and completion contracts; edits cannot silently discard them.

Retrieved experience code_versions may include available_skills resolved by exact source hash in this
episode's robot/library snapshot. Use these references to read the relevant recorded version directly;
multiple contracts may share a source hash, so inspect the chosen manifest. Availability is not applicability.
When an earlier attempt is marked verified=false, its COMPLETE task was not verified; inspect its
recorded stages for useful measured progress and the actual stopping condition. Reuse established
clearance, jaw conventions and contact logic when applicable. Combine those with later corrected
tracking/seating logic instead of repeating every historical failed probe. A composed program is still
a new candidate and needs its own fresh-scene checks and full-task verification.

For reuse, prefer check_skill followed by a code intervention with skill_id/skill_version/parameters
when the existing contract and current scene match. If adaptation is needed, use import_skill_program
to copy the exact snapshot source into this workspace, then edit_program. This avoids emitting the
entire source again. The imported version and its patches record origin_skill; importing never executes
or promotes the source. Keep observation-derived geometry, explicit stage checks and stable helper
functions, replacing obsolete step guards and historical frame dependencies with current measurements
and checked preconditions. Do not simply delete a guard to force an old trajectory to run.

write_skill_package and edit_program accept parameter_schema (a JSON Schema object with local references
only) and parameter_sources. Main parameters, or legacy ctx.parameters, contain values that legitimately
change between scenes. The host validates invocation parameters; edits inherit the schema unless
explicitly updated. Executed candidates retain the parameter contract for direct skill
calls in later episodes. A parameter contract does not establish the reliability of its perception.

Program tools: observe(), capabilities(), ground_target(target,camera,observation_step,bbox,confidence),
query_geometry(grounding_id), geometry(camera,bbox), check(condition), stage(name),
note(facts=[...],hypotheses=[...]), act(action12), move_eef(target,steps=1),
move_relative(delta_position,gripper_closed,delta_rotation_vector=[0,0,0],steps=1),
set_gripper(closed,steps=1), move_base(motion,gripper_closed,steps=1), stop(reason).
ctx.parameters and ctx.completion_condition carry the validated invocation. Native act is one step;
other motion tools are 1..5 steps, all counted against the same attempt and episode budgets.
check(condition) returns {condition, passed, evidence, observation_step}; test result["passed"] is True.
Never test the dictionary's truthiness or compare the dictionary itself with True. passed=None is unknown.
stop revokes authority immediately and ends the attempt; no later code is executed through tools.

ctx.read_observation(camera,step=None) loads recorded NumPy rgb/state and optional depth_m arrays,
plus metadata including calibration and its alignment flag. Current and past observations only;
do not invent missing depth or treat unverified extrinsics as reliable. Use NumPy/Pillow/SciPy for
local computation. A raw depth array is in meters, not a colorized depth image.

with ctx.stage(name,preconditions=[...],completion_condition=...) as stage checks prerequisites
before the body and verifies completion afterward. Failed checks raise an exception. The conditions
must have installed independent verifiers. stage.record(**measured) stores program-reported values;
those values cannot approve success. Stages without a completion check are labelled unchecked.
Legacy ctx.stage(name) is only a marker. Factor your code into functions that receive the same ctx.
Use `with ctx.stage(...)` for nested helper stages: the child returns to its parent, and each declared
completion condition is independently checked. Parent/child stages share all action and time budgets.
Legacy bare markers should remain at the top level; they are not scoped contexts. At most 32 contexts
may be nested, and cancellation closes all active stage records without granting more actions.

Organize recovery into measured stages, and inspect actual outcomes after each short action segment.
Use current observations for parameters; do not hardcode a successful layout's absolute coordinates.
geometry returns visible surface measurements, not verified grasp points or collision-free plans.
If depth/calibration is unreliable, request a fresh observation/viewpoint or return UNRESOLVED.
Use facts for observations and hypotheses for explanations. Preserve failed attempts and revised code.

The default Linux worker uses Landlock and a seccomp syscall allowlist before loading generated code.
Only runtime libraries and current exported evidence are readable; host file writes, network,
process spawning and execution are denied. Tools execute through the host broker and Harness.
If confinement cannot be established the attempt fails; trusted mode is an explicit developer option.
