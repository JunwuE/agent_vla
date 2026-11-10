"""Observable five-way review criteria; categories advise Astra, never execute interventions."""

VERSION = "robot-supervision-v2"

CRITERIA = {
    "KEEP": (
        "The current task and phase remain appropriate, and the available evidence supports normal "
        "execution or a normal unfinished phase without a concrete need for Astra review. Approach, "
        "reorientation, retreat before another grasp, settling, and holding contact can be valid progress. "
        "An incomplete goal, a first attempt, a single obscured view, or short stationary motion alone "
        "does not establish failure. Use task-relative visual changes and measured history, not only "
        "distance to an object or elapsed time."
    ),
    "REFINE": (
        "There is concrete evidence of a correctable execution or guidance problem while the current "
        "subtask goal and prerequisites remain valid: the robot targets the wrong region, attempts an "
        "empty grasp, repeatedly loses needed contact, or commands motion without the expected measured "
        "or visible effect. Evidence must distinguish the problem from ordinary approach or deliberate "
        "holding. Ask Astra to reassess execution of the same goal. This category does not require more "
        "detailed language: Astra may retain the instruction, use Code Policy, or decide no action is needed."
    ),
    "SWITCH": (
        "Evidence indicates the ENTIRE current executable subtask meets its stated completion condition. "
        "For example, a door is visibly closed and remains closed, or the required object is stably "
        "carried/placed as the goal specifies. Completing an internal approach/alignment phase, closing "
        "the gripper, reaching a pose, or receiving a high model reward alone is insufficient. This is "
        "normal task progression, not failure. Astra and the installed verifier must confirm before advancing."
    ),
    "REPLAN": (
        "Observed scene changes or unmet/lost prerequisites make the current subtask or remaining order "
        "inappropriate: another goal must be handled first, the intended target/state changed, or the "
        "remaining plan conflicts with the actual scene. Identify the dependency or plan-level mismatch "
        "rather than merely a bad grasp or contact. A local execution failure usually leaves the goal "
        "valid; repeated failures alone do not prove that VLA lacks the skill or that the plan must change."
    ),
    "ESCALATE": (
        "Astra review is warranted but the evidence does not support a specific correction, completion, "
        "or replanning diagnosis: material disagreement between views/history, uncertain outcome after "
        "a critical action, loss of essential target visibility, or a persistent anomaly whose phase "
        "meaning is unclear. Missing information is relevant only when it prevents judging a necessary "
        "condition; an ordinarily unfinished phase or one uninformative view alone is not enough. "
        "Preserve uncertainty rather than inventing a failure cause or declaring completion."
    ),
}

INSTRUCTIONS = (
    "Choose the best-supported category for whether this execution NOW needs Astra review, using only "
    "the supplied causal history and current goal/phase/completion condition. Read frame_map: images "
    "at the same step are synchronized camera views, not successive moments. Compare each camera "
    "through time and use the other views to resolve occlusion; camera motion is not object motion. "
    "Commands are intentions; measured robot motion and visible target changes are separate evidence. "
    "A closed gripper does not prove grasping, and end-effector stillness does not prove task stagnation. "
    "Memory rule_signals contain measured trigger evidence, never authoritative failure labels. "
    "Check whether low motion is explained by settling, holding contact, object motion or the current "
    "phase. Do not ignore persistent unexplained command/measurement mismatch solely because this is "
    "the first attempt. Prior interventions and Flora scores/rationales are historical assessments, "
    "not physical proof. SWITCH refers to the whole executable subtask, not an internal phase. "
    "All non-KEEP categories request Astra review; they do not themselves pause VLA, change language, "
    "select a tool, or execute recovery. Select exactly one category."
)
