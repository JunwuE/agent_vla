---
name: eef-feedback-control
description: Bounded OSC EEF command segments from the current measured pose.
status: draft
robot: PandaOmron
entry: code/run.py
version: 1
uses: ["rgbd-surface-geometry"]
---
# eef-feedback-control

## When to use
Bounded OSC EEF command segments from the current measured pose.
Only the measured operating envelopes in the source programs are supported. This refactor is a draft.

## Preconditions
Fresh metric depth, aligned optical-to-OSC-arm-base calibration, and actual measured robot state.
The closing procedure checks its nearly upright door and contact/clearance prerequisites before motion.
Geometry functions do not establish target identity, grasp success or collision freedom.

## Procedure
Read actual pose, clamp translation/rotation, execute a short segment. The caller checks contact/object progress.
All commands use the supplied ctx and the existing Harness; no simulator handles or object-state oracle.

## Code
`code/run.py:main(ctx, **params)` returns `{ok, measured, why}`.
Python modules use the package name `eef_feedback_control`. Import dependencies by package name.
position: meters in OSC base; quaternion: xyzw; closed: bool; steps: 1..5. This helper is not a grasp/placement skill.
Named fixed bounds live in params.py. Scene selections are explicit inputs, never last episode's coordinates.

## Checks
Keep the original entry, depth, alignment, motion, observation and progress checks unchanged.
The host independently checks the whole assigned subtask. A helper's return does not release VLA.

### Perception checkpoints
| Step | Evidence | Check | On failure |
|---|---|---|---|
| Entry | current RGB-D/calibration | selected surfaces and required state are consistent | stop with reason |
| Segments | new observations and actual pose | original tracking/progress bounds | stop; return evidence to Astra |
| Finish | installed task verifier | full subtask success | remain code-owned if unresolved |

## Failure modes
Stale selections, mixed depth, occlusion, lost tracked surfaces, missing calibration, incompatible posture,
blocked motion and insufficient object progress remain possible. Fresh parameters are the first adaptation
when supported; change controller logic only for an observed incompatibility. Do not weaken guards to run.

## Evidence
Source programs: recovery-6e0569a16adc/6e0569a16adc2f86, recovery-2ff9c7e5cdd3/2ff9c7e5cdd39598.
Their original logs remain immutable. Source history is provenance, not a fresh physical run of this package.
No promotion to verified is implied. Refactor replay receipts are kept under runs/rax-code-policy-20261031.
