---
name: open-drawer-from-misaligned-open-gripper
description: Bounded RGB-D realignment, engagement and pull of a horizontal drawer
  handle.
status: draft
robot: PandaOmron
entry: code/run.py
version: 4
uses:
- eef-feedback-control
- rgbd-surface-geometry
- drawer-pull-from-handle-hold
- open-drawer-from-missed-grasp
---
# Open drawer from misaligned open gripper
## When to use
An open gripper is below a visible horizontal drawer handle and repeated alignment corrections have stalled.
## Preconditions
Stationary base; current right-camera RGB-D; explicitly selected handle and nearby flat drawer-front regions; measured OSC pose and fingerpad geometry. No established grasp. Only task_success terminates the whole subtask.
## Procedure
Measure front plane and silver handle; retract from front; orient vertical jaws; approach measured handle center; close; test outward coupling; pull in short measured segments while checking task_success.
## Code
code/run.py main delegates perception and controller modules. Uses pinned shared helpers with the same ctx.
## Checks
Fresh geometry in osc_arm_base, plane residual and normal, handle foreground separation, horizontal span, pad metadata, pose tracking, aperture, observed handle translation and shared step budget. No base commands.
### Perception checkpoints
Measure at entry, after clearance and approach, and every pull segment. Fresh region selection uses RGB intensity plus measured depth and geometric consistency. Occlusion or ambiguity returns unresolved.
### Resume after interrupted rotation
Optional clearance_start_position is the measured OSC EEF position at the original clearance entry, in meters. With a stationary base, reconstruct the original 0.16 m outward / 0.12 m upward target using the freshly measured normal. Resume only if current position is within the unchanged POSE_TOL of that target and the original open-gripper and geometry checks pass. This avoids accumulating clearance on retries; all subsequent pose, engagement, coupling and task_success checks remain unchanged. Current execution showed continuing rotation at the segment limit, not a verified grasp.
## Failure modes
Small silver pixels may mix with the cabinet, robot occlusion can lose the handle, approach can be blocked, and closure does not prove contact. Return unresolved with evidence rather than assume a grasp.
## Evidence
New experimental whole-task procedure. Steps 555 and 700 show open fingers below the handle; steps 500 to 700 have only about 7.5 mm net wrist motion despite 218 mm path. Dependencies are unverified format migrations, not evidence of drawer success.

Development revision: Use >=4 pixel handle strips, skip individually invalid fragments; preserve geometric gates. Not yet physically validated.

Revision: After correcting minimum depth samples, test exterior/upward clearance before rotation; retain tracking and grasp checks.

Revision: Compute handle scan from one current RGB-D frame; keep the same valid-depth, projection and geometric filters, avoiding per-patch RPC/grounding calls.

Revision: After measured regrasp, reuse the same current-view front selector and handle-hold pull controller that completed seed 7; retain dense geometry and all coupling/task checks.
