---
name: open-drawer-from-misaligned-handle
description: Bounded RGB-D alignment, grasp and pull from an open misaligned gripper.
status: draft
robot: PandaOmron
entry: code/run.py
version: 3
uses:
- eef-feedback-control
- rgbd-surface-geometry
---
## When to use
A closed drawer with a visible horizontal neutral-colored bar and an open gripper stalled beside it.
## Preconditions
Current wrist RGB-D selections identify the bar and drawer face. Fresh reliable metric geometry, measured pad boxes and OSC pose are mandatory. No held object. Drawer face is approximately vertical; bar is horizontal.
## Procedure
Measure face and bar, retreat outward, rotate jaws perpendicular to the bar, track the bar again, align outside the drawer, approach with checked face clearance, close and pull in bounded increments. Stop on missed grasp or tracking ambiguity.
## Code
code/run.py main(ctx, **params). Dependencies are pinned, share ctx and do not grant extra action budget.
## Checks
Whole-task condition is task_success. Numerical pose convergence, plane residual, bar direction, pad clearance and independent bar translation are checked. A closed jaw is not grasp proof. Limit 390 actions, 0.28 m pull, each primitive 3 steps or fewer.
### Perception checkpoints
Fresh geometry at entry, after retreat/orientation, before seating, after closing, and after each pull segment. Re-segment the current image; old boxes are never reused after motion. Track only an elongated neutral component whose measured axis matches the handle, excluding components touching the lower image edge. Unknown or ambiguous tracking stops the procedure.
## Failure modes
Occlusion, poor segmentation, inaccurate selected surface, blocked tracking, no coupled handle motion, missed closure, unsafe pad clearance. No return to VLA during unfinished recovery.
## Evidence
New experimental drawer procedure. Current observation shows open pads outside handle after ineffective positioning. Dependencies are unverified migrated helpers; their source history is not drawer success.

Development revision: Retreat upward 6 cm before wrist reorientation; retain pose tolerances. Not yet physically validated.

Revision: Test larger exterior/upward clearance (0.16/0.12 m) before wrist rotation; unchanged convergence tolerances, 550-step internal bound.

Revision: Choose the closer of two equivalent parallel-jaw orientations; preserve clearance and convergence checks.
