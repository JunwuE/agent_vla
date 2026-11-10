---
name: drawer-pull-from-handle-hold
description: Probe an existing drawer-handle hold and continue a measured outward
  pull.
status: draft
robot: PandaOmron
entry: code/run.py
version: 2
uses:
- eef-feedback-control
- rgbd-surface-geometry
---
# Drawer pull from handle hold

## When to use
A gripper has stopped at the selected drawer handle with closure maintained, while the drawer remains closed.
This is an experimental whole-task controller, not a verified grasp procedure.

## Preconditions
Current selected RGB-D patches lie on the same drawer front, with a nearby visible handle.
Reliable OSC-base metric geometry, measured control-site pose and fingerpad kinematics are required.
The initial pull is a contact experiment; closure is not a grasp claim.

## Procedure
Fit the selected planar front at entry and orient its normal toward the measured gripper.
Maintain orientation, base pose and closure. Pull outward in short feedback segments.
After every segment remeasure the front and compare its normal displacement with measured EEF displacement.
Stop on separation, nonprogress, tracking inconsistency, base motion or the action/travel bound.
Continue coupled pulling until task_success passes. Never hand an unfinished task back to VLA.

## Code
code/run.py main(ctx, **params) returns {ok, measured, why}.
Uses pinned public helper imports with the same ctx. No independent action lease.

## Checks
Only task_success is an installed completion verifier.
Plane geometry and displacement checks are program measurements, not independent success proof.
Maximum 300 steps, 0.30 m outward travel, 0.008 m target advance, 3 steps per segment.
A 0.012 m EEF/front displacement discrepancy ends the probe.

### Perception checkpoints
Entry: fresh four front patches, handle patch and measured robot pad geometry.
Each segment: fresh same image patches, fit residual, plane normal, EEF pose and base state.
Selected wrist patches remain usable only while the same planar drawer front remains visible.
A changed surface or lost tracking returns unresolved.

## Failure modes
False engagement, mixed surfaces, occlusion, slipping, blocked robot motion and insufficient opening.
No automatic regrasp is attempted by this version.

## Evidence
At episode snapshot 615, the front patches fit a plane with second singular value 0.0193 m.
Gripper closure persisted without opening from the prior review at 535.
Dependencies have format-migration evidence only; this composition is unexecuted.

Development revision: Extend bounded pull travel from 0.30 to 0.45 m; preserve slip, action and time bounds. Not yet physically validated.

Revision: Allow independently observed handle camera; retain validated wrist front tracking and 0.45 m bound.
