---
name: open-drawer-from-missed-grasp
description: Reacquire a visible horizontal drawer handle after an empty grasp.
status: draft
robot: PandaOmron
entry: code/run.py
version: 2
uses:
- eef-feedback-control
- rgbd-surface-geometry
- drawer-pull-from-handle-hold
---
## When to use
A horizontal drawer handle remains visible and the withdrawn closed gripper has missed it. Fixed base, aligned depth, visible handle end patches and planar drawer face are required.
## Preconditions
Current measured control-site pose and fingerpad geometry. Selected regions and seed points identify this drawer only; they are remeasured before use. No attached load is expected from the visibly separated empty gripper.
## Procedure
Fit the face from fresh patch centroids, track two exposed handle-end top surfaces, open fingers, align vertical jaw axis at outward clearance, seat around the measured handle surface and close. Test measured nonempty aperture, then pull in bounded segments while measuring handle motion. At most two contact-depth hypotheses are tested, 360 actions total.
## Code
code/run.py calls named perception and controller modules with pinned helpers and the same ctx.
## Checks
Only task_success can complete this assigned subtask. Pose convergence, aperture and surface displacement are diagnostics, not success verifiers.
### Perception checkpoints
Entry and every approach/insertion boundary: fresh RGB-D geometry with reliable depth, face residual and handle end agreement. Pull: reselect end surface strips near their prior metric points and stop for lost correspondence or lack of coupled motion.
## Failure modes
Occluded end patches, mixed depth, plane mismatch, unreachable pose, empty grasp or no coupled drawer motion return unresolved. Contact centers below the visible top surface are bounded hypotheses, not measured hidden geometry. Do not relax bounds to force execution.
## Evidence
Current episode steps 608 and 680 show closed fingers withdrawn below an unchanged drawer. Step 680 surface selections separate protruding handle top near z=0.096 m from the drawer face. Imported helper evidence is format migration only; this new procedure is unverified.

Development revision: Project expected 3D points into current calibrated views, jointly match both endpoints; preserve bar checks. Not yet physically validated.

Revision: Reuse the successful handle-hold pull package after seating; select current visible wrist front patches instead of tracking handle ends out of view.
