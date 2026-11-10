---
name: open-drawer-from-missed-handle
description: Bounded RGB-D handle reacquisition and drawer pull after missed engagement.
status: draft
robot: PandaOmron
entry: code/run.py
version: 2
uses:
- eef-feedback-control
- rgbd-surface-geometry
---
## When to use
Closed gripper is visibly clear of a closed drawer handle; stationary base and fixed external camera.
## Preconditions
Current selected drawer and handle regions, reliable aligned geometry in osc_arm_base, unobstructed exterior approach, measured pad geometry. No grasp is assumed.
## Procedure
Fit selected drawer plane, segment protruding handle surfaces, align open pads, close, and test a short pull before continuing bounded opening. Entire task remains code-owned.
## Code
code/run.py main; parameters are image selections. Dependencies provide bounded motion and decoding.
## Checks
Plane residual, explicitly aligned optical-to-OSC-base dense depth calibration, handle protrusion/extent, pad opening, measured pose tracking, independent task_success after segments. One-pixel geometry calls are invalid and are not used. Max 390 physical steps.
### Perception checkpoints
Remeasure selected surfaces at entry and handle before engagement. After test pull require outward translation of multiple drawer-front patches. Missing/occluded geometry returns unresolved.
## Failure modes
Small pixels, mixed surfaces, occlusion, robot obstruction, missed contact, no coupled object motion.
## Evidence
New experimental controller. Step 710 selected front plane fit residual 0.00023 m; handle geometry must be resolved at runtime. Dependencies are unverified format migrations, not demonstrated drawer skills.

Revision: Select the measured horizontal band within the U-shaped handle ROI before PCA; preserve residual, length and depth gates.
