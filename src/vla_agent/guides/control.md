# PandaOmron control

By default these motion tools are called inside Code Policy. Direct Astra eef/edit actions are only
available when explicitly enabled in the experiment's allowed_interventions. Check capabilities.

Inspect capabilities and controller calibration before requesting motion. Policy state16 contains a
policy EEF pose; it is not the OSC EEF site pose. Only eef_control_pose is a valid source for absolute
targets. That pose is in the moving osc_arm_base frame, in meters, with quaternion_xyzw orientation.

EEF corrections specify target.position, target.quaternion_xyzw, target.gripper_closed (boolean),
target.base_motion [x,y,yaw,torso] and target.control_mode arm/base. Execute 1..5 feedback steps.
The default total position/rotation limits are 0.05 m and 0.35 rad; read actual configured limits.
The position limit is a vector norm, not a per-axis allowance. Arm mode requires zero mobile commands.
Base mode copies the measured EEF pose, uses bounded normalized mobile inputs and holds the native
controller's desired arm goal. Mobile inputs are not distances. Neither mode performs collision checking.

For edit: request preview_vla, inspect the prefix, optionally call preview_edit, then submit the exact
proposal_id, steps and edit. delta_position is a total metric offset distributed across the prefix;
delta_rotation_vector is a composed rotation, not Euler angles. The host preserves base/mode channels
and reports controller clipping. This edits a nominal command path, not a predicted physical trajectory.
The proposal becomes invalid after execution, takeover or a language/subtask version change.

A closed gripper command or accurate EEF tracking does not establish object grasping or task success.

## Primitive composition

Keep repeated motion primitives inside Code Policy; Astra selects and revises complete procedures.
Existing callable primitives are ctx.move_eef, ctx.move_relative, ctx.set_gripper and ctx.move_base.
move_to / move_pose can be structured wrappers around these, but are not additional registered APIs.
rotate_wrist / rotate_pitch need explicit axes and reference frames; EEF orientation control is not
independent wrist-joint control. Do not assume a default tool axis or confuse tool/base composition.
A release procedure must check current support/constraint and clearance conditions before opening,
then observe the effect. Unknown conditions require observation; gripper opening is not release proof.
Only installed verifiers can be passed to ctx.check. Existing OSC motion has no collision planner.

For grasp orientation, inspect controller.gripper_kinematics in capabilities/observe. When available,
jaw_separation_axis_eef is the unoriented jaw opening axis in the measured EEF control-site frame.
Rotate it with eef_control_pose.quaternion_xyzw to express it in osc_arm_base. Finger-body rotations
are included: never assume that a finger joint's local Y axis equals the control site's Y axis.
This robot calibration does not identify contact points, object thickness, or successful grasping.
Always compare new RGB and measured state, then use an installed independent completion verifier.
# Robot finger geometry

When available, controller.gripper_kinematics.fingerpad_geometry provides the two declared robot pad
boxes from the robot model and measured joints, in the current eef_control_site frame. Each box has
center_eef, rotation_eef and half_extents_m. Transform using the current eef_control_pose; validate
availability, matching step, frame and source. This can initialize robot-finger geometry without
requiring an occluded fingertip to be segmented again from RGB-D. It does not measure object contact,
force or grasp success. Target geometry still comes from current RGB-D; verify object motion and the
installed task completion condition independently. Absence of this optional geometry is unknown.

