"""Measured OSC site control and command-space edits, adapted from gpt_as_policy.

No simulator object truth, FK prediction or IK is used here. Policy state16 is
deliberately never interpreted as the OSC control pose.
"""

from typing import Literal

import numpy as np
from pydantic import Field, StrictBool, StrictInt, model_validator
from scipy.spatial.transform import Rotation

from .contracts import Model, validate_actions

Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


class EEFTarget(Model):
    frame: Literal["osc_arm_base"] = "osc_arm_base"
    position: Vector3
    quaternion_xyzw: Quaternion
    gripper_closed: StrictBool
    base_motion: tuple[float, float, float, float] = (0, 0, 0, 0)
    control_mode: Literal["arm", "base"] = "arm"

    @model_validator(mode="after")
    def unit_quaternion(self):
        if abs(np.linalg.norm(self.quaternion_xyzw) - 1) > 0.01:
            raise ValueError("quaternion_xyzw must be unit length")
        return self


class ActionEdit(Model):
    delta_position: Vector3
    delta_rotation_vector: Vector3
    gripper: Literal["keep", "open", "closed"] = "keep"


class EEFRequest(Model):
    target: EEFTarget
    steps: StrictInt = Field(default=1, ge=1, le=5)


class RelativeRequest(Model):
    delta_position: Vector3
    delta_rotation_vector: Vector3 = (0, 0, 0)
    gripper_closed: StrictBool
    steps: StrictInt = Field(default=1, ge=1, le=5)


class GripperRequest(Model):
    closed: StrictBool
    steps: StrictInt = Field(default=1, ge=1, le=5)


class MobileRequest(Model):
    motion: tuple[float, float, float, float]
    gripper_closed: StrictBool
    steps: StrictInt = Field(default=1, ge=1, le=5)


class OSCControl:
    def __init__(self, observation):
        metadata, pose = observation.controller, observation.eef_control_pose
        if not metadata or not pose:
            raise ValueError("Measured OSC site pose and controller calibration are unavailable")
        if (
            metadata.get("name"),
            metadata.get("input_type"),
            metadata.get("input_ref_frame"),
            metadata.get("impedance_mode"),
        ) != ("OSC_POSE", "delta", "base", "fixed"):
            raise ValueError("Unsupported controller contract")
        if pose.get("frame") != "osc_arm_base" or pose.get("step") != observation.step:
            raise ValueError("Stale or incompatible measured OSC site pose")
        self.position = np.asarray(pose["position"], float)
        quat = np.asarray(pose["quaternion_xyzw"], float)
        if (
            self.position.shape != (3,)
            or quat.shape != (4,)
            or not np.isfinite(self.position).all()
            or not np.isfinite(quat).all()
            or abs(np.linalg.norm(quat) - 1) > 0.01
        ):
            raise ValueError("Invalid measured OSC site pose")
        self.rotation = Rotation.from_quat(quat)
        for name in ("input_min", "input_max", "output_min", "output_max"):
            value = np.asarray(metadata[name], float)
            if value.shape != (6,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid controller {name}")
            setattr(self, name, value)
        if np.any(self.input_max <= self.input_min) or np.any(self.output_max <= self.output_min):
            raise ValueError("Controller scaling ranges must increase")
        if np.any(self.input_min < -1) or np.any(self.input_max > 1):
            raise ValueError("Controller input range incompatible with canonical actions")
        self.max_position_delta = float(metadata.get("max_position_delta", 0.05))
        self.max_rotation_delta = float(metadata.get("max_rotation_delta", 0.35))
        self.max_mobile_command = float(metadata.get("max_mobile_command", 0.25))
        if not (
            0 < self.max_position_delta <= 0.1
            and 0 < self.max_rotation_delta <= 1
            and 0 < self.max_mobile_command <= 1
        ):
            raise ValueError("Invalid correction limits")

    def scale(self, values):
        clipped = np.clip(values, self.input_min, self.input_max)
        return self.output_min + (clipped - self.input_min) / (self.input_max - self.input_min) * (
            self.output_max - self.output_min
        )

    def unscale(self, delta):
        values = self.input_min + (delta - self.output_min) / (self.output_max - self.output_min) * (
            self.input_max - self.input_min
        )
        return np.clip(values, self.input_min, self.input_max)

    def errors(self, target):
        return np.r_[
            np.asarray(target.position) - self.position,
            (Rotation.from_quat(target.quaternion_xyzw) * self.rotation.inv()).as_rotvec(),
        ]

    def validate_target(self, target):
        error = self.errors(target)
        if np.linalg.norm(error[:3]) > self.max_position_delta + 1e-8:
            raise ValueError("EEF target exceeds position correction limit")
        if np.linalg.norm(error[3:]) > self.max_rotation_delta + 1e-8:
            raise ValueError("EEF target exceeds rotation correction limit")
        if np.max(np.abs(target.base_motion)) > self.max_mobile_command:
            raise ValueError("Mobile command exceeds correction limit")
        if target.control_mode == "base":
            if np.linalg.norm(error) > 1e-5:
                raise ValueError("Base mode must copy the current OSC site pose")
        elif np.any(target.base_motion):
            raise ValueError("Arm mode requires zero mobile commands")

    def target_action(self, target):
        error = self.errors(target)
        requested = error.copy()
        if target.control_mode == "base":
            requested[:] = 0
        for part, limit in ((slice(0, 3), self.max_position_delta), (slice(3, 6), self.max_rotation_delta)):
            norm = np.linalg.norm(requested[part])
            if norm > limit:
                requested[part] *= limit / norm
        arm = self.unscale(requested)
        action = np.r_[
            arm, float(target.gripper_closed), target.base_motion, float(target.control_mode == "base")
        ]
        diagnostics = dict(
            measured_position=self.position.tolist(),
            error=error.tolist(),
            requested_delta=requested.tolist(),
            applied_delta=self.scale(arm).tolist(),
            clipped=not np.allclose(self.scale(arm), requested),
        )
        return validate_actions([action])[0], diagnostics

    def relative_target(self, delta_position, delta_rotation_vector, gripper_closed):
        return EEFTarget(
            position=self.position + np.asarray(delta_position),
            quaternion_xyzw=(Rotation.from_rotvec(delta_rotation_vector) * self.rotation).as_quat(),
            gripper_closed=gripper_closed,
        )

    def preview(self, actions):
        return [
            dict(
                step_offset=i,
                osc_delta=self.scale(a[:6]).tolist(),
                gripper_closed=bool(a[6] >= 0.5),
                base_motion=a[7:11].tolist(),
                control_mode="base" if a[11] >= 0.5 else "arm",
            )
            for i, a in enumerate(validate_actions(actions))
        ]

    def edit(self, actions, edit):
        actions = validate_actions(actions, maximum=5)
        dp, dr = np.array(edit.delta_position), np.array(edit.delta_rotation_vector)
        if np.linalg.norm(dp) > self.max_position_delta or np.linalg.norm(dr) > self.max_rotation_delta:
            raise ValueError("Edit exceeds total correction limits")
        output, diagnostics = actions.copy(), []
        previous = Rotation.identity()
        for i, action in enumerate(actions):
            alpha = (i + 1) / len(actions)
            delta = self.scale(action[:6])
            bias = Rotation.from_rotvec(alpha * dr)
            wanted = np.r_[
                delta[:3] + dp / len(actions),
                (bias * Rotation.from_rotvec(delta[3:]) * previous.inv()).as_rotvec(),
            ]
            if np.any(dp) or np.any(dr):
                output[i, :6] = self.unscale(wanted)
            if edit.gripper != "keep":
                output[i, 6] = float(edit.gripper == "closed")
            applied = self.scale(output[i, :6])
            diagnostics.append(
                dict(
                    step_offset=i,
                    alpha=alpha,
                    requested_delta=wanted.tolist(),
                    applied_delta=applied.tolist(),
                    clipped=not np.allclose(wanted, applied),
                )
            )
            previous = bias
        return validate_actions(output, maximum=5), diagnostics


def available_capabilities(obs):
    result = ["native_action12", "rgb"]
    try:
        OSCControl(obs)
    except (ValueError, KeyError, TypeError):
        pass
    else:
        result += ["osc_eef", "osc_edit", "mobile_base", "gripper"]
    if any(
        c.get("step") == obs.step
        and c.get("alignment_verified") is True
        and c.get("frame") == "optical_to_osc_arm_base"
        and camera in obs.depth_m
        for camera, c in obs.calibration.items()
    ):
        result.append("rgbd")
    return result


def robot_capabilities(harness):
    """The same current robot contract is visible to the planner and code worker."""
    obs = harness.obs
    return dict(capabilities=available_capabilities(obs),
                verifiers=getattr(harness.env, 'verifier_conditions', ['task_success']),
                controller=obs.controller, eef_control_pose=obs.eef_control_pose,
                observation_step=obs.step)


async def execute_target(harness, memory, lease, request, intervention_id, step_executor=None):
    """One short feedback-controlled segment, with all native actions going through Harness."""
    harness.check(lease)
    initial = await harness.env.observe()
    OSCControl(initial).validate_target(request.target)
    harness.log.event(
        "eef_target",
        intervention_id=intervention_id,
        target=request.target,
        steps=request.steps,
        before_step=initial.step,
    )
    executed = 0
    for _ in range(request.steps):
        harness.check(lease)
        observation = await harness.env.observe()
        if observation.terminal:
            break
        action, diagnostics = OSCControl(observation).target_action(request.target)
        harness.log.event(
            "eef_tracking", intervention_id=intervention_id, before_step=observation.step, **diagnostics
        )
        if step_executor is None:
            await harness.execute(action, lease, memory, intervention_id)
        else:
            await step_executor(action)
        executed += 1
    observation = await harness.env.observe()
    error = OSCControl(observation).errors(request.target)
    return dict(
        executed_steps=executed,
        observation=observation.summary(),
        position_error_m=float(np.linalg.norm(error[:3])),
        rotation_error_rad=float(np.linalg.norm(error[3:])),
        note="Robot tracking only; object interaction success requires independent verification",
    )
