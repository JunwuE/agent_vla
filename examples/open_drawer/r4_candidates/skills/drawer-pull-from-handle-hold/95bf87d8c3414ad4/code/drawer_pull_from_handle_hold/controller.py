import numpy as np
from scipy.spatial.transform import Rotation
from eef_feedback_control import pose, move_bounded
from .perception import front, surface
from .params import *

def passed(ctx):
    return ctx.check("task_success")["passed"] is True

def execute(ctx, params):
    measured = {}
    try:
        obs = pose(ctx)
        start_step = obs["step"]
        if start_step != params["observation_step"]:
            raise ValueError("Selection snapshot changed; remeasure scene parameters")
        if passed(ctx):
            return {"ok": True, "measured": measured, "why": "Installed task verifier passed at entry"}
        p0 = obs["eef"].copy()
        q0 = np.asarray(obs["eef_control_pose"]["quaternion_xyzw"], float)
        if obs["eef_control_pose"]["frame"] != "osc_arm_base":
            raise ValueError("Incompatible measured control pose")
        base0 = np.asarray(obs["state"][7:10], float)
        camera, boxes = params["camera"], params["front_boxes"]
        with ctx.stage("measure_current_handle_hold") as stage:
            c0, normal, residual = front(ctx, camera, boxes, start_step, eef=p0)
            gap = float(np.dot(p0-c0, normal))
            handle, _ = surface(ctx, params.get("handle_camera", camera), params["handle_box"], start_step)
            if not 0.005 < gap < 0.070 or np.linalg.norm(handle-p0) > 0.080:
                raise ValueError("Measured gripper is outside handle-hold entry envelope")
            kin = obs["controller"]["gripper_kinematics"]["fingerpad_geometry"]
            if not kin.get("available") or kin.get("step") != start_step or kin.get("frame") != "eef_control_site":
                raise ValueError("Missing current fingerpad geometry")
            pads = kin["pads"]
            pad_gap = float(np.linalg.norm(np.asarray(pads[0]["center_eef"])-np.asarray(pads[1]["center_eef"])))
            if not 0.010 < pad_gap < 0.085:
                raise ValueError("Finger separation incompatible with hold probe")
            measured.update(entry_gap_m=gap, pad_center_separation_m=pad_gap,
                            outward_normal=normal.tolist(), plane_residual_m=residual)
            stage.record(**measured)
        stalled = 0
        last_pull = 0.0
        with ctx.stage("probe_and_pull_with_surface_feedback") as stage:
            while True:
                obs = pose(ctx)
                if passed(ctx):
                    return {"ok": True, "measured": measured, "why": "Installed task_success passed"}
                if obs["step"]-start_step+SEGMENT_STEPS > MAX_STEPS:
                    return {"ok": False, "measured": measured, "why": "Bounded pull action limit reached"}
                actual = float(np.dot(obs["eef"]-p0, normal))
                if actual >= MAX_PULL_M:
                    return {"ok": False, "measured": measured, "why": "Pull travel bound reached without task_success"}
                target = p0 + normal*min(actual+ADVANCE_M, MAX_PULL_M)
                move_bounded(ctx, target, q0, True, start_step=start_step,
                             limit=MAX_STEPS, steps=SEGMENT_STEPS)
                after = pose(ctx)
                if passed(ctx):
                    return {"ok": True, "measured": measured, "why": "Installed task_success passed after segment"}
                if np.linalg.norm(np.asarray(after["state"][7:10])-base0) > 0.005:
                    raise ValueError("Base frame changed during surface tracking")
                rotation_error = (Rotation.from_quat(after["eef_control_pose"]["quaternion_xyzw"])*Rotation.from_quat(q0).inv()).magnitude()
                if rotation_error > 0.10:
                    raise ValueError("Orientation departed from measured hold")
                center, n_now, residual = front(ctx, camera, boxes, after["step"], outward=normal)
                if np.dot(n_now, normal) < np.cos(MAX_NORMAL_ANGLE_RAD):
                    raise ValueError("Drawer plane normal changed; tracking ambiguous")
                pull = float(np.dot(after["eef"]-p0, normal))
                drawer = float(np.dot(center-c0, normal))
                transverse = float(np.linalg.norm(after["eef"]-p0-normal*pull))
                measured.update(step=after["step"], eef_pull_m=pull, drawer_front_motion_m=drawer,
                                relative_separation_m=pull-drawer, transverse_m=transverse,
                                plane_residual_m=residual)
                stage.record(**measured)
                if abs(pull-drawer) > MAX_SLIP_M:
                    return {"ok": False, "measured": measured, "why": "Front and gripper motion not coupled; stop contact experiment"}
                if drawer < -0.004 or transverse > 0.010:
                    raise ValueError("Unexpected drawer or lateral gripper motion")
                if pull-last_pull < 0.0003:
                    stalled += 1
                else:
                    stalled = 0
                if stalled >= 6:
                    return {"ok": False, "measured": measured, "why": "No measured outward progress across six bounded segments"}
                last_pull = pull
    except (ValueError, KeyError, TypeError) as error:
        return {"ok": False, "measured": measured, "why": str(error)}
