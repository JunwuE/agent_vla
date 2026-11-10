import json
import numpy as np
from scipy.spatial.transform import Rotation

def run(ctx):
    goal = "task_success"

    def unpack(value):
        if isinstance(value, str):
            return json.loads(value.split("\ndata:image", 1)[0])
        return value

    def checked_done():
        result = unpack(ctx.check(goal))
        return result.get("passed") is True

    def unresolved(reason):
        ctx.note(facts=[reason], hypotheses=["The full door-closing goal remains unverified; code retains ownership."])
        return "UNRESOLVED"

    if checked_done():
        return "SUBTASK_DONE"
    start = unpack(ctx.observe())
    caps = unpack(ctx.capabilities())
    if start.get("step") != 1075:
        return unresolved("Initial snapshot changed; the selected image regions require reselection.")
    if goal not in caps.get("verifiers", []):
        return unresolved("The whole-task verifier is unavailable.")
    if not all(c in caps.get("capabilities", []) for c in ["rgbd", "osc_eef", "gripper"]):
        return unresolved("Required observation or control capability is unavailable.")

    def rgb_now():
        data = ctx.read_observation("right")
        rgb = np.asarray(data["rgb"])
        if rgb.shape != (256, 256, 3):
            raise ValueError("Unexpected camera image shape")
        rgb = rgb.astype(float)
        if np.max(rgb) <= 1.0:
            rgb = rgb * 255.0
        return rgb

    initial_rgb = rgb_now()
    selections = {
        "handle": {
            "context": [89, 117, 128, 133],
            "patch": [98, 122, 105, 125],
            "label": "central silver toaster oven door handle",
            "offset": [0, 0]
        },
        "oven": {
            "context": [93, 85, 151, 113],
            "patch": [115, 94, 123, 99],
            "label": "oven upper surface behind the handle",
            "offset": [0, 0]
        }
    }
    for item in selections.values():
        x0, y0, x1, y1 = item["context"]
        item["template"] = initial_rgb[y0:y1, x0:x1].copy()
        if float(np.std(item["template"])) < 2.0:
            return unresolved("A selected visual tracking region lacks sufficient texture.")

    previous_points = {}

    def measure_scene():
        obs = unpack(ctx.observe())
        rgb = rgb_now()
        points = {}
        for name, item in selections.items():
            x0, y0, x1, y1 = item["context"]
            old_x, old_y = item["offset"]
            candidates = []
            for dy in range(old_y - 4, old_y + 5):
                for dx in range(old_x - 4, old_x + 5):
                    if abs(dx) > 10 or abs(dy) > 10:
                        continue
                    if x0+dx < 0 or y0+dy < 0 or x1+dx > 256 or y1+dy > 256:
                        continue
                    crop = rgb[y0+dy:y1+dy, x0+dx:x1+dx]
                    error = float(np.mean((crop - item["template"]) ** 2))
                    candidates.append((error, dx, dy))
            if not candidates:
                raise ValueError("Visual tracking search has no valid region")
            error, dx, dy = min(candidates)
            if error > 900.0:
                raise ValueError("Selected target appearance changed beyond the tracking bound")
            item["offset"] = [dx, dy]
            a, b, c, d = item["patch"]
            grounding = unpack(ctx.ground_target(
                target=item["label"], camera="right",
                observation_step=obs["step"],
                bbox=[a+dx, b+dy, c+dx, d+dy], confidence=0.85))
            geom = unpack(ctx.query_geometry(grounding["grounding_id"]))
            if geom.get("reliable") is not True or geom.get("frame") != "osc_arm_base":
                raise ValueError("Aligned metric geometry is unavailable or unreliable")
            if geom.get("observation_step") != obs["step"] or geom.get("valid_fraction", 0) < 0.9:
                raise ValueError("Geometry is stale or lacks valid depth")
            point = np.asarray(geom["centroid_m"], dtype=float)
            lo = np.asarray(geom["min_m"], dtype=float)
            hi = np.asarray(geom["max_m"], dtype=float)
            if point.shape != (3,) or not np.all(np.isfinite(point)):
                raise ValueError("Invalid measured surface point")
            if hi[2] - lo[2] > 0.05 or geom.get("depth_iqr_m", 1.0) > 0.06:
                raise ValueError("Selected patch contains inconsistent visible surfaces")
            if name in previous_points:
                maximum_shift = 0.045 if name == "handle" else 0.020
                if float(np.linalg.norm(point - previous_points[name])) > maximum_shift:
                    raise ValueError("Tracked surface jumped beyond the short-segment metric bound")
            # Refresh only after this image selection passes aligned-depth checks.
            item["template"] = rgb[y0+dy:y1+dy, x0+dx:x1+dx].copy()
            previous_points[name] = point.copy()
            points[name] = point
        vector = points["oven"] - points["handle"]
        vector[2] = 0.0
        separation = float(np.linalg.norm(vector))
        if not 0.08 < separation < 0.35:
            raise ValueError("Current geometry no longer matches a raised door near the oven")
        points["inward"] = vector / separation
        points["separation"] = separation
        points["eef"] = np.asarray(obs["eef_control_pose"]["position"], dtype=float)
        points["quaternion"] = np.asarray(obs["eef_control_pose"]["quaternion_xyzw"], dtype=float)
        points["step"] = obs["step"]
        return points

    def budget_check():
        current = unpack(ctx.observe())
        if current["step"] - start["step"] + 2 > 280:
            raise ValueError("This revision reached its 280-step action limit")

    def relative(delta, closed):
        budget_check()
        return ctx.move_relative(delta_position=np.asarray(delta).tolist(),
                                 gripper_closed=closed, steps=2)

    def hold_grip(closed):
        budget_check()
        return ctx.set_gripper(closed=closed, steps=2)

    def pose_segment(position, quaternion, closed):
        budget_check()
        return ctx.move_eef(target={
            "position": np.asarray(position).tolist(),
            "quaternion_xyzw": np.asarray(quaternion).tolist(),
            "gripper_closed": closed, "base_motion": [0, 0, 0, 0],
            "control_mode": "arm"
        }, steps=2)

    def bound(delta, limit):
        length = float(np.linalg.norm(delta))
        return delta if length <= limit else delta * (limit / length)

    try:
        with ctx.stage("remeasure persistent left-edge contact") as stage:
            baseline = measure_scene()
            stage.record(handle_m=baseline["handle"].tolist(),
                         eef_m=baseline["eef"].tolist(),
                         initial_contact_depth_m=float(np.dot(
                             baseline["eef"] - baseline["handle"], baseline["inward"])))

        # The handle followed the preceding horizontal retreat. Test an upward
        # clearance direction before asking for further outward displacement.
        with ctx.stage("clear possible hooked contact above measured handle and oven lip") as stage:
            hold_grip(False)
            if checked_done():
                return "SUBTASK_DONE"
            cleared_up = False
            stagnant = 0
            for index in range(30):
                before = measure_scene()
                clearance_height = max(float(before["oven"][2]) + 0.025,
                                       float(before["handle"][2]) + 0.065)
                error = clearance_height - float(before["eef"][2])
                if error < 0.005:
                    cleared_up = True
                    break
                relative([0.0, 0.0, min(0.012, error)], False)
                if checked_done():
                    return "SUBTASK_DONE"
                after = measure_scene()
                upward = float(after["eef"][2] - before["eef"][2])
                stage.record(segment=index, observation_step=after["step"],
                             measured_upward_m=upward,
                             clearance_height_m=clearance_height)
                stagnant = stagnant + 1 if upward < 0.0005 else 0
                if stagnant >= 3:
                    return unresolved("The upward clearance motion is blocked.")
                if after["handle"][2] < baseline["handle"][2] - 0.035:
                    return unresolved("Door descended beyond the raised-door recovery envelope.")
            if not cleared_up:
                return unresolved("The bounded upward move did not establish clearance height.")

        # Back away from the current contact before changing approach or adding pressure.
        with ctx.stage("release and withdraw to the outside of the handle") as stage:
            hold_grip(False)
            if checked_done():
                return "SUBTASK_DONE"
            stagnant = 0
            cleared = False
            for index in range(40):
                before = measure_scene()
                signed_depth = float(np.dot(before["eef"] - before["handle"], before["inward"]))
                if signed_depth <= -0.075:
                    cleared = True
                    break
                delta = -before["inward"] * min(0.020, signed_depth + 0.080)
                relative(delta, False)
                if checked_done():
                    return "SUBTASK_DONE"
                after = measure_scene()
                retreat = float(np.dot(before["eef"] - after["eef"], before["inward"]))
                stage.record(segment=index, observation_step=after["step"],
                             measured_outward_m=retreat,
                             prior_contact_depth_m=signed_depth)
                if after["handle"][2] < baseline["handle"][2] - 0.035:
                    return unresolved("Door descended during release; a fresh open-door contact plan is needed.")
                stagnant = stagnant + 1 if retreat < 0.001 else 0
                if stagnant >= 3:
                    return unresolved("The measured retreat is blocked; no further arm pressure was added.")
            if not cleared:
                return unresolved("The bounded retreat did not establish the outside approach prerequisite.")

        with ctx.stage("close fingers in the measured outside position") as stage:
            for index in range(8):
                hold_grip(True)
                if checked_done():
                    return "SUBTASK_DONE"
                current = measure_scene()
                stage.record(segment=index, observation_step=current["step"],
                             eef_m=current["eef"].tolist())
                if float(np.dot(current["eef"]-current["handle"], current["inward"])) > -0.050:
                    return unresolved("The outside clearance was lost during finger closure.")

        with ctx.stage("orient the closed gripper toward the door from outside") as stage:
            aligned = False
            for index in range(30):
                current = measure_scene()
                z_axis = current["inward"]
                y_axis = np.array([0.0, 0.0, 1.0])
                x_axis = np.cross(y_axis, z_axis)
                x_axis /= np.linalg.norm(x_axis)
                desired = Rotation.from_matrix(np.column_stack((x_axis, y_axis, z_axis)))
                measured = Rotation.from_quat(current["quaternion"])
                error = (desired * measured.inv()).as_rotvec()
                angle = float(np.linalg.norm(error))
                if angle < 0.07:
                    aligned = True
                    break
                if float(np.dot(current["eef"]-current["handle"], current["inward"])) > -0.050:
                    return unresolved("Insufficient measured outside clearance for reorientation.")
                next_rotation = Rotation.from_rotvec(bound(error, 0.20)) * measured
                pose_segment(current["eef"], next_rotation.as_quat(), True)
                if checked_done():
                    return "SUBTASK_DONE"
                after = measure_scene()
                stage.record(segment=index, angle_before_rad=angle,
                             observation_step=after["step"])
            if not aligned:
                return unresolved("Reorientation did not reach the measured approach tolerance.")

        # The translation target is recomputed from the currently visible handle.
        for stage_name, standoff, max_segments in [
                ("center outside the handle", 0.080, 60),
                ("approach the central handle from outside", 0.015, 40)]:
            with ctx.stage(stage_name) as stage:
                reached = False
                stagnant = 0
                for index in range(max_segments):
                    before = measure_scene()
                    target = before["handle"] - standoff * before["inward"]
                    delta = target - before["eef"]
                    error = float(np.linalg.norm(delta))
                    if error < 0.007:
                        reached = True
                        break
                    relative(bound(delta, 0.020), True)
                    if checked_done():
                        return "SUBTASK_DONE"
                    after = measure_scene()
                    movement = float(np.linalg.norm(after["eef"] - before["eef"]))
                    stage.record(segment=index, target_error_before_m=error,
                                 measured_movement_m=movement,
                                 observation_step=after["step"])
                    if after["handle"][2] < baseline["handle"][2] - 0.035:
                        return unresolved("Door moved outside the raised-door approach envelope.")
                    stagnant = stagnant + 1 if movement < 0.001 else 0
                    if stagnant >= 3:
                        return unresolved("The new measured approach is blocked.")
                if not reached:
                    return unresolved("The bounded approach did not reach its measured target.")

        with ctx.stage("seat door from central outside contact") as stage:
            stagnant = 0
            for index in range(8):
                before = measure_scene()
                target = before["handle"] + before["inward"] * 0.015
                target[2] += 0.003
                relative(bound(target - before["eef"], 0.012), True)
                if checked_done():
                    return "SUBTASK_DONE"
                after = measure_scene()
                change = before["separation"] - after["separation"]
                stage.record(segment=index, observation_step=after["step"],
                             measured_handle_separation_change_m=change)
                if after["handle"][2] < baseline["handle"][2] - 0.035:
                    return unresolved("The door descended during central seating.")
                stagnant = stagnant + 1 if change < 0.0005 else 0
                if stagnant >= 3:
                    return unresolved("Central contact did not establish consistent closing progress.")
        if checked_done():
            return "SUBTASK_DONE"
        return unresolved("Repositioned closing attempt remains unverified.")
    except (ValueError, KeyError, TypeError) as error:
        return unresolved("Recovery prerequisite or observation failed: " + str(error))
