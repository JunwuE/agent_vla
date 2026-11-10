import json
import numpy as np

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
    if start.get("step") != 1025:
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
            "context": [89, 111, 128, 128],
            "patch": [99, 117, 106, 120],
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
            points[name] = point
        vector = points["oven"] - points["handle"]
        vector[2] = 0.0
        separation = float(np.linalg.norm(vector))
        if not 0.08 < separation < 0.35:
            raise ValueError("Current geometry no longer matches a raised door near the oven")
        points["inward"] = vector / separation
        points["separation"] = separation
        points["eef"] = np.asarray(obs["eef_control_pose"]["position"], dtype=float)
        points["step"] = obs["step"]
        return points

    try:
        with ctx.stage("measure current raised door contact") as stage:
            baseline = measure_scene()
            stage.record(handle_m=baseline["handle"].tolist(),
                         oven_m=baseline["oven"].tolist(),
                         inward=baseline["inward"].tolist())

        # Test the finger-interference hypothesis without first increasing arm pressure.
        with ctx.stage("change finger contact while holding measured pose") as stage:
            for index in range(4):
                before = unpack(ctx.observe())
                ctx.set_gripper(closed=True, steps=5)
                if checked_done():
                    return "SUBTASK_DONE"
                current = measure_scene()
                displacement = float(np.linalg.norm(
                    current["eef"] - np.asarray(before["eef_control_pose"]["position"], dtype=float)))
                stage.record(segment=index, observation_step=current["step"],
                             eef_displacement_m=displacement,
                             handle_height_m=float(current["handle"][2]))
                if displacement > 0.025:
                    return unresolved("Unexpected end-effector displacement during the contact change.")
                if current["handle"][2] < baseline["handle"][2] - 0.025:
                    return unresolved("Measured handle height dropped; the closing-contact assumption no longer holds.")

        # Geometry is reselected and measured after each bounded segment.
        stagnant = 0
        for index in range(8):
            with ctx.stage("measured final seating segment " + str(index)) as stage:
                before = measure_scene()
                inward = before["inward"]
                delta = inward * 0.010
                delta[2] = 0.0015
                ctx.move_relative(delta_position=delta.tolist(),
                                  gripper_closed=True, steps=5)
                if checked_done():
                    return "SUBTASK_DONE"
                after = measure_scene()
                tracking = float(np.dot(after["eef"] - before["eef"], inward))
                seating_change = before["separation"] - after["separation"]
                stage.record(observation_step=after["step"],
                             requested_delta_m=delta.tolist(),
                             measured_eef_inward_m=tracking,
                             handle_body_separation_change_m=seating_change)
                if after["handle"][2] < baseline["handle"][2] - 0.025:
                    return unresolved("The handle moved downward beyond the raised-door bound.")
                if float(np.linalg.norm(after["eef"] - before["eef"])) > 0.03:
                    return unresolved("Measured movement exceeded this segment's expected range.")
                if tracking < 0.0005 or seating_change < 0.0005:
                    stagnant += 1
                else:
                    stagnant = 0
                if stagnant >= 3:
                    return unresolved("Three measured seating segments did not establish consistent inward handle progress.")

        if checked_done():
            return "SUBTASK_DONE"
        return unresolved("Bounded closing recovery ended without whole-task verification.")
    except (ValueError, KeyError, TypeError) as error:
        return unresolved("Recovery prerequisite or observation failed: " + str(error))
