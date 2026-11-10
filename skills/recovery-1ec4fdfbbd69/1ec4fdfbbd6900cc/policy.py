def run(ctx):
    import json
    import numpy as np
    from scipy.spatial.transform import Rotation

    goal = "task_success"
    limit = 195

    def unpack(value):
        return json.loads(value.split("\ndata:image", 1)[0]) if isinstance(value, str) else value

    def done():
        return unpack(ctx.check(goal)).get("passed") is True

    def unresolved(reason):
        ctx.note(facts=[reason], hypotheses=["The whole door-closing goal remains unverified; code retains ownership."])
        return "UNRESOLVED"

    if done():
        return "SUBTASK_DONE"
    start = unpack(ctx.observe())
    caps = unpack(ctx.capabilities())
    if start.get("step") != 872:
        return unresolved("Initial image selections are stale; no action executed.")
    if goal not in caps.get("verifiers", []) or not all(c in caps.get("capabilities", []) for c in ("rgbd", "osc_eef", "gripper")):
        return unresolved("Required geometry, control, or whole-task verifier is unavailable.")

    def rgb_now():
        data = ctx.read_observation("right")
        rgb = np.asarray(data["rgb"], dtype=float)
        if rgb.shape != (256, 256, 3):
            raise ValueError("Unexpected image shape")
        return rgb * 255.0 if np.max(rgb) <= 1.0 else rgb

    def metric(label, box, step, zspread=0.04):
        g = unpack(ctx.ground_target(target=label, camera="right",
            observation_step=step, bbox=list(box), confidence=0.85))
        m = unpack(ctx.query_geometry(g["grounding_id"]))
        if m.get("reliable") is not True or m.get("frame") != "osc_arm_base":
            raise ValueError("Metric geometry unavailable or unreliable for " + label)
        if m.get("observation_step") != step or m.get("valid_fraction", 0) < 0.95:
            raise ValueError("Stale or invalid depth for " + label)
        p = np.asarray(m["centroid_m"], dtype=float)
        lo, hi = np.asarray(m["min_m"]), np.asarray(m["max_m"])
        if p.shape != (3,) or not np.all(np.isfinite(p)):
            raise ValueError("Invalid surface point")
        if m.get("depth_iqr_m", 1.0) > 0.045 or hi[2]-lo[2] > zspread:
            raise ValueError("Mixed surfaces in " + label)
        return p

    selections = {
        "rim": {"label": "outer door rim right-center",
                "context": [62, 174, 89, 190], "patch": [69, 181, 76, 183]},
        "oven": {"label": "fixed oven upper housing",
                 "context": [50, 88, 105, 117], "patch": [65, 98, 75, 102]}
    }
    initial_rgb = rgb_now()
    for item in selections.values():
        a,b,c,d = item["context"]
        item["template"] = initial_rgb[b:d,a:c].copy()
        item["offset"] = (0,0)
        if np.std(item["template"]) < 2.0:
            return unresolved("Selected visual context lacks sufficient texture.")

    previous = {}
    def measure():
        obs = unpack(ctx.observe())
        rgb = rgb_now()
        result = {"step": obs["step"], "pose": obs["eef_control_pose"]}
        for key,item in selections.items():
            a,b,c,d = item["context"]
            ox,oy = item["offset"]
            best = None
            for dy in range(oy-4,oy+5):
                for dx in range(ox-4,ox+5):
                    if min(a+dx,b+dy) < 0 or c+dx > 256 or d+dy > 256:
                        continue
                    patch = rgb[b+dy:d+dy,a+dx:c+dx]
                    err = float(np.mean((patch-item["template"])**2))
                    candidate = (err,dx,dy)
                    if best is None or candidate < best:
                        best = candidate
            if best is None or best[0] > 1400.0:
                raise ValueError("Visual target lost or obscured: " + key)
            err,dx,dy = best
            x0,y0,x1,y1 = item["patch"]
            p = metric(item["label"], [x0+dx,y0+dy,x1+dx,y1+dy], obs["step"])
            if key in previous and np.linalg.norm(p-previous[key]) > (0.035 if key == "rim" else 0.015):
                raise ValueError("Tracked metric surface jumped: " + key)
            item["offset"] = (dx,dy)
            item["template"] = rgb[b+dy:d+dy,a+dx:c+dx].copy()
            previous[key] = p.copy()
            result[key] = p
        result["eef"] = np.asarray(result["pose"]["position"], dtype=float)
        return result

    def act_pose(position, quaternion, closed, steps=4):
        obs = unpack(ctx.observe())
        if obs["step"]-start["step"]+steps > limit:
            raise ValueError("Revision action bound reached")
        pose = obs["eef_control_pose"]
        p = np.asarray(pose["position"],dtype=float)
        delta = np.asarray(position)-p
        length = float(np.linalg.norm(delta))
        if length > 0.020:
            delta *= 0.020/length
        measured = Rotation.from_quat(pose["quaternion_xyzw"])
        error = (Rotation.from_quat(quaternion)*measured.inv()).as_rotvec()
        angle = float(np.linalg.norm(error))
        if angle > 0.12:
            error *= 0.12/angle
        q = (Rotation.from_rotvec(error)*measured).as_quat()
        ctx.move_eef(target={"position":(p+delta).tolist(),"quaternion_xyzw":q.tolist(),
            "gripper_closed":closed,"base_motion":[0,0,0,0],"control_mode":"arm"},steps=steps)

    try:
        with ctx.stage("measure current outside alignment and rim clearance") as stage:
            baseline = measure()
            a = metric("outer rim left-center",[38,176,45,179],baseline["step"])
            rear = metric("hinge-side rim",[59,150,67,153],baseline["step"])
            counter = metric("countertop right of door",[99,177,108,183],baseline["step"])
            up = np.array([0.,0.,1.])
            side = baseline["rim"]-a
            side[2] = 0.
            span = float(np.linalg.norm(side))
            if not 0.10 < span < 0.25:
                return unresolved("Rim line could not be established.")
            side /= span
            inward = np.cross(side,up)
            if np.dot(inward,rear-baseline["rim"]) < 0:
                inward = -inward
            inward /= np.linalg.norm(inward)
            radius = float(np.dot(rear-baseline["rim"],inward))
            pivot = baseline["rim"]+radius*inward
            pivot[2] = rear[2]
            axis = np.cross(up,inward)
            q0 = np.asarray(baseline["pose"]["quaternion_xyzw"])
            rotation0 = Rotation.from_quat(q0)
            direction_error = float(np.dot(rotation0.as_matrix()[:,2],inward))
            jaw_vertical = abs(float(rotation0.as_matrix()[2,1]))
            clearance = float(baseline["eef"][2]-counter[2])
            lateral = float(np.dot(baseline["eef"]-baseline["rim"],side))
            height_offset = float(baseline["eef"][2]-baseline["rim"][2])
            if direction_error < 0.97 or jaw_vertical < 0.97:
                return unresolved("Measured gripper orientation does not support a straight rim approach.")
            if not 0.060 < clearance < 0.10 or not 0.0 < height_offset < 0.025:
                return unresolved("Current gripper height does not satisfy the bounded approach envelope.")
            if not 0.14 < radius < 0.30:
                return unresolved("Invalid measured hinge distance.")
            stage.record(eef_m=baseline["eef"].tolist(),rim_m=baseline["rim"].tolist(),
                inward=inward.tolist(),jaw_vertical=jaw_vertical,
                forward_alignment=direction_error,counter_clearance_m=clearance,
                rim_height_offset_m=height_offset)

        def guard(current):
            if np.linalg.norm(current["oven"]-baseline["oven"]) > 0.015:
                raise ValueError("Oven reference displaced beyond the experiment bound")

        with ctx.stage("advance straight toward the rim at current height and orientation") as stage:
            stagnant = 0
            reached = False
            for index in range(24):
                before = measure()
                guard(before)
                target = before["rim"]+lateral*side+0.006*inward+height_offset*up
                error = target-before["eef"]
                distance = float(np.linalg.norm(error))
                if distance < 0.005:
                    reached = True
                    break
                if index < 3:
                    # Isolate forward translation before changing lateral alignment.
                    forward = float(np.dot(error,inward))
                    delta = min(0.015,max(0.,forward))*inward
                else:
                    delta = error*min(1.,0.015/max(distance,1e-9))
                act_pose(before["eef"]+delta,q0,False)
                if done():
                    return "SUBTASK_DONE"
                after = measure()
                guard(after)
                actual = float(np.dot(after["eef"]-before["eef"],inward))
                stage.record(step=after["step"],target_error_before_m=distance,
                    forward_motion_m=actual,measured_pose=after["pose"])
                stagnant = stagnant+1 if actual < 0.0007 else 0
                if stagnant >= 3:
                    return unresolved("Straight forward approach is also blocked; no extra pressure or descent applied.")
                if abs(float((after["rim"]-after["oven"]-baseline["rim"]+baseline["oven"])[2])) > 0.012:
                    return unresolved("Door moved during the approach; reassess contact from fresh views.")
            if not reached:
                return unresolved("Straight approach did not reach the rim within its stage bound.")

        with ctx.stage("close vertically opposed fingers at the measured rim") as stage:
            for index in range(4):
                before = measure()
                guard(before)
                act_pose(before["eef"],q0,True)
                if done():
                    return "SUBTASK_DONE"
                after = measure()
                stage.record(step=after["step"],rim_m=after["rim"].tolist(),eef_m=after["eef"].tolist())

        with ctx.stage("verify rim engagement through bounded upward response") as stage:
            initial_lift = measure()
            engaged = False
            for index in range(7):
                before = measure()
                act_pose(before["eef"]+0.012*up,q0,True)
                if done():
                    return "SUBTASK_DONE"
                after = measure()
                guard(after)
                rim_rise = float((after["rim"]-after["oven"]-initial_lift["rim"]+initial_lift["oven"])[2])
                eef_rise = float(after["eef"][2]-initial_lift["eef"][2])
                stage.record(step=after["step"],rim_rise_m=rim_rise,eef_rise_m=eef_rise)
                if rim_rise > 0.010:
                    engaged = True
                    break
                if eef_rise > 0.035:
                    break
            if not engaged:
                return unresolved("The newly approached fingers did not establish measurable door lift.")

        def angle_of(current):
            h = pivot+current["oven"]-baseline["oven"]
            v = current["rim"]-h
            horizontal = -float(np.dot(v,inward))
            r = float(np.hypot(horizontal,v[2]))
            if not 0.70*radius < r < 1.30*radius:
                raise ValueError("Tracked rim is inconsistent with the measured hinge")
            return float(np.arctan2(v[2],horizontal))

        with ctx.stage("continue measured hinged closure") as stage:
            initial = measure()
            initial_angle = angle_of(initial)
            contact_radius = max(0.12,float(np.dot(pivot-initial["eef"],inward)))
            q_at_lift = Rotation.from_quat(initial["pose"]["quaternion_xyzw"])
            window_angle = initial_angle
            for index in range(30):
                before = measure()
                guard(before)
                angle = angle_of(before)
                if angle < window_angle-0.10:
                    return unresolved("Door rotation reversed; engagement no longer established.")
                if angle < 1.43:
                    delta = 0.014*(np.sin(angle)*inward+np.cos(angle)*up)
                    desired = Rotation.from_rotvec(axis*(angle-initial_angle+0.014/contact_radius))*q_at_lift
                else:
                    delta = 0.010*inward
                    desired = Rotation.from_quat(before["pose"]["quaternion_xyzw"])
                act_pose(before["eef"]+delta,desired.as_quat(),True)
                if done():
                    return "SUBTASK_DONE"
                after = measure()
                new_angle = angle_of(after)
                stage.record(step=after["step"],door_angle_rad=new_angle,rim_m=after["rim"].tolist())
                if index%3 == 2:
                    if new_angle-window_angle < 0.025:
                        return unresolved("Closing stopped producing measurable hinge rotation.")
                    window_angle = new_angle
        return unresolved("Revision bound reached before whole-task closure verification.")
    except (ValueError,KeyError,TypeError) as error:
        return unresolved("Straight approach or closing prerequisite failed: "+str(error))
