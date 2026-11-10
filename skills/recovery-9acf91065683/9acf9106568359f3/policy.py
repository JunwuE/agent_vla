def run(ctx):
    import json
    import numpy as np
    from scipy.spatial.transform import Rotation

    goal = "task_success"
    limit = 190

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
    if start.get("step") != 650:
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

    def move(delta, rot=None, steps=3):
        obs = unpack(ctx.observe())
        if obs["step"]-start["step"]+steps > limit:
            raise ValueError("Revision action budget reached")
        p = obs["eef_control_pose"]
        delta = np.asarray(delta,dtype=float)
        length = float(np.linalg.norm(delta))
        if length > 0.018:
            delta *= 0.018/length
        q = np.asarray(p["quaternion_xyzw"],dtype=float)
        if rot is not None:
            rv = np.asarray(rot,dtype=float)
            angle = float(np.linalg.norm(rv))
            if angle > 0.10:
                rv *= 0.10/angle
            q = (Rotation.from_rotvec(rv)*Rotation.from_quat(q)).as_quat()
        ctx.move_eef(target={"position": (np.asarray(p["position"])+delta).tolist(),
            "quaternion_xyzw": q.tolist(), "gripper_closed": True,
            "base_motion": [0,0,0,0], "control_mode": "arm"}, steps=steps)

    try:
        with ctx.stage("measure current open-door rim and hinge direction") as stage:
            baseline = measure()
            a = metric("outer door rim left-center", [38,176,45,179], baseline["step"])
            rear = metric("hinge-side door rim", [59,150,67,153], baseline["step"])
            side = baseline["rim"]-a
            side[2] = 0.0
            if not 0.10 < np.linalg.norm(side) < 0.25:
                return unresolved("Rim samples do not establish a consistent door width direction.")
            side /= np.linalg.norm(side)
            up = np.array([0.,0.,1.])
            inward = np.cross(side,up)
            if np.dot(inward,rear-baseline["rim"]) < 0:
                inward = -inward
            inward /= np.linalg.norm(inward)
            radius = float(np.dot(rear-baseline["rim"],inward))
            if not 0.14 < radius < 0.30 or abs(baseline["rim"][2]-rear[2]) > 0.035:
                return unresolved("Current geometry is not a lowered door with a usable hinge direction.")
            pivot = baseline["rim"]+radius*inward
            pivot[2] = rear[2]
            axis = np.cross(up,inward)
            contact_radius = float(np.dot(pivot-baseline["eef"],inward))
            if not 0.08 < contact_radius < 0.28 or abs(baseline["eef"][2]-baseline["rim"][2]) > 0.06:
                return unresolved("Measured current gripper pose is outside the edge-lift experiment envelope.")
            stage.record(rim_m=baseline["rim"].tolist(), hinge_side_m=rear.tolist(),
                inward=inward.tolist(), radius_m=radius, contact_radius_m=contact_radius,
                eef_m=baseline["eef"].tolist())

        with ctx.stage("test finger engagement at the existing edge contact") as stage:
            for i in range(4):
                move([0,0,0],steps=3)
                if done():
                    return "SUBTASK_DONE"
                current = measure()
                stage.record(step=current["step"], rim_m=current["rim"].tolist(),
                    eef_m=current["eef"].tolist())
                if np.linalg.norm(current["oven"]-baseline["oven"]) > 0.015:
                    return unresolved("Oven reference moved beyond the experiment bound.")

        with ctx.stage("bounded upward engagement test") as stage:
            before_lift = measure()
            raised = False
            for i in range(8):
                before = measure()
                move([0,0,0.012],steps=3)
                if done():
                    return "SUBTASK_DONE"
                after = measure()
                rim_rise = float((after["rim"]-after["oven"]-baseline["rim"]+baseline["oven"])[2])
                eef_rise = float(after["eef"][2]-before_lift["eef"][2])
                stage.record(step=after["step"], rim_rise_m=rim_rise,
                    eef_rise_m=eef_rise)
                if rim_rise > 0.010:
                    raised = True
                    break
                if eef_rise > 0.040:
                    break
                if i >= 2 and np.linalg.norm(after["eef"]-before["eef"]) < 0.0007:
                    return unresolved("Upward test is blocked; further pressure stopped.")
            if not raised:
                return unresolved("Finger closure and bounded lift did not produce measurable upward door motion.")

        def door_angle(current):
            h = pivot + current["oven"]-baseline["oven"]
            v = current["rim"]-h
            outward_length = -float(np.dot(v,inward))
            height = float(v[2])
            r = float(np.hypot(outward_length,height))
            if not 0.70*radius < r < 1.30*radius:
                raise ValueError("Tracked rim no longer matches the measured hinge geometry")
            return float(np.arctan2(height,outward_length))

        with ctx.stage("follow measured upward door rotation to closure") as stage:
            current = measure()
            window_angle = door_angle(current)
            previous_angle = window_angle
            for i in range(45):
                before = measure()
                angle = door_angle(before)
                if angle < previous_angle-0.10:
                    return unresolved("Door fell back during closing; engagement no longer established.")
                tangent = np.sin(angle)*inward+np.cos(angle)*up
                if angle < 1.43:
                    delta = 0.014*tangent
                    rotation = axis*min(0.08,0.014/contact_radius)
                else:
                    delta = 0.010*inward
                    rotation = None
                move(delta,rotation,steps=3)
                if done():
                    return "SUBTASK_DONE"
                after = measure()
                new_angle = door_angle(after)
                stage.record(step=after["step"], door_angle_rad=new_angle,
                    rim_m=after["rim"].tolist(),
                    eef_motion_m=float(np.linalg.norm(after["eef"]-before["eef"])))
                if np.linalg.norm(after["oven"]-baseline["oven"]) > 0.015:
                    return unresolved("Fixed oven reference displaced; closing stopped.")
                if i%4 == 3:
                    if new_angle-window_angle < 0.025:
                        return unresolved("Closing interaction no longer produces measurable hinge rotation.")
                    window_angle = new_angle
                previous_angle = new_angle
            return unresolved("Bounded closing arc ended without whole-task verification.")
    except (ValueError, KeyError, TypeError) as error:
        return unresolved("Observation or stage prerequisite failed: " + str(error))
