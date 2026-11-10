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
    if start.get("step") != 680:
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

    def segment(target_position, target_quaternion=None, closed=True, steps=4):
        obs = unpack(ctx.observe())
        if obs["step"]-start["step"]+steps > limit:
            raise ValueError("Revision action budget reached; retain code ownership")
        pose = obs["eef_control_pose"]
        p = np.asarray(pose["position"],dtype=float)
        delta = np.asarray(target_position)-p
        length = float(np.linalg.norm(delta))
        if length > 0.025:
            delta *= 0.025/length
        q = np.asarray(pose["quaternion_xyzw"],dtype=float)
        if target_quaternion is not None:
            measured = Rotation.from_quat(q)
            desired = Rotation.from_quat(target_quaternion)
            rv = (desired*measured.inv()).as_rotvec()
            angle = float(np.linalg.norm(rv))
            if angle > 0.18:
                rv *= 0.18/angle
            q = (Rotation.from_rotvec(rv)*measured).as_quat()
        ctx.move_eef(target={"position":(p+delta).tolist(),"quaternion_xyzw":q.tolist(),
            "gripper_closed":closed,"base_motion":[0,0,0,0],"control_mode":"arm"},steps=steps)

    def compact(value):
        if isinstance(value,dict):
            return {str(k):compact(v) for k,v in value.items() if k not in ("rgb","depth_m","state")}
        if isinstance(value,np.ndarray):
            return value.tolist() if value.size <= 64 else {"shape":list(value.shape),"dtype":str(value.dtype)}
        if isinstance(value,(list,tuple)):
            return [compact(v) for v in value] if len(value)<=64 else {"length":len(value)}
        if isinstance(value,(str,int,float,bool)) or value is None:
            return value
        return str(value)

    try:
        with ctx.stage("remeasure unengaged open door from the current state") as stage:
            baseline = measure()
            a = metric("outer door rim left-center",[38,176,45,179],baseline["step"])
            rear = metric("hinge-side rim",[59,150,67,153],baseline["step"])
            side = baseline["rim"]-a
            side[2] = 0.
            width_span = float(np.linalg.norm(side))
            if not 0.10 < width_span < 0.25:
                return unresolved("Rim direction is inconsistent.")
            side /= width_span
            up = np.array([0.,0.,1.])
            inward = np.cross(side,up)
            if np.dot(inward,rear-baseline["rim"]) < 0:
                inward = -inward
            inward /= np.linalg.norm(inward)
            center_shift = 0.5*(a-baseline["rim"])
            center_shift[2] = 0.
            if abs(rear[2]-baseline["rim"][2]) > 0.035:
                return unresolved("Door is no longer lowered; outside approach requires replanning.")
            stage.record(eef_m=baseline["eef"].tolist(),rim_m=baseline["rim"].tolist(),
                inward=inward.tolist(),target_center_shift_m=center_shift.tolist())

        def guard(current):
            rim_change = current["rim"]-current["oven"]-baseline["rim"]+baseline["oven"]
            if np.linalg.norm(current["oven"]-baseline["oven"]) > 0.015:
                raise ValueError("Oven reference displaced during outside approach")
            if np.linalg.norm(rim_change) > 0.015:
                raise ValueError("Door moved during approach; fresh contact assessment required")

        def center(current):
            return current["rim"]+center_shift

        def translation_stage(name, target_fn, quaternion=None, closed=True, max_segments=18):
            stagnant = 0
            with ctx.stage(name) as stage:
                for index in range(max_segments):
                    before = measure()
                    guard(before)
                    target = np.asarray(target_fn(before),dtype=float)
                    err = float(np.linalg.norm(target-before["eef"]))
                    if err < 0.006:
                        stage.record(step=before["step"],target_error_m=err,position_m=before["eef"].tolist())
                        return True
                    segment(target,quaternion,closed)
                    if done():
                        return "DONE"
                    after = measure()
                    guard(after)
                    motion = float(np.linalg.norm(after["eef"]-before["eef"]))
                    stage.record(step=after["step"],target_error_before_m=err,
                        measured_motion_m=motion,position_m=after["eef"].tolist())
                    stagnant = stagnant+1 if motion < 0.001 else 0
                    if stagnant >= 3:
                        raise ValueError("Outside approach motion is blocked in " + name)
            return False

        # Withdraw along the measured door normal while retaining current height
        # and lateral coordinate, then recenter outside the door.
        def retreat_target(current):
            p = current["eef"].copy()
            signed = float(np.dot(p-center(current),inward))
            return p+(-0.10-signed)*inward

        result = translation_stage("withdraw above the front edge",retreat_target)
        if result == "DONE":
            return "SUBTASK_DONE"
        if result is not True:
            return unresolved("Outside clearance not reached within the stage bound.")

        clearance_height = max(float(baseline["eef"][2]),float(baseline["rim"][2])+0.065)
        def outside_center(current):
            p = center(current)-0.10*inward
            p[2] = clearance_height+float(current["oven"][2]-baseline["oven"][2])
            return p

        result = translation_stage("center outside the open door",outside_center)
        if result == "DONE":
            return "SUBTASK_DONE"
        if result is not True:
            return unresolved("Outside centering remains incomplete.")

        x_axis = np.cross(up,inward)
        desired = Rotation.from_matrix(np.column_stack((x_axis,up,inward)))
        with ctx.stage("orient jaws vertically from the outside clearance") as stage:
            aligned = False
            for index in range(20):
                current = measure()
                guard(current)
                if float(np.dot(current["eef"]-center(current),inward)) > -0.075:
                    return unresolved("Insufficient measured outside clearance for reorientation.")
                measured = Rotation.from_quat(current["pose"]["quaternion_xyzw"])
                angle = float(np.linalg.norm((desired*measured.inv()).as_rotvec()))
                if angle < 0.055:
                    aligned = True
                    break
                segment(outside_center(current),desired.as_quat(),True)
                if done():
                    return "SUBTASK_DONE"
                after = measure()
                guard(after)
                stage.record(step=after["step"],angle_before_rad=angle,
                    measured_pose=after["pose"])
            if not aligned:
                return unresolved("Reorientation did not reach its tolerance.")

        with ctx.stage("open jaws in the outside clearance") as stage:
            for index in range(4):
                current = measure()
                guard(current)
                segment(outside_center(current),desired.as_quat(),False)
                if done():
                    return "SUBTASK_DONE"
            current = measure()
            stage.record(step=current["step"],pose=current["pose"])

        def low_outside(current):
            p = center(current)-0.10*inward
            p[2] = current["rim"][2]-0.003
            return p
        result = translation_stage("lower open jaws outside the door rim",low_outside,desired.as_quat(),False)
        if result == "DONE":
            return "SUBTASK_DONE"
        if result is not True:
            return unresolved("Outside rim-height alignment remains incomplete.")

        def near_outside(current):
            p = center(current)-0.035*inward
            p[2] = current["rim"][2]-0.003
            return p
        result = translation_stage("approach measured front rim with open jaws",near_outside,desired.as_quat(),False)
        if result == "DONE":
            return "SUBTASK_DONE"
        if result is not True:
            return unresolved("Bounded approach remains incomplete.")

        current = measure()
        ctx.note(facts=["Outside pregrasp reached at step "+str(current["step"]),
                       "Measured OSC pose: "+json.dumps(current["pose"]),
                       "Current tracked door rim: "+json.dumps(current["rim"].tolist())],
            hypotheses=["Fresh fingertip-to-rim inspection is required before advancing onto the rim. This is a stage of the same code-owned whole-task recovery."])
        for camera in ("right","wrist"):
            record = ctx.read_observation(camera)
            ctx.note(facts=["Observation metadata for "+camera+": "+json.dumps(compact(record))[:14000]],hypotheses=[])
        return "SUBTASK_DONE" if done() else "UNRESOLVED"
    except (ValueError,KeyError,TypeError) as error:
        return unresolved("Outside approach prerequisite failed: "+str(error))
