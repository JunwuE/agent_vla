def run(ctx):
    import json
    import numpy as np
    from scipy.spatial.transform import Rotation

    goal = "task_success"
    limit = 196

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
    if start.get("step") != 1140:
        return unresolved("Initial image selections are stale; no action executed.")
    if goal not in caps.get("verifiers", []) or not all(c in caps.get("capabilities", []) for c in ("rgbd", "osc_eef", "gripper")):
        return unresolved("Required geometry, control, or whole-task verifier is unavailable.")

    def rgb_now():
        data = ctx.read_observation("right")
        rgb = np.asarray(data["rgb"], dtype=float)
        if rgb.shape != (256, 256, 3):
            raise ValueError("Unexpected image shape")
        return rgb * 255.0 if np.max(rgb) <= 1.0 else rgb

    def metric(label, box, step, zspread=0.04, camera="right"):
        g = unpack(ctx.ground_target(target=label, camera=camera,
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

    def act_pose(position, quaternion, closed, steps=2):
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

    def calibration():
        data = ctx.read_observation("wrist")
        meta = data.get("metadata", {})
        cal = meta.get("calibration", {})
        current = unpack(ctx.observe())
        if cal.get("alignment_verified") is not True or cal.get("step") != current["step"]:
            raise ValueError("Current wrist calibration is unavailable or stale")
        if cal.get("frame") != "optical_to_osc_arm_base" or data.get("depth_m") is None:
            raise ValueError("Aligned wrist metric depth is unavailable")
        return np.asarray(cal["intrinsics"],float), np.asarray(cal["camera_to_base"],float)

    def project(point, intrinsics, transform):
        optical = transform[:3,:3].T @ (point-transform[:3,3])
        if optical[2] <= 0.02:
            raise ValueError("Finger surface is behind the measured camera")
        pixel = intrinsics @ optical
        return pixel[:2]/pixel[2]

    finger_models = []
    def finger_surfaces(current):
        intrinsics, transform = calibration()
        rotation = Rotation.from_quat(current["pose"]["quaternion_xyzw"])
        state = unpack(ctx.observe())["state"]
        points = []
        for item in finger_models:
            local = item["offset"].copy()
            local[0] += item["sign"]*abs(float(state[item["joint"]]))
            expected = current["eef"]+rotation.apply(local)
            uv = project(expected,intrinsics,transform)
            u,v = int(round(float(uv[0]))), int(round(float(uv[1])))
            if min(u,v)<4 or max(u,v)>251:
                raise ValueError("Projected fingertip marker is outside the wrist image")
            point = metric(item["name"],[u-2,v-2,u+3,v+3],current["step"],0.018,"wrist")
            if np.linalg.norm(point-expected)>0.008:
                raise ValueError("New fingertip depth does not match the current measured joint and camera model")
            points.append(point)
        vector = points[0]-points[1]
        separation = float(np.linalg.norm(vector))
        if not 0.045 < separation < 0.095:
            raise ValueError("Current finger separation is outside the bounded approach model")
        if abs(float(vector[2]))/separation < 0.975:
            raise ValueError("Fresh finger surfaces do not confirm vertical separation")
        return points[0],points[1],state[-2:]

    try:
        with ctx.stage("remeasure the constrained finger and the visible gap above the rail") as stage:
            baseline = measure()
            up = np.array([0.,0.,1.])
            second = metric("second front rim section",[56,179,62,181],baseline["step"])
            rear = metric("hinge-side rim",[59,150,67,153],baseline["step"])
            rail_top = metric("upper visible surface of projecting rail",[101,150,105,160],baseline["step"],0.010,"wrist")
            lip = metric("front lip above the gap",[142,165,146,173],baseline["step"],0.010,"wrist")
            side = baseline["rim"]-second
            side[2] = 0.
            side /= np.linalg.norm(side)
            inward = np.cross(side,up)
            if np.dot(inward,rear-baseline["rim"]) < 0:
                inward = -inward
            radius = float(np.dot(rear-baseline["rim"],inward))
            if not 0.14 < radius < 0.30:
                return unresolved("Measured hinge geometry is not applicable")
            pivot = baseline["rim"]+radius*inward
            pivot[2] = rear[2]
            axis = np.cross(up,inward)
            gap_height = float(lip[2]-rail_top[2])
            if not 0.022 < gap_height < 0.045:
                return unresolved("Visible rail-to-lip gap does not support the revised approach")
            lower_height = float(rail_top[2]+0.55*gap_height)
            rotation = Rotation.from_quat(baseline["pose"]["quaternion_xyzw"])
            if -rotation.as_matrix()[2,0] < 0.98:
                return unresolved("The achieved vertical jaw orientation is no longer present")
            q_hold = baseline["pose"]["quaternion_xyzw"]
            q0 = Rotation.from_matrix(np.column_stack((-up,np.cross(inward,-up),inward))).as_quat()
            state = unpack(ctx.observe())["state"]
            intrinsics, transform = calibration()
            for name,box,joint,sign in [
                ("lower finger current surface",[79,200,86,205],-1,1.),
                ("upper finger current surface",[202,201,209,205],-2,-1.)
            ]:
                point = metric(name,box,baseline["step"],0.015,"wrist")
                local = rotation.inv().apply(point-baseline["eef"])
                if np.linalg.norm(local)>0.060 or abs(local[1])>0.020 or abs(local[2])>0.020:
                    return unresolved("Fresh fingertip selection does not belong to the expected near-hand surface")
                projected = project(point,intrinsics,transform)
                center = np.array([(box[0]+box[2]-1)/2.,(box[1]+box[3]-1)/2.])
                if np.linalg.norm(projected-center)>3.:
                    return unresolved("Metric projection does not reproduce the selected fingertip")
                local[0] -= sign*abs(float(state[joint]))
                finger_models.append({"name":name,"offset":local,"joint":joint,"sign":sign})
            lower,upper,joints = finger_surfaces(baseline)
            stage.record(step=baseline["step"],lower_tip_m=lower.tolist(),upper_tip_m=upper.tolist(),
                rail_top_m=rail_top.tolist(),lip_m=lip.tolist(),visible_gap_m=gap_height,
                revised_lower_finger_height_m=lower_height,finger_positions=joints)

        def guard(current):
            if np.linalg.norm(current["oven"]-baseline["oven"])>0.015:
                raise ValueError("Oven reference moved beyond the bounded experiment")
            lateral = float(np.dot(current["eef"]-current["rim"],side))
            if not -0.28 < lateral < -0.03:
                raise ValueError("Hand moved outside the selected interior door section")

        with ctx.stage("withdraw from constrained rail contact without further insertion") as stage:
            target = baseline["eef"]-0.040*inward
            window = baseline["eef"].copy()
            reached = False
            for index in range(14):
                before = measure()
                guard(before)
                remaining = float(np.dot(before["eef"]-target,inward))
                if remaining < 0.006:
                    reached = True
                    break
                if before["eef"][2]<baseline["eef"][2]-0.015:
                    return unresolved("Withdrawal caused further excessive downward hand displacement")
                act_pose(target,q_hold,False)
                if done():
                    return "SUBTASK_DONE"
                after = measure()
                stage.record(step=after["step"],measured_pose=after["pose"],
                    finger_positions=unpack(ctx.observe())["state"][-2:])
                if index%4==3:
                    if float(np.dot(window-after["eef"],inward))<0.003:
                        return unresolved("Withdrawal failed to clear the constrained contact")
                    window=after["eef"].copy()
            if not reached:
                if float(np.dot(measure()["eef"]-target,inward))>0.008:
                    return unresolved("Outside clearance not reached at the stage bound")

        with ctx.stage("raise the lower fingertip into the measured rail-to-lip gap") as stage:
            current = measure()
            lower,upper,joints=finger_surfaces(current)
            outside_depth=float(np.dot(lower-current["rim"],inward))
            if outside_depth>-0.025:
                return unresolved("Lower finger is not sufficiently outside for vertical repositioning")
            target_height=lower_height
            window=lower.copy()
            reached=False
            for index in range(18):
                before=measure()
                guard(before)
                lower,upper,joints=finger_surfaces(before)
                depth=float(np.dot(lower-before["rim"],inward))
                # Track the reference housing to account for small measured frame changes.
                height=target_height+float(before["oven"][2]-baseline["oven"][2])
                error=height-float(lower[2])
                if abs(error)<0.003 and abs(depth-outside_depth)<0.007 and min(abs(float(joints[0])),abs(float(joints[1])))>0.036:
                    reached=True
                    break
                delta=(outside_depth-depth)*inward+error*up
                act_pose(before["eef"]+delta,q0,False)
                if done():
                    return "SUBTASK_DONE"
                after=measure()
                lower_after,upper_after,joints_after=finger_surfaces(after)
                stage.record(step=after["step"],lower_tip_m=lower_after.tolist(),
                    target_lower_height_m=height,finger_positions=joints_after)
                if index%4==3:
                    if np.linalg.norm(lower_after-window)<0.002:
                        return unresolved("Higher approach alignment stopped producing measured fingertip movement")
                    window=lower_after.copy()
            if not reached:
                return unresolved("Higher lower-finger alignment remains incomplete")

        with ctx.stage("approach above the rail with two-step contact feedback") as stage:
            current=measure()
            lower,upper,joints=finger_surfaces(current)
            open_joints=np.abs(np.asarray(joints,float))
            window_depth=float(np.dot(lower-current["rim"],inward))
            reached=False
            for index in range(22):
                before=measure()
                guard(before)
                lower,upper,joints=finger_surfaces(before)
                depth=float(np.dot(lower-before["rim"],inward))
                target_height=lower_height+float(before["oven"][2]-baseline["oven"][2])
                height_error=target_height-float(lower[2])
                if abs(height_error)>0.006:
                    return unresolved("Lower finger left the measured gap-height envelope")
                if depth>=0.006:
                    reached=True
                    break
                if np.max(open_joints-np.abs(np.asarray(joints,float)))>0.004:
                    return unresolved("A finger is constrained before adequate measured insertion")
                delta=min(0.010,0.009-depth)*inward+height_error*up
                act_pose(before["eef"]+delta,q0,False)
                if done():
                    return "SUBTASK_DONE"
                after=measure()
                lower_after,upper_after,joints_after=finger_surfaces(after)
                new_depth=float(np.dot(lower_after-after["rim"],inward))
                stage.record(step=after["step"],lower_tip_m=lower_after.tolist(),
                    insertion_m=new_depth,finger_positions=joints_after)
                if after["eef"][2]<before["eef"][2]-0.004:
                    return unresolved("Inward approach caused opposite downward hand motion")
                if index%4==3:
                    if new_depth-window_depth<0.002:
                        return unresolved("Revised approach stopped producing inward fingertip progress")
                    window_depth=new_depth
            if not reached:
                return unresolved("Higher insertion did not reach its bounded grasp position")
            current=measure()
            lower,upper,joints=finger_surfaces(current)
            stage.record(step=current["step"],lower_tip_m=lower.tolist(),upper_tip_m=upper.tolist(),
                finger_positions=joints)

        with ctx.stage("close vertically opposed fingers at the measured rim") as stage:
            for index in range(8):
                before = measure()
                guard(before)
                act_pose(before["eef"],q0,True)
                if done():
                    return "SUBTASK_DONE"
                after = measure()
                stage.record(step=after["step"],rim_m=after["rim"].tolist(),eef_m=after["eef"].tolist(),finger_positions=unpack(ctx.observe())["state"][-2:])

        with ctx.stage("verify rim engagement through bounded upward response") as stage:
            initial_lift = measure()
            engaged = False
            for index in range(10):
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
            for index in range(80):
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
                if index%5 == 4:
                    if new_angle-window_angle < 0.020:
                        return unresolved("Closing stopped producing measurable hinge rotation.")
                    window_angle = new_angle
        return unresolved("Revision bound reached before whole-task closure verification.")
    except (ValueError,KeyError,TypeError) as error:
        return unresolved("Measured-jaw approach or closing prerequisite failed: "+str(error))
