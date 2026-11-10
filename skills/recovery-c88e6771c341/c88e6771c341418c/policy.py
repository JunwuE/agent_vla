def run(ctx):
    import json
    import numpy as np
    from scipy.spatial.transform import Rotation

    goal = ctx.completion_condition
    limit = 400
    params = ctx.parameters

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
    if start.get("step") != params["observation_step"]:
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
        "rim": {"label": "exposed outer door silver rim",
                "context": params["rim_context"], "patch": params["rim_box"]},
        "oven": {"label": "fixed oven roof reference",
                 "context": params["oven_context"], "patch": params["oven_box"]}
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
            # Roof texture previously drifted along a planar surface.
            # Hold its selected pixels and original boundary context fixed.
            y_offsets = [0] if key == "oven" else range(oy-4,oy+5)
            x_offsets = [0] if key == "oven" else range(ox-4,ox+5)
            for dy in y_offsets:
                for dx in x_offsets:
                    if min(a+dx,b+dy) < 0 or c+dx > 256 or d+dy > 256:
                        continue
                    patch = rgb[b+dy:d+dy,a+dx:c+dx]
                    err = float(np.mean((patch-item["template"])**2))
                    candidate = (err,dx,dy)
                    if best is None or candidate < best:
                        best = candidate
            if best is None or best[0] > (450.0 if key == "oven" else 1400.0):
                raise ValueError("Visual target lost or obscured: " + key)
            err,dx,dy = best
            x0,y0,x1,y1 = item["patch"]
            p = metric(item["label"], [x0+dx,y0+dy,x1+dx,y1+dy], obs["step"])
            if key in previous and np.linalg.norm(p-previous[key]) > (0.035 if key == "rim" else 0.015):
                raise ValueError("Tracked metric surface jumped: " + key)
            item["offset"] = (dx,dy)
            if key != "oven":
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
        if length > 0.028:
            delta *= 0.028/length
        measured = Rotation.from_quat(pose["quaternion_xyzw"])
        error = (Rotation.from_quat(quaternion)*measured.inv()).as_rotvec()
        angle = float(np.linalg.norm(error))
        if angle > 0.22:
            error *= 0.22/angle
        q = (Rotation.from_rotvec(error)*measured).as_quat()
        ctx.move_eef(target={"position":(p+delta).tolist(),"quaternion_xyzw":q.tolist(),
            "gripper_closed":closed,"base_motion":[0,0,0,0],"control_mode":"arm"},steps=steps)

    def fingertips(current):
        points = [
            metric("open fingertip A", params["finger_a_box"], current["step"], 0.018, "wrist"),
            metric("open fingertip B", params["finger_b_box"], current["step"], 0.018, "wrist")
        ]
        vector = points[0]-points[1]
        width = float(np.linalg.norm(vector))
        if not 0.067 < width < 0.095:
            raise ValueError("Open fingertip surfaces no longer match the selected pair")
        rotation = Rotation.from_quat(current["pose"]["quaternion_xyzw"])
        local_axis = rotation.inv().apply(vector/width)
        midpoint = 0.5*(points[0]+points[1])
        local_midpoint = rotation.inv().apply(midpoint-current["eef"])
        if abs(local_axis[0]) < 0.98 or np.linalg.norm(local_midpoint) > 0.025:
            raise ValueError("Observed finger geometry differs from the measured local-x model")
        return midpoint, vector/width, width, min(points[0][2],points[1][2])

    try:
        with ctx.stage("remeasure actual fingertips and lowered door") as stage:
            baseline = measure()
            second = metric("second front rim section", params["second_box"], baseline["step"], 0.010, params["second_camera"])
            rear = metric("hinge-side rim", params["rear_box"], baseline["step"], 0.010)
            counter = metric("countertop beside the door", params["counter_box"], baseline["step"], 0.010)
            up = np.array([0.,0.,1.])
            side = baseline["rim"]-second
            side[2] = 0.
            span = float(np.linalg.norm(side))
            if not 0.05 < span < 0.25:
                return unresolved("Two current rim selections do not define a usable edge")
            side /= span
            inward = np.cross(side,up)
            if np.dot(inward,rear-baseline["rim"]) < 0:
                inward = -inward
            radius = float(np.dot(rear-baseline["rim"],inward))
            if not 0.14 < radius < 0.30:
                return unresolved("Measured hinge-side distance is outside the closing model")
            pivot = baseline["rim"]+radius*inward
            pivot[2] = rear[2]
            axis = np.cross(up,inward)
            midpoint, jaw_axis, width, lowest = fingertips(baseline)
            local_offset = Rotation.from_quat(baseline["pose"]["quaternion_xyzw"]).inv().apply(midpoint-baseline["eef"])
            if not 0.035 < baseline["rim"][2]-counter[2] < 0.09:
                return unresolved("Measured rim and countertop do not support this outside approach")
            if baseline["rim"][2]-0.5*width-counter[2] < 0.009:
                return unresolved("Measured open lower finger would lack countertop clearance")
            # Select the nearer vertical direction for the measured local x jaw axis.
            jaw_goal = -up if jaw_axis[2] < 0 else up
            y_goal = np.cross(inward,jaw_goal)
            y_goal /= np.linalg.norm(y_goal)
            desired_rotation = Rotation.from_matrix(np.column_stack((jaw_goal,y_goal,inward)))
            q0 = desired_rotation.as_quat()
            stage.record(step=baseline["step"],measured_jaw_axis=jaw_axis.tolist(),
                measured_local_axis=Rotation.from_quat(baseline["pose"]["quaternion_xyzw"]).inv().apply(jaw_axis).tolist(),
                fingertip_midpoint_m=midpoint.tolist(),open_width_m=width,
                local_midpoint_offset_m=local_offset.tolist(),inward=inward.tolist(),
                rim_m=baseline["rim"].tolist(),counter_height_m=float(counter[2]))

        def guard(current):
            if np.linalg.norm(current["oven"]-baseline["oven"]) > 0.015:
                raise ValueError("Oven reference displaced beyond the experiment bound")
            lateral = float(np.dot(current["eef"]-current["rim"],side))
            if not -0.28 < lateral < -0.03:
                raise ValueError("Hand moved outside the selected interior rim section")

        def pose_error(current, goal_position, goal_q):
            distance = float(np.linalg.norm(goal_position-current["eef"]))
            rotation = Rotation.from_quat(current["pose"]["quaternion_xyzw"])
            angle = float((Rotation.from_quat(goal_q)*rotation.inv()).magnitude())
            return distance, angle

        with ctx.stage("measure the projecting handle and gap below the door lip") as stage:
            current = measure()
            guard(current)
            tip,jaw_axis,width,lowest = fingertips(current)
            if abs(float(np.dot(jaw_axis,up))) < 0.985:
                return unresolved("Current measured jaws no longer retain vertical alignment")
            handle = metric("upper visible handle bar surface",params["handle_box"],current["step"],0.008,"wrist")
            lip = metric("door lip above handle gap",params["lip_box"],current["step"],0.008,"wrist")
            gap = float(lip[2]-handle[2])
            if not 0.025 < gap < 0.045 or abs(float(lip[2]-current["rim"][2]))>0.012:
                return unresolved("Measured handle-to-lip gap does not support the revised approach")
            lower_goal = float(handle[2]+0.55*gap)
            initial_depth = float(np.dot(tip-current["rim"],inward))
            if not -0.045 < initial_depth < -0.010:
                return unresolved("Current fingertip depth does not match the stopped exterior approach")
            stage.record(step=current["step"],handle_m=handle.tolist(),lip_m=lip.tolist(),
                lower_tip_height_m=lowest,lower_goal_m=lower_goal,visible_gap_m=gap,
                fingertip_depth_m=initial_depth)
            hold_q = current["pose"]["quaternion_xyzw"]

        with ctx.stage("withdraw from the blocked handle approach") as stage:
            target_depth = min(-0.055,initial_depth-0.025)
            window = initial_depth
            reached = False
            for index in range(16):
                before=measure()
                guard(before)
                tip,jaw_axis,width,lowest=fingertips(before)
                depth=float(np.dot(tip-before["rim"],inward))
                if depth < target_depth+0.004:
                    reached=True
                    break
                act_pose(before["eef"]+max(-0.012,target_depth-depth)*inward,hold_q,False,steps=2)
                if done():
                    return "SUBTASK_DONE"
                after=measure()
                guard(after)
                new_tip,_,_,_=fingertips(after)
                new_depth=float(np.dot(new_tip-after["rim"],inward))
                relative_rim=after["rim"]-after["oven"]-baseline["rim"]+baseline["oven"]
                if np.linalg.norm(relative_rim)>0.015:
                    return unresolved("Door reference moved during outward clearance; remeasure before height correction")
                stage.record(step=after["step"],tip_depth_m=new_depth,eef_m=after["eef"].tolist())
                if index%4==3:
                    if window-new_depth<0.002:
                        return unresolved("Withdrawal from blocked handle contact stopped making progress")
                    window=new_depth
            if not reached:
                return unresolved("Measured outward handle clearance was not established")

        with ctx.stage("raise the lower fingertip into the measured handle-to-lip gap") as stage:
            current=measure()
            tip,_,_,_=fingertips(current)
            outside_depth=float(np.dot(tip-current["rim"],inward))
            window=tip.copy()
            reached=False
            for index in range(20):
                before=measure()
                guard(before)
                tip,jaw_axis,width,lowest=fingertips(before)
                depth=float(np.dot(tip-before["rim"],inward))
                goal_height=lower_goal+float(before["oven"][2]-baseline["oven"][2])
                error=goal_height-lowest
                if depth > -0.040:
                    return unresolved("Lost outside handle clearance during height correction")
                if abs(error)<0.003 and abs(depth-outside_depth)<0.005:
                    reached=True
                    break
                delta=(outside_depth-depth)*inward+float(np.clip(error,-0.012,0.012))*up
                act_pose(before["eef"]+delta,q0,False,steps=2)
                if done():
                    return "SUBTASK_DONE"
                after=measure()
                new_tip,_,_,new_low=fingertips(after)
                stage.record(step=after["step"],lower_tip_height_m=new_low,target_lower_height_m=goal_height)
                if index%4==3:
                    if np.linalg.norm(new_tip-window)<0.002:
                        return unresolved("Gap-height alignment stopped making measured progress")
                    window=new_tip.copy()
            if not reached:
                return unresolved("Lower finger gap-height correction remains incomplete")

        with ctx.stage("insert above the projecting handle with measured clearance") as stage:
            current=measure()
            tip,jaw_axis,width,lowest=fingertips(current)
            window_depth=float(np.dot(tip-current["rim"],inward))
            open_joints=np.abs(np.asarray(unpack(ctx.observe())["state"][-2:],float))
            reached=False
            for index in range(36):
                before=measure()
                guard(before)
                tip,jaw_axis,width,lowest=fingertips(before)
                depth=float(np.dot(tip-before["rim"],inward))
                goal_height=lower_goal+float(before["oven"][2]-baseline["oven"][2])
                error=goal_height-lowest
                if abs(error)>0.006 or abs(float(np.dot(jaw_axis,up)))<0.975:
                    return unresolved("Lower finger left the measured gap or vertical jaw envelope")
                if np.max(open_joints-np.abs(np.asarray(unpack(ctx.observe())["state"][-2:],float)))>0.004:
                    return unresolved("A finger became constrained before measured insertion")
                if depth>=0.006:
                    reached=True
                    break
                delta=min(0.010,0.009-depth)*inward+error*up
                act_pose(before["eef"]+delta,q0,False,steps=2)
                if done():
                    return "SUBTASK_DONE"
                after=measure()
                new_tip,_,_,new_low=fingertips(after)
                new_depth=float(np.dot(new_tip-after["rim"],inward))
                stage.record(step=after["step"],tip_depth_m=new_depth,lower_tip_height_m=new_low,
                    finger_positions=unpack(ctx.observe())["state"][-2:])
                if after["eef"][2]<before["eef"][2]-0.004:
                    return unresolved("Higher insertion caused opposing downward hand motion")
                if index%4==3:
                    if new_depth-window_depth<0.002:
                        return unresolved("Higher approach stopped producing measured insertion")
                    window_depth=new_depth
            if not reached:
                return unresolved("Higher approach did not reach the bounded grasp position")
            current=measure()
            tip,jaw_axis,width,lowest=fingertips(current)
            stage.record(step=current["step"],pregrasp_tip_m=tip.tolist(),lower_tip_height_m=lowest)

        with ctx.stage("close vertically opposed fingers at the measured rim") as stage:
            for index in range(4):
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
            for index in range(60):
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
        return unresolved("Measured-jaw approach or closing prerequisite failed: "+str(error))
