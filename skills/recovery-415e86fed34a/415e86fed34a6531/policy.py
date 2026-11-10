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
        data = ctx.read_observation(params["tracking_camera"])
        rgb = np.asarray(data["rgb"], dtype=float)
        if rgb.shape != (256, 256, 3):
            raise ValueError("Unexpected image shape")
        return rgb * 255.0 if np.max(rgb) <= 1.0 else rgb

    def metric(label, box, step, zspread=0.04, camera=None, with_bounds=False):
        camera = params["tracking_camera"] if camera is None else camera
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
        return (p,lo,hi) if with_bounds else p

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

    # Paired rim tracking uses current calibrated depth; fixed roof pixels prevent
    # the planar-template drift observed in the imported candidate.
    selections["rim2"] = {"label":"second exposed outer rim section",
        "context":params["second_context"],"patch":params["second_track_box"],"offset":(0,0)}
    a,b,c,d = params["second_context"]
    selections["rim2"]["template"] = initial_rgb[b:d,a:c].copy()
    for key in ("rim","rim2"):
        x0,y0,x1,y1=selections[key]["patch"]
        selections[key]["color"]=np.median(initial_rgb[y0:y1,x0:x1].reshape(-1,3),axis=0)
    previous = {}
    tracking_model = {}
    cache = {}
    def measure():
        obs = unpack(ctx.observe())
        if cache.get("step") == obs["step"]:
            return cache.copy()
        data=ctx.read_observation(params["tracking_camera"])
        cal=data.get("metadata",{}).get("calibration",{})
        if cal.get("step")!=obs["step"] or cal.get("alignment_verified") is not True or cal.get("frame")!="optical_to_osc_arm_base":
            raise ValueError("Current aligned depth calibration unavailable")
        rgb=np.asarray(data["rgb"],float)
        if np.max(rgb)<=1.0:
            rgb=rgb*255.0
        depth=np.asarray(data.get("depth_m"),float)
        k=np.asarray(cal["intrinsics"],float)
        transform=np.asarray(cal["camera_to_base"],float)
        if depth.shape!=(256,256) or rgb.shape!=(256,256,3) or k.shape!=(3,3) or transform.shape!=(4,4):
            raise ValueError("Unexpected calibrated RGB-D shape")
        def surface(box):
            x0,y0,x1,y1=box
            z=depth[y0:y1,x0:x1]
            if z.size<5 or not np.all(np.isfinite(z)) or np.any(z<=0.02):
                return None
            if float(np.percentile(z,75)-np.percentile(z,25))>0.045:
                return None
            vv,uu=np.mgrid[y0:y1,x0:x1]
            optical=np.stack(((uu-k[0,2])*z/k[0,0],(vv-k[1,2])*z/k[1,1],z),axis=-1).reshape(-1,3)
            cloud=optical@transform[:3,:3].T+transform[:3,3]
            p=np.median(cloud,axis=0)
            if float(np.ptp(cloud[:,2]))>0.010 or np.max(np.linalg.norm(cloud-p,axis=1))>0.025:
                return None
            return p
        def consistent(key,p,result):
            if key in previous and np.linalg.norm(p-previous[key])>(0.025 if key!="oven" else 0.015):
                return False
            if key!="oven" and tracking_model:
                shift=result["oven"]-tracking_model["oven"]
                lateral=abs(float(np.dot(p-tracking_model[key]-shift,tracking_model["side"])))
                vector=p-tracking_model["pivot"]-shift
                radial=float(np.hypot(np.dot(vector,tracking_model["inward"]),vector[2]))
                if lateral>0.020 or abs(radial-tracking_model["radius"])>0.020:
                    return False
            return True
        result={"step":obs["step"],"pose":obs["eef_control_pose"]}
        for key in ("oven","rim","rim2"):
            item=selections[key]
            a,b,c,d=item["context"]
            ox,oy=item["offset"]
            x0,y0,x1,y1=item["patch"]
            candidates=[]
            first=key not in previous
            y_offsets=[0] if key=="oven" or first else range(oy-5,oy+6)
            x_offsets=[0] if key=="oven" or first else range(ox-5,ox+6)
            for dy in y_offsets:
                for dx in x_offsets:
                    if min(a+dx,b+dy,x0+dx,y0+dy)<0 or max(c+dx,x1+dx)>256 or max(d+dy,y1+dy)>256:
                        continue
                    context=rgb[b+dy:d+dy,a+dx:c+dx]
                    err=float(np.mean((context-item["template"])**2))
                    box=[x0+dx,y0+dy,x1+dx,y1+dy]
                    if key=="oven":
                        if err<=450.0:
                            candidates.append((err,dx,dy,None))
                        continue
                    p=surface(box)
                    if p is None or not consistent(key,p,result):
                        continue
                    if key=="rim2" and tracking_model:
                        separation=p-result["rim"]
                        if abs(float(separation[2]))>0.012 or abs(float(np.dot(separation,tracking_model["inward"])))>0.012:
                            continue
                    color=np.median(rgb[y0+dy:y1+dy,x0+dx:x1+dx].reshape(-1,3),axis=0)
                    color_error=float(np.sqrt(np.mean((color-item["color"])**2)))
                    continuity=0.0 if first else float(np.linalg.norm(p-previous[key]))
                    score=continuity*continuity+0.00002*min(err/1400.0,4.0)+0.00002*min((color_error/70.0)**2,4.0)
                    candidates.append((score,dx,dy,p))
            candidates.sort(key=lambda candidate:candidate[0])
            accepted=None
            for score,dx,dy,estimate in candidates[:4]:
                try:
                    p=metric(item["label"],[x0+dx,y0+dy,x1+dx,y1+dy],obs["step"],0.010 if key!="oven" else 0.008)
                    if estimate is not None and np.linalg.norm(p-estimate)>0.006:
                        continue
                    if not consistent(key,p,result):
                        continue
                    accepted=(p,dx,dy)
                    break
                except (ValueError,KeyError,TypeError):
                    continue
            if accepted is None:
                raise ValueError("No calibrated surface candidate for "+key)
            p,dx,dy=accepted
            item["offset"]=(dx,dy)
            if key!="oven":
                item["template"]=rgb[b+dy:d+dy,a+dx:c+dx].copy()
            previous[key]=p.copy()
            result[key]=p
        result["eef"]=np.asarray(result["pose"]["position"],dtype=float)
        cache.clear()
        cache.update(result)
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
            second = metric("second front rim section", params["second_box"], baseline["step"], 0.010, "left")
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
            # Keep vertical jaw geometry, with a selectable measured horizontal heading.
            jaw_goal = -up if jaw_axis[2] < 0 else up
            approach_heading = inward.copy()
            if params["wrist_heading_mode"] == "preserve_current_horizontal":
                forward = Rotation.from_quat(baseline["pose"]["quaternion_xyzw"]).apply([0.,0.,1.])
                approach_heading = forward-float(np.dot(forward,up))*up
                heading_norm = float(np.linalg.norm(approach_heading))
                if heading_norm < 0.90:
                    return unresolved("Current wrist heading is insufficiently horizontal for preserved-heading alignment")
                approach_heading /= heading_norm
                if float(np.dot(approach_heading,inward)) < np.sqrt(0.5):
                    return unresolved("Current horizontal wrist heading faces too far along the rim")
            y_goal = np.cross(approach_heading,jaw_goal)
            y_goal /= np.linalg.norm(y_goal)
            desired_rotation = Rotation.from_matrix(np.column_stack((jaw_goal,y_goal,approach_heading)))
            q0 = desired_rotation.as_quat()
            stage.record(wrist_heading_mode=params["wrist_heading_mode"],
                approach_heading=approach_heading.tolist(),
                heading_normal_component=float(np.dot(approach_heading,inward)))
            # Runtime capabilities may omit controller metadata. Prefer current
            # observation calibration, otherwise use the same-step host snapshot.
            kin=start.get("controller",{}).get("gripper_kinematics",{})
            calibration_source="runtime_observation"
            if not kin:
                kin=params["jaw_calibration"]
                calibration_source="same_step_host_snapshot"
            jaw_local=np.asarray(kin.get("jaw_separation_axis_eef",[]),float)
            if kin.get("available") is not True or kin.get("step")!=start["step"] or kin.get("frame")!="eef_control_site" or kin.get("source")!="robot_joint_kinematics":
                return unresolved("Current controller jaw calibration provenance is unavailable")
            if jaw_local.shape!=(3,) or not np.all(np.isfinite(jaw_local)) or not 0.99<float(np.linalg.norm(jaw_local))<1.01 or abs(float(jaw_local[0]))<0.98:
                return unresolved("Controller jaw calibration does not support local-x alignment")
            observed_local=Rotation.from_quat(baseline["pose"]["quaternion_xyzw"]).inv().apply(jaw_axis)
            if abs(float(np.dot(observed_local,jaw_local)))<0.98:
                return unresolved("Current fingertip depth disagrees with controller jaw calibration")
            stage.record(jaw_calibration_source=calibration_source,controller_jaw_axis_eef=jaw_local.tolist(),observed_jaw_axis_eef=observed_local.tolist())
            if abs(float(baseline["rim"][2]-baseline["rim2"][2]))>0.008 or abs(float(np.dot(baseline["rim"]-baseline["rim2"],inward)))>0.012:
                return unresolved("Independent rim selections disagree")
            tracking_model.update(pivot=pivot,radius=radius,side=side,inward=inward,
                oven=baseline["oven"].copy(),rim=baseline["rim"].copy(),rim2=baseline["rim2"].copy())
            stage.record(step=baseline["step"],measured_jaw_axis=jaw_axis.tolist(),
                measured_local_axis=Rotation.from_quat(baseline["pose"]["quaternion_xyzw"]).inv().apply(jaw_axis).tolist(),
                fingertip_midpoint_m=midpoint.tolist(),open_width_m=width,
                local_midpoint_offset_m=local_offset.tolist(),inward=inward.tolist(),
                rim_m=baseline["rim"].tolist(),counter_height_m=float(counter[2]))

        def guard(current):
            if np.linalg.norm(current["oven"]-baseline["oven"]) > 0.015:
                raise ValueError("Oven reference displaced beyond the experiment bound")
            if abs(float(current["rim"][2]-current["rim2"][2]))>0.012 or abs(float(np.dot(current["rim"]-current["rim2"],inward)))>0.012:
                raise ValueError("Separate measured rim sections disagree")
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
            handle,handle_lo,handle_hi = metric("upper visible handle bar surface",params["handle_box"],current["step"],0.008,"wrist",True)
            lip,lip_lo,lip_hi = metric("outer door lip above handle gap",params["lip_box"],current["step"],0.008,"wrist",True)
            lip2,lip2_lo,lip2_hi = metric("second outer door lip surface",params["contact_second_box"],current["step"],0.008,"wrist",True)
            if abs(float(np.dot(lip-lip2,inward)))>0.008 or abs(float(lip[2]-lip2[2]))>0.008:
                return unresolved("Independent outer lip surfaces disagree")
            if abs(float(lip[2]-current["rim"][2]))>0.012 or abs(float(np.dot(lip-current["rim"],inward)))>0.045:
                return unresolved("Outer lip is inconsistent with the tracked door rim")
            if abs(float(np.dot(tip-lip,side)))>0.12 or abs(float(np.dot(handle-lip,side)))>0.06:
                return unresolved("Selected handle gap is too far from the current approach")
            _,fa_lo,fa_hi = metric("finger A thickness",params["finger_a_box"],current["step"],0.018,"wrist",True)
            _,fb_lo,fb_hi = metric("finger B thickness",params["finger_b_box"],current["step"],0.018,"wrist",True)
            finger_halfheight=0.5*max(float(fa_hi[2]-fa_lo[2]),float(fb_hi[2]-fb_lo[2]))
            gap_bottom=float(handle_hi[2])
            gap_top=float(min(lip_lo[2],lip2_lo[2]))
            gap=gap_top-gap_bottom
            if not 0.001<finger_halfheight<0.009 or gap>0.045:
                return unresolved("Finger thickness or visible gap is outside the measured model")
            clearance=finger_halfheight+0.003
            lower_goal=0.5*(gap_top+gap_bottom)
            height_tolerance=0.0006
            # Centering error plus feedback headroom; physical clearance is unchanged.
            if min(lower_goal-gap_bottom,gap_top-lower_goal)<clearance+height_tolerance+0.0005:
                return unresolved("Visible gap lacks measured finger thickness and clearance allowance")
            contact_lip=0.5*(lip+lip2)
            def contact_surface(current):
                shift=current["rim"]-baseline["rim"]
                if np.linalg.norm(shift)>0.006:
                    raise ValueError("Door moved during gap approach; current contact lip must be reselected")
                return contact_lip+shift
            initial_depth = float(np.dot(tip-contact_surface(current),inward))
            if not -0.045 < initial_depth < -0.010:
                return unresolved("Current fingertip depth does not match the stopped exterior approach")
            stage.record(step=current["step"],handle_m=handle.tolist(),lip_m=lip.tolist(),
                lower_tip_height_m=lowest,lower_goal_m=lower_goal,visible_gap_m=gap,
                fingertip_depth_m=initial_depth)
            hold_q = current["pose"]["quaternion_xyzw"]
            q0 = hold_q

        with ctx.stage("withdraw from the blocked handle approach") as stage:
            target_depth = min(-0.055,initial_depth-0.025)
            window = initial_depth
            reached = False
            for index in range(16):
                before=measure()
                guard(before)
                tip,jaw_axis,width,lowest=fingertips(before)
                depth=float(np.dot(tip-contact_surface(before),inward))
                if depth < target_depth+0.004:
                    reached=True
                    break
                act_pose(before["eef"]+max(-0.012,target_depth-depth)*inward,hold_q,False,steps=2)
                if done():
                    return "SUBTASK_DONE"
                after=measure()
                guard(after)
                new_tip,_,_,_=fingertips(after)
                new_depth=float(np.dot(new_tip-contact_surface(after),inward))
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
            tip,_,_,lowest=fingertips(current)
            outside_depth=float(np.dot(tip-contact_surface(current),inward))
            window_error=abs(lower_goal+float(current["oven"][2]-baseline["oven"][2])-lowest)
            reached=False
            for index in range(20):
                before=measure()
                guard(before)
                tip,jaw_axis,width,lowest=fingertips(before)
                depth=float(np.dot(tip-contact_surface(before),inward))
                goal_height=lower_goal+float(before["oven"][2]-baseline["oven"][2])
                error=goal_height-lowest
                if depth > -0.040:
                    return unresolved("Lost outside handle clearance during height correction")
                if abs(error)<height_tolerance and abs(depth-outside_depth)<0.005:
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
                    next_error=abs(lower_goal+float(after["oven"][2]-baseline["oven"][2])-new_low)
                    if next_error>height_tolerance and window_error-next_error<0.0002:
                        return unresolved("Gap-height alignment stopped reducing measured height error")
                    window_error=next_error
            if not reached:
                return unresolved("Lower finger gap-height correction remains incomplete")

        with ctx.stage("insert with separate height correction and inward advance") as stage:
            current=measure()
            tip,jaw_axis,width,lowest=fingertips(current)
            window_depth=float(np.dot(tip-contact_surface(current),inward))
            open_joints=np.abs(np.asarray(unpack(ctx.observe())["state"][-2:],float))
            reached=False
            advances=0
            correction_count=0
            correction_anchor=None
            correction_window=None
            for index in range(120):
                before=measure()
                guard(before)
                tip,jaw_axis,width,lowest=fingertips(before)
                depth=float(np.dot(tip-contact_surface(before),inward))
                shift=float(before["oven"][2]-baseline["oven"][2])
                goal_height=lower_goal+shift
                error=goal_height-lowest
                if abs(error)>0.003 or lowest < gap_bottom+shift+clearance or lowest > gap_top+shift-clearance or abs(float(np.dot(jaw_axis,up)))<0.975:
                    return unresolved("Lower finger left the measured gap or vertical jaw envelope")
                if np.max(open_joints-np.abs(np.asarray(unpack(ctx.observe())["state"][-2:],float)))>0.004:
                    return unresolved("A finger became constrained before measured insertion")
                if abs(error)>height_tolerance:
                    if correction_anchor is None:
                        correction_anchor=before["eef"].copy()
                        correction_window=abs(error)
                        correction_count=0
                    if np.linalg.norm((before["eef"]-correction_anchor)[:2])>0.005:
                        return unresolved("Height-only correction lost its horizontal anchor")
                    delta=correction_anchor-before["eef"]
                    delta[2]=error
                    act_pose(before["eef"]+delta,q0,False,steps=2)
                    if done():
                        return "SUBTASK_DONE"
                    after=measure()
                    guard(after)
                    _,_,_,after_low=fingertips(after)
                    next_error=abs(lower_goal+float(after["oven"][2]-baseline["oven"][2])-after_low)
                    correction_count+=1
                    stage.record(step=after["step"],action="height_only",height_error_m=next_error)
                    if correction_count%4==0 and next_error>height_tolerance:
                        if correction_window-next_error<0.0002:
                            return unresolved("Height-only insertion correction stopped reducing measured error")
                        correction_window=next_error
                    if correction_count>=12 and next_error>height_tolerance:
                        return unresolved("Height-only insertion correction reached its bound")
                    continue
                correction_anchor=None
                if depth>=0.003:
                    reached=True
                    break
                # One short inward step, then fresh fingertip depth and height.
                delta=min(0.004,0.006-depth)*inward
                act_pose(before["eef"]+delta,q0,False,steps=1)
                if done():
                    return "SUBTASK_DONE"
                after=measure()
                new_tip,_,_,new_low=fingertips(after)
                new_depth=float(np.dot(new_tip-contact_surface(after),inward))
                advances+=1
                stage.record(step=after["step"],action="inward",tip_depth_m=new_depth,
                    lower_tip_height_m=new_low,finger_positions=unpack(ctx.observe())["state"][-2:])
                if after["eef"][2]<before["eef"][2]-0.004:
                    return unresolved("Higher insertion caused opposing downward hand motion")
                if advances%8==0:
                    if new_depth-window_depth<0.002:
                        return unresolved("Separate-height approach stopped producing measured insertion")
                    window_depth=new_depth
            if not reached:
                return unresolved("Separate-height approach did not reach the bounded grasp position")
            current=measure()
            tip,jaw_axis,width,lowest=fingertips(current)
            contact=contact_surface(current)
            depth=float(np.dot(tip-contact,inward))
            highest=2.0*tip[2]-lowest
            if not 0.0025<depth<0.018 or not 0.008<contact[2]-lowest<0.035 or highest-contact[2]<0.015:
                return unresolved("Measured fingers do not straddle and overlap the outer lip")
            final_shift=float(current["oven"][2]-baseline["oven"][2])
            if lowest<gap_bottom+final_shift+clearance or lowest>gap_top+final_shift-clearance or abs(float(np.dot(jaw_axis,up)))<0.975:
                return unresolved("Final measured grasp pose lost the gap or vertical jaw clearance")
            stage.record(step=current["step"],pregrasp_tip_m=tip.tolist(),lower_tip_height_m=lowest,
                outer_lip_overlap_m=depth,contact_lip_m=contact.tolist())

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
            for index in range(180):
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
                act_pose(before["eef"]+delta,desired.as_quat(),True,steps=2)
                if done():
                    return "SUBTASK_DONE"
                after = measure()
                guard(after)
                new_angle = angle_of(after)
                stage.record(step=after["step"],door_angle_rad=new_angle,rim_m=after["rim"].tolist())
                if index%5 == 4:
                    if new_angle-window_angle < 0.018:
                        return unresolved("Closing stopped producing measurable hinge rotation.")
                    window_angle = new_angle
        return unresolved("Revision bound reached before whole-task closure verification.")
    except (ValueError,KeyError,TypeError) as error:
        return unresolved("Measured-jaw approach or closing prerequisite failed: "+str(error))
