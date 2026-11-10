def run(ctx):
    import json
    import numpy as np
    from scipy.spatial.transform import Rotation

    goal = ctx.completion_condition
    limit = 400
    params = ctx.parameters
    camera = params["camera"]  # Fresh selected view; never reuse historical surface coordinates.

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
        data = ctx.read_observation(camera)
        rgb = np.asarray(data["rgb"], dtype=float)
        if rgb.shape != (256, 256, 3):
            raise ValueError("Unexpected image shape")
        return rgb * 255.0 if np.max(rgb) <= 1.0 else rgb

    def metric(label, box, step, zspread=0.04, camera=camera):
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
        if item["label"] != "fixed oven roof reference" and np.std(item["template"]) < 2.0:
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
        data=ctx.read_observation(camera)
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
                        # Fixed selected pixels, fresh depth and displacement guards
                        # remain mandatory; surrounding arm appearance is not a veto.
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
        obs = unpack(ctx.observe())
        if obs["step"] != current["step"]:
            raise ValueError("Finger geometry and pose are not contemporaneous")
        kin = obs.get("controller",{}).get("gripper_kinematics",{})
        geom = kin.get("fingerpad_geometry",{})
        if geom.get("available") is not True or geom.get("step") != current["step"] or geom.get("frame") != "eef_control_site" or geom.get("source") != "robot_model_and_measured_joint_kinematics":
            raise ValueError("Current measured fingerpad boxes unavailable")
        pads = geom.get("pads",[])
        if len(pads) != 2:
            raise ValueError("Expected two measured fingerpads")
        rotation = Rotation.from_quat(current["pose"]["quaternion_xyzw"])
        pads = sorted(pads, key=lambda pad: pad["center_eef"][0], reverse=True)
        points, lower = [], []
        for pad in pads:
            center = np.asarray(pad["center_eef"],float)
            orient = np.asarray(pad["rotation_eef"],float)
            half = np.asarray(pad["half_extents_m"],float)
            if center.shape!=(3,) or orient.shape!=(3,3) or half.shape!=(3,) or not np.all(np.isfinite(center)) or not np.all(np.isfinite(orient)) or not np.all(np.isfinite(half)) or np.any(half<=0) or np.max(half)>0.020:
                raise ValueError("Invalid measured fingerpad geometry")
            point = current["eef"] + rotation.apply(center)
            extent_z = float(np.sum(np.abs((rotation.as_matrix()@orient)[2])*half))
            points.append(point)
            lower.append(float(point[2]-extent_z))
        vector=points[0]-points[1]
        width=float(np.linalg.norm(vector))
        if not 0.035<width<0.095:
            raise ValueError("Measured finger separation is outside approach bounds")
        midpoint=0.5*(points[0]+points[1])
        local_axis=rotation.inv().apply(vector/width)
        if local_axis[0]<0.98 or np.linalg.norm(rotation.inv().apply(midpoint-current["eef"]))>0.030:
            raise ValueError("Measured pads disagree with control-site jaw geometry")
        # Robot geometry only. Independent rim lift still establishes engagement.
        return midpoint,vector/width,width,min(lower)

    try:
        with ctx.stage("remeasure actual fingertips and lowered door") as stage:
            baseline = measure()
            second = metric("second front rim section", params["second_box"], baseline["step"], 0.010, camera)
            rear = metric("hinge-side rim", params["rear_box"], baseline["step"], 0.010)
            counter = metric("countertop beside the door", params["counter_box"], baseline["step"], 0.010)
            up = np.array([0.,0.,1.])
            side = baseline["rim"]-second
            side[2] = 0.
            span = float(np.linalg.norm(side))
            if not 0.05 < span < 0.25:
                return unresolved("Two current rim selections do not define a usable edge")
            side /= span
            # Orient the selected edge away from the current hand; preserve the
            # same signed lateral clearance envelope across mirrored views.
            if float(np.dot(baseline["eef"]-baseline["rim"],side)) > 0:
                side = -side
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
            if jaw_local.shape!=(3,) or not np.all(np.isfinite(jaw_local)) or not 0.99<float(np.linalg.norm(jaw_local))<1.01 or float(jaw_local[0])>-0.98:
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

        # Preserve completed release across revisions using current joint and pad evidence.
        release_state=np.asarray(unpack(ctx.observe())["state"][-2:],float)
        if release_state[0]>0.032 and release_state[1]<-0.032:
            current=measure()
            _,_,release_width,_=fingertips(current)
            if not 0.067<release_width<0.095:
                return unresolved("Open joint state disagrees with current measured pad width")
            if float(np.dot(current["eef"]-current["rim"],inward))>-0.035:
                return unresolved("Already released hand lacks measured exterior clearance")
            with ctx.stage("confirm current release without repeating lateral retreat") as stage:
                stage.record(step=current["step"],finger_positions=release_state.tolist(),
                             measured_pad_width_m=release_width)
        else:
            with ctx.stage("reverse lateral approach to release the constrained finger") as stage:
                origin=measure()
                hold=origin["pose"]["quaternion_xyzw"]
                origin_depth=float(np.dot(origin["eef"]-origin["rim"],inward))
                if origin_depth>-0.035:
                    return unresolved("Lateral release requires measured exterior rim clearance")
                target=origin["eef"]-0.080*side
                released=False
                window=origin["eef"].copy()
                for index in range(20):
                    before=measure()
                    if np.linalg.norm(before["oven"]-baseline["oven"])>0.015:
                        return unresolved("Oven reference moved during lateral release")
                    if max(abs(float(before[key][2]-baseline[key][2])) for key in ("rim","rim2"))>0.006:
                        return unresolved("Door moved during unverified lateral release")
                    depth=float(np.dot(before["eef"]-before["rim"],inward))
                    travel=float(np.dot(origin["eef"]-before["eef"],side))
                    if abs(depth-origin_depth)>0.015 or abs(float(before["eef"][2]-origin["eef"][2]))>0.015 or not -0.010<travel<0.090:
                        return unresolved("Lateral release left its measured motion corridor")
                    qpos=np.asarray(unpack(ctx.observe())["state"][-2:],float)
                    if travel>=0.020 and qpos[0]>0.032 and qpos[1]<-0.032:
                        _,_,width,_=fingertips(before)
                        if not 0.067<width<0.095:
                            return unresolved("Lateral release joints and pad width disagree")
                        released=True
                        stage.record(step=before["step"],reverse_travel_m=travel,finger_positions=qpos.tolist())
                        break
                    if travel>=0.078:
                        return unresolved("Bounded lateral retreat did not release both fingers")
                    act_pose(target,hold,False,steps=2)
                    if done():
                        return "SUBTASK_DONE"
                    after=measure()
                    stage.record(step=after["step"],position_m=after["eef"].tolist(),
                                 finger_positions=unpack(ctx.observe())["state"][-2:])
                    if index%4==3:
                        if float(np.dot(window-after["eef"],side))<0.003:
                            return unresolved("Reverse lateral release stopped making measured progress")
                        window=after["eef"].copy()
                if not released:
                    return unresolved("Lateral release reached its action bound")
        withdrawal_anchor=measure()["eef"].copy()
        initial_lateral = float(np.dot(withdrawal_anchor-measure()["rim"],side))
        approach_mode = "withdraw"

        def guard(current):
            if np.linalg.norm(current["oven"]-baseline["oven"]) > 0.015:
                raise ValueError("Oven reference displaced beyond the experiment bound")
            if abs(float(current["rim"][2]-current["rim2"][2]))>0.012 or abs(float(np.dot(current["rim"]-current["rim2"],inward)))>0.012:
                raise ValueError("Separate measured rim sections disagree")
            lateral = float(np.dot(current["eef"]-current["rim"],side))
            if approach_mode == "withdraw":
                if abs(lateral-initial_lateral)>0.015 or abs(float(current["eef"][2]-withdrawal_anchor[2]))>0.015:
                    raise ValueError("Exterior withdrawal left its measured initial lateral/height corridor")
            elif approach_mode == "center":
                if not min(initial_lateral,-0.075)-0.015 < lateral < max(initial_lateral,-0.075)+0.015:
                    raise ValueError("Exterior centering left the measured start-to-target corridor")
                if float(np.dot(current["eef"]-current["rim"],inward))>-0.035:
                    raise ValueError("Exterior centering lost rim clearance")
            elif not -0.28 < lateral < -0.03:
                raise ValueError("Hand moved outside the selected interior rim section")

        def pose_error(current, goal_position, goal_q):
            distance = float(np.linalg.norm(goal_position-current["eef"]))
            rotation = Rotation.from_quat(current["pose"]["quaternion_xyzw"])
            angle = float((Rotation.from_quat(goal_q)*rotation.inv()).magnitude())
            return distance, angle

        with ctx.stage("withdraw outside the rim before changing the measured jaw axis") as stage:
            hold_q = baseline["pose"]["quaternion_xyzw"]
            # Withdraw at current height before lifting or shifting laterally.
            current=measure()
            tip,_,_,_=fingertips(current)
            depth = float(np.dot(tip-current["rim"],inward))
            outside_position = current["eef"]+(-params["outside_clearance_m"]-depth)*inward
            stage.record(outside_target_m=outside_position.tolist())
            window = baseline["eef"].copy()
            reached = False
            for index in range(10):
                before = measure()
                guard(before)
                distance, angle = pose_error(before,outside_position,hold_q)
                if distance < 0.007:
                    reached = True
                    break
                act_pose(outside_position,hold_q,False)
                if done():
                    return "SUBTASK_DONE"
                after = measure()
                guard(after)
                stage.record(step=after["step"],remaining_m=float(np.linalg.norm(outside_position-after["eef"])))
                if index%3 == 2:
                    if np.linalg.norm(after["eef"]-window) < 0.003:
                        return unresolved("Outside withdrawal stopped producing measured translation")
                    window = after["eef"].copy()
            if not reached:
                after = measure()
                if np.linalg.norm(outside_position-after["eef"]) > 0.009:
                    return unresolved("Outside withdrawal did not reach its bounded clearance")

        with ctx.stage("confirm both fingers released after exterior withdrawal") as stage:
            released=False
            for index in range(5):
                current=measure()
                guard(current)
                if float(np.dot(current["eef"]-current["rim"],inward))>-0.040:
                    return unresolved("Outside release lacks measured rim clearance")
                qpos=np.asarray(unpack(ctx.observe())["state"][-2:],float)
                if qpos[0]>0.032 and qpos[1]<-0.032:
                    tip,axis_now,width,lowest=fingertips(current)
                    if not 0.067<width<0.095:
                        return unresolved("Released joint state lacks matching open fingertip depth")
                    if current["rim"][2]+0.020-0.5*width<counter[2]+0.009:
                        return unresolved("Measured open fingers lack counter clearance for the planned descent")
                    stage.record(step=current["step"],finger_positions=qpos.tolist(),visible_width_m=width)
                    released=True
                    break
                act_pose(current["eef"],hold_q,False,steps=2)
                if done():
                    return "SUBTASK_DONE"
            if not released:
                return unresolved("Both fingers did not release within the exterior opening bound")

        approach_mode = "interior"
        with ctx.stage("restore upper clearance after verified exterior release") as stage:
            current=measure()
            target=current["eef"].copy()
            target[2]=max(float(target[2]),float(current["rim"][2]+0.069))
            window=current["eef"].copy()
            reached=False
            for index in range(12):
                before=measure()
                guard(before)
                qpos=np.asarray(unpack(ctx.observe())["state"][-2:],float)
                if qpos[0]<=0.032 or qpos[1]>=-0.032:
                    return unresolved("A finger became constrained during upper clearance restoration")
                if float(np.dot(before["eef"]-before["rim"],inward))>-0.040:
                    return unresolved("Upper clearance restoration lost exterior distance")
                if np.linalg.norm(before["eef"]-target)<0.003:
                    reached=True
                    break
                act_pose(target,hold_q,False,steps=2)
                if done():
                    return "SUBTASK_DONE"
                after=measure()
                stage.record(step=after["step"],height_m=float(after["eef"][2]))
                if index%4==3:
                    if float(after["eef"][2]-window[2])<0.002:
                        return unresolved("Upper clearance restoration stopped making progress")
                    window=after["eef"].copy()
            if not reached and np.linalg.norm(measure()["eef"]-target)>0.004:
                return unresolved("Upper clearance restoration reached its action bound")

        with ctx.stage("verify achieved upper clearance at the current lateral section") as stage:
            current=measure()
            tip,jaw_axis,width,lowest=fingertips(current)
            lateral=float(np.dot(current["eef"]-current["rim"],side))
            depth=float(np.dot(tip-current["rim"],inward))
            if not -0.28<lateral<-0.03 or not -0.095<depth<-0.035:
                return unresolved("Current achieved position lacks the existing lateral/descent envelope")
            if current["eef"][2]<current["rim"][2]+0.065:
                return unresolved("Current upper clearance is below the original height target")
            obs=unpack(ctx.observe())
            geom=obs["controller"]["gripper_kinematics"]["fingerpad_geometry"]
            sweep=max(float(np.linalg.norm(np.asarray(pad["center_eef"],float)))+
                      float(np.linalg.norm(np.asarray(pad["half_extents_m"],float)))
                      for pad in geom["pads"])
            if current["eef"][2]-sweep<current["rim"][2]+0.009:
                return unresolved("Measured pad rotation sweep lacks vertical rim clearance")
            if float(np.dot(current["eef"]-current["rim"],inward))+sweep>-0.009:
                return unresolved("Measured pad rotation sweep lacks exterior rim clearance")
            qpos=np.asarray(obs["state"][-2:],float)
            if qpos[0]<=0.032 or qpos[1]>=-0.032:
                return unresolved("A finger remains constrained before wrist alignment")
            stage.record(step=current["step"],lateral_m=lateral,outside_tip_depth_m=depth,
                         pad_sweep_radius_m=sweep,lowest_pad_m=lowest,finger_positions=qpos.tolist())
        approach_mode = "interior"
        guard(measure())
        with ctx.stage("rotate the measured local-x jaw axis vertically outside the door") as stage:
            current = measure()
            anchor = current["eef"].copy()
            reached = False
            last_angle = None
            for index in range(18):
                before = measure()
                guard(before)
                qpos=np.asarray(unpack(ctx.observe())["state"][-2:],float)
                if qpos[0]<=0.032 or qpos[1]>=-0.032:
                    return unresolved("A finger became constrained during exterior wrist alignment")
                distance, angle = pose_error(before,anchor,q0)
                if angle < 0.065 and distance < 0.014:
                    reached = True
                    break
                if float(np.dot(before["eef"]-before["rim"],inward)) > -0.032:
                    return unresolved("Outside rotation lost its rim clearance")
                if before["eef"][2] < baseline["rim"][2]+0.032:
                    return unresolved("Outside rotation lost its vertical clearance")
                act_pose(anchor,q0,False)
                if done():
                    return "SUBTASK_DONE"
                after = measure()
                distance, new_angle = pose_error(after,anchor,q0)
                stage.record(step=after["step"],orientation_error_rad=new_angle,anchor_error_m=distance)
                if distance > 0.035:
                    return unresolved("Orientation change displaced the hand beyond its outside envelope")
                if index%4 == 3:
                    if last_angle is not None and last_angle-new_angle < 0.035:
                        return unresolved("Vertical jaw reorientation stopped making measured progress")
                    last_angle = new_angle
            if not reached:
                current = measure()
                if pose_error(current,anchor,q0)[1] > 0.075:
                    return unresolved("Jaw reorientation remains incomplete at its stage bound")
            current = measure()
            midpoint,jaw_axis,width,lowest = fingertips(current)
            if abs(float(np.dot(jaw_axis,up))) < 0.985:
                return unresolved("Fresh fingertip depth does not confirm vertical jaw separation")
            stage.record(step=current["step"],observed_jaw_axis=jaw_axis.tolist(),
                observed_fingertip_midpoint_m=midpoint.tolist())

        with ctx.stage("reconcile measured fingertip standoff after wrist alignment") as stage:
            origin=measure()
            origin_tip,_,_,_=fingertips(origin)
            target_depth=-params["outside_clearance_m"]
            initial_depth=float(np.dot(origin_tip-origin["rim"],inward))
            if abs(initial_depth-target_depth)>0.020:
                return unresolved("Post-rotation standoff correction exceeds its bounded translation")
            window_error=abs(initial_depth-target_depth)
            reached=False
            for index in range(10):
                before=measure()
                guard(before)
                tip,jaw_axis,width,lowest=fingertips(before)
                depth=float(np.dot(tip-before["rim"],inward))
                error=target_depth-depth
                obs=unpack(ctx.observe())
                qpos=np.asarray(obs["state"][-2:],float)
                if qpos[0]<=0.032 or qpos[1]>=-0.032 or abs(float(np.dot(jaw_axis,up)))<0.985:
                    return unresolved("Post-rotation correction lost open vertical fingers")
                if abs(float(before["eef"][2]-origin["eef"][2]))>0.015 or abs(float(np.dot(before["eef"]-origin["eef"],side)))>0.015:
                    return unresolved("Post-rotation correction left its lateral or height corridor")
                geom=obs["controller"]["gripper_kinematics"]["fingerpad_geometry"]
                sweep=max(float(np.linalg.norm(np.asarray(pad["center_eef"],float)))+
                          float(np.linalg.norm(np.asarray(pad["half_extents_m"],float)))
                          for pad in geom["pads"])
                if before["eef"][2]-sweep<before["rim"][2]+0.009:
                    return unresolved("Post-rotation correction lacks upper pad sweep clearance")
                if float(np.dot(before["eef"]-before["rim"],inward))+sweep>-0.009:
                    return unresolved("Post-rotation correction lacks exterior pad sweep clearance")
                if max(abs(float(before[key][2]-origin[key][2])) for key in ("rim","rim2"))>0.006:
                    return unresolved("Door moved during exterior standoff correction")
                stage.record(step=before["step"],measured_tip_depth_m=depth,standoff_error_m=error)
                if abs(error)<0.003:
                    reached=True
                    break
                delta=float(np.clip(error,-0.008,0.008))*inward
                if float(np.dot(before["eef"]+delta-before["rim"],inward))+sweep>-0.009:
                    return unresolved("Proposed standoff correction loses exterior sweep clearance")
                act_pose(before["eef"]+delta,q0,False,steps=2)
                if done():
                    return "SUBTASK_DONE"
                after=measure()
                new_tip,_,_,_=fingertips(after)
                new_error=abs(target_depth-float(np.dot(new_tip-after["rim"],inward)))
                if index%3==2:
                    if window_error-new_error<0.002:
                        return unresolved("Post-rotation standoff correction stopped making measured progress")
                    window_error=new_error
            if not reached:
                return unresolved("Post-rotation standoff correction reached its action bound")

        with ctx.stage("lower open vertical fingers outside the rim using measured fingertip centers") as stage:
            current = measure()
            tip,_,_,_ = fingertips(current)
            outside_depth = float(np.dot(tip-current["rim"],inward))
            if not -0.095 < outside_depth < -0.035:
                return unresolved("Actual fingertips are not clear of the rim for descent")
            window = tip.copy()
            reached = False
            for index in range(14):
                before = measure()
                guard(before)
                tip,jaw_axis,width,lowest = fingertips(before)
                depth = float(np.dot(tip-before["rim"],inward))
                height_error = float(before["rim"][2]+0.020-tip[2])
                delta = (outside_depth-depth)*inward+height_error*up
                if abs(height_error)<0.004 and abs(depth-outside_depth)<0.007:
                    reached = True
                    break
                if lowest < counter[2]+0.008:
                    return unresolved("Observed lower fingertip reached the countertop-clearance bound")
                act_pose(before["eef"]+delta,q0,False)
                if done():
                    return "SUBTASK_DONE"
                after = measure()
                tip_after,_,_,lowest_after = fingertips(after)
                stage.record(step=after["step"],tip_height_m=float(tip_after[2]),
                    rim_height_m=float(after["rim"][2]),lowest_fingertip_m=float(lowest_after))
                if index%3 == 2:
                    if np.linalg.norm(tip_after-window) < 0.002:
                        return unresolved("Outside fingertip descent stopped making measured progress")
                    window = tip_after.copy()
            if not reached:
                current = measure()
                tip,_,_,_ = fingertips(current)
                if abs(tip[2]-current["rim"][2]-0.020)>0.006:
                    return unresolved("Outside fingertip height alignment remains incomplete")

        with ctx.stage("insert the measured vertical finger pair across the front rim") as stage:
            window_depth = None
            reached = False
            for index in range(28):
                before = measure()
                guard(before)
                tip,jaw_axis,width,lowest = fingertips(before)
                depth = float(np.dot(tip-before["rim"],inward))
                height_error = float(before["rim"][2]+0.020-tip[2])
                if abs(float(np.dot(jaw_axis,up))) < 0.975 or abs(height_error)>0.012:
                    return unresolved("Current finger orientation or height left the insertion envelope")
                if lowest < counter[2]+0.007:
                    return unresolved("Lower fingertip has insufficient measured counter clearance")
                if depth >= 0.003:
                    reached = True
                    break
                open_before = np.abs(np.asarray(unpack(ctx.observe())["state"][-2:],float))
                delta = min(0.008,0.006-depth)*inward+height_error*up
                act_pose(before["eef"]+delta,q0,False,steps=2)
                if done():
                    return "SUBTASK_DONE"
                after = measure()
                tip_after,_,_,_ = fingertips(after)
                new_depth = float(np.dot(tip_after-after["rim"],inward))
                open_after = np.abs(np.asarray(unpack(ctx.observe())["state"][-2:],float))
                if np.max(open_before-open_after)>0.004 or after["eef"][2]<before["eef"][2]-0.004:
                    return unresolved("Insertion constrained a finger or displaced the hand downward")
                stage.record(step=after["step"],fingertip_insertion_m=new_depth,
                    tip_height_error_m=float(tip_after[2]-after["rim"][2]))
                if index%3 == 2:
                    if window_depth is not None and new_depth-window_depth < 0.003:
                        return unresolved("Rim insertion stopped producing measured advance")
                    window_depth = new_depth
            if not reached:
                current = measure()
                tip,_,_,_ = fingertips(current)
                if float(np.dot(tip-current["rim"],inward))<0.003:
                    return unresolved("Rim insertion remains incomplete at its stage bound")
            current = measure()
            tip,jaw_axis,width,lowest = fingertips(current)
            depth=float(np.dot(tip-current["rim"],inward))
            highest=2.0*tip[2]-lowest
            if not 0.0025 < depth < 0.018 or not 0.008 < current["rim"][2]-lowest < 0.035 or highest-current["rim"][2]<0.015:
                return unresolved("Measured fingers do not straddle and overlap the current rim")
            stage.record(step=current["step"],pregrasp_tip_m=tip.tolist(),
                observed_jaw_axis=jaw_axis.tolist(),finger_positions=unpack(ctx.observe())["state"][-2:])

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
