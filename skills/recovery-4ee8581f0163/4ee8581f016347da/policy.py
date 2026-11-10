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

    frame_cache = {}
    def read_frame(step=None):
        if step is None:
            return ctx.read_observation("right")
        if frame_cache.get("step") != step:
            frame_cache.clear()
            frame_cache.update(step=step, data=ctx.read_observation("right", step=step))
        return frame_cache["data"]

    def rgb_now(step=None):
        data = read_frame(step)
        rgb = np.asarray(data["rgb"], dtype=float)
        if rgb.shape != (256, 256, 3):
            raise ValueError("Unexpected image shape")
        return rgb * 255.0 if np.max(rgb) <= 1.0 else rgb

    def metric(label, box, step, zspread=0.04, camera="right"):
        if step < start["step"]:
            return historical_surface({"observation_step":step}, box, zspread)
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

    def initialize_selections(settings, step=None):
        selections = {
            "rim": {"label": "exposed outer door silver rim",
                    "context": settings["rim_context"], "patch": settings["rim_box"]},
            "oven": {"label": "fixed oven roof reference",
                     "context": settings["oven_context"], "patch": settings["oven_box"]}
        }
        initial_rgb = rgb_now(step)
        for item in selections.values():
            a,b,c,d = item["context"]
            item["template"] = initial_rgb[b:d,a:c].copy()
            item["offset"] = (0,0)
            if np.std(item["template"]) < 2.0:
                raise ValueError("Selected visual context lacks sufficient texture.")

        # Paired rim tracking uses current calibrated depth; fixed roof pixels prevent
        # the planar-template drift observed in the imported candidate.
        selections["rim2"] = {"label":"second exposed outer rim section",
            "context":settings["second_context"],"patch":settings["second_track_box"],"offset":(0,0)}
        a,b,c,d = settings["second_context"]
        selections["rim2"]["template"] = initial_rgb[b:d,a:c].copy()
        for key in ("rim","rim2"):
            x0,y0,x1,y1=selections[key]["patch"]
            selections[key]["color"]=np.median(initial_rgb[y0:y1,x0:x1].reshape(-1,3),axis=0)
        return selections

    selections = initialize_selections(params)
    previous = {}
    tracking_model = {}
    cache = {}
    def measure(history_step=None):
        obs = unpack(ctx.observe())
        if history_step is not None:
            if not 0 <= history_step < start["step"]:
                raise ValueError("Historical replay step outside recorded interval")
            obs = dict(obs, step=history_step)
        if cache.get("step") == obs["step"]:
            return cache.copy()
        data=read_frame(history_step)
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
            if key!="oven" and "pivot" in tracking_model:
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
                        stable_fraction=float(np.mean(np.max(np.abs(context-item["template"]),axis=2)<35.0))
                        if err<=450.0 or stable_fraction>=0.85:
                            candidates.append((err,dx,dy,None))
                        else:
                            ctx.note(facts=["Fixed roof context rejected: mse=%.3f, stable_fraction=%.3f"%(err,stable_fraction)])
                        continue
                    p=surface(box)
                    if p is None or not consistent(key,p,result):
                        continue
                    if key=="rim2" and "pair_delta" in tracking_model:
                        separation=p-result["rim"]
                        if abs(float(separation[2]))>0.012 or abs(float(np.dot(separation,tracking_model["inward"])))>0.012:
                            continue
                        if "pair_delta" in tracking_model and abs(float(np.dot(separation-tracking_model["pair_delta"],tracking_model["side"])))>0.012:
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
        roof2=metric("independent fixed roof surface",params["roof_second_box"],obs["step"],0.008)
        if abs(float(roof2[2]-result["oven"][2]))>0.008:
            raise ValueError("Independent roof surfaces disagree in height")
        if "roof2" in previous:
            if np.linalg.norm(roof2-previous["roof2"])>0.015:
                raise ValueError("Independent roof surface moved beyond temporal bound")
        if "roof_pair" not in tracking_model:
            tracking_model["roof_pair"]=roof2-result["oven"]
            tracking_model["roof2_initial"]=roof2.copy()
        if np.linalg.norm(roof2-result["oven"]-tracking_model["roof_pair"])>0.008 or np.linalg.norm(roof2-tracking_model["roof2_initial"])>0.015:
            raise ValueError("Independent roof rigidity or displacement check failed")
        previous["roof2"]=roof2.copy()
        result["roof2"]=roof2
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

    def historical_surface(sample, box, max_zspread):
        data=read_frame(sample["observation_step"])
        cal=data.get("metadata",{}).get("calibration",{})
        if cal.get("step")!=sample["observation_step"] or cal.get("alignment_verified") is not True or cal.get("frame")!="optical_to_osc_arm_base":
            raise ValueError("Historical arc calibration unavailable")
        depth=np.asarray(data.get("depth_m"),float)
        k=np.asarray(cal["intrinsics"],float)
        transform=np.asarray(cal["camera_to_base"],float)
        if depth.shape!=(256,256) or k.shape!=(3,3) or transform.shape!=(4,4) or not np.all(np.isfinite(k)) or not np.all(np.isfinite(transform)) or min(k[0,0],k[1,1])<=0:
            raise ValueError("Invalid historical calibration")
        x0,y0,x1,y1=box
        z=depth[y0:y1,x0:x1]
        if z.size<5 or not np.all(np.isfinite(z)) or np.any(z<=0.02) or np.percentile(z,75)-np.percentile(z,25)>0.045:
            raise ValueError("Historical arc patch has unreliable depth")
        vv,uu=np.mgrid[y0:y1,x0:x1]
        optical=np.stack(((uu-k[0,2])*z/k[0,0],(vv-k[1,2])*z/k[1,1],z),axis=-1).reshape(-1,3)
        cloud=optical@transform[:3,:3].T+transform[:3,3]
        if np.ptp(cloud[:,2])>max_zspread:
            raise ValueError("Historical arc patch contains mixed surfaces")
        return np.median(cloud,axis=0)

    try:
        with ctx.stage("measure current door frame and calibrate closure arc") as stage:
            baseline=measure()
            up=np.array([0.,0.,1.])
            edge=baseline["rim"]-baseline["rim2"]
            edge[2]=0.
            span=float(np.linalg.norm(edge))
            if not 0.05<span<0.18 or abs(float(baseline["rim"][2]-baseline["rim2"][2]))>0.006:
                return unresolved("Current raised rim sections do not define a consistent edge")
            edge/=span
            inward=np.cross(edge,up)
            if np.dot(inward,baseline["oven"]-baseline["rim"])<0:
                inward=-inward
            side=np.cross(inward,up)
            history=params["lowered_reference"]
            if not history["observation_step"]<baseline["step"]:
                return unresolved("Lowered-door reference must precede current observation")
            old_rim=historical_surface(history,history["rim_box"],0.010)
            old_roof=historical_surface(history,params["oven_box"],0.008)
            if np.linalg.norm(old_rim-np.asarray(history["rim_m"]))>0.006 or np.linalg.norm(old_roof-baseline["oven"])>0.015:
                return unresolved("Lowered-door reference does not reproduce recorded geometry")
            old_rim=old_rim-old_roof+baseline["oven"]
            selection=unpack(ctx.ground_target(target="far visible door side frame, with background separated by calibrated depth",
                camera="right",observation_step=baseline["step"],bbox=params["side_frame_box"],confidence=0.85))
            selected_geometry=unpack(ctx.query_geometry(selection["grounding_id"]))
            if selected_geometry.get("reliable") is not True or selected_geometry.get("observation_step")!=baseline["step"] or selected_geometry.get("valid_fraction",0)<0.95:
                return unresolved("Selected frame region lacks valid current depth")
            data=ctx.read_observation("right")
            cal=data.get("metadata",{}).get("calibration",{})
            if cal.get("step")!=baseline["step"] or cal.get("alignment_verified") is not True or cal.get("frame")!="optical_to_osc_arm_base":
                return unresolved("Current frame calibration unavailable")
            depth=np.asarray(data.get("depth_m"),float)
            k=np.asarray(cal["intrinsics"],float)
            transform=np.asarray(cal["camera_to_base"],float)
            if depth.shape!=(256,256) or k.shape!=(3,3) or transform.shape!=(4,4) or not np.all(np.isfinite(k)) or not np.all(np.isfinite(transform)):
                return unresolved("Invalid frame depth calibration")
            x0,y0,x1,y1=params["side_frame_box"]
            z=depth[y0:y1,x0:x1]
            if not np.all(np.isfinite(z)) or np.any(z<=0.02):
                return unresolved("Frame region contains missing depth")
            vv,uu=np.mgrid[y0:y1,x0:x1]
            optical=np.stack(((uu-k[0,2])*z/k[0,0],(vv-k[1,2])*z/k[1,1],z),axis=-1).reshape(-1,3)
            cloud=optical@transform[:3,:3].T+transform[:3,3]
            along=(cloud-baseline["rim"])@inward
            # The fixed-front branch extends above the lowered rim. Separate it
            # by measured inward position, not height alone. Include the visible
            # radial frame down to the measured lowered-rim height; retain the
            # crossbar exclusion and all direction-resolution quality checks.
            front=metric("fixed oven front plane below hinge lip",
                params["fixed_front_box"],baseline["step"],0.010)
            front_along=float(np.dot(front-baseline["rim"],inward))
            if not 0.08<front_along<0.25 or front[2]>old_rim[2]+0.010:
                return unresolved("Independent fixed-front selection is inconsistent with the observed open door")
            mask=(cloud[:,2]>old_rim[2])&(cloud[:,2]<baseline["rim"][2]-0.020)&(along>0.015)&(along<0.18)&(along<front_along-0.020)
            frame=cloud[mask]
            stage.record(step=baseline["step"],fixed_front_m=front.tolist(),
                fixed_front_inward_m=front_along,frame_candidate_count=len(frame),
                spatially_selected_frame_points_m=frame.tolist())
            if len(frame)<20:
                return unresolved("Too few measured side-frame points above the fixed-front depth branch")
            plane=np.column_stack(((frame-baseline["rim"])@inward,frame[:,2]-baseline["rim"][2]))
            mean=np.mean(plane,axis=0)
            _,singular,vectors=np.linalg.svd(plane-mean,full_matrices=False)
            direction=vectors[0]
            if direction[1]<0:
                direction=-direction
            normal=np.array([-direction[1],direction[0]])
            residual=np.abs((plane-mean)@normal)
            length=float(np.ptp((plane-mean)@direction))
            lateral_width=float(np.ptp(frame@side))
            angle=float(np.arctan2(direction[1],-direction[0]))
            stage.record(step=baseline["step"],frame_points=len(frame),frame_span_m=length,
                frame_lateral_width_m=lateral_width,frame_rms_m=float(np.sqrt(np.mean(residual**2))),
                frame_max_residual_m=float(np.max(residual)),frame_direction_plane=direction.tolist(),
                measured_door_angle_rad=angle,selected_frame_points_m=frame.tolist())
            if not 0.080<length<0.25 or lateral_width>0.040 or np.max(residual)>0.006 or np.sqrt(np.mean(residual**2))>0.003 or not 0.30<angle<1.50:
                return unresolved("Current side-frame points do not support a well-resolved door direction")
            radial=direction[0]*inward+direction[1]*up
            chord=baseline["rim"]-old_rim
            chord-=float(np.dot(chord,side))*side
            denominator=2.*float(np.dot(chord,radial))
            if np.linalg.norm(chord)<0.10 or denominator<=0.04:
                return unresolved("Current direction and measured door displacement cannot condition a hinge arc")
            radius=float(np.dot(chord,chord)/denominator)
            pivot=baseline["rim"]-radius*radial
            if not 0.14<radius<0.30 or not 0.02<float(np.dot(pivot-baseline["rim"],inward))<0.28 or not -0.28<float(pivot[2]-baseline["rim"][2])<-0.04:
                return unresolved("Frame-derived hinge arc has implausible measured dimensions")
            secondary_radius=float(np.hypot(np.dot(baseline["rim2"]-pivot,inward),baseline["rim2"][2]-pivot[2]))
            if abs(secondary_radius-radius)>0.008:
                return unresolved("Separate current rim disagrees with frame-derived arc")
            points=[old_rim,baseline["rim"].copy()]
            for anchor in params["execution_anchors"]:
                roof=historical_surface({"observation_step":anchor["step"]},params["oven_box"],0.008)
                if not history["observation_step"]<anchor["step"]<baseline["step"] or np.linalg.norm(roof-baseline["oven"])>0.015:
                    return unresolved("Independent execution anchor is outside calibrated interval")
                point=np.asarray(anchor["rim_m"])-roof+baseline["oven"]
                v=point-pivot
                arc_error=abs(float(np.hypot(np.dot(v,inward),v[2]))-radius)
                stage.record(anchor_step=anchor["step"],independent_anchor_error_m=arc_error)
                if arc_error>0.008:
                    return unresolved("Independent executed rim measurement disagrees with frame-derived arc")
                points.append(point)
            angles=np.array([np.arctan2((point-pivot)[2],-np.dot(point-pivot,inward)) for point in points])
            axis=np.cross(up,inward)
            joints=np.abs(np.asarray(unpack(ctx.observe())["state"][-2:],float))
            if not 0.004<float(np.sum(joints))<0.050:
                return unresolved("Measured finger opening is incompatible with the continuing contact experiment")
            if abs(float(baseline["eef"][2]-baseline["rim"][2]))>0.040 or abs(float(np.dot(baseline["eef"]-baseline["rim"],inward)))>0.065:
                return unresolved("Measured hand is outside the raised-rim contact envelope")
            tracking_model.update(pivot=pivot,radius=radius,side=side,inward=inward,
                oven=baseline["oven"].copy(),rim=baseline["rim"].copy(),rim2=baseline["rim2"].copy(),
                pair_delta=baseline["rim2"]-baseline["rim"])
            stage.record(step=baseline["step"],rim_m=baseline["rim"].tolist(),
                rim2_m=baseline["rim2"].tolist(),observed_path_m=[p.tolist() for p in points],
                pivot_m=pivot.tolist(),radius_m=radius,inward=inward.tolist(),
                calibration_angles_rad=angles.tolist(),secondary_radius_m=secondary_radius,
                finger_positions=joints.tolist(),eef_m=baseline["eef"].tolist())
            ctx.note(facts=["Current calibrated side-frame direction, remeasured lowered rim, paired current rim and independent executed rim anchors support the closure arc."],
                hypotheses=["Temporal rim-template replay was not a sufficiently stable material-point track. Direct current frame geometry supports this distinct continuation only while independent object progress and task_success checks hold."])

        def guard(current):
            if np.linalg.norm(current["oven"]-baseline["oven"])>0.015:
                raise ValueError("Oven reference displaced beyond the experiment bound")
            separation=current["rim2"]-current["rim"]
            if abs(float(separation[2]))>0.012 or abs(float(np.dot(separation,inward)))>0.012:
                raise ValueError("Separate measured rim sections disagree")
            if abs(float(np.dot(separation-tracking_model["pair_delta"],side)))>0.012:
                raise ValueError("Separated rim trackers changed their lateral spacing")
            lateral=float(np.dot(current["eef"]-current["rim"],side))
            if not -0.28<lateral<-0.03:
                raise ValueError("Hand moved outside the selected rim contact section")
            if abs(float(current["eef"][2]-current["rim"][2]))>0.060 or abs(float(np.dot(current["eef"]-current["rim"],inward)))>0.085:
                raise ValueError("Hand lost its measured raised-rim contact envelope")

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
            contact_radius = max(0.12,float(np.hypot(np.dot(pivot-initial["eef"],inward),initial["eef"][2]-pivot[2])))
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
