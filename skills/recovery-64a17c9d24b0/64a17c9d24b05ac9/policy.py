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

    def rgb_now(camera="right"):
        data = ctx.read_observation(camera)
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

    rim_camera = params.get("rim_camera", "right")
    selections = {
        "rim": {"label": "exposed outer door silver rim", "camera": rim_camera,
                "context": params["rim_context"], "patch": params["rim_box"]},
        "oven": {"label": "fixed oven roof reference", "camera": "right",
                 "context": params["oven_context"], "patch": params["oven_box"]},
        "rim2": {"label": "second exposed outer rim section", "camera": params["second_camera"],
                 "context": params["second_context"], "patch": params["second_track_box"]}
    }
    selections["rim3"] = {"label": "third separated thin silver door lip", "camera": rim_camera,
                          "context": params["third_context"], "patch": params["third_box"]}
    if not params.get("resume_supported", False):
        return unresolved("Three-region continuation requires the retained raised-door arc")
    if params.get("diagnostic_only", False):
        return unresolved("Historical diagnostic is complete; use the calibrated whole-task branch.")

    initial_images = {camera: rgb_now(camera) for camera in {v["camera"] for v in selections.values()}}
    for key, item in selections.items():
        initial_rgb = initial_images[item["camera"]]
        a,b,c,d = item["context"]
        item["template"] = initial_rgb[b:d,a:c].copy()
        item["offset"] = (0,0)
        if np.std(item["template"]) < 2.0:
            return unresolved("Selected visual context lacks sufficient texture.")
        if key != "oven":
            x0,y0,x1,y1 = item["patch"]
            item["color"] = np.median(initial_rgb[y0:y1,x0:x1].reshape(-1,3),axis=0)
    previous = {}
    tracking_model = {}
    cache = {}
    # Plane observations identify a rotating face, not a tracked material point.
    # The lip coordinates returned below are rigid-model reconstructions anchored
    # to recorded lip depth; all support/progress records distinguish that inference.
    plane_model = {}
    def calibrated_frame(camera, step):
        data = ctx.read_observation(camera, step=step)
        cal = data.get("metadata", {}).get("calibration", {})
        if (cal.get("step") != step or cal.get("alignment_verified") is not True
                or cal.get("frame") != "optical_to_osc_arm_base"):
            raise ValueError("Aligned metric calibration unavailable")
        rgb = np.asarray(data["rgb"], float)
        depth = np.asarray(data.get("depth_m"), float)
        k = np.asarray(cal["intrinsics"], float)
        tf = np.asarray(cal["camera_to_base"], float)
        if (rgb.shape != (256,256,3) or depth.shape != (256,256)
                or k.shape != (3,3) or tf.shape != (4,4)
                or not np.all(np.isfinite(k)) or not np.all(np.isfinite(tf))
                or min(k[0,0],k[1,1]) <= 0):
            raise ValueError("Invalid calibrated RGB-D arrays")
        if np.max(rgb) <= 1:
            rgb = 255*rgb
        return rgb,depth,k,tf

    def patch_cloud(frame, box, minimum=5):
        _,dep,k,tf = frame
        x0,y0,x1,y1 = box
        if not (0 <= x0 < x1 <= 256 and 0 <= y0 < y1 <= 256):
            return None
        z = dep[y0:y1,x0:x1]
        if z.size < minimum or not np.all(np.isfinite(z)) or np.any(z <= 0.02):
            return None
        if np.percentile(z,75)-np.percentile(z,25) > 0.045:
            return None
        vv,uu = np.mgrid[y0:y1,x0:x1]
        optical = np.stack(((uu-k[0,2])*z/k[0,0],(vv-k[1,2])*z/k[1,1],z),axis=-1).reshape(-1,3)
        cloud = optical@tf[:3,:3].T+tf[:3,3]
        p = np.median(cloud,axis=0)
        if np.ptp(cloud[:,2]) > 0.010 or np.max(np.linalg.norm(cloud-p,axis=1)) > 0.025:
            return None
        return cloud

    def face_plane(cloud, direction):
        if cloud is None or len(cloud) < 8:
            return None
        center = cloud.mean(axis=0)
        _,singular,vt = np.linalg.svd(cloud-center,full_matrices=False)
        normal = vt[-1]
        if np.dot(normal,direction) < 0:
            normal = -normal
        residual = float(np.max(np.abs((cloud-center)@normal)))
        # A genuinely planar face is distinct from a patch crossing its bevel.
        # The recorded face residuals are <1.1um; crossing patches are 50..593um.
        if singular[1] < 0.002 or residual > 0.00001:
            return None
        return np.median(cloud,axis=0),normal,residual

    def initialize_plane_model(frame, roof):
        inward = np.asarray(params["arc_reference"]["inward"],float)
        if inward.shape != (3,) or not np.all(np.isfinite(inward)) or abs(np.linalg.norm(inward)-1)>0.001 or abs(inward[2])>0.001:
            raise ValueError("Invalid measured horizontal direction")
        side = np.cross(np.array([0.,0.,1.]),inward)
        records = params["historical_face_samples"]
        if not 12 <= len(records) <= 32:
            raise ValueError("Insufficient bounded historical face observations")
        rows = []
        rejected = []
        frames = {start["step"]:frame}
        for record in records:
            step = record["observation_step"]
            if step > start["step"]:
                raise ValueError("Future calibration observation prohibited")
            historical = frames.get(step)
            if historical is None:
                historical = calibrated_frame(rim_camera,step)
                frames[step] = historical
            roof_cloud = patch_cloud(historical,params["oven_box"])
            if roof_cloud is None or np.ptp(roof_cloud[:,2]) > 0.008:
                raise ValueError("Historical roof is unavailable")
            old_roof = np.median(roof_cloud,axis=0)
            if np.linalg.norm(old_roof-roof)>0.015:
                raise ValueError("Historical roof drift exceeds retained bound")
            face = face_plane(patch_cloud(historical,record["box"]),inward)
            if face is None:
                rejected.append(step)
                continue
            p,n,residual = face
            if abs(float(np.dot(n,side))) > 0.01:
                raise ValueError("Historical planar face axis differs from measured edge")
            p = p+roof-old_roof
            theta = float(np.arctan2(-n[2],np.dot(n,inward)))
            rows.append((step,p,n,theta,residual))
        rows.sort(key=lambda row:row[0])
        if len(rows)<12 or rows[-1][3]-rows[0][3]<0.30:
            raise ValueError("Insufficient independent plane rotation for hinge estimation")
        if any(b[3]<a[3]-0.01 or b[3]-a[3]>0.12 for a,b in zip(rows,rows[1:])):
            raise ValueError("Historical plane orientation is discontinuous")
        def fit(rs):
            design = np.asarray([[np.dot(r[2],inward),r[2][2],-1.] for r in rs])
            target = np.asarray([np.dot(r[2],r[1]) for r in rs])
            solution,_,rank,singular = np.linalg.lstsq(design,target,rcond=None)
            if rank != 3 or not np.all(np.isfinite(solution)):
                raise ValueError("Rotating-plane hinge is degenerate")
            residual = float(np.max(np.abs(design@solution-target)))
            return solution,residual,singular
        solution,residual,singular = fit(rows)
        if residual > 0.0001:
            raise ValueError("Clean face planes do not share a consistent hinge")
        split=len(rows)//2
        early,_,_ = fit(rows[:split])
        late,_,_ = fit(rows[split:])
        split_difference = float(np.linalg.norm(early[:2]-late[:2]))
        leave_one = [fit(rows[:i]+rows[i+1:])[0] for i in range(len(rows))]
        jackknife = max(float(np.linalg.norm(v[:2]-solution[:2])) for v in leave_one)
        # Preserve the existing 20mm geometric uncertainty bound, and require
        # stability of the full observed arc under every single-row omission.
        if split_difference>0.020 or jackknife>0.002:
            raise ValueError("Hinge estimate is too sensitive to the observed arc")
        hu,hz,offset = solution
        if not 0.14 < offset < 0.30:
            raise ValueError("Measured plane-to-hinge distance is outside the retained model")
        hinge = hu*inward+np.array([0.,0.,hz])
        lip_step=params["historical_lip_step"]
        matching=[r for r in rows if r[0]==lip_step]
        acquisition=params["calibration_observation_step"]
        checkpoints=params["bridge_checkpoints"]
        chain=[acquisition]+[v["observation_step"] for v in checkpoints]+[start["step"]]
        if (len(matching)!=1 or lip_step>=acquisition or acquisition-lip_step>4
                or acquisition!=rows[-1][0] or not 1<=len(checkpoints)<=8
                or any(not 0<b-a<=20 for a,b in zip(chain,chain[1:]))):
            raise ValueError("Lip acquisition or bounded observed checkpoint chain is invalid")
        lip_frame=frames[lip_step]
        lip_roof=np.median(patch_cloud(lip_frame,params["oven_box"]),axis=0)
        lips=[]
        for box in params["historical_lip_boxes"]:
            cloud=patch_cloud(lip_frame,box,minimum=4)
            if cloud is None:
                raise ValueError("Recorded lip geometry is unavailable")
            lips.append(np.median(cloud,axis=0)+roof-lip_roof)
        if len(lips)!=3:
            raise ValueError("Three separated recorded lip regions are required")
        for a,b in ((0,1),(0,2),(1,2)):
            sep=lips[a]-lips[b]
            if not 0.05<np.linalg.norm(sep[:2])<0.25 or abs(sep[2])>0.008 or abs(np.dot(sep,inward))>0.012:
                raise ValueError("Recorded lip regions are incompatible")
        radius=float(np.hypot(np.dot(lips[0]-hinge,inward),lips[0][2]-hz))
        if not 0.14<radius<0.30:
            raise ValueError("Anchored lip radius lies outside the retained model")
        plane_model.update(inward=inward,side=side,hinge=hinge,offset=float(offset),
            roof=roof.copy(),lips=lips,anchor_angle=matching[0][3],radius=radius,
            uncertainty=max(0.006,split_difference),last_normals={},first_points={})
        # Preserve the original lip-anchor acquisition check. At a later restart,
        # replay every intervening two-step calibrated observation rather than
        # pretending the old lip selection is fresh or resetting model provenance.
        last_p,last_n,last_theta=rows[-1][1:4]
        box=list(records[[r["observation_step"] for r in records].index(acquisition)]["box"])
        bridge_steps=sorted(set(list(range(acquisition+2,start["step"]+1,2))+
            [v["observation_step"] for v in checkpoints]+[start["step"]]))
        checkpoint_angles={v["observation_step"]:v["face_angle"] for v in checkpoints}
        for bridge_step in bridge_steps:
            bf=calibrated_frame(rim_camera,bridge_step)
            rc=patch_cloud(bf,params["oven_box"])
            if rc is None or np.ptp(rc[:,2])>0.008:
                raise ValueError("Bridge roof depth unavailable")
            br=np.median(rc,axis=0)
            if np.linalg.norm(br-roof)>0.015:
                raise ValueError("Bridge roof drift exceeds retained bound")
            options=[]
            x0,y0,x1,y1=box
            for dy in range(-5,6):
                for dx in range(-5,6):
                    bb=[x0+dx,y0+dy,x1+dx,y1+dy]
                    f=face_plane(patch_cloud(bf,bb),last_n)
                    if f is None:
                        continue
                    bp,bn,_=f
                    bp=bp+roof-br
                    bt=float(np.arctan2(-bn[2],np.dot(bn,inward)))
                    if (np.linalg.norm(bp-last_p)>0.025 or abs(bt-last_theta)>0.06
                            or abs(np.dot(bn,side))>0.01
                            or abs(np.dot(bn,hinge-bp)-offset)>0.020):
                        continue
                    options.append((float(np.linalg.norm(bp-last_p)),bb,bp,bn,bt))
            if not options:
                raise ValueError("Recorded bridge lost continuous calibrated face geometry")
            options.sort(key=lambda row:row[0])
            _,box,last_p,last_n,last_theta=options[0]
            if bridge_step in checkpoint_angles and abs(last_theta-checkpoint_angles[bridge_step])>0.01:
                raise ValueError("Recorded checkpoint does not match remeasured plane orientation")
        plane_model["bridge_angle"]=last_theta
        ctx.note(facts=["Runtime validated face continuity across recorded bridge steps "+str(bridge_steps)],
            hypotheses=["Historical continuity is not evidence that current support contact persists."])
        ctx.note(facts=["Runtime rotating-face calibration uses "+str(len(rows))+" planar observations; excluded edge-crossing steps "+str(rejected),
            "Hinge coordinates (inward,height), plane offset: "+str(solution.tolist())+
            "; max residual "+str(residual)+" m; split center difference "+str(split_difference)+
            " m; leave-one-out sensitivity "+str(jackknife)+" m; design singular values "+str(singular.tolist())],
            hypotheses=["Recorded lip points are rigid-model anchors, not current visible rim measurements or evidence of current contact. Fresh three-region face rotation must validate support."])

    def measure():
        obs=unpack(ctx.observe())
        if cache.get("step")==obs["step"]:
            return cache.copy()
        frames={camera:calibrated_frame(camera,obs["step"])
            for camera in set(v["camera"] for v in selections.values())}
        frame=frames[rim_camera]
        rgb=frames[selections["oven"]["camera"]][0]
        oven=selections["oven"]
        a,b,c,d=oven["context"]
        if float(np.mean((rgb[b:d,a:c]-oven["template"])**2))>450.0:
            raise ValueError("Fixed oven appearance changed")
        roof=metric(oven["label"],oven["patch"],obs["step"],0.008,oven["camera"])
        if not plane_model:
            initialize_plane_model(frame,roof)
        model=plane_model
        inward,side=model["inward"],model["side"]
        shift=roof-model["roof"]
        if np.linalg.norm(shift)>0.015:
            raise ValueError("Fixed roof drift exceeds retained limit")
        hinge=model["hinge"]+shift
        options={}
        for key in ("rim","rim2","rim3"):
            item=selections[key]
            frame=frames[item["camera"]]
            rgb=frame[0]
            a,b,c,d=item["context"]
            x0,y0,x1,y1=item["patch"]
            ox,oy=item["offset"]
            first=key not in previous
            choices=[]
            offsets=[(0,0)] if first else [(dx,dy) for dy in range(oy-5,oy+6) for dx in range(ox-5,ox+6)]
            for dx,dy in offsets:
                if min(a+dx,b+dy,x0+dx,y0+dy)<0 or max(c+dx,x1+dx)>256 or max(d+dy,y1+dy)>256:
                    continue
                box=[x0+dx,y0+dy,x1+dx,y1+dy]
                face=face_plane(patch_cloud(frame,box),model["last_normals"].get(key,inward))
                if face is None:
                    continue
                p,n,residual=face
                if abs(float(np.dot(n,side)))>0.01:
                    continue
                theta=float(np.arctan2(-n[2],np.dot(n,inward)))
                if not 0.0<theta<1.65 or (first and abs(theta-model["bridge_angle"])>0.01):
                    continue
                if abs(float(np.dot(n,hinge-p))-model["offset"])>0.020:
                    continue
                if not first:
                    if np.linalg.norm(p-previous[key])>0.025:
                        continue
                    if abs(float(np.dot(p-model["first_points"][key]-shift,side)))>0.020:
                        continue
                    old_theta=float(np.arctan2(-model["last_normals"][key][2],np.dot(model["last_normals"][key],inward)))
                    if abs(theta-old_theta)>0.06:
                        continue
                context=rgb[b+dy:d+dy,a+dx:c+dx]
                err=float(np.mean((context-item["template"])**2))
                color=np.median(rgb[y0+dy:y1+dy,x0+dx:x1+dx].reshape(-1,3),axis=0)
                ce=float(np.sqrt(np.mean((color-item["color"])**2)))
                continuity=0.0 if first else float(np.linalg.norm(p-previous[key]))
                score=continuity**2+0.00002*min(err/1400.,4.)+0.00002*min((ce/70.)**2,4.)
                choices.append((score,p,n,theta,dx,dy,box,residual))
            choices.sort(key=lambda row:row[0])
            valid=[]
            for candidate in choices[:4]:
                try:
                    m=metric("planar outer door bar "+key,candidate[6],obs["step"],0.010,item["camera"])
                    if np.linalg.norm(m-candidate[1])<=0.006:
                        valid.append(candidate)
                except (ValueError,KeyError,TypeError):
                    continue
            if not valid:
                raise ValueError("No continuous calibrated planar door face for "+key)
            options[key]=valid
        best=None
        for one in options["rim"]:
            for two in options["rim2"]:
                for three in options["rim3"]:
                    group=(one,two,three)
                    good=True
                    for i,j in ((0,1),(0,2),(1,2)):
                        sep=group[i][1]-group[j][1]
                        if (not 0.05<np.linalg.norm(sep[:2])<0.25 or abs(sep[2])>0.012
                                or abs(np.dot(sep,inward))>0.012 or abs(group[i][3]-group[j][3])>0.01
                                or abs(np.dot(group[i][2],sep))>0.002):
                            good=False
                            break
                    score=sum(v[0] for v in group)
                    if good and (best is None or score<best[0]):
                        best=(score,group)
        if best is None:
            raise ValueError("Three separated face planes do not agree")
        result={"step":obs["step"],"pose":obs["eef_control_pose"],"oven":roof,
                "eef":np.asarray(obs["eef_control_pose"]["position"],float),
                "face_angles":{},"face_points":{},"lip_inference_error":0.0}
        for index,(key,candidate) in enumerate(zip(("rim","rim2","rim3"),best[1])):
            _,p,n,theta,dx,dy,box,residual=candidate
            turn=theta-model["anchor_angle"]
            reconstructed=model["hinge"]+Rotation.from_rotvec(side*turn).apply(model["lips"][index]-model["hinge"])+shift
            # Up to 6mm recorded-patch uncertainty plus full-arc fit sensitivity.
            error=0.006+2*model["uncertainty"]*abs(np.sin(turn/2))
            result[key]=reconstructed
            result["lip_inference_error"]=max(result["lip_inference_error"],float(error))
            result["face_angles"][key]=theta
            result["face_points"][key]=p.tolist()
            previous[key]=p.copy()
            model["last_normals"][key]=n.copy()
            if key not in model["first_points"]:
                model["first_points"][key]=p.copy()
            item=selections[key]
            a,b,c,d=item["context"]
            item["offset"]=(dx,dy)
            item["template"]=frames[item["camera"]][0][b+dy:d+dy,a+dx:c+dx].copy()
        cache.clear()
        cache.update(result)
        return result

    def act_pose(position, quaternion, closed, steps=4):
        obs = unpack(ctx.observe())
        if (obs["step"]-start["step"]+steps > limit
                or obs["step"]-params["closing_started_step"]+steps>330):
            raise ValueError("Revision or original closing-phase action bound reached")
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
        kin = obs.get("controller", {}).get("gripper_kinematics", {})
        geom = kin.get("fingerpad_geometry", {})
        if obs.get("step") != current["step"] or geom.get("step") != current["step"] or geom.get("available") is not True:
            raise ValueError("Current measured fingerpad geometry unavailable")
        if geom.get("frame") != "eef_control_site" or geom.get("source") != "robot_model_and_measured_joint_kinematics":
            raise ValueError("Unverified fingerpad geometry provenance")
        pads = geom.get("pads", [])
        if len(pads) != 2:
            raise ValueError("Expected two measured fingerpads")
        pads = sorted(pads, key=lambda pad: pad["center_eef"][0])
        rotation = Rotation.from_quat(current["pose"]["quaternion_xyzw"])
        points = []
        lower_bounds = []
        for pad in pads:
            center = np.asarray(pad["center_eef"], float)
            local_rotation = np.asarray(pad["rotation_eef"], float)
            half = np.asarray(pad["half_extents_m"], float)
            if center.shape != (3,) or local_rotation.shape != (3,3) or half.shape != (3,) or not np.all(np.isfinite(center)) or not np.all(np.isfinite(local_rotation)) or not np.all(np.isfinite(half)) or np.any(half <= 0):
                raise ValueError("Invalid measured pad box")
            point = current["eef"] + rotation.apply(center)
            vertical_extent = float(np.sum(np.abs((rotation.as_matrix() @ local_rotation)[2]) * half))
            points.append(point)
            lower_bounds.append(float(point[2]-vertical_extent))
        vector = points[1]-points[0]
        width = float(np.linalg.norm(vector))
        if not 0.067 < width < 0.095:
            raise ValueError("Current jaws are not open within the approach envelope")
        midpoint = 0.5*(points[0]+points[1])
        local_axis = rotation.inv().apply(vector/width)
        if abs(local_axis[0]) < 0.98 or np.linalg.norm(rotation.inv().apply(midpoint-current["eef"])) > 0.025:
            raise ValueError("Measured pad geometry differs from the local-x model")
        return midpoint, vector/width, width, min(lower_bounds)

    try:
        with ctx.stage("remeasure fingertips and calibrate rotating door face") as stage:
            baseline = measure()
            up = np.array([0.,0.,1.])
            resume = params.get("resume_supported", False)
            if resume:
                if not params.get("supported_contact",False):
                    return unresolved("Plane continuation requires the complete supported closing branch")
                inward=plane_model["inward"].copy()
                side=plane_model["side"].copy()
                pivot=plane_model["hinge"].copy()
                radius=plane_model["radius"]
                counter=metric("countertop beside the door",params["counter_box"],baseline["step"],0.010,rim_camera)
                for key in ("rim","rim2","rim3"):
                    vector=baseline[key]-pivot
                    radial=float(np.hypot(np.dot(vector,inward),vector[2]))
                    if abs(radial-radius)>0.020:
                        return unresolved("Recorded lip anchors disagree on radius")
                stage.record(geometry_source="calibrated rotating face planes plus recorded lip anchors",
                    radius_m=radius,pivot_m=pivot.tolist(),
                    reconstructed_lip_m=baseline["rim"].tolist(),
                    reconstruction_uncertainty_m=baseline["lip_inference_error"],
                    measured_face_angles_rad=baseline["face_angles"],
                    measured_face_points_m=baseline["face_points"])
            else:
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
            if resume:
                if lowest < counter[2]+0.007:
                    return unresolved("Measured support pads lack current countertop clearance")
            elif not 0.035 < baseline["rim"][2]-counter[2] < 0.09:
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
            if jaw_local.shape!=(3,) or not np.all(np.isfinite(jaw_local)) or not 0.99<float(np.linalg.norm(jaw_local))<1.01 or abs(float(jaw_local[0]))<0.98:
                return unresolved("Controller jaw calibration does not support local-x alignment")
            observed_local=Rotation.from_quat(baseline["pose"]["quaternion_xyzw"]).inv().apply(jaw_axis)
            if abs(float(np.dot(observed_local,jaw_local)))<0.98:
                return unresolved("Current pad kinematics disagrees with controller jaw calibration")
            stage.record(jaw_calibration_source=calibration_source,controller_jaw_axis_eef=jaw_local.tolist(),observed_jaw_axis_eef=observed_local.tolist())
            if abs(float(baseline["rim"][2]-baseline["rim2"][2]))>0.008 or abs(float(np.dot(baseline["rim"]-baseline["rim2"],inward)))>0.012:
                return unresolved("Independent rim selections disagree")
            if abs(float(baseline["rim3"][2]-baseline["rim"][2]))>0.008 or abs(float(np.dot(baseline["rim3"]-baseline["rim"],inward)))>0.012:
                return unresolved("Third rim sample disagrees with initial support geometry")
            tracking_model.update(pivot=pivot,radius=radius,side=side,inward=inward,
                oven=baseline["oven"].copy(),rim=baseline["rim"].copy(),rim2=baseline["rim2"].copy(),rim3=baseline["rim3"].copy())
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
            for key in ("rim2","rim3"):
                sep=current[key]-current["rim"]
                if abs(float(sep[2]))>0.012 or abs(float(np.dot(sep,inward)))>0.012:
                    raise ValueError("Three measured rim regions disagree")
            lateral = float(np.dot(current["eef"]-current["rim"],side))
            if not -0.28 < lateral < -0.03:
                raise ValueError("Hand moved outside the selected interior rim section")

        def pose_error(current, goal_position, goal_q):
            distance = float(np.linalg.norm(goal_position-current["eef"]))
            rotation = Rotation.from_quat(current["pose"]["quaternion_xyzw"])
            angle = float((Rotation.from_quat(goal_q)*rotation.inv()).magnitude())
            return distance, angle

        if params.get("supported_contact", False):
            # Resume the already observed closing phase. Its initial support
            # experiment completed before the one-step visibility interruption.
            # This checkpoint is evidence to test continuation, never contact proof.
            with ctx.stage("validate recent closing checkpoint using both cameras") as stage:
                current=measure()
                guard(current)
                cp=params["closing_checkpoint"]
                joints=np.abs(np.asarray(unpack(ctx.observe())["state"][-2:],float))
                old_joints=np.abs(np.asarray(cp["finger_joints"],float))
                if (not 0<current["step"]-cp["observation_step"]<=2
                        or params["closing_started_step"]>cp["observation_step"]
                        or np.min(old_joints)<0.032
                        or np.any(joints<old_joints-0.0002)):
                    return unresolved("Recent closing checkpoint or bounded finger transition is invalid")
                if np.linalg.norm(current["eef"]-np.asarray(cp["position"]))>0.003:
                    return unresolved("Current hand is outside the recent closing checkpoint")
                changes={key:current["face_angles"][key]-cp["face_angles"][key] for key in cp["face_angles"]}
                if min(changes.values()) < -0.025 or max(changes.values()) > 0.060:
                    return unresolved("Fresh door orientations do not continue the recorded closing phase")
                cp_roof_cloud=patch_cloud(calibrated_frame("right",cp["observation_step"]),params["oven_box"])
                if cp_roof_cloud is None or np.ptp(cp_roof_cloud[:,2])>0.008:
                    return unresolved("Recorded closing checkpoint roof is unavailable")
                cp_roof=np.median(cp_roof_cloud,axis=0)
                if np.linalg.norm(cp_roof-current["oven"])>0.015:
                    return unresolved("Closing checkpoint fixed reference has drifted")
                cp_relative=np.asarray(cp["reconstructed_lip_m"])-cp_roof
                if float((current["rim"]-current["oven"]-cp_relative)[2]) < -0.006:
                    return unresolved("Door reopened since the latest closing checkpoint")
                if fingertips(current)[3]<counter[2]+0.007:
                    return unresolved("Current pads lack countertop clearance")
                stage.record(step=current["step"],finger_joints_m=joints.tolist(),
                    measured_face_angle_changes_rad=changes,measured_face_points_m=current["face_points"])

            # Test translation-free settling at the current measured orientation.
            # Preserve the earlier angular corridor rather than resetting it.
            ref=params["wrist_corridor_reference"]
            q_origin=np.asarray(ref["quaternion_xyzw"],float)
            if (q_origin.shape!=(4,) or abs(np.linalg.norm(q_origin)-1)>0.001
                    or not ref["observation_step"]<=cp["observation_step"]):
                return unresolved("Original wrist corridor reference is invalid")
            wrist_origin=Rotation.from_quat(q_origin)
            settle_rotation=Rotation.from_quat(current["pose"]["quaternion_xyzw"])
            if float((settle_rotation*wrist_origin.inv()).magnitude())>0.060:
                return unresolved("Current wrist is outside the retained angular corridor")
            with ctx.stage("test open-gripper settling while holding the current pose") as stage:
                anchor=current["eef"].copy()
                initial_joints=joints.copy()
                initial_angles=current["face_angles"].copy()
                initial_relative=current["rim"]-current["oven"]
                restored=False
                for index in range(13):
                    before=measure()
                    guard(before)
                    rotation=Rotation.from_quat(before["pose"]["quaternion_xyzw"])
                    joints=np.abs(np.asarray(unpack(ctx.observe())["state"][-2:],float))
                    if (np.any(joints<initial_joints-0.0002)
                            or np.linalg.norm(before["eef"]-anchor)>0.003
                            or float((rotation*wrist_origin.inv()).magnitude())>0.060
                            or fingertips(before)[3]<counter[2]+0.007):
                        return unresolved("Current wrist relief crossed compression, pose or clearance bounds")
                    if (min(before["face_angles"][key]-initial_angles[key] for key in initial_angles)<-0.025
                            or float((before["rim"]-before["oven"]-initial_relative)[2]) < -0.006):
                        return unresolved("Door reopened during current wrist relief")
                    stage.record(step=before["step"],finger_joints_m=joints.tolist(),
                        position_drift_m=float(np.linalg.norm(before["eef"]-anchor)),
                        measured_face_angles_rad=before["face_angles"])
                    if np.min(joints)>=0.0325:
                        restored=True
                        break
                    if index==12:
                        break
                    act_pose(anchor,settle_rotation.as_quat(),False,steps=1)
                    if done():
                        return "SUBTASK_DONE"
                if not restored:
                    return unresolved("Twelve-step current relief failed to restore opening")
            ctx.note(facts=["Closing checkpoint and current opening were remeasured; continuing with one-step object and finger feedback."],
                hypotheses=["Translation-free settling may restore clearance without more counter-rotation. Fresh three-region rotation remains required; restored opening is not contact evidence."])

            def support_angle(current):
                h = pivot+current["oven"]-baseline["oven"]
                v = current["rim"]-h
                horizontal = -float(np.dot(v,inward))
                measured_radius = float(np.hypot(horizontal,v[2]))
                if not 0.70*radius < measured_radius < 1.30*radius:
                    raise ValueError("Supported rim inconsistent with local arc reference")
                return float(np.arctan2(v[2],horizontal))

            with ctx.stage("continue closing with measured open-hand support") as stage:
                initial = measure()
                first_angle = support_angle(initial)
                last_angle = first_angle
                window_angle = first_angle
                initial_rotation = Rotation.from_quat(initial["pose"]["quaternion_xyzw"])
                fresh_angles=initial["face_angles"].copy()
                hold_rotation=settle_rotation
                fresh_response=False
                pause=None
                for index in range(330):
                    before = measure()
                    guard(before)
                    angle = support_angle(before)
                    if before["step"]-params["closing_started_step"]>=330:
                        return unresolved("Original closing-phase action budget reached")
                    actual_rotation=Rotation.from_quat(before["pose"]["quaternion_xyzw"])
                    if (float((actual_rotation*wrist_origin.inv()).magnitude())>0.060
                            or fingertips(before)[3]<counter[2]+0.007):
                        return unresolved("Closing left the bounded wrist or pad-clearance corridor")
                    if angle < last_angle-0.06 or angle < window_angle-0.10:
                        return unresolved("Supported door rotation reversed")
                    joints = np.abs(np.asarray(unpack(ctx.observe())["state"][-2:],float))
                    if np.min(joints) < 0.032:
                        return unresolved("Open support fingers became constrained")
                    # Stop advancing before the unchanged compression limit.
                    # Pause commands keep one fixed position and orientation;
                    # actual opening, not elapsed settling time, releases the pause.
                    if np.min(joints)<0.0325:
                        if pause is None:
                            pause={"position":before["eef"].copy(),"step":before["step"],
                                "joints":joints.copy(),"angles":before["face_angles"].copy(),
                                "relative":before["rim"]-before["oven"]}
                        if (before["step"]-pause["step"]>=12
                                or np.linalg.norm(before["eef"]-pause["position"])>0.003
                                or np.any(joints<pause["joints"]-0.0002)):
                            return unresolved("Translation pause failed its opening or position bounds")
                        if (min(before["face_angles"][key]-pause["angles"][key] for key in pause["angles"])<-0.025
                                or float((before["rim"]-before["oven"]-pause["relative"])[2])<-0.006):
                            return unresolved("Door reopened during a translation pause")
                        target_position=pause["position"]
                        mode="open_gripper_pose_hold"
                    else:
                        pause=None
                        if angle < 1.43:
                            delta=0.010*(np.sin(angle)*inward+np.cos(angle)*up)
                        else:
                            delta=0.008*inward
                        target_position=before["eef"]+delta
                        mode="closing_translation"
                    act_pose(target_position,hold_rotation.as_quat(),False,steps=1)
                    if done():
                        return "SUBTASK_DONE"
                    after = measure()
                    guard(after)
                    after_joints=np.abs(np.asarray(unpack(ctx.observe())["state"][-2:],float))
                    if np.min(after_joints)<0.032:
                        return unresolved("Open support fingers became constrained after one-step closing")
                    if (float((Rotation.from_quat(after["pose"]["quaternion_xyzw"])*wrist_origin.inv()).magnitude())>0.060
                            or fingertips(after)[3]<counter[2]+0.007):
                        return unresolved("Measured closing crossed the wrist or counter bound")
                    if pause is not None:
                        if (np.linalg.norm(after["eef"]-pause["position"])>0.003
                                or np.any(after_joints<pause["joints"]-0.0002)
                                or min(after["face_angles"][key]-pause["angles"][key] for key in pause["angles"])<-0.025
                                or float((after["rim"]-after["oven"]-pause["relative"])[2])<-0.006):
                            return unresolved("Measured translation pause crossed its retained corridor")
                    last_angle = support_angle(after)
                    stage.record(step=after["step"],action_mode=mode,finger_joints_m=after_joints.tolist(),door_angle_rad=last_angle,
                        reconstructed_lip_m=after["rim"].tolist(),measured_face_angles_rad=after["face_angles"],
                        measured_face_points_m=after["face_points"],eef_m=after["eef"].tolist())
                    if index%10 == 9:
                        if not fresh_response:
                            changes={key:after["face_angles"][key]-fresh_angles[key] for key in fresh_angles}
                            if min(changes.values())<0.025:
                                return unresolved("Reopened fingers did not produce fresh three-region door rotation")
                            fresh_response=True
                        if last_angle-window_angle < 0.018:
                            return unresolved("Open-hand closing stopped producing measured door rotation")
                        window_angle = last_angle
            return unresolved("Supported closing reached its bounded limit without task verification")

        return unresolved("This revision requires the observed supported closing phase.")
    except (ValueError,KeyError,TypeError) as error:
        return unresolved("Measured-jaw approach or closing prerequisite failed: "+str(error))
