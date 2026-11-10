def run(ctx):
    import json
    import numpy as np
    from scipy.spatial.transform import Rotation

    goal = ctx.completion_condition
    params = ctx.parameters
    limit = 300
    def decode(value):
        return json.loads(value.split("\ndata:image",1)[0]) if isinstance(value,str) else value
    def done():
        return decode(ctx.check(goal)).get("passed") is True
    def unresolved(reason):
        ctx.note(facts=[reason],hypotheses=["The complete door-closing task remains assigned to code; only task_success establishes completion."])
        return "UNRESOLVED"
    if done():
        return "SUBTASK_DONE"
    start = decode(ctx.observe())
    caps = decode(ctx.capabilities())
    if start.get("step") != params["observation_step"]:
        return unresolved("Fresh image selections are stale; no action executed.")
    if goal not in caps.get("verifiers",[]) or not all(c in caps.get("capabilities",[]) for c in ("rgbd","osc_eef","gripper")):
        return unresolved("Required metric observations, control or whole-task verifier unavailable.")

    def record(camera,step):
        data = ctx.read_observation(camera)
        cal = data.get("metadata",{}).get("calibration",{})
        if cal.get("step") != step or cal.get("alignment_verified") is not True or cal.get("frame") != "optical_to_osc_arm_base":
            raise ValueError("Current aligned calibration is unavailable for "+camera)
        if data.get("depth_m") is None:
            raise ValueError("Metric depth missing for "+camera)
        rgb = np.asarray(data["rgb"],float)
        if np.max(rgb)<=1.:
            rgb=rgb*255.
        depth=np.asarray(data["depth_m"],float)
        if depth.shape!=(256,256) or rgb.shape!=(256,256,3):
            raise ValueError("Unexpected RGB-D shape")
        return rgb,depth,np.asarray(cal["intrinsics"],float),np.asarray(cal["camera_to_base"],float)

    def metric(label,camera,box,step):
        g=decode(ctx.ground_target(target=label,camera=camera,observation_step=step,bbox=list(box),confidence=0.86))
        m=decode(ctx.query_geometry(g["grounding_id"]))
        if m.get("reliable") is not True or m.get("frame")!="osc_arm_base" or m.get("observation_step")!=step or m.get("valid_fraction",0)<0.95:
            raise ValueError("Unreliable or stale surface depth: "+label)
        point=np.asarray(m["centroid_m"],float)
        if point.shape!=(3,) or not np.all(np.isfinite(point)):
            raise ValueError("Invalid measured surface")
        return point,m

    def project(point,k,t):
        p=t[:3,:3].T @ (point-t[:3,3])
        if p[2]<=0.02:
            raise ValueError("Tracked surface is behind the camera")
        uv=k@p
        return uv[:2]/uv[2]

    def point_patch(box,depth,k,t):
        x0,y0,x1,y1=box
        vv,uu=np.mgrid[y0:y1,x0:x1]
        zz=depth[y0:y1,x0:x1]
        if zz.size<3 or not np.all(np.isfinite(zz)) or np.any(zz<=0.02):
            return None
        optical=np.stack(((uu-k[0,2])*zz/k[0,0],(vv-k[1,2])*zz/k[1,1],zz),axis=-1).reshape(-1,3)
        points=optical@t[:3,:3].T+t[:3,3]
        return np.median(points,axis=0),points

    def pose():
        obs=decode(ctx.observe())
        obs["eef"]=np.asarray(obs["eef_control_pose"]["position"],float)
        return obs

    def act(position,quaternion,closed,steps=2):
        obs=pose()
        if obs["step"]-start["step"]+steps>limit:
            raise ValueError("Shared revision action bound reached")
        delta=np.asarray(position)-obs["eef"]
        length=float(np.linalg.norm(delta))
        if length>0.020:
            delta*=0.020/length
        measured=Rotation.from_quat(obs["eef_control_pose"]["quaternion_xyzw"])
        rv=(Rotation.from_quat(quaternion)*measured.inv()).as_rotvec()
        angle=float(np.linalg.norm(rv))
        if angle>0.14:
            rv*=0.14/angle
        target_rotation=Rotation.from_rotvec(rv)*measured
        ctx.move_eef(target={"position":(obs["eef"]+delta).tolist(),
            "quaternion_xyzw":target_rotation.as_quat().tolist(),
            "gripper_closed":closed,"base_motion":[0,0,0,0],"control_mode":"arm"},steps=steps)

    try:
        with ctx.stage("remeasure nearly upright door before releasing the constrained contact") as stage:
            initial=pose()
            rim_a,ma=metric("raised near rim right","right",params["rim_box"],initial["step"])
            rim_left,mb=metric("raised second rim right","right",params["second_box"],initial["step"])
            oven0,mo=metric("fixed oven top","right",params["oven_box"],initial["step"])
            for measured in (ma,mb,mo):
                if measured.get("depth_iqr_m",1.)>0.035 or float(np.asarray(measured["max_m"])[2]-np.asarray(measured["min_m"])[2])>0.010:
                    return unresolved("Selected current surface contains mixed depth")
            up=np.array([0.,0.,1.])
            side=rim_left-rim_a
            side[2]=0.
            span=float(np.linalg.norm(side))
            if not 0.04<span<0.16 or abs(float(rim_a[2]-rim_left[2]))>0.008:
                return unresolved("Current measured rim sections disagree")
            side/=span
            inward=np.cross(side,up)
            if np.dot(inward,oven0-rim_a)<0:
                inward=-inward
            initial_rim=(rim_a+rim_left)/2.
            if not -0.010 < float(oven0[2]-initial_rim[2]) < 0.060:
                return unresolved("Door upper edge is outside the near-upright roof height band")
            markers=[]
            for box,point in [(params["rim_box"],rim_a),(params["second_box"],rim_left)]:
                rgb,depth,k,t=record("right",initial["step"])
                x0,y0,x1,y1=box
                color=np.median(rgb[y0:y1,x0:x1].reshape(-1,3),axis=0)
                markers.append({"camera":"right","point":point.copy(),"initial":point.copy(),
                    "color":color,"missing":0})
            q_hold=initial["eef_control_pose"]["quaternion_xyzw"]
            def pad_corners(obs):
                geom=obs.get("controller",{}).get("gripper_kinematics",{}).get("fingerpad_geometry",{})
                if geom.get("available") is not True or geom.get("step")!=obs["step"] or geom.get("frame")!="eef_control_site":
                    raise ValueError("Fresh measured fingerpad boxes unavailable")
                corners=[]
                for pad in geom["pads"]:
                    center=np.asarray(pad["center_eef"],float)
                    matrix=np.asarray(pad["rotation_eef"],float)
                    half=np.asarray(pad["half_extents_m"],float)
                    for x in (-1,1):
                        for y in (-1,1):
                            for z in (-1,1):
                                corners.append(center+matrix@(half*np.array([x,y,z])))
                if len(corners)!=16:
                    raise ValueError("Expected two measured fingerpad boxes")
                return np.asarray(corners)
            initial_corners=pad_corners(initial)
            # This sphere is robot pad geometry, not object/contact evidence.
            # A fixed extra band accommodates opening before high-clearance rotation.
            sweep_radius=float(np.max(np.linalg.norm(initial_corners,axis=1)))+0.025
            if not 0.030 < sweep_radius < 0.100:
                return unresolved("Measured pad sweep is outside the bounded clearance model")
            contact_lateral=0.5*float(np.dot(initial["eef"]+initial_rim,side))
            last_eef=initial["eef"].copy()
            last_step=initial["step"]
            last_oven=oven0.copy()
            cache={"step":last_step,"eef":last_eef.copy(),"pose":initial["eef_control_pose"],
                "oven":oven0.copy(),"rim":initial_rim.copy(),"advance":0.0,"views":2}
            stage.record(step=initial["step"],near_rim_m=rim_a.tolist(),far_rim_m=rim_left.tolist(),
                roof_m=oven0.tolist(),inward=inward.tolist(),pad_sweep_radius_m=sweep_radius,
                finger_positions=initial["state"][-2:])
            ctx.note(facts=["Current measured rim strips agree in height near the roof. No physical hinge center is established; track local door advance relative to the oven without an arc assumption."],
                hypotheses=["A constrained finger may obstruct seating. Preserve current wrist orientation during bounded upward release, then recontact from the exterior after measured pad clearance."])

        def measure():
            nonlocal last_eef,last_step,last_oven,cache
            current=pose()
            if current["step"]==last_step:
                return cache
            oven,om=metric("fixed oven top","right",params["oven_box"],current["step"])
            if np.linalg.norm(oven-oven0)>0.015 or om.get("depth_iqr_m",1.)>0.035:
                raise ValueError("Oven reference displaced or became unreliable")
            shift=oven-oven0
            hand_delta=current["eef"]-last_eef
            normal_delta=hand_delta-np.dot(hand_delta,side)*side
            points=[]
            advances=[]
            for item in markers:
                camera=item["camera"]
                rgb,depth,k,t=record(camera,current["step"])
                # Prediction only selects an image neighborhood. Every accepted
                # point below is fresh metric depth, never the predicted position.
                expected=item["point"]+(oven-last_oven)
                # Hand motion is deliberately excluded during release/retreat:
                # it is not evidence that the door moved.
                uv=project(expected,k,t)
                u0,v0=int(round(float(uv[0]))),int(round(float(uv[1])))
                candidates=[]
                for v in range(max(1,v0-5),min(254,v0+6)):
                    for u in range(max(2,u0-5),min(253,u0+6)):
                        box=[u-2,v,u+3,v+1]
                        patch=point_patch(box,depth,k,t)
                        if patch is None:
                            continue
                        p,cloud=patch
                        if np.linalg.norm(p-expected)>0.028 or np.linalg.norm(p-item["point"])>0.032:
                            continue
                        if np.max(np.linalg.norm(cloud-p,axis=1))>0.012:
                            continue
                        lateral=abs(float(np.dot(p-item["initial"]-shift,side)))
                        if lateral>0.008:
                            continue
                        relative=p-item["initial"]-shift
                        advance=float(np.dot(relative,inward))
                        if advance < -0.040 or advance > 0.065 or abs(float(relative[2]))>0.035:
                            continue
                        if abs(float(p[2]-expected[2]))>0.008 or abs(float(np.dot(p-expected,inward)))>0.020:
                            continue
                        color=np.median(rgb[v:v+1,u-1:u+2].reshape(-1,3),axis=0)
                        color_error=float(np.sqrt(np.mean((color-item["color"])**2)))
                        if color_error>75.:
                            continue
                        score=float(np.linalg.norm(p-expected)**2)+2.*lateral*lateral+4.*float(p[2]-expected[2])**2+0.00002*(color_error/75.)**2
                        candidates.append((score,box,p,color))
                candidates.sort(key=lambda x:x[0])
                accepted=None
                for _,box,p,color in candidates[:3]:
                    try:
                        actual,m=metric("tracked raised rim "+camera,camera,box,current["step"])
                        spread=np.asarray(m["max_m"])-np.asarray(m["min_m"])
                        if m.get("depth_iqr_m",1.)>0.015 or spread[2]>0.015 or np.linalg.norm(actual-p)>0.006:
                            continue
                        if np.linalg.norm(actual-expected)>0.028:
                            continue
                        accepted=(actual,color)
                        break
                    except (ValueError,KeyError,TypeError):
                        continue
                if accepted is None:
                    item["missing"]+=1
                    continue
                actual,color=accepted
                item["point"]=actual
                item["color"]=0.8*item["color"]+0.2*color
                item["missing"]=0
                advances.append(float(np.dot(actual-item["initial"]-shift,inward)))
                points.append(actual)
            if not points:
                raise ValueError("Neither selected rim section provides a reliable current metric measurement")
            # One section may be occluded by the exterior approach. A fresh,
            # reliable metric point is still mandatory at every segment.
            if len(advances)==2 and abs(advances[0]-advances[1])>0.006:
                raise ValueError("Current rim sections disagree on door advance")
            advance=float(np.median(advances))
            rim=np.mean(points,axis=0)
            rim+=(float(np.dot(initial_rim+shift,side))-float(np.dot(rim,side)))*side
            if advance < -0.040 or rim[2]<initial_rim[2]-0.035:
                raise ValueError("Door fell outside the measured near-upright recovery envelope")
            last_eef=current["eef"].copy()
            last_step=current["step"]
            last_oven=oven.copy()
            cache={"step":last_step,"eef":current["eef"],"pose":current["eef_control_pose"],
                "oven":oven,"rim":rim,"advance":advance,"views":len(points)}
            return cache

        class VerifiedDone(Exception):
            pass

        def segment(target,closed):
            act(target,q_hold,closed,steps=2)
            if done():
                raise VerifiedDone()
            return measure()

        def guarded_reach(name,target_fn,closed,max_segments,tolerance=0.005,guard_fn=None):
            with ctx.stage(name) as stage:
                window=measure()["eef"].copy()
                for index in range(max_segments):
                    before=measure()
                    if guard_fn is not None:
                        guard_fn(before)
                    target=np.asarray(target_fn(before),float)
                    error=float(np.linalg.norm(target-before["eef"]))
                    if error<tolerance:
                        stage.record(step=before["step"],target_error_m=error,
                            eef_m=before["eef"].tolist(),rim_m=before["rim"].tolist())
                        return
                    after=segment(target,closed)
                    if guard_fn is not None:
                        guard_fn(after)
                    stage.record(step=after["step"],target_error_before_m=error,
                        eef_m=after["eef"].tolist(),rim_m=after["rim"].tolist(),
                        door_inward_advance_m=after["advance"],valid_markers=after["views"])
                    if index%4==3:
                        if float(np.linalg.norm(after["eef"]-window))<0.002:
                            raise ValueError("Four segments failed to move the hand during "+name)
                        window=after["eef"].copy()
                before=measure()
                if np.linalg.norm(np.asarray(target_fn(before))-before["eef"])>tolerance+0.002:
                    raise ValueError("Measured approach did not reach its bounded stage target: "+name)

        def height(c):
            return max(float(c["oven"][2]),float(c["rim"][2]))+sweep_radius+0.018

        def at_lateral(point,c):
            shifted=contact_lateral+float(np.dot(c["oven"]-oven0,side))
            return point+(shifted-float(np.dot(point,side)))*side

        def outside_target(c,standoff,z):
            p=at_lateral(c["rim"].copy()-standoff*inward,c)
            p[2]=z
            return p

        def outside_guard(c):
            depth=float(np.dot(c["eef"]-c["rim"],inward))
            if depth>-0.045:
                raise ValueError("Outside clearance was lost before lowering the closed gripper")

        def high_guard(c):
            if c["eef"][2]<height(c)-0.009:
                raise ValueError("Measured high clearance was lost during repositioning")
            current=pose()
            corners=pad_corners(current)
            radius_now=float(np.max(np.linalg.norm(corners,axis=1)))
            if radius_now>sweep_radius:
                raise ValueError("Current measured pad sweep exceeds clearance model")
            rotated=corners@Rotation.from_quat(current["eef_control_pose"]["quaternion_xyzw"]).as_matrix().T+current["eef"]
            if float(np.min(rotated[:,2]))<max(float(c["oven"][2]),float(c["rim"][2]))+0.009:
                raise ValueError("Measured pad boxes lack vertical clearance")

        def contact_guard(c):
            lateral=contact_lateral+float(np.dot(c["oven"]-oven0,side))
            if abs(float(np.dot(c["eef"],side))-lateral)>0.014:
                raise ValueError("Exterior approach lost measured lateral alignment")
            if abs(float(c["eef"][2]-c["rim"][2]+0.006))>0.015:
                raise ValueError("Exterior contact height left the observed rim band")

        try:
            with ctx.stage("release fingers at the current measured contact") as stage:
                hold=measure()["eef"].copy()
                for index in range(6):
                    current=segment(hold,False)
                    if np.linalg.norm(current["eef"]-hold)>0.012:
                        raise ValueError("Finger release displaced the hand beyond the bounded hold envelope")
                    joints=np.abs(np.asarray(decode(ctx.observe())["state"][-2:],float))
                    stage.record(step=current["step"],finger_positions=joints.tolist(),
                        eef_m=current["eef"].tolist(),rim_m=current["rim"].tolist())
                    if float(np.min(joints))>0.035:
                        break
                # Full opening may be constrained by the inner finger. The
                # following upward move tests clearance without an inward push.

            def lift_target(c):
                p=c["eef"].copy()
                p[2]=height(c)
                return p
            guarded_reach("lift released fingers above the measured oven lip",
                          lift_target,False,32,tolerance=0.005)

            joints=np.abs(np.asarray(decode(ctx.observe())["state"][-2:],float))
            if float(np.min(joints))<0.032:
                raise ValueError("Measured fingers remain constrained after the bounded upward release")
            high_guard(measure())

            guarded_reach("withdraw and align above the exterior rim",
                lambda c: outside_target(c,0.065,height(c)),
                False,28,tolerance=0.005,guard_fn=high_guard)

            with ctx.stage("close fingers after measured exterior clearance") as stage:
                outside_guard(measure())
                hold=measure()["eef"].copy()
                closed=False
                for index in range(9):
                    current=segment(hold,True)
                    outside_guard(current)
                    high_guard(current)
                    joints=np.abs(np.asarray(decode(ctx.observe())["state"][-2:],float))
                    stage.record(step=current["step"],finger_positions=joints.tolist())
                    if float(np.max(joints))<0.006:
                        closed=True
                        break
                if not closed:
                    raise ValueError("Finger closure remained constrained outside the door")

            with ctx.stage("align closed fingers downward with measured high exterior clearance") as stage:
                current=measure()
                outside_guard(current)
                high_guard(current)
                hold=current["eef"].copy()
                measured_q=Rotation.from_quat(current["pose"]["quaternion_xyzw"])
                direction=measured_q.apply(np.array([0.,0.,1.]))
                cross=np.cross(direction,-up)
                sine=float(np.linalg.norm(cross))
                cosine=float(np.clip(np.dot(direction,-up),-1.,1.))
                # Larger orientation change is permitted only here, after the
                # complete measured pad sweep clears the roof and door.
                high_guard(current)
                if sine<1e-8 and cosine<0:
                    raise ValueError("Antipodal wrist correction is not defined")
                correction=Rotation.from_rotvec(cross*(np.arctan2(sine,cosine)/max(sine,1e-9)))
                desired=correction*measured_q
                q_hold=desired.as_quat().tolist()
                aligned=False
                for index in range(40):
                    current=measure()
                    outside_guard(current)
                    high_guard(current)
                    angular_error=float((desired*Rotation.from_quat(current["pose"]["quaternion_xyzw"]).inv()).magnitude())
                    if angular_error<0.035:
                        aligned=True
                        break
                    current=segment(hold,True)
                    outside_guard(current)
                    high_guard(current)
                    if np.linalg.norm(current["eef"]-hold)>0.010:
                        raise ValueError("Exterior wrist alignment displaced the hand beyond its hold bound")
                    stage.record(step=current["step"],orientation_error_before_rad=angular_error,
                        eef_m=current["eef"].tolist())
                if not aligned:
                    angular_error=float((desired*Rotation.from_quat(measure()["pose"]["quaternion_xyzw"]).inv()).magnitude())
                    if angular_error>=0.035:
                        raise ValueError("Bounded exterior wrist alignment did not converge")
                if np.max(np.abs(np.asarray(decode(ctx.observe())["state"][-2:],float)))>=0.006:
                    raise ValueError("Fingers did not remain closed during exterior wrist alignment")

            guarded_reach("lower the closed gripper outside the door",
                lambda c: outside_target(c,0.065,float(c["rim"][2])-0.006),
                True,22,tolerance=0.005,guard_fn=outside_guard)

            guarded_reach("approach the visible rim from its exterior",
                lambda c: outside_target(c,0.022,float(c["rim"][2])-0.006),
                True,20,tolerance=0.004,guard_fn=contact_guard)

            with ctx.stage("seat the door with measured exterior contact") as stage:
                window=measure()
                window_rim=window["rim"].copy()
                window_oven=window["oven"].copy()
                contact_start=window["eef"].copy()
                stagnant=0
                for index in range(30):
                    before=measure()
                    contact_guard(before)
                    depth=float(np.dot(before["eef"]-before["rim"],inward))
                    if depth>0.027 or float(np.dot(before["eef"]-contact_start,inward))>0.085:
                        raise ValueError("Exterior seating reached its bounded penetration or travel envelope")
                    delta=0.010*inward
                    delta[2]=float(np.clip(before["rim"][2]-0.006-before["eef"][2],-0.004,0.004))
                    target=at_lateral(before["eef"]+delta,before)
                    after=segment(target,True)
                    stage.record(step=after["step"],rim_m=after["rim"].tolist(),
                        eef_m=after["eef"].tolist(),door_inward_advance_m=after["advance"],
                        valid_markers=after["views"])
                    if index%4==3:
                        progress=float(np.dot((after["rim"]-after["oven"])-(window_rim-window_oven),inward))
                        stage.record(step=after["step"],four_segment_door_inward_progress_m=progress)
                        # The first window can bridge the measured air gap.
                        if index>=7 and progress<0.0015:
                            stagnant+=1
                        else:
                            stagnant=0
                        if stagnant>=1:
                            raise ValueError("Four exterior closing segments produced no measurable door advance")
                        window_rim=after["rim"].copy()
                        window_oven=after["oven"].copy()
            return unresolved("The complete exterior seating attempt remains unverified")
        except VerifiedDone:
            return "SUBTASK_DONE"
    except (ValueError,KeyError,TypeError) as error:
        return unresolved("Exterior closure prerequisite or feedback failed: "+str(error))
