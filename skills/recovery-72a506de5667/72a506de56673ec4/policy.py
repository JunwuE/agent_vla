def run(ctx):
    import json
    import numpy as np
    from scipy.spatial.transform import Rotation

    goal = "task_success"
    limit = 196
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
    if start.get("step") != 1208:
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

    def act(position,quaternion,steps=2):
        obs=pose()
        if obs["step"]-start["step"]+steps>limit:
            raise ValueError("Shared revision action bound reached")
        delta=np.asarray(position)-obs["eef"]
        length=float(np.linalg.norm(delta))
        if length>0.024:
            delta*=0.024/length
        measured=Rotation.from_quat(obs["eef_control_pose"]["quaternion_xyzw"])
        rv=(Rotation.from_quat(quaternion)*measured.inv()).as_rotvec()
        angle=float(np.linalg.norm(rv))
        if angle>0.14:
            rv*=0.14/angle
        target_rotation=Rotation.from_rotvec(rv)*measured
        ctx.move_eef(target={"position":(obs["eef"]+delta).tolist(),
            "quaternion_xyzw":target_rotation.as_quat().tolist(),
            "gripper_closed":True,"base_motion":[0,0,0,0],"control_mode":"arm"},steps=steps)

    try:
        with ctx.stage("remeasure raised rim in both cameras while preserving current contact") as stage:
            initial=pose()
            rim_a,_=metric("raised rim right thin strip","right",[63,175,69,176],initial["step"])
            rim_b,_=metric("raised rim second right strip","right",[70,176,75,177],initial["step"])
            rim_left,_=metric("raised rim left thin strip","left",[143,181,150,182],initial["step"])
            oven0,_=metric("fixed oven top","right",[65,98,75,102],initial["step"])
            rear,_=metric("visible hinge-side reference","right",[59,150,67,153],initial["step"])
            up=np.array([0.,0.,1.])
            side=rim_b-rim_a
            side[2]=0.
            span=float(np.linalg.norm(side))
            if not 0.025<span<0.10:
                return unresolved("Fresh thin rim strips do not establish an edge direction")
            side/=span
            inward=np.cross(side,up)
            if np.dot(inward,rear-rim_a)<0:
                inward=-inward
            axis=np.cross(up,inward)
            radius=float(np.dot(rear-rim_a,inward))
            if not 0.15<radius<0.29 or not 0.012<rim_a[2]-rear[2]<0.075:
                return unresolved("Current door does not satisfy the slightly raised closing prerequisite")
            if abs(float(rim_left[2]-rim_a[2]))>0.008:
                return unresolved("Fresh cameras disagree on raised rim height")
            pivot=rim_a+radius*inward
            pivot[2]=rear[2]
            markers=[]
            for camera,box,point in [
                ("right",[63,175,69,176],rim_a),
                ("left",[143,181,150,182],rim_left)
            ]:
                rgb,depth,k,t=record(camera,initial["step"])
                x0,y0,x1,y1=box
                color=np.median(rgb[y0:y1,x0:x1].reshape(-1,3),axis=0)
                markers.append({"camera":camera,"point":point.copy(),"initial":point.copy(),
                    "color":color,"missing":0})
            initial_q=Rotation.from_quat(initial["eef_control_pose"]["quaternion_xyzw"])
            initial_angle=float(np.arctan2(rim_a[2]-pivot[2],-np.dot(rim_a-pivot,inward)))
            last_eef=initial["eef"].copy()
            last_step=initial["step"]
            last_oven=oven0.copy()
            last_angle=initial_angle
            cache={"step":last_step,"eef":last_eef.copy(),"pose":initial["eef_control_pose"],
                "oven":oven0.copy(),"rim":rim_a.copy(),"angle":initial_angle,"views":2}
            stage.record(step=initial["step"],rim_right_m=rim_a.tolist(),rim_left_m=rim_left.tolist(),
                rear_surface_proxy_m=rear.tolist(),initial_angle_rad=initial_angle,
                finger_positions=initial["state"][-2:])
            ctx.note(facts=["Two current camera selections locate the raised top rim near the hand height."],
                hypotheses=["The rear visible surface is a hinge-location proxy, not an independently measured joint center. Short-segment object response determines whether closure continues."])

        def measure():
            nonlocal last_eef,last_step,last_oven,last_angle,cache
            current=pose()
            if current["step"]==last_step:
                return cache
            oven,om=metric("fixed oven top","right",[65,98,75,102],current["step"])
            if np.linalg.norm(oven-oven0)>0.015 or om.get("depth_iqr_m",1.)>0.035:
                raise ValueError("Oven reference displaced or became unreliable")
            shift=oven-oven0
            hand_delta=current["eef"]-last_eef
            normal_delta=hand_delta-np.dot(hand_delta,side)*side
            points=[]
            angles=[]
            for item in markers:
                camera=item["camera"]
                rgb,depth,k,t=record(camera,current["step"])
                # Prediction only selects an image neighborhood. Every accepted
                # point below is fresh metric depth, never the predicted position.
                expected=item["point"]+normal_delta
                expected+=np.dot(oven-last_oven,side)*side
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
                        if np.linalg.norm(p-expected)>0.018 or np.linalg.norm(p-item["point"])>0.025:
                            continue
                        if np.max(np.linalg.norm(cloud-p,axis=1))>0.012:
                            continue
                        lateral=abs(float(np.dot(p-item["initial"]-shift,side)))
                        if lateral>0.008:
                            continue
                        vector=p-pivot-shift
                        radial=float(np.hypot(np.dot(vector,inward),vector[2]))
                        if not 0.75*radius<radial<1.30*radius:
                            continue
                        theta=float(np.arctan2(vector[2],-np.dot(vector,inward)))
                        if abs(theta-last_angle)>0.10:
                            continue
                        color=np.median(rgb[v:v+1,u-1:u+2].reshape(-1,3),axis=0)
                        color_error=float(np.sqrt(np.mean((color-item["color"])**2)))
                        if color_error>75.:
                            continue
                        score=float(np.linalg.norm(p-expected)**2)+2.*lateral*lateral+0.00002*(color_error/75.)**2
                        candidates.append((score,box,p,color))
                candidates.sort(key=lambda x:x[0])
                accepted=None
                for _,box,p,color in candidates[:3]:
                    try:
                        actual,m=metric("tracked raised rim "+camera,camera,box,current["step"])
                        spread=np.asarray(m["max_m"])-np.asarray(m["min_m"])
                        if m.get("depth_iqr_m",1.)>0.015 or spread[2]>0.015 or np.linalg.norm(actual-p)>0.006:
                            continue
                        if np.linalg.norm(actual-expected)>0.018:
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
                vector=actual-pivot-shift
                angles.append(float(np.arctan2(vector[2],-np.dot(vector,inward))))
                points.append(actual)
            if not points:
                raise ValueError("Neither camera provides a reliable current rim measurement")
            if max(item["missing"] for item in markers)>4:
                raise ValueError("One rim view remained unavailable across five measured segments")
            if len(angles)==2 and abs(angles[0]-angles[1])>0.050:
                raise ValueError("Current rim cameras disagree on door angle")
            angle=float(np.median(angles))
            rim=np.mean(points,axis=0)
            last_eef=current["eef"].copy()
            last_step=current["step"]
            last_oven=oven.copy()
            last_angle=angle
            cache={"step":last_step,"eef":current["eef"],"pose":current["eef_control_pose"],
                "oven":oven,"rim":rim,"angle":angle,"views":len(points)}
            return cache

        with ctx.stage("continue the measured hinge-closing motion from established contact") as stage:
            window_angle=initial_angle
            window_eef=initial["eef"].copy()
            contact_radius=max(0.15,float(np.dot(pivot-initial["eef"],inward)))
            for index in range(98):
                before=measure()
                angle=before["angle"]
                if angle<initial_angle-0.06 or angle<window_angle-0.06:
                    return unresolved("Door rotation reversed; current engagement is no longer established")
                if angle<1.43:
                    arc_step=0.018 if index>=6 else 0.012
                    delta=arc_step*(np.sin(angle)*inward+np.cos(angle)*up)
                    desired=Rotation.from_rotvec(axis*(angle-initial_angle+arc_step/contact_radius))*initial_q
                else:
                    delta=0.010*inward+0.001*up
                    desired=Rotation.from_quat(before["pose"]["quaternion_xyzw"])
                act(before["eef"]+delta,desired.as_quat(),steps=2)
                if done():
                    return "SUBTASK_DONE"
                after=measure()
                stage.record(step=after["step"],door_angle_rad=after["angle"],
                    rim_m=after["rim"].tolist(),eef_m=after["eef"].tolist(),valid_views=after["views"],
                    finger_positions=decode(ctx.observe())["state"][-2:])
                if index%5==4:
                    advance=after["angle"]-window_angle
                    hand_move=float(np.linalg.norm(after["eef"]-window_eef))
                    if advance<0.018:
                        return unresolved("Five closing segments produced insufficient measured door rotation")
                    if hand_move>0.060 and advance<0.06:
                        return unresolved("Hand motion is no longer matched by door motion")
                    window_angle=after["angle"]
                    window_eef=after["eef"].copy()
            return unresolved("Closing remains unverified at the 196-step revision bound")
    except (ValueError,KeyError,TypeError) as error:
        return unresolved("Raised-door closure observation or control prerequisite failed: "+str(error))
