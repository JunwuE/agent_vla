import json
import numpy as np
from scipy.spatial.transform import Rotation

def run(ctx):
    goal = "task_success"
    def unpack(value):
        return json.loads(value.split("\ndata:image", 1)[0]) if isinstance(value, str) else value
    def done():
        return unpack(ctx.check(goal)).get("passed") is True
    def unresolved(reason):
        ctx.note(facts=[reason], hypotheses=["The whole closing task remains assigned to code until independently verified."])
        return "UNRESOLVED"
    if done():
        return "SUBTASK_DONE"
    initial = unpack(ctx.observe())
    if initial.get("step") != 1233:
        return unresolved("The selected left-camera regions require fresh selection.")
    caps = unpack(ctx.capabilities())
    if goal not in caps.get("verifiers", []) or not all(
            c in caps.get("capabilities", []) for c in ["rgbd", "osc_eef", "gripper"]):
        return unresolved("Required verifier or capability is unavailable.")

    def rgb_now():
        rgb = np.asarray(ctx.read_observation("left")["rgb"], dtype=float)
        if rgb.shape != (256, 256, 3):
            raise ValueError("Unexpected left-camera image dimensions")
        return rgb * 255.0 if np.max(rgb) <= 1.0 else rgb

    selections = {
        "left": {"context": [110,109,126,120], "patch": [115,114,122,116],
                 "label": "exposed silver handle section near left end"},
        "right": {"context": [135,110,145,119], "patch": [139,115,145,117],
                  "label": "exposed silver handle section left of gripper"},
        "oven": {"context": [131,87,171,108], "patch": [143,95,153,99],
                 "label": "oven upper surface behind the handle"}
    }
    first_rgb = rgb_now()
    for item in selections.values():
        a,b,c,d = item["context"]
        item["template"] = first_rgb[b:d,a:c].copy()
        item["offset"] = [0,0]
    previous = {}
    previous_normal = None
    contact_offset = None
    initial_handle_span = None

    def measure():
        nonlocal previous_normal, contact_offset, initial_handle_span
        obs = unpack(ctx.observe())
        rgb = rgb_now()
        points = {}
        updates = {}
        for name,item in selections.items():
            a,b,c,d = item["context"]
            ox,oy = item["offset"]
            choices = []
            for dy in range(oy-4,oy+5):
                for dx in range(ox-4,ox+5):
                    if abs(dx)>14 or abs(dy)>14 or min(a+dx,b+dy)<0 or max(c+dx,d+dy)>256:
                        continue
                    crop=rgb[b+dy:d+dy,a+dx:c+dx]
                    loss=float(np.mean((crop-item["template"])**2))
                    score=loss+8.0*((dx-ox)**2+(dy-oy)**2)
                    choices.append((score,loss,dx,dy))
            if not choices:
                raise ValueError("No valid target tracking region")
            score,loss,dx,dy=min(choices)
            if loss>900.0:
                raise ValueError("Selected "+name+" tracking context became obscured")
            x0,y0,x1,y1=item["patch"]
            ground=unpack(ctx.ground_target(target=item["label"],camera="left",
                observation_step=obs["step"],bbox=[x0+dx,y0+dy,x1+dx,y1+dy],confidence=0.85))
            geo=unpack(ctx.query_geometry(ground["grounding_id"]))
            if geo.get("reliable") is not True or geo.get("frame")!="osc_arm_base":
                raise ValueError("Metric geometry is unavailable or unreliable")
            if geo.get("observation_step")!=obs["step"] or geo.get("valid_fraction",0)<0.9:
                raise ValueError("Geometry is stale or lacks valid depth")
            p=np.asarray(geo["centroid_m"],dtype=float)
            lo=np.asarray(geo["min_m"],dtype=float)
            hi=np.asarray(geo["max_m"],dtype=float)
            if not np.all(np.isfinite(p)) or geo.get("depth_iqr_m",1.0)>0.035 or hi[2]-lo[2]>0.025:
                raise ValueError("Selected patch contains inconsistent visible depths")
            if name in previous and np.linalg.norm(p-previous[name])>0.030:
                raise ValueError("A tracked surface exceeded the short-step motion bound")
            points[name]=p
            updates[name]=(dx,dy,rgb[b+dy:d+dy,a+dx:c+dx].copy())
        tangent=points["right"]-points["left"]
        length=float(np.linalg.norm(tangent[:2]))
        if not 0.08<length<0.18 or abs(tangent[2])>0.015:
            raise ValueError("The two selected patches do not define the expected horizontal handle")
        tangent[2]=0.0
        tangent/=np.linalg.norm(tangent)
        if initial_handle_span is None:
            initial_handle_span=length
        elif abs(length-initial_handle_span)>0.020:
            raise ValueError("Measured spacing between handle sections changed")
        pose=obs["eef_control_pose"]
        if pose.get("step")!=obs["step"] or pose.get("frame")!="osc_arm_base":
            raise ValueError("Current OSC control pose is unavailable")
        eef=np.asarray(pose["position"],dtype=float)
        if contact_offset is None:
            contact_offset=float(np.dot(eef-points["left"],tangent))
        center=points["left"]+contact_offset*tangent
        center[2]=(points["right"][2]+points["left"][2])*0.5
        normal=np.cross(np.array([0.0,0.0,1.0]),tangent)
        if np.dot(normal,points["oven"]-center)<0:
            normal=-normal
        if previous_normal is not None and np.dot(normal,previous_normal)<0.98:
            raise ValueError("Measured handle normal changed abruptly")
        gap=float(np.dot(points["oven"]-center,normal))
        if not 0.06<gap<0.30:
            raise ValueError("Current door position is outside the final-seating envelope")
        for name,(dx,dy,template) in updates.items():
            selections[name]["offset"]=[dx,dy]
            selections[name]["template"]=template
            previous[name]=points[name].copy()
        previous_normal=normal.copy()
        pose=obs["eef_control_pose"]
        return {"step":obs["step"],"handle":center,"normal":normal,"tangent":tangent,
                "gap":gap,"left":points["left"],"eef":eef,
                "quaternion":np.asarray(pose["quaternion_xyzw"],dtype=float)}

    def move(delta):
        obs=unpack(ctx.observe())
        if obs["step"]-initial["step"]+2>160:
            raise ValueError("This revision reached its 160-step limit")
        delta=np.asarray(delta,dtype=float)
        length=float(np.linalg.norm(delta))
        if length>0.012:
            delta*=0.012/length
        ctx.move_relative(delta_position=delta.tolist(),gripper_closed=True,steps=2)

    try:
        with ctx.stage("confirm exposed handle geometry and preserve current contact line") as stage:
            start=measure()
            far_ground=unpack(ctx.ground_target(
                target="visible right silver handle strip for initial contact-line bounds",
                camera="left",observation_step=start["step"],
                bbox=[163,116,171,118],confidence=0.85))
            far_geo=unpack(ctx.query_geometry(far_ground["grounding_id"]))
            if (far_geo.get("reliable") is not True or far_geo.get("frame")!="osc_arm_base"
                    or far_geo.get("observation_step")!=start["step"]
                    or far_geo.get("valid_fraction",0)<0.9
                    or far_geo.get("depth_iqr_m",1)>0.015):
                return unresolved("The initial contact-line bounds lack reliable geometry.")
            far=np.asarray(far_geo["centroid_m"],dtype=float)
            span=float(np.dot(far-start["left"],start["tangent"]))
            deviation=far-start["left"]-span*start["tangent"]
            if (not np.all(np.isfinite(far)) or np.linalg.norm(deviation)>0.012
                    or not 0.04<contact_offset<span-0.025):
                return unresolved("The current approach does not project safely inside the measured handle span.")
            depth=float(np.dot(start["eef"]-start["handle"],start["normal"]))
            side=float(np.dot(start["eef"]-start["handle"],start["tangent"]))
            approach_axis=Rotation.from_quat(start["quaternion"]).apply([0,0,1])
            if not -0.15<depth<0.015 or abs(side)>0.12 or abs(start["eef"][2]-start["handle"][2])>0.08:
                return unresolved("Current measured pose does not satisfy the outside approach prerequisites.")
            if np.dot(approach_axis,start["normal"])<0.85:
                return unresolved("Current measured gripper orientation is not aligned with the door.")
            stage.record(eef_depth_m=depth,lateral_offset_m=side,
                         handle_m=start["handle"].tolist(),inward=start["normal"].tolist(),
                         contact_offset_m=contact_offset,measured_handle_span_m=span)

        with ctx.stage("complete current approach while tracking exposed handle sections") as stage:
            reached=False
            stagnant=0
            for index in range(45):
                before=measure()
                target=before["handle"]-before["normal"]*0.008
                delta=target-before["eef"]
                if np.linalg.norm(delta)<0.007:
                    reached=True
                    break
                move(delta)
                if done():
                    return "SUBTASK_DONE"
                after=measure()
                moved=float(np.linalg.norm(after["eef"]-before["eef"]))
                closing=float(np.dot(after["handle"]-before["handle"],before["normal"]))
                stage.record(segment=index,observation_step=after["step"],
                             eef_motion_m=moved,handle_inward_change_m=closing)
                stagnant=stagnant+1 if moved<0.0004 and closing<0.0004 else 0
                if stagnant>=4:
                    return unresolved("Approach stopped producing measured arm or door progress.")
                if after["handle"][2]<start["handle"][2]-0.030:
                    return unresolved("The door descended out of the final-seating envelope.")
            if not reached:
                return unresolved("The bounded current-state approach did not reach the handle.")

        with ctx.stage("close and verify the whole door task from current contact") as stage:
            stagnant=0
            for index in range(35):
                before=measure()
                delta=before["normal"]*0.010
                delta[2]=float(np.clip(before["handle"][2]+0.001-before["eef"][2],-0.002,0.003))
                move(delta)
                if done():
                    return "SUBTASK_DONE"
                after=measure()
                closing=float(np.dot(after["handle"]-before["handle"],before["normal"]))
                stage.record(segment=index,observation_step=after["step"],
                             handle_inward_change_m=closing,
                             eef_m=after["eef"].tolist())
                if np.dot(after["eef"]-start["eef"],start["normal"])>0.160:
                    return unresolved("Measured inward travel reached the recovery bound.")
                if after["handle"][2]<start["handle"][2]-0.030:
                    return unresolved("The door descended during the closing push.")
                stagnant=stagnant+1 if closing<0.0003 else 0
                if stagnant>=5:
                    return unresolved("The new contact did not produce consistent door-closing progress.")
        return "SUBTASK_DONE" if done() else unresolved("The bounded final-seating attempt remains unverified.")
    except (ValueError,KeyError,TypeError) as error:
        return unresolved("Recovery observation or prerequisite failed: "+str(error))
