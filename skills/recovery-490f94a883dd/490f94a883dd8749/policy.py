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
        ctx.note(facts=[reason], hypotheses=["The entire door-closing subtask remains assigned to code until task_success is verified."])
        return "UNRESOLVED"

    if done():
        return "SUBTASK_DONE"
    initial = unpack(ctx.observe())
    if initial.get("step") != 1267:
        return unresolved("Current image selections require fresh grounding.")
    caps = unpack(ctx.capabilities())
    if goal not in caps.get("verifiers", []) or not all(
            name in caps.get("capabilities", []) for name in ["rgbd", "osc_eef", "gripper"]):
        return unresolved("Required capabilities or completion verifier are unavailable.")

    def rgb_now():
        rgb = np.asarray(ctx.read_observation("left")["rgb"], dtype=float)
        if rgb.shape != (256,256,3):
            raise ValueError("Unexpected RGB dimensions")
        return rgb*255.0 if np.max(rgb)<=1.0 else rgb

    selections = {
        "handle": {"context":[110,108,126,117], "patch":[115,112,121,114],
                   "label":"exposed silver handle near its left end"},
        "body_left": {"context":[115,105,121,107], "patch":[115,105,121,107],
                      "label":"fixed oven upper front left surface"},
        "body_right": {"context":[177,106,183,108], "patch":[177,106,183,108],
                       "label":"fixed oven upper front right surface"}
    }
    first_rgb=rgb_now()
    for item in selections.values():
        a,b,c,d=item["context"]
        x0,y0,x1,y1=item["patch"]
        item["template"]=first_rgb[b:d,a:c].copy()
        item["brightness"]=float(np.mean(first_rgb[y0:y1,x0:x1]))
        item["offset"]=[0,0]
    previous={}
    anchors={}
    initial_normal=None
    initial_span=None

    def geometry_at(label,bbox,step):
        ground=unpack(ctx.ground_target(target=label,camera="left",
            observation_step=step,bbox=bbox,confidence=0.85))
        return unpack(ctx.query_geometry(ground["grounding_id"]))

    def valid_geometry(geo,step,name):
        if (geo.get("reliable") is not True or geo.get("frame")!="osc_arm_base"
                or geo.get("observation_step")!=step or geo.get("valid_fraction",0)<0.9):
            return False
        p=np.asarray(geo["centroid_m"],dtype=float)
        lo=np.asarray(geo["min_m"],dtype=float)
        hi=np.asarray(geo["max_m"],dtype=float)
        limit=0.015 if name=="handle" else 0.012
        if not np.all(np.isfinite(p)) or geo.get("depth_iqr_m",1)>limit or hi[2]-lo[2]>limit:
            return False
        if name in previous:
            motion_limit=0.020 if name=="handle" else 0.006
            if np.linalg.norm(p-previous[name])>motion_limit:
                return False
        if name!="handle" and name in anchors and np.linalg.norm(p-anchors[name])>0.008:
            return False
        return True

    def measure():
        nonlocal initial_normal,initial_span
        obs=unpack(ctx.observe())
        pose=obs["eef_control_pose"]
        if pose.get("step")!=obs["step"] or pose.get("frame")!="osc_arm_base":
            raise ValueError("Fresh measured OSC control pose is unavailable")
        rgb=rgb_now()
        points={}
        updates={}
        for name,item in selections.items():
            a,b,c,d=item["context"]
            x0,y0,x1,y1=item["patch"]
            ox,oy=item["offset"]
            choices=[]
            # The two oven patches are freshly remeasured at their original
            # pixels. They remain fixed only while both appearance and metric
            # anchor checks support this; handle tracking may translate.
            dy_choices=range(oy-3,oy+4) if name=="handle" else [0]
            dx_choices=range(ox-3,ox+4) if name=="handle" else [0]
            for dy in dy_choices:
                for dx in dx_choices:
                    if abs(dx)>12 or abs(dy)>12 or min(a+dx,b+dy)<0 or max(c+dx,d+dy)>256:
                        continue
                    crop=rgb[b+dy:d+dy,a+dx:c+dx]
                    loss=float(np.mean((crop-item["template"])**2))
                    brightness=float(np.mean(rgb[y0+dy:y1+dy,x0+dx:x1+dx]))
                    if loss<=900.0 and abs(brightness-item["brightness"])<=25.0:
                        score=loss+8.0*((dx-ox)**2+(dy-oy)**2)
                        choices.append((score,loss,dx,dy))
            choices.sort()
            accepted=None
            # Only nearby, appearance-matched candidates of the selected surface
            # may replace a patch rejected by current measured depth.
            for score,loss,dx,dy in choices[:5]:
                if score>choices[0][0]+200.0:
                    break
                geo=geometry_at(item["label"],[x0+dx,y0+dy,x1+dx,y1+dy],obs["step"])
                if valid_geometry(geo,obs["step"],name):
                    accepted=(np.asarray(geo["centroid_m"],dtype=float),dx,dy)
                    break
            if accepted is None:
                raise ValueError("No visible, depth-consistent candidate for "+name)
            p,dx,dy=accepted
            points[name]=p
            updates[name]=(dx,dy,rgb[b+dy:d+dy,a+dx:c+dx].copy())

        tangent=points["body_right"]-points["body_left"]
        span=float(np.linalg.norm(tangent[:2]))
        if not 0.28<span<0.39 or abs(tangent[2])>0.015:
            raise ValueError("The measured oven references no longer define its horizontal front")
        tangent[2]=0.0
        tangent/=np.linalg.norm(tangent)
        normal=np.cross(np.array([0.0,0.0,1.0]),tangent)
        body=(points["body_left"]+points["body_right"])*0.5
        if np.dot(normal,body-points["handle"])<0:
            normal=-normal
        if initial_normal is None:
            initial_normal=normal.copy()
            initial_span=span
        elif np.dot(normal,initial_normal)<0.995 or abs(span-initial_span)>0.012:
            raise ValueError("The observed oven orientation or reference spacing changed")
        gap=float(np.dot(body-points["handle"],normal))
        if not 0.005<gap<0.10:
            raise ValueError("The handle left the measured final-seating region")

        for name,(dx,dy,template) in updates.items():
            selections[name]["offset"]=[dx,dy]
            selections[name]["template"]=template
            previous[name]=points[name].copy()
            if name!="handle" and name not in anchors:
                anchors[name]=points[name].copy()
        return {"step":obs["step"],"handle":points["handle"],"body":body,
                "normal":normal,"tangent":tangent,"gap":gap,
                "eef":np.asarray(pose["position"],dtype=float),
                "quaternion":np.asarray(pose["quaternion_xyzw"],dtype=float)}

    try:
        with ctx.stage("validate current near-contact geometry and closing direction") as stage:
            start=measure()
            middle_geo=geometry_at("exposed silver handle middle upper strip",
                                   [128,112,134,114],start["step"])
            if not valid_geometry(middle_geo,start["step"],"handle_initial"):
                return unresolved("The initial handle direction lacks reliable measured depth.")
            middle=np.asarray(middle_geo["centroid_m"],dtype=float)
            handle_axis=middle-start["handle"]
            handle_span=float(np.linalg.norm(handle_axis[:2]))
            if not 0.045<handle_span<0.09 or abs(handle_axis[2])>0.012:
                return unresolved("The exposed handle sections do not define the expected bar.")
            handle_axis[2]=0.0
            handle_axis/=np.linalg.norm(handle_axis)
            if np.dot(handle_axis,start["tangent"])<0.995:
                return unresolved("Current handle and oven directions disagree.")
            depth=float(np.dot(start["eef"]-start["handle"],start["normal"]))
            side=float(np.dot(start["eef"]-start["handle"],start["tangent"]))
            height=float(start["eef"][2]-start["handle"][2])
            axis=Rotation.from_quat(start["quaternion"]).apply([0,0,1])
            if not -0.04<depth<0.012 or not 0.10<side<0.25 or not -0.005<height<0.025:
                return unresolved("Current hand pose does not meet the near-contact prerequisites.")
            if np.dot(axis,start["normal"])<0.90:
                return unresolved("Current hand orientation is inconsistent with inward seating.")
            start["contact_side"]=side
            held_quaternion=start["quaternion"].tolist()
            height_offset=max(0.0,height)
            stage.record(eef_depth_m=depth,contact_offset_m=side,height_offset_m=height,
                         handle_front_gap_m=start["gap"],
                         normal=start["normal"].tolist())

        with ctx.stage("seat the door from current contact and verify the whole task") as stage:
            history=[start]
            current=start
            for index in range(60):
                if current["step"]-initial["step"]+2>120:
                    return unresolved("This revision reached its 120-step limit.")
                if done():
                    return "SUBTASK_DONE"
                delta=current["normal"]*0.008
                delta[2]=float(np.clip(
                    current["handle"][2]+height_offset-current["eef"][2],0.0,0.002))
                target={"position":(current["eef"]+delta).tolist(),
                        "quaternion_xyzw":held_quaternion,
                        "gripper_closed":True,"base_motion":[0,0,0,0],"control_mode":"arm"}
                ctx.move_eef(target=target,steps=2)
                if done():
                    return "SUBTASK_DONE"
                after=measure()
                closing=float(np.dot(after["handle"]-current["handle"],current["normal"]))
                eef_motion=float(np.linalg.norm(after["eef"]-current["eef"]))
                depth=float(np.dot(after["eef"]-after["handle"],after["normal"]))
                side=float(np.dot(after["eef"]-after["handle"],after["tangent"]))
                stage.record(segment=index,observation_step=after["step"],
                             handle_inward_change_m=closing,eef_motion_m=eef_motion,
                             eef_depth_m=depth,handle_front_gap_m=after["gap"])
                if (np.dot(after["eef"]-start["eef"],start["normal"])>0.060
                        or depth>0.018 or abs(side-start["contact_side"])>0.04):
                    return unresolved("Measured travel reached the contact-region bound.")
                if (after["handle"][2]<start["handle"][2]-0.015
                        or after["eef"][2]>start["eef"][2]+0.030
                        or after["eef"][2]-after["handle"][2]>0.030
                        or after["gap"]>start["gap"]+0.015):
                    return unresolved("The measured door or hand left the seating envelope.")
                history.append(after)
                if len(history)>=7:
                    earlier=history[-7]
                    door_progress=float(np.dot(after["handle"]-earlier["handle"],after["normal"]))
                    arm_progress=float(np.dot(after["eef"]-earlier["eef"],after["normal"]))
                    if door_progress<0.001 and arm_progress<0.003:
                        return unresolved("Six short seating segments produced no useful measured progress.")
                if len(history)>=13:
                    earlier=history[-13]
                    door_progress=float(np.dot(after["handle"]-earlier["handle"],after["normal"]))
                    if door_progress<0.001:
                        return unresolved("Twelve short segments did not establish inward door progress.")
                current=after
        return "SUBTASK_DONE" if done() else unresolved("Bounded seating remains unverified.")
    except (ValueError,KeyError,TypeError) as error:
        return unresolved("Current-state recovery stopped: "+str(error))
