from drawer_pull_from_handle_hold.controller import execute as pull_held_drawer
from open_drawer_from_missed_grasp.controller import held_parameters
import numpy as np
from scipy.spatial.transform import Rotation
from eef_feedback_control import pose, move_bounded
from rgbd_surface_geometry import decode
from .perception import plane,handle
from .params import *

def pads(obs):
    k=obs["controller"]["gripper_kinematics"]
    g=k["fingerpad_geometry"]
    if not g.get("available") or g.get("step")!=obs["step"] or g.get("frame")!="eef_control_site":
        raise ValueError("Missing fresh measured fingerpad geometry")
    p=np.array([v["center_eef"] for v in g["pads"]])
    if len(p)!=2: raise ValueError("Expected two pads")
    return p.mean(0),float(np.linalg.norm(p[1]-p[0]))

def execute(ctx,camera,handle_roi,front_roi,tracking_roi):
    start=pose(ctx)["step"]; report={}
    def done():
        return decode(ctx.check("task_success"))["passed"] is True
    def bound():
        if pose(ctx)["step"]-start+2>MAX_STEPS: raise ValueError("Recovery action bound reached")
    def goto(target,q,closed,max_steps=70):
        first=pose(ctx); initial=np.linalg.norm(np.asarray(target)-first["eef"])
        olderr=None; stalled=0
        for _ in range(max_steps//2):
            bound()
            ob=pose(ctx)
            err=float(np.linalg.norm(target-ob["eef"]))
            angle=float((Rotation.from_quat(q)*Rotation.from_quat(ob["eef_control_pose"]["quaternion_xyzw"]).inv()).magnitude())
            if err<POSE_TOL and angle<ROT_TOL: return
            delta=target-ob["eef"]
            if np.linalg.norm(delta)>MAX_SEGMENT: delta*=MAX_SEGMENT/np.linalg.norm(delta)
            move_bounded(ctx,ob["eef"]+delta,q,closed,start_step=start,limit=MAX_STEPS,steps=2)
            now=pose(ctx)
            if np.linalg.norm(now["eef"]-ob["eef"])>0.06: raise ValueError("Unexpected measured motion")
            metric=err+0.04*angle
            stalled=stalled+1 if olderr is not None and metric>olderr-0.0004 else 0
            olderr=metric
            if stalled>=8: raise ValueError("Pose tracking stalled")
            if done(): return
        raise ValueError("Pose target not reached within segment bound")
    try:
        if done(): return {"ok":True,"measured":{},"why":"Installed verifier passed at entry"}
        ob=pose(ctx); offset,sep=pads(ob)
        with ctx.stage("measure_drawer") as stage:
            c,n,res=plane(ctx,camera,front_roi)
            h,hinfo=handle(ctx,camera,handle_roi,c,n)
            report.update(plane=c.tolist(),normal=n.tolist(),handle=h.tolist(),plane_residual=res)
            stage.record(**report)
        if sep<0.060:
            with ctx.stage("establish_open_missed_grasp") as stage:
                ob=pose(ctx)
                midpoint,_=pads(ob)
                actual=ob["eef"]+Rotation.from_quat(ob["eef_control_pose"]["quaternion_xyzw"]).apply(midpoint)
                if h[2]-actual[2]<0.060 or np.linalg.norm(h-actual)<0.080:
                    raise ValueError("Cannot establish separated below-handle entry")
                initial=ob["eef"].copy()
                for _ in range(10):
                    bound(); ctx.set_gripper(closed=False,steps=2)
                    ob=pose(ctx)
                    if np.linalg.norm(ob["eef"]-initial)>0.015:
                        raise ValueError("Unexpected wrist motion during opening")
                    _,sep=pads(ob)
                    if sep>=0.060: break
                if sep<0.060: raise ValueError("Open-entry prerequisite not achieved")
                h,hinfo=handle(ctx,camera,handle_roi,c,n,expected=h)
                stage.record(open_pad_separation_m=sep)
        with ctx.stage("clear_and_align") as stage:
            ob=pose(ctx); q0=ob["eef_control_pose"]["quaternion_xyzw"]
            goto(ob["eef"]+0.16*n+np.array([0.,0.,0.12]),q0,False,max_steps=100)
            up=np.array([0.,0.,1.]); up-=n*np.dot(up,n); up/=np.linalg.norm(up)
            # Control-site x is the jaw separation line; z points from wrist toward pads.
            z=-n; x=up; y=np.cross(z,x)
            rot=Rotation.from_matrix(np.column_stack((x,y,z))); q=rot.as_quat()
            goto(pose(ctx)["eef"],q,False)
            h,hinfo=handle(ctx,camera,handle_roi,c,n,expected=h)
            midpoint,_=pads(pose(ctx))
            pre=h+0.075*n-rot.apply(midpoint)
            goto(pre,q,False)
            stage.record(eef=pose(ctx)["eef"].tolist(),target_handle=h.tolist())
        with ctx.stage("seat_and_close") as stage:
            h,hinfo=handle(ctx,camera,handle_roi,c,n,expected=h)
            midpoint,_=pads(pose(ctx))
            # Seat 6 mm behind the measured outer silver surface, retaining cabinet clearance.
            seat=h-0.006*n-rot.apply(midpoint)
            if np.dot((seat+rot.apply(midpoint))-c,n)<0.016:
                raise ValueError("Insufficient measured cabinet clearance")
            goto(seat,q,False)
            if done(): return {"ok":True,"measured":report,"why":"Installed verifier passed"}
            before=pose(ctx)
            actual=before["eef"]+Rotation.from_quat(before["eef_control_pose"]["quaternion_xyzw"]).apply(pads(before)[0])
            if np.linalg.norm(actual-(h-0.006*n))>0.012: raise ValueError("Pad center misses handle")
            # Closure is a probe, never a grasp claim.
            for _ in range(10):
                bound(); ctx.set_gripper(closed=True,steps=2)
            _,sep=pads(pose(ctx))
            report["closed_pad_separation"]=sep
            if sep<0.010 or sep>0.055: raise ValueError("Closure does not support handle engagement")
            stage.record(**report)
        with ctx.stage("pull_with_reused_surface_skill"):
            pts=np.asarray(hinfo["points"])
            up=np.array([0.,0.,1.]); tangent=np.cross(up,n);tangent/=np.linalg.norm(tangent)
            along=pts@tangent
            anchors=pts[[int(np.argmin(along)),int(np.argmax(along))]]
            return pull_held_drawer(ctx,held_parameters(ctx,c,n,anchors))
    except (ValueError,KeyError,TypeError) as exc:
        ctx.note(facts=["Recovery stopped: "+str(exc)],hypotheses=["Alignment, occlusion or contact may require a measured revision."])
        return {"ok":False,"measured":report,"why":str(exc)}
