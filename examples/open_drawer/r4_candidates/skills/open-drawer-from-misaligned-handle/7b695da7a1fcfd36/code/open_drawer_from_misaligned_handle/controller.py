import numpy as np
from scipy.spatial.transform import Rotation
from eef_feedback_control import pose, move_bounded
from rgbd_surface_geometry import decode
from .perception import initialize, track
from .params import *

class Complete(Exception): pass

def success(ctx):
    return decode(ctx.check("task_success")).get("passed") is True

def pads(obs):
    k=obs["controller"]["gripper_kinematics"]["fingerpad_geometry"]
    if not k.get("available") or k["step"]!=obs["step"]:
        raise ValueError("Missing current measured pads")
    return k["pads"]

def drive(ctx,target,q,closed,start,loops=24):
    target=np.asarray(target,float)
    if target.shape!=(3,) or not np.all(np.isfinite(target)): raise ValueError("Invalid target")
    old=None
    stagnant=0
    for _ in range(loops):
        o=pose(ctx)
        if success(ctx): raise Complete()
        cur=Rotation.from_quat(o["eef_control_pose"]["quaternion_xyzw"])
        rv=(Rotation.from_quat(q)*cur.inv()).as_rotvec()
        err=target-o["eef"]
        if np.linalg.norm(err)<POSITION_TOL_M and np.linalg.norm(rv)<ANGLE_TOL_RAD: return o
        metric=np.linalg.norm(err)+0.07*np.linalg.norm(rv)
        if old is not None and old-metric<0.00015: stagnant+=1
        else: stagnant=0
        if stagnant>=9: raise ValueError("Measured pose convergence stalled")
        old=metric
        d=err*min(1,MAX_SEGMENT_M/max(np.linalg.norm(err),1e-9))
        rv=rv*min(1,MAX_ROTATION_RAD/max(np.linalg.norm(rv),1e-9))
        rot=(Rotation.from_rotvec(rv)*cur).as_quat()
        move_bounded(ctx,o["eef"]+d,rot,closed,start_step=start,limit=MAX_STEPS,steps=SEGMENT_STEPS)
    raise ValueError("Bounded pose convergence failed")

def hold(ctx,closed,start,count):
    for _ in range(count):
        o=pose(ctx)
        if success(ctx): raise Complete()
        move_bounded(ctx,o["eef"],o["eef_control_pose"]["quaternion_xyzw"],closed,start_step=start,limit=MAX_STEPS,steps=3)

def clearance(o,point,q,plane,normal):
    r=Rotation.from_quat(q).as_matrix()
    for pad in pads(o):
        center=point+r@np.asarray(pad["center_eef"])
        axes=r@np.asarray(pad["rotation_eef"])
        extent=np.abs(normal@axes)@np.asarray(pad["half_extents_m"])
        if np.dot(center-plane,normal)-extent<0.006:
            raise ValueError("Insufficient measured pad-to-face clearance")

def execute(ctx, observation_step, handle_boxes, face_boxes):
    measured={}
    try:
        o=pose(ctx)
        start=o["step"]
        if start!=observation_step: raise ValueError("Entry selection is stale")
        if success(ctx): return {"ok":True,"measured":{},"why":"Verifier already passes"}
        pp=pads(o)
        sep=np.linalg.norm(np.asarray(pp[0]["center_eef"])-np.asarray(pp[1]["center_eef"]))
        if sep<0.060: raise ValueError("Entry requires open empty gripper")
        c,plane,n,t=initialize(ctx,handle_boxes,face_boxes)
        measured.update(initial_handle=c.tolist(),outward_normal=n.tolist())
        q0=o["eef_control_pose"]["quaternion_xyzw"]
        with ctx.stage("retreat_from_misalignment"):
            retreat=o["eef"]+RETREAT_M*n+np.array([0.,0.,0.12])
            drive(ctx,retreat,q0,False,start)
        # Local z is finger extension. Local x is the measured jaw separation axis.
        z=-n
        x=np.cross(t,z); x=x/np.linalg.norm(x)
        if x[2]>0: x=-x
        y=np.cross(z,x); y=y/np.linalg.norm(y)
        matrix=np.column_stack([x,y,z])
        # Parallel jaws have an unsigned separation axis. Pick the equivalent
        # orientation nearer the measured wrist, keeping finger extension fixed.
        candidates=[Rotation.from_matrix(matrix),Rotation.from_matrix(matrix@np.diag([-1.,-1.,1.]))]
        current=Rotation.from_quat(pose(ctx)["eef_control_pose"]["quaternion_xyzw"])
        q=min(candidates,key=lambda r:(r*current.inv()).magnitude()).as_quat()
        with ctx.stage("orient_clear_of_drawer"):
            drive(ctx,retreat,q,False,start)
            c=track(ctx,c,t)
        # Goal is measured bar surface plus bounded seating offset, with pad-center correction.
        with ctx.stage("align_outside_handle"):
            o=pose(ctx)
            r=Rotation.from_quat(q).as_matrix()
            offset=r@np.mean([p["center_eef"] for p in pads(o)],axis=0)
            target=c-SEAT_M*n-offset
            pre=target+PREGRASP_M*n
            clearance(o,pre,q,plane,n)
            drive(ctx,pre,q,False,start)
            c=track(ctx,c,t)
            target=c-SEAT_M*n-offset
            clearance(pose(ctx),target,q,plane,n)
        with ctx.stage("seat_and_close"):
            drive(ctx,target,q,False,start)
            # Current geometry can be briefly occluded by fingers; pose and face clearance remain checked.
            hold(ctx,True,start,5)
            o=pose(ctx)
            pp=pads(o)
            sep=np.linalg.norm(np.asarray(pp[0]["center_eef"])-np.asarray(pp[1]["center_eef"]))
            measured["closed_pad_separation_m"]=float(sep)
            if not 0.009<sep<0.060: raise ValueError("Closure does not support handle capture")
            before=track(ctx,c,t)
        with ctx.stage("pull_with_object_feedback"):
            origin=pose(ctx)["eef"].copy()
            last=before.copy()
            traveled=0.0
            for j in range(14):
                if success(ctx): raise Complete()
                distance=min(MAX_PULL_M,0.02*(j+1))
                drive(ctx,origin+distance*n,q,True,start,loops=12)
                expected=before+distance*n
                fresh=track(ctx,expected,t)
                shift=float(np.dot(fresh-before,n))
                measured["handle_outward_motion_m"]=shift
                measured["pull_command_m"]=distance
                if j==0 and shift<0.006:
                    raise ValueError("Initial pull did not move independently tracked handle")
                if j>0 and np.dot(fresh-last,n)<0.004:
                    raise ValueError("Drawer stopped following gripper")
                if np.linalg.norm((fresh-before)-shift*n)>0.020:
                    raise ValueError("Unexpected transverse object motion")
                last=fresh
            hold(ctx,True,start,2)
        return {"ok":success(ctx),"measured":measured,"why":"Bounded pull finished; host verifier decides"}
    except Complete:
        return {"ok":True,"measured":measured,"why":"Installed task_success passed"}
    except ValueError as e:
        ctx.note(facts=[str(e)],hypotheses=["Recovery did not establish all required measured prerequisites"])
        return {"ok":False,"measured":measured,"why":str(e)}
