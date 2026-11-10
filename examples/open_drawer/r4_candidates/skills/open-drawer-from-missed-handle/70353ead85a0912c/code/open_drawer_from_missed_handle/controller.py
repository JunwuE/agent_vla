import numpy as np
from scipy.spatial.transform import Rotation
from eef_feedback_control import pose, move_bounded
from rgbd_surface_geometry import decode
from .perception import plane,handle
from .params import LIMIT,PRE_CLEARANCE,POSE_TOL,MAX_PULL

def execute(ctx,camera,handle_box,front_boxes,observation_step):
    start=pose(ctx)["step"]; measured={}
    def done():return decode(ctx.check("task_success")).get("passed") is True
    def result(ok,why):return {"ok":bool(ok),"measured":measured,"why":why}
    def travel(target,q,closed,cap=120):
        initial=pose(ctx); begin=initial["step"]; bad=0; previous=None
        while pose(ctx)["step"]-begin<cap:
            o=pose(ctx)
            if done():return True
            d=float(np.linalg.norm(np.asarray(target)-o["eef"]))
            a=float(np.linalg.norm((Rotation.from_quat(q)*Rotation.from_quat(o["eef_control_pose"]["quaternion_xyzw"]).inv()).as_rotvec()))
            if d<POSE_TOL and a<0.09:return False
            if previous is not None:
                bad=bad+1 if np.linalg.norm(o["eef"]-previous)<0.0003 and d>0.012 else 0
            if bad>=10:raise ValueError("Measured approach blocked")
            previous=o["eef"].copy()
            move_bounded(ctx,target,q,closed,start_step=start,limit=LIMIT,steps=2)
        raise ValueError("Stage failed to converge within action bound")
    try:
        if done():return result(True,"Verifier already passed")
        if start!=observation_step:raise ValueError("Scene selection step changed before entry")
        with ctx.stage("measure_drawer_and_handle") as st:
            c,n,basepts=plane(ctx,camera,front_boxes)
            h,axis,length=handle(ctx,camera,handle_box,c,n)
            if abs(n[2])>0.15 or abs(axis[2])>0.35:raise ValueError("Expected vertical drawer and horizontal handle")
            measured.update(handle=h.tolist(),normal=n.tolist(),handle_length=length)
            st.record(**measured)
        # Local X is the measured jaw axis; local Z points from the palm toward the handle.
        kin=pose(ctx)["controller"]["gripper_kinematics"]
        if not kin.get("available") or abs(abs(kin["jaw_separation_axis_eef"][0])-1)>0.01:raise ValueError("Unsupported jaw calibration")
        z=-n; x=np.cross(axis,z);x/=np.linalg.norm(x)
        if x[2]<0:x=-x
        y=np.cross(z,x)
        q=Rotation.from_matrix(np.column_stack((x,y,z))).as_quat()
        pre=h+PRE_CLEARANCE*n
        with ctx.stage("open_and_align_clear_of_cabinet") as st:
            o=pose(ctx)
            if (o["eef"]-c)@n<0.12:raise ValueError("Entry lacks measured cabinet clearance")
            # Align laterally at the current exterior standoff, then approach.
            away=pre+max(0.0,float((o["eef"]-pre)@n))*n
            if travel(away,q,False,150):return result(True,"Verified while aligning")
            if travel(pre,q,False,150):return result(True,"Verified while approaching")
            o=pose(ctx); pads=o["controller"]["gripper_kinematics"]["fingerpad_geometry"]
            if not pads.get("available") or pads["step"]!=o["step"]:raise ValueError("Fresh pad geometry missing")
            span=np.linalg.norm(np.asarray(pads["pads"][0]["center_eef"])-np.asarray(pads["pads"][1]["center_eef"]))
            if span<0.045:raise ValueError("Pads did not open")
            st.record(pad_span=float(span))
        with ctx.stage("remeasure_and_engage_handle") as st:
            c2,n2,p2=plane(ctx,camera,front_boxes,reference=c+n)
            if np.linalg.norm(c2-c)>0.015:raise ValueError("Drawer changed before engagement")
            h2,axis2,length2=handle(ctx,camera,handle_box,c2,n2,expected=h)
            if np.linalg.norm(h2-h)>0.018:raise ValueError("Handle target occluded or displaced")
            # Visible front of bar to bar center, retaining clearance from backing plane.
            pad_center=h2-0.006*n
            pads=pose(ctx)["controller"]["gripper_kinematics"]["fingerpad_geometry"]["pads"]
            mid=np.mean([p["center_eef"] for p in pads],axis=0)
            goal=pad_center-Rotation.from_quat(q).apply(mid)
            if (pad_center-c2)@n<0.018:raise ValueError("Insufficient handle backing clearance")
            if travel(goal,q,False,80):return result(True,"Verified during seating")
            for _ in range(5):
                if pose(ctx)["step"]-start+3>LIMIT:raise ValueError("Action bound before closure")
                ctx.set_gripper(closed=True,steps=3)
            grasp=pose(ctx)["eef"].copy()
            measured["closure_pose"]=grasp.tolist()
            st.record(closure_pose=grasp.tolist())
        with ctx.stage("test_coupled_drawer_motion") as st:
            if travel(grasp+0.035*n,q,True,50):return result(True,"Drawer verifier passed")
            cc,nn,pp=plane(ctx,camera,front_boxes,reference=c+n)
            progress=float((cc-c)@n)
            shifts=(pp-basepts)@n
            measured["test_drawer_translation"]=progress
            if progress<0.012 or progress>0.065 or np.count_nonzero(shifts>0.01)<3:
                raise ValueError("Short pull did not establish coupled drawer motion")
            st.record(drawer_translation=progress)
        with ctx.stage("continue_bounded_opening") as st:
            for dist in np.arange(0.06,MAX_PULL+0.001,0.025):
                if travel(grasp+float(dist)*n,q,True,30):return result(True,"task_success passed")
                if done():return result(True,"task_success passed")
            st.record(verifier_passed=done())
        return result(done(),"Bounded pull ended; only task_success establishes completion")
    except (ValueError,KeyError,TypeError) as exc:
        measured["stopped_step"]=pose(ctx)["step"]
        return result(done(),"Unresolved: "+str(exc))
