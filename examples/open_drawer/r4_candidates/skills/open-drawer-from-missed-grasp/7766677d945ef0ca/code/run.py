import numpy as np
from scipy.spatial.transform import Rotation
from eef_feedback_control import pose
from open_drawer_from_missed_grasp.perception import face,track,pads
from open_drawer_from_missed_grasp.controller import done,goto,grip,held_parameters
from drawer_pull_from_handle_hold.controller import execute as pull_held_drawer
from open_drawer_from_missed_grasp.params import CONTACT_DEPTHS,MAX_PULL,LIMIT

def main(ctx,camera,observation_step,face_boxes,handle_boxes,handle_seeds):
    measured={}
    try:
        o=pose(ctx); start=o["step"]
        if done(ctx): return {"ok":True,"measured":measured,"why":"Installed verifier passed at entry"}
        if start!=observation_step: raise ValueError("Reselect inputs on the current snapshot")
        if np.linalg.norm(o["eef"]-np.mean(handle_seeds,axis=0))>0.45:
            raise ValueError("Handle outside bounded reacquisition envelope")
        with ctx.stage("measure_drawer"):
            c,n=face(ctx,camera,face_boxes,o["eef"])
            anchors,boxes=track(ctx,camera,handle_boxes,handle_seeds,n,c,True)
            top=anchors.mean(0)
            if np.dot(top-o["eef"],-n)<0.08: raise ValueError("Withdrawn entry clearance not established")
            up=np.array([0.,0.,1.]); forward=-n
            jaw=up-n*np.dot(up,n); jaw/=np.linalg.norm(jaw)
            side=np.cross(forward,jaw); side/=np.linalg.norm(side)
            q=Rotation.from_matrix(np.column_stack((jaw,side,forward))).as_quat()
            measured.update(face_normal=n.tolist(),handle_top=top.tolist())
        for attempt,depth in enumerate(CONTACT_DEPTHS):
            with ctx.stage("align_and_seat"):
                grip(ctx,False,start)
                center=top-jaw*depth
                padmean,_=pads(pose(ctx))
                offset=Rotation.from_quat(q).apply(padmean)
                if goto(ctx,center+n*0.075-offset,q,False,start,80):
                    return {"ok":True,"measured":measured,"why":"Installed verifier passed"}
                anchors,boxes=track(ctx,camera,boxes,anchors,n,c)
                top=anchors.mean(0); center=top-jaw*depth
                padmean,_=pads(pose(ctx)); offset=Rotation.from_quat(q).apply(padmean)
                # A short seating hypothesis into the measured protruding handle,
                # bounded away from the independently measured drawer face.
                seat=center-n*0.004-offset
                if np.dot(seat-c,n)<0.024: raise ValueError("Insufficient face clearance")
                if goto(ctx,center+n*0.025-offset,q,False,start,40):
                    return {"ok":True,"measured":measured,"why":"Installed verifier passed"}
                if goto(ctx,seat,q,False,start,40):
                    return {"ok":True,"measured":measured,"why":"Installed verifier passed"}
                sep=grip(ctx,True,start)
                measured["attempt_"+str(attempt)]={"pad_separation":sep,"contact_depth_hypothesis":depth}
                if sep<0.012 or sep>0.065:
                    # Empty closure is not engagement. Retreat before another trial.
                    goto(ctx,seat+n*0.075,q,False,start,40)
                    anchors,boxes=track(ctx,camera,boxes,anchors,n,c)
                    top=anchors.mean(0)
                    continue
            with ctx.stage("pull_with_reused_surface_skill"):
                return pull_held_drawer(ctx,held_parameters(ctx,c,n,anchors))
        ok=done(ctx)
        return {"ok":ok,"measured":measured,"why":"Verifier passed" if ok else "Bounded grasp/pull completed without task verification"}
    except Exception as exc:
        return {"ok":False,"measured":measured,"why":str(exc)}
