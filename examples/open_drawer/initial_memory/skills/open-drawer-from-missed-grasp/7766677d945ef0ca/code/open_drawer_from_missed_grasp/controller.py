import numpy as np
from scipy.spatial.transform import Rotation
from eef_feedback_control import pose,move_bounded
from rgbd_surface_geometry import decode
from .params import LIMIT,POSE_TOL,ANGLE_TOL
from .perception import pads

def done(ctx):
    return decode(ctx.check("task_success"))["passed"] is True

def goto(ctx,target,q,closed,start,cap=64):
    target=np.array(target); previous=None; stagnant=0
    for _ in range(cap//2):
        o=pose(ctx)
        if done(ctx): return True
        err=np.linalg.norm(target-o["eef"])
        angle=(Rotation.from_quat(q)*Rotation.from_quat(o["eef_control_pose"]["quaternion_xyzw"]).inv()).magnitude()
        if err<POSE_TOL and angle<ANGLE_TOL: return False
        if previous is not None:
            stagnant=stagnant+1 if np.linalg.norm(o["eef"]-previous)<0.0003 and err>0.007 and angle<ANGLE_TOL else 0
        if stagnant>=6: raise ValueError("Measured motion blocked")
        previous=o["eef"].copy()
        move_bounded(ctx,target,q,closed,start_step=start,limit=LIMIT,steps=2)
    raise ValueError("Pose convergence bound reached")

def grip(ctx,closed,start):
    for _ in range(5):
        o=pose(ctx)
        move_bounded(ctx,o["eef"],o["eef_control_pose"]["quaternion_xyzw"],closed,start_step=start,limit=LIMIT,steps=4)
    o=pose(ctx); _,sep=pads(o)
    if not closed and sep<0.070: raise ValueError("Fingers did not open")
    return sep

def held_parameters(ctx, face_c, normal, handle_points):
    """Choose current visible surfaces, then reuse the checked hold/pull procedure."""
    from itertools import combinations
    from rgbd_surface_geometry import project, point_patch
    obs=pose(ctx); step=obs['step']
    raw=ctx.read_observation('wrist')
    cal=raw['metadata']['calibration']
    if cal.get('step')!=step or cal.get('alignment_verified') is not True:
        raise ValueError('Current wrist calibration unavailable')
    k=np.asarray(cal['intrinsics']); t=np.asarray(cal['camera_to_base'])
    depth=np.asarray(raw['depth_m']); available=[]
    for y in range(20,220,28):
        for x in range(20,220,28):
            box=[x,y,x+8,y+8]; patch=point_patch(box,depth,k,t)
            if patch is None: continue
            p,points=patch
            if abs((p-face_c)@normal)>0.006: continue
            if np.max(np.abs((points-p)@normal))>0.004: continue
            available.append((box,p))
    # Select four well-spread measured patches; no guessed object coordinates.
    best=None
    if len(available)>20:
        indices=np.linspace(0,len(available)-1,20).astype(int)
        available=[available[i] for i in indices]
    for chosen in combinations(available,4):
        pts=np.array([a[1] for a in chosen]); _,s,v=np.linalg.svd(pts-pts.mean(0))
        if s[1]<0.012 or max(abs((pts-pts.mean(0))@v[-1]))>0.002: continue
        if best is None or s[1]>best[0]: best=(s[1],[a[0] for a in chosen])
    if best is None: raise ValueError('No visible noncollinear drawer-front patches in wrist view')
    handle_match=None
    for view in ('wrist','left','right'):
        raw=ctx.read_observation(view); cal=raw['metadata']['calibration']
        if cal.get('step')!=step or cal.get('alignment_verified') is not True: continue
        k=np.asarray(cal['intrinsics']);t=np.asarray(cal['camera_to_base']);depth=np.asarray(raw['depth_m'])
        for anchor in handle_points:
            try:u,v=np.rint(project(anchor,k,t)).astype(int)
            except ValueError:continue
            for dy in range(-3,4):
                for dx in range(-3,4):
                    box=[int(u+dx-2),int(v+dy),int(u+dx+2),int(v+dy+1)]
                    if min(box)<0 or box[2]>depth.shape[1] or box[3]>depth.shape[0]:continue
                    patch=point_patch(box,depth,k,t)
                    if patch is None:continue
                    error=float(np.linalg.norm(patch[0]-anchor))
                    if error>0.012:continue
                    if handle_match is None or error<handle_match[0]:handle_match=(error,view,box)
    if handle_match is None:raise ValueError('No current measured handle patch for hold entry')
    return dict(observation_step=step,camera='wrist',front_boxes=best[1],
                handle_camera=handle_match[1],handle_box=handle_match[2])
