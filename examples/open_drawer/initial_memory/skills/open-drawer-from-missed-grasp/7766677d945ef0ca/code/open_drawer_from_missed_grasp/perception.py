import numpy as np
from rgbd_surface_geometry import decode

def patch(ctx,camera,box):
    g=decode(ctx.geometry(camera=camera,bbox=[int(v) for v in box]))
    if not g.get("reliable") or g.get("frame")!="osc_arm_base":
        return None
    return g

def face(ctx,camera,boxes,eef):
    pts=[]
    for b in boxes:
        g=patch(ctx,camera,b)
        if g is None: raise ValueError("Unreliable drawer face depth")
        pts.append(g["centroid_m"])
    pts=np.array(pts); c=pts.mean(0)
    _,s,v=np.linalg.svd(pts-c,full_matrices=False)
    n=v[-1]
    if np.dot(n,eef-c)<0: n=-n
    if s[1]<0.015 or max(abs((pts-c)@n))>0.004 or abs(n[2])>0.15:
        raise ValueError("Face plane is degenerate or not vertical")
    return c,n

def track(ctx,camera,boxes,anchors,n,face_c,initial=False):
    """Match the two measured handle points in current calibrated RGB-D views.

    The caller's expected positions only center the search; depth measurements
    and unchanged rigid-bar checks establish correspondence. Never use predicted
    points as measurements. Try another view when a handle leaves one image.
    """
    from rgbd_surface_geometry import project, point_patch
    observation=decode(ctx.observe())
    step=observation['step']
    anchors=np.asarray(anchors,float)
    candidates=[[],[]]
    for view in dict.fromkeys((camera,'left','right','wrist')):
        raw=ctx.read_observation(view)
        meta=raw['metadata']; cal=meta.get('calibration',{})
        if (meta.get('step')!=step or cal.get('step')!=step
            or cal.get('frame')!='optical_to_osc_arm_base'
            or cal.get('alignment_verified') is not True):
            continue
        depth=np.asarray(raw['depth_m'],float)
        k=np.asarray(cal['intrinsics'],float); t=np.asarray(cal['camera_to_base'],float)
        height,width=depth.shape
        for i,anchor in enumerate(anchors):
            try: uv=project(anchor,k,t)
            except ValueError: continue
            u,v=np.rint(uv).astype(int)
            for y in range(max(0,v-8),min(height-1,v+9)):
                for x in range(max(0,u-6),min(width-3,u+7)):
                    box=[x,y,x+4,y+1]
                    z=depth[y:y+1,x:x+4]
                    if np.percentile(z,75)-np.percentile(z,25)>0.012: continue
                    patch=point_patch(box,depth,k,t)
                    if patch is None: continue
                    p=patch[0]; d=p-anchor
                    if abs(d[2])>0.006 or abs(d[1])>0.025 or np.linalg.norm(d)>0.045: continue
                    if np.dot(p-face_c,n)<0.020: continue
                    cost=float(np.linalg.norm(d)+2*abs(d[2]))
                    candidates[i].append((cost,p,box,view))
    if not all(candidates): raise ValueError('Handle points not visible in current calibrated views')
    pairs=[]
    for a in sorted(candidates[0],key=lambda x:x[0])[:30]:
        for b in sorted(candidates[1],key=lambda x:x[0])[:30]:
            delta=a[1]-b[1]
            if abs(delta[2])>0.008 or abs(np.dot(delta,n))>0.018 or not 0.045<np.linalg.norm(delta)<0.16:
                continue
            if abs(np.linalg.norm(delta)-np.linalg.norm(anchors[0]-anchors[1]))>0.015: continue
            pairs.append((a[0]+b[0],a,b))
    if not pairs: raise ValueError('Handle-end correspondence inconsistent across current views')
    _,a,b=min(pairs,key=lambda p:p[0])
    return np.array([a[1],b[1]]),[a[2],b[2]]

def pads(obs):
    k=obs["controller"]["gripper_kinematics"]
    f=k["fingerpad_geometry"]
    if not f.get("available") or f.get("step")!=obs["step"] or f.get("frame")!="eef_control_site":
        raise ValueError("Fresh measured pad boxes unavailable")
    p=np.array([v["center_eef"] for v in f["pads"]])
    return p.mean(0),float(np.linalg.norm(p[0]-p[1]))
