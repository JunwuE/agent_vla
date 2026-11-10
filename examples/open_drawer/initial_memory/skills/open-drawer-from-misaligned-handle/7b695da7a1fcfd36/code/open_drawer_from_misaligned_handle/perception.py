import numpy as np
from scipy import ndimage
from rgbd_surface_geometry import decode

def measure(ctx, box, label="handle"):
    o=decode(ctx.observe())
    g=decode(ctx.ground_target(target=label,camera="wrist",observation_step=o["step"],bbox=[int(x) for x in box],confidence=0.95))
    gid=g.get("grounding_id",g.get("grounding",{}).get("grounding_id"))
    if not gid:
        raise ValueError("No grounding id")
    d=decode(ctx.query_geometry(gid))
    d=d.get("geometry",d)
    if not d.get("reliable") or d.get("frame")!="osc_arm_base" or d.get("depth_iqr_m",1)>0.012:
        raise ValueError("Unreliable selected depth")
    return np.asarray(d["centroid_m"],float)

def initialize(ctx, handle_boxes, face_boxes):
    bars=np.array([measure(ctx,b) for b in handle_boxes])
    face=np.array([measure(ctx,b,"drawer face") for b in face_boxes])
    p=face.mean(0)
    _,sv,v=np.linalg.svd(face-p)
    n=v[-1]
    if sv[1]<0.012 or np.max(np.abs((face-p)@n))>0.003:
        raise ValueError("Drawer face fit degenerate")
    e=np.asarray(decode(ctx.observe())["eef_control_pose"]["position"])
    if np.dot(n,e-p)<0: n=-n
    t=bars[-1]-bars[0]
    length=np.linalg.norm(t)
    if not 0.035<length<0.25: raise ValueError("Bar span invalid")
    t=t/length
    if abs(t[2])>0.20 or abs(n[2])>0.20 or abs(np.dot(t,n))>0.15:
        raise ValueError("Not a horizontal drawer bar")
    c=bars[1]
    gap=np.dot(c-p,n)
    if not 0.018<gap<0.080: raise ValueError("Handle-to-face gap invalid")
    if np.linalg.norm(c-e)>0.22: raise ValueError("Handle too far from entry pose")
    return c,p,n,t

def track(ctx, expected, tangent):
    raw=ctx.read_observation("wrist")
    rgb=np.asarray(raw["rgb"])
    if rgb.dtype.kind=="f" and rgb.max()<=1.0: rgb=rgb*255
    a=rgb[:,:,:3].astype(float)
    mask=(a.min(2)>135)&((a.max(2)-a.min(2))<45)
    lab,num=ndimage.label(mask)
    candidates=[]
    for idx in range(1,num+1):
        yy,xx=np.nonzero(lab==idx)
        if len(xx)<65 or np.max(yy)>=rgb.shape[0]-3: continue
        uv=np.column_stack([xx,yy]).astype(float)
        cov=np.cov(uv.T)
        vals,vec=np.linalg.eigh(cov)
        if vals[-1]<100 or vals[-1]/max(vals[0],1)<3.0: continue
        axis=vec[:,-1]
        ss=(uv-uv.mean(0))@axis
        dist=ndimage.distance_transform_edt(lab==idx)
        samples=[]
        for frac in [0.25,0.50,0.75]:
            target=np.quantile(ss,frac)
            score=np.abs(ss-target)+12/(dist[yy,xx]+0.1)
            k=int(np.argmin(score))
            x,y=int(xx[k]),int(yy[k])
            if dist[y,x]<2: continue
            samples.append([x-1,y-1,x+2,y+2])
        if len(samples)<3: continue
        candidates.append((len(xx),samples))
    found=[]
    for _,boxes in sorted(candidates,reverse=True)[:5]:
        try:
            pts=np.array([measure(ctx,b,"tracked handle bar") for b in boxes])
        except ValueError:
            continue
        d=pts[-1]-pts[0]
        length=np.linalg.norm(d)
        if not 0.025<length<0.30 or abs(np.dot(d/length,tangent))<0.92: continue
        center=pts.mean(0)
        line=center+tangent*np.dot(expected-center,tangent)
        if np.linalg.norm(line-expected)>0.035: continue
        if np.max(np.linalg.norm(np.cross(pts-center,tangent),axis=1))>0.008: continue
        found.append(line)
    if not found: raise ValueError("No independently tracked handle")
    found.sort(key=lambda p:np.linalg.norm(p-expected))
    if len(found)>1 and np.linalg.norm(found[1]-found[0])>0.012:
        raise ValueError("Ambiguous handle tracking")
    return found[0]
