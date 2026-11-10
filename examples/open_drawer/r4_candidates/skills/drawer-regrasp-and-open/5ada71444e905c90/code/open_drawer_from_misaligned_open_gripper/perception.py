import numpy as np
from rgbd_surface_geometry import decode
from eef_feedback_control import pose
from .params import PLANE_RESIDUAL, HANDLE_STANDOFF

def geom(ctx,camera,box):
    obs=pose(ctx)
    g=decode(ctx.ground_target(target="selected drawer surface",camera=camera,observation_step=obs["step"],bbox=[int(x) for x in box],confidence=0.85))
    gid=g.get("grounding_id",g.get("grounding",{}).get("grounding_id"))
    if gid is None: raise ValueError("Missing grounding identifier")
    d=decode(ctx.query_geometry(gid))
    d=d.get("geometry",d)
    if not d.get("reliable") or d.get("frame")!="osc_arm_base":
        raise ValueError("Unreliable current surface geometry")
    return np.asarray(d["centroid_m"],float),d

def plane(ctx,camera,box):
    x0,y0,x1,y1=box; xm=(x0+x1)//2; ym=(y0+y1)//2
    boxes=[[x0,y0,xm,ym],[xm,y0,x1,ym],[x0,ym,xm,y1],[xm,ym,x1,y1]]
    pts=np.array([geom(ctx,camera,b)[0] for b in boxes])
    c=pts.mean(0); _,s,v=np.linalg.svd(pts-c)
    if s[1]<0.005: raise ValueError("Degenerate cabinet plane")
    n=v[-1]
    if np.dot(n,pose(ctx)["eef"]-c)<0: n=-n
    residual=float(np.max(np.abs((pts-c)@n)))
    if residual>PLANE_RESIDUAL or abs(n[2])>0.15: raise ValueError("Inconsistent vertical cabinet plane")
    return c,n,residual

def handle(ctx,camera,roi,c,n,expected=None):
    raw=ctx.read_observation(camera)
    query=dense_geometry(raw)
    rgb=np.asarray(raw["rgb"])
    if rgb.ndim!=3 or rgb.shape[2]<3: raise ValueError("Missing RGB")
    if rgb.max()<=1.0: rgb=rgb*255.0
    x0,y0,x1,y1=[int(v) for v in roi]
    pts=[]
    for y in range(y0,y1):
        row=rgb[y,x0:x1,:3].astype(float)
        mask=(row.min(1)>100)&((row.max(1)-row.min(1))<65)
        inds=np.flatnonzero(mask)
        for run in np.split(inds,np.where(np.diff(inds)>1)[0]+1):
            if len(run)<4: continue
            for start in range(0,len(run)-3,4):
                xx=int(run[start])+x0
                try: p,d=query([xx,y,xx+4,y+1])
                except ValueError: continue
                extent=np.asarray(d["max_m"])-np.asarray(d["min_m"])
                if np.dot(np.abs(n),extent)>0.012: continue
                stand=float(np.dot(p-c,n))
                if expected is None:
                    if not HANDLE_STANDOFF[0]<stand<HANDLE_STANDOFF[1]: continue
                elif np.linalg.norm(p-expected)>0.105 or abs(np.dot(p-expected,n))>0.035:
                    continue
                pts.append(p)
    if len(pts)<3: raise ValueError("Handle surface insufficient or occluded")
    pts=np.array(pts)
    up=np.array([0.,0.,1.]); up-=n*np.dot(up,n); up/=np.linalg.norm(up)
    tangent=np.cross(up,n); tangent/=np.linalg.norm(tangent)
    heights=pts@up
    if expected is None:
        best=max(heights,key=lambda h:np.sum(np.abs(heights-h)<0.009))
    else: best=float(expected@up)
    pts=pts[np.abs(heights-best)<0.010]
    if len(pts)<3: raise ValueError("Handle height is inconsistent")
    along=pts@tangent
    span=float(np.ptp(along))
    if span<0.040 or span>0.17: raise ValueError("Insufficient horizontal handle span")
    center=np.median(pts,axis=0)
    center+=tangent*((float(np.min(along))+float(np.max(along)))/2-center@tangent)
    if expected is not None:
        if not np.any(along<expected@tangent-0.015) or not np.any(along>expected@tangent+0.015):
            raise ValueError("Cannot track both exposed sides of handle")
        if abs((center-expected)@up)>0.012: raise ValueError("Handle vertical tracking mismatch")
    return center,{"points":pts.tolist(),"span_m":span}

def dense_geometry(raw):
    """One current RGB-D read; equivalent four-valid-pixel metric surface queries."""
    meta=raw['metadata']; cal=meta['calibration']
    if (cal.get('step')!=meta['step'] or cal.get('frame')!='optical_to_osc_arm_base'
            or cal.get('alignment_verified') is not True):
        raise ValueError('Calibration frame or timestamp mismatch')
    depth=np.asarray(raw['depth_m'],dtype=float)
    k=np.asarray(cal['intrinsics'],dtype=float)
    t=np.asarray(cal['camera_to_base'],dtype=float)
    if (k.shape!=(3,3) or t.shape!=(4,4) or not np.isfinite(k).all() or not np.isfinite(t).all()
            or k[0,0]<=0 or k[1,1]<=0 or abs(np.linalg.det(k))<1e-10
            or not np.allclose(t[3],[0,0,0,1])
            or not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-4)):
        raise ValueError('Invalid calibration')
    y,x=np.mgrid[:depth.shape[0],:depth.shape[1]]
    pixels=np.stack((x.ravel(),y.ravel(),np.ones(depth.size)))
    xyz=(np.linalg.inv(k)@pixels)*depth.ravel()
    points=(t[:3,:3]@xyz+t[:3,3:4]).T.reshape(*depth.shape,3)
    valid=np.isfinite(depth)&(depth>0)&(depth<10)
    def query(box):
        x0,y0,x1,y1=box
        if not 0<=x0<x1<=depth.shape[1] or not 0<=y0<y1<=depth.shape[0]:
            raise ValueError('bbox outside image')
        mask=valid[y0:y1,x0:x1]
        if np.count_nonzero(mask)<4:raise ValueError('Insufficient valid metric depth')
        pts=points[y0:y1,x0:x1][mask]
        return np.median(pts,axis=0),dict(min_m=pts.min(0),max_m=pts.max(0),
            count=len(pts),frame='osc_arm_base',observation_step=meta['step'])
    return query
