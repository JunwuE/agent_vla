import numpy as np
from rgbd_surface_geometry import decode, point_patch
from .params import PLANE_RESIDUAL, PROTRUSION_MIN, PROTRUSION_MAX

def measure(ctx,camera,box,label):
    obs=decode(ctx.observe())
    g=decode(ctx.ground_target(target=label,camera=camera,observation_step=obs["step"],bbox=[int(x) for x in box],confidence=0.85))
    if "grounding" in g: g=g["grounding"]
    v=decode(ctx.query_geometry(g["grounding_id"]))
    if "geometry" in v: v=v["geometry"]
    if v.get("reliable") is not True or v.get("frame")!="osc_arm_base":
        return None
    return np.asarray(v["centroid_m"],float)

def plane(ctx,camera,boxes,reference=None):
    pts=[measure(ctx,camera,b,"selected drawer front surface") for b in boxes]
    if any(p is None for p in pts): raise ValueError("Drawer plane depth unavailable")
    pts=np.asarray(pts); c=pts.mean(0)
    _,s,v=np.linalg.svd(pts-c,full_matrices=False); n=v[-1]
    if s[1]<0.015 or np.max(np.abs((pts-c)@n))>PLANE_RESIDUAL:
        raise ValueError("Drawer plane fit is degenerate or mixed")
    ref=np.asarray(decode(ctx.observe())["eef_control_pose"]["position"]) if reference is None else reference
    if np.dot(n,ref-c)<0:n=-n
    return c,n,pts

def handle(ctx,camera,box,c,n,expected=None):
    raw=decode(ctx.read_observation(camera))
    entries=[]
    def walk(d,path=""):
        if isinstance(d,dict):
            for key,value in d.items():
                entries.append((path+str(key),str(key).lower(),value))
                if isinstance(value,dict):walk(value,path+str(key)+".")
    walk(raw)
    depth=raw.get("depth_m")
    if depth is None:raise ValueError("No metric depth; observation keys="+str(list(raw)))
    aligned=any("align" in key and value is True for _,key,value in entries)
    frame=any(isinstance(value,str) and "osc_arm_base" in value for _,key,value in entries)
    kk=None;tt=None
    for path,key,value in entries:
        if key in ("intrinsics","camera_intrinsics","k","intrinsic_matrix"):
            try:
                a=np.asarray(value,float)
                if a.shape==(3,3):kk=a
            except (TypeError,ValueError):pass
        if key in ("camera_to_base","t_camera_to_base","optical_to_base","t_optical_to_base","camera_to_osc_arm_base","optical_to_osc_arm_base","t_camera_to_osc_arm_base","t_optical_to_osc_arm_base"):
            try:
                a=np.asarray(value,float)
                if a.shape==(4,4):tt=a
                if "osc_arm_base" in key:frame=True
            except (TypeError,ValueError):pass
    if not aligned or not frame or kk is None or tt is None:
        raise ValueError("Explicit aligned optical-to-OSC calibration missing; keys="+str([p for p,_,_ in entries if not p.startswith("rgb")])[:1800])
    patch=point_patch(box,np.asarray(depth,float),kk,tt)
    if patch is None:raise ValueError("Invalid metric depth in selected handle ROI")
    allpts=patch[1];offset=(allpts-c)@n
    pts=allpts[(offset>PROTRUSION_MIN)&(offset<PROTRUSION_MAX)]
    if expected is not None:pts=pts[np.linalg.norm(pts-expected,axis=1)<0.10]
    if len(pts)<7:raise ValueError("Insufficient reliable protruding handle pixels")
    # A handle ROI includes its vertical mounting legs. Select the observed
    # horizontal band before fitting a bar, rather than fitting the entire U.
    up=np.array([0.,0.,1.]); up-=n*np.dot(up,n); up/=np.linalg.norm(up)
    heights=pts@up
    height=max(heights,key=lambda h:np.count_nonzero(np.abs(heights-h)<0.006))
    pts=pts[np.abs(heights-height)<0.007]
    if len(pts)<7:raise ValueError("Insufficient observed horizontal handle band")
    center=np.median(pts,axis=0)
    _,s,v=np.linalg.svd(pts-center,full_matrices=False)
    axis=v[0]-np.dot(v[0],n)*n;axis/=np.linalg.norm(axis)
    along=(pts-center)@axis
    extent=float(np.quantile(along,0.9)-np.quantile(along,0.1))
    if not 0.045<extent<0.20 or s[1]>0.65*s[0]:raise ValueError("Handle surface is not a bounded bar")
    # Grasp bar midpoint, excluding the handle mounting ends.
    center+=axis*float((np.quantile(along,0.1)+np.quantile(along,0.9))/2)
    return center,axis,extent
