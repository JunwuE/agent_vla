import json
import numpy as np

def decode(value):
    return json.loads(value.split("\ndata:image",1)[0]) if isinstance(value,str) else value


def project(point,k,t):
    p=t[:3,:3].T @ (point-t[:3,3])
    if p[2]<=0.02:
        raise ValueError("Tracked surface is behind the camera")
    uv=k@p
    return uv[:2]/uv[2]


def point_patch(box,depth,k,t):
    x0,y0,x1,y1=box
    vv,uu=np.mgrid[y0:y1,x0:x1]
    zz=depth[y0:y1,x0:x1]
    if zz.size<3 or not np.all(np.isfinite(zz)) or np.any(zz<=0.02):
        return None
    optical=np.stack(((uu-k[0,2])*zz/k[0,0],(vv-k[1,2])*zz/k[1,1],zz),axis=-1).reshape(-1,3)
    points=optical@t[:3,:3].T+t[:3,3]
    return np.median(points,axis=0),points
