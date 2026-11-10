def run(ctx):
    import json
    import numpy as np

    def unpack(value):
        return json.loads(value.split("\ndata:image",1)[0]) if isinstance(value,str) else value

    def done():
        return unpack(ctx.check("task_success")).get("passed") is True

    def compact(value):
        if isinstance(value,dict):
            return {str(k):compact(v) for k,v in value.items() if k not in ("rgb","depth_m","state")}
        if isinstance(value,np.ndarray):
            return value.tolist() if value.size<=64 else {"shape":list(value.shape),"dtype":str(value.dtype)}
        if isinstance(value,(list,tuple)):
            return [compact(v) for v in value] if len(value)<=64 else {"length":len(value)}
        if isinstance(value,(str,int,float,bool)) or value is None:
            return value
        return str(value)

    if done():
        return "SUBTASK_DONE"
    start = unpack(ctx.observe())
    if start.get("step") != 1032:
        ctx.note(facts=["Initial finger-observation setup is stale; no motion executed."],hypotheses=[])
        return "UNRESOLVED"
    pose = start["eef_control_pose"]
    with ctx.stage("open fingers at the current raised pose to measure their actual axis") as stage:
        for index in range(4):
            ctx.move_eef(target={"position":pose["position"],"quaternion_xyzw":pose["quaternion_xyzw"],
                "gripper_closed":False,"base_motion":[0,0,0,0],"control_mode":"arm"},steps=4)
            if done():
                return "SUBTASK_DONE"
            current = unpack(ctx.observe())
            displacement = float(np.linalg.norm(np.asarray(current["eef_control_pose"]["position"])-np.asarray(pose["position"])))
            stage.record(step=current["step"],measured_pose=current["eef_control_pose"],
                finger_positions=current["state"][-2:],eef_displacement_m=displacement)
            if displacement > 0.015:
                ctx.note(facts=["Unexpected arm displacement during the finger-only observation stage."],hypotheses=[])
                return "UNRESOLVED"
    for camera in ("left","right","wrist"):
        observation = ctx.read_observation(camera)
        ctx.note(facts=["Current observation metadata for "+camera+": "+json.dumps(compact(observation))[:14000]],hypotheses=[])
    ctx.note(facts=["Finger-opening observation stage ended at step "+str(current["step"])+"; no door grasp or closure has been established."],
        hypotheses=["The actual finger separation axis must be measured from fresh RGB-D before selecting a rim-thickness grasp orientation. Code retains the complete original closing subtask."])
    return "SUBTASK_DONE" if done() else "UNRESOLVED"
