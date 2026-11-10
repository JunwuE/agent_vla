def run(ctx):
    """Open or close while requesting the current measured OSC site pose."""
    ctx.stage("set gripper and hold pose")
    ctx.set_gripper(**ctx.parameters)
    ctx.note(facts=["Gripper command executed"],
             hypotheses=["Finger closure alone does not prove a grasp"])
    ctx.stage("independent verification")
    return "RECOVERED" if ctx.check(ctx.completion_condition)["passed"] is True else "UNRESOLVED"
