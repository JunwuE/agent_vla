def run(ctx):
    """Short native base/torso correction; motion values are normalized, not meters."""
    ctx.stage("mobile correction with desired arm goal held")
    ctx.move_base(**ctx.parameters)
    ctx.stage("independent verification")
    return "RECOVERED" if ctx.check(ctx.completion_condition)["passed"] is True else "UNRESOLVED"
