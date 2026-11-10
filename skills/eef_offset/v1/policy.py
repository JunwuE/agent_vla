def run(ctx):
    """A bounded relative move, recomputed from the current OSC site pose."""
    ctx.stage("measure and move")
    result = ctx.move_relative(**ctx.parameters)
    ctx.note(facts=["Measured final EEF position error: " + str(result["position_error_m"])],
             hypotheses=["Robot motion alone does not establish object recovery"])
    ctx.stage("independent verification")
    return "RECOVERED" if ctx.check(ctx.completion_condition)["passed"] is True else "UNRESOLVED"
