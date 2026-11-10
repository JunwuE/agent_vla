"""Planner limits are recorded stops, not model or simulator failures."""


class PlannerBudget(RuntimeError):
    pass
