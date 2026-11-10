"""Astra's evidence-backed execution phases, independent of VLA language and action leases."""

from typing import Literal

from pydantic import Field

from .contracts import Model, ReviewEvidence


class Phase(Model):
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    goal: str = Field(min_length=1, max_length=500)
    prerequisites: list[str] = Field(default_factory=list, max_length=8)
    progress_signals: list[str] = Field(min_length=1, max_length=8)
    completion_signals: list[str] = Field(min_length=1, max_length=8)
    failure_signals: list[str] = Field(default_factory=list, max_length=8)


class PhasePlan(Model):
    expected_revision: int = Field(ge=0, strict=True)
    phases: list[Phase] = Field(min_length=1, max_length=12)
    active_phase: str
    reason: str = Field(min_length=1, max_length=1000)


class PhaseAssessment(Model):
    revision: int = Field(ge=1, strict=True)
    phase_id: str
    status: Literal["unknown", "progressing", "stalled", "complete"]
    evidence: list[ReviewEvidence] = Field(min_length=1, max_length=12)
    hypotheses: list[str] = Field(default_factory=list, max_length=8)
    next_phase: str | None = None


class PhaseMemory:
    def __init__(self):
        self.revision = 0
        self.phases = []
        self.active = None
        self.started_step = 0
        self.assessments = []
        self.last_progress_step = 0

    def set_plan(self, plan, step):
        ids = [p.id for p in plan.phases]
        if plan.expected_revision != self.revision:
            raise ValueError("Phase plan revision changed")
        if len(set(ids)) != len(ids) or plan.active_phase not in ids:
            raise ValueError("Phase ids must be unique and include the active phase")
        self.revision += 1
        self.phases = [p.model_dump() for p in plan.phases]
        self.active = plan.active_phase
        self.started_step = step
        return self.summary()

    def assess(self, assessment, step, available_steps):
        if assessment.revision != self.revision or assessment.phase_id != self.active:
            raise ValueError("Stale phase assessment")
        if any(
            e.observation_step > step
            or e.observation_step < self.started_step
            or e.observation_step not in available_steps
            for e in assessment.evidence
        ):
            raise ValueError("Phase evidence must refer to recorded observations of this phase")
        if assessment.next_phase is not None:
            if assessment.status != "complete" or assessment.next_phase not in {p["id"] for p in self.phases}:
                raise ValueError("Phase transition needs completion evidence and a known destination")
            if assessment.next_phase == self.active:
                raise ValueError("Destination is already the active phase")
        record = dict(
            **assessment.model_dump(),
            observation_step=step,
            source="astra_assessment",
            independently_verified=False,
        )
        self.assessments = (self.assessments + [record])[-16:]
        if assessment.status in {"progressing", "complete"}:
            self.last_progress_step = max(e.observation_step for e in assessment.evidence)
        if assessment.next_phase:
            self.active = assessment.next_phase
            self.started_step = step
        self.revision += 1
        return record

    def summary(self):
        return dict(
            revision=self.revision,
            active_phase=self.active,
            started_step=self.started_step,
            phases=self.phases,
            recent_assessments=self.assessments[-4:],
            last_progress_step=self.last_progress_step,
            authority="phase beliefs only; no language update, action handoff or task success",
        )
