"""Versioned contracts. Native commands and measured state are distinct objects."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class DecisionLabel(StrEnum):
    KEEP = "KEEP"
    REFINE = "REFINE"
    SWITCH = "SWITCH"
    REPLAN = "REPLAN"
    ESCALATE = "ESCALATE"


class Source(StrEnum):
    VLA = "vla"
    ASTRA = "astra_local"
    CODE = "code_policy"


class Binding(Model):
    episode_id: str
    subtask_version: int
    instruction_version: int
    control_epoch: int
    observation_step: int
    observation_time: float
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)


class Subtask(Model):
    name: str
    instruction: str = Field(min_length=1)
    completion_condition: str = Field(min_length=1)
    preconditions: list[str] = Field(default_factory=list, max_length=12)


class Task(Model):
    instruction: str = Field(min_length=1)
    subtasks: list[Subtask] = Field(min_length=1)
    robot: str = "PandaOmron"
    language_catalog: list[Subtask] = Field(default_factory=list)


class Decision(Model):
    binding: Binding
    label: DecisionLabel
    probabilities: dict[str, float] = Field(default_factory=dict)
    trigger: str = "periodic"
    fallback_reason: str | None = None


class Verification(Model):
    condition: str
    passed: bool | None  # None means unobservable / not implemented, never success.
    evidence: list[str] = Field(default_factory=list)
    observation_step: int


class Intervention(Model):
    kind: Literal["continue", "refine", "switch", "replan", "local", "eef", "edit", "code", "observe", "stop"]
    reason: str
    instruction: str | None = None
    instruction_id: str | None = None
    remaining_plan: list[Subtask] | None = None
    actions: list[list[float]] | None = None
    completion_condition: str | None = None
    skill_id: str | None = None
    skill_version: str | None = None
    code: str | None = None
    program_version: str | None = None
    hypotheses: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    target: dict[str, Any] | None = None
    edit: dict[str, Any] | None = None
    proposal_id: str | None = None
    steps: int = Field(default=1, ge=1, le=5, strict=True)


class ReviewEvidence(Model):
    observation_step: int = Field(ge=0, strict=True)
    fact: str = Field(min_length=1)


class ExecutionReview(Model):
    """A read-only assessment, never an executable action or a language update."""

    action: Literal["continue", "request_intervention"]
    assessment: Literal["progressing", "uncertain", "failed_attempt", "stalled_progress", "wrong_target",
                        "subtask_complete", "plan_invalid", "unsafe"]
    phase: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    evidence: list[ReviewEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def intervention_needs_evidence(self):
        if self.action == "request_intervention" and (
            self.assessment in {"progressing", "uncertain"} or not self.evidence
        ):
            raise ValueError("Intervention needs a specific problem/transition and observed evidence")
        if self.assessment == "stalled_progress" and len({e.observation_step for e in self.evidence}) < 2:
            raise ValueError("Stalled progress requires evidence from at least two distinct observations")
        return self


@dataclass
class Observation:
    episode_id: str
    step: int
    timestamp: float  # monotonic capture time, for online response expiry
    sim_time: float
    state: np.ndarray  # canonical EEF-first 16D state
    rgb: dict[str, np.ndarray]
    depth_m: dict[str, np.ndarray] = field(default_factory=dict)
    calibration: dict[str, dict[str, Any]] = field(default_factory=dict)
    facts: dict[str, Any] = field(default_factory=dict)
    events: tuple[str, ...] = ()
    terminal: bool = False
    success: bool = False
    eef_control_pose: dict[str, Any] = field(default_factory=dict)
    controller: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.state = np.asarray(self.state, dtype=np.float64).copy()
        if self.state.shape != (16,) or not np.isfinite(self.state).all():
            raise ValueError("Observation requires finite canonical 16D state")
        if not np.isfinite([self.timestamp, self.sim_time]).all():
            raise ValueError("Nonfinite observation timestamp")
        for name, rgb in self.rgb.items():
            if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
                raise ValueError(f"{name}: expected HWC uint8 RGB")

    @property
    def key(self):
        return f"{self.episode_id}:{self.step}"

    def summary(self):
        return dict(
            episode_id=self.episode_id,
            step=self.step,
            timestamp=self.timestamp,
            sim_time=self.sim_time,
            state=self.state.tolist(),
            facts=self.facts,
            events=list(self.events),
            terminal=self.terminal,
            success=self.success,
            cameras=list(self.rgb),
            depth_available=list(self.depth_m),
            eef_control_pose=self.eef_control_pose,
            controller=self.controller,
        )


@dataclass(frozen=True)
class Lease:
    episode_id: str
    epoch: int
    source: Source


@dataclass
class Chunk:
    binding: Binding
    actions: np.ndarray


def validate_actions(value, *, maximum=50):
    actions = np.asarray(value, dtype=np.float64)
    if (
        actions.ndim != 2
        or actions.shape[1] != 12
        or not 1 <= len(actions) <= maximum
        or not np.isfinite(actions).all()
    ):
        raise ValueError(f"Expected 1..{maximum} finite 12D actions")
    # Canonical representation uses normalized native OSC / mobile commands.
    if np.any(np.abs(actions[:, :6]) > 1) or np.any(np.abs(actions[:, 7:11]) > 1):
        raise ValueError("Native motion command outside [-1, 1]")
    if np.any((actions[:, [6, 11]] < 0) | (actions[:, [6, 11]] > 1)):
        raise ValueError("Canonical gripper and mode must lie in [0, 1]")
    return actions.copy()


def now():
    return time.monotonic()
