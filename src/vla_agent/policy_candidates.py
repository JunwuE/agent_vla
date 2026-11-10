"""Astra's bounded policy proposal surface. Candidates never affect a running episode."""

import hashlib
from pathlib import Path

from pydantic import Field, StrictBool, model_validator

from .contracts import Model
from .data import dumps


class PolicyPatch(Model):
    decision_interval_steps: int | None = Field(default=None, ge=10, le=200, strict=True)
    decision_keep_threshold: float | None = Field(default=None, ge=0, le=1)
    decision_min_margin: float | None = Field(default=None, ge=0, le=0.5)
    decision_history_seconds: list[float] | None = Field(default=None, min_length=1, max_length=5)
    audit_interval_steps: int | None = Field(default=None, ge=100, le=1000, strict=True)
    flora_with_rationale: StrictBool | None = None

    @model_validator(mode="after")
    def validate_offsets(self):
        if not self.model_dump(exclude_none=True):
            raise ValueError("Empty policy proposal")
        if self.decision_history_seconds is not None:
            offsets = self.decision_history_seconds
            if any(not 0 <= t <= 15 for t in offsets) or 0 not in offsets:
                raise ValueError("History offsets must be within 0..15 seconds and include the present")
        return self


class PolicyProposal(Model):
    patch: PolicyPatch
    hypothesis: str = Field(min_length=1, max_length=2000)
    evidence_event_ids: list[int] = Field(min_length=1, max_length=20)
    expected_effect: str = Field(min_length=1, max_length=2000)


def save_candidate(root, proposal, episode_id, available_ids):
    if any(i not in available_ids for i in proposal.evidence_event_ids):
        raise ValueError("Policy evidence must reference existing episode events")
    record = dict(
        schema="vla_agent.policy_candidate.v1",
        episode_id=episode_id,
        status="candidate_requires_independent_evaluation",
        **proposal.model_dump(),
    )
    digest = hashlib.sha256(dumps(record).encode()).hexdigest()
    directory = Path(root) / "policy_candidates"
    directory.mkdir(exist_ok=True)
    path = directory / f"{digest}.json"
    if not path.exists():
        path.write_text(dumps(record))
    return dict(version=digest, path=str(path), status=record["status"], applied=False)


def apply_experimental_patch(config, patch):
    """Host-only experiment builder; all execution budgets/protections remain unchanged."""
    result = config.model_copy(deep=True)
    for field, value in patch.model_dump(exclude_none=True).items():
        if field.startswith("decision_"):
            setattr(result.decision, field[len("decision_") :], value)
        elif field == "audit_interval_steps":
            result.runtime.audit_interval_steps = value
        else:
            result.flora.with_rationale = value
    return type(config).model_validate(result.model_dump())
