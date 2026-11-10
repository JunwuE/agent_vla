"""Host-owned stage evidence. Program-reported measurements are labelled separately."""

import uuid

from pydantic import Field

from .contracts import Model


class StageSpec(Model):
    name: str = Field(min_length=1, max_length=120)
    preconditions: list[str] = Field(default_factory=list, max_length=16)
    completion_condition: str | None = None
    parent_stage_id: str | None = None


class StageTracker:
    def __init__(self, harness, intervention_id):
        self.harness, self.intervention_id = harness, intervention_id
        self.rows = []
        self.active = None
        self.parents = []

    async def start(self, args):
        spec = StageSpec.model_validate(args)
        if self.active and spec.parent_stage_id != self.active['stage_id']:
            expected_parent = self.parents[-1]['stage_id'] if self.parents else None
            if spec.parent_stage_id != expected_parent:
                raise ValueError('Stage parent does not match the active context')
            if self.active["completion_condition"]:
                raise ValueError("Finish the active checked stage before starting another")
            self.close(None)
        if spec.parent_stage_id is not None:
            if self.active is None or self.active['stage_id'] != spec.parent_stage_id:
                raise ValueError('Stage parent does not match the active context')
            if len(self.parents) >= 31:
                raise ValueError('Stage nesting exceeds 32 contexts')
            self.parents.append(self.active)
        elif self.active is not None:
            raise ValueError('Finish the nested stage context before starting a top-level stage')
        row = dict(
            stage_id=uuid.uuid4().hex,
            **spec.model_dump(),
            start_step=self.harness.obs.step,
            end_step=None,
            status="active",
            checks=[],
            program_measurements=[],
            error=None,
        )
        self.rows.append(row)
        self.active = row
        available = getattr(self.harness.env, "verifier_conditions", ["task_success"])
        for condition in spec.preconditions + (
            [spec.completion_condition] if spec.completion_condition else []
        ):
            if condition not in available:
                self.close("No installed verifier: " + condition)
                raise ValueError("Stage references an unavailable verifier: " + condition)
        for condition in spec.preconditions:
            proof = await self.harness.verify(condition)
            row["checks"].append(dict(phase="precondition", **proof.model_dump()))
            if proof.passed is not True:
                self.close("Stage precondition not verified: " + condition)
                raise ValueError("Stage precondition not verified: " + condition)
        self.harness.log.event("stage_begin", intervention_id=self.intervention_id, stage=row.copy())
        return dict(stage_id=row["stage_id"], observation_step=self.harness.obs.step)

    async def finish(self, stage_id, error=None, measured=None):
        if self.active is None or self.active["stage_id"] != stage_id:
            raise ValueError("Stage completion does not match the active stage")
        row = self.active
        if measured:
            row["program_measurements"].append(measured)
        if error is None and row["completion_condition"]:
            proof = await self.harness.verify(row["completion_condition"])
            row["checks"].append(dict(phase="completion", **proof.model_dump()))
            if proof.passed is not True:
                error = "Stage completion not verified: " + row["completion_condition"]
        self.close(error)
        return dict(ok=error is None, independently_verified=row["status"] == "verified", stage=row)

    def close(self, error):
        if self.active is None:
            return
        row = self.active
        self.active = self.parents.pop() if self.parents else None
        row["end_step"] = self.harness.obs.step
        row["error"] = error
        row["status"] = (
            "failed"
            if error
            else "verified"
            if any(c["phase"] == "completion" and c["passed"] is True for c in row["checks"])
            else "unchecked"
        )
        self.harness.log.event("stage_end", intervention_id=self.intervention_id, stage=row)

    def close_all(self, error):
        while self.active is not None:
            self.close(error)

    @property
    def failed_stage(self):
        return next((row["name"] for row in reversed(self.rows) if row["status"] == "failed"), None)
