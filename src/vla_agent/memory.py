from __future__ import annotations

import copy
import uuid
from collections import deque
from dataclasses import dataclass, field

from .contracts import Binding, Observation, Subtask, Task
from .phases import PhaseMemory
from .skills import SkillLibrary as SkillLibrary


@dataclass
class WorkingMemory:
    task: Task
    episode_id: str
    subtask_index: int = 0
    subtask_version: int = 0
    instruction_version: int = 0
    pending_instruction: str | None = None
    instruction: str = ""
    observations: deque = field(default_factory=lambda: deque(maxlen=320))
    action_history: deque = field(default_factory=lambda: deque(maxlen=320))
    interventions: deque = field(default_factory=lambda: deque(maxlen=10))
    notes: deque = field(default_factory=lambda: deque(maxlen=20))
    current_code: dict | None = None
    reviews: deque = field(default_factory=lambda: deque(maxlen=8))
    reward_assessments: deque = field(default_factory=lambda: deque(maxlen=8))
    attempts: deque = field(default_factory=lambda: deque(maxlen=12))
    attempt_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    attempt_number: int = 1
    attempt_started_step: int = 0
    attempt_status: str = "running"
    last_problem: dict | None = None
    language_catalog: dict = field(default_factory=dict)
    phases: PhaseMemory = field(default_factory=PhaseMemory)
    groundings: dict = field(default_factory=dict)
    retrieved_experiences: list = field(default_factory=list)
    task_scope: dict = field(default_factory=dict)
    task_memory: dict | None = None
    executor: str = "vla"
    code_assignment: dict | None = None
    rule_signals: list = field(default_factory=list)
    completed_subtasks: list = field(default_factory=list)

    def __post_init__(self):
        self.instruction = self.current.instruction
        for command in self.task.subtasks + self.task.language_catalog:
            if command.name in self.language_catalog and self.language_catalog[command.name] != command:
                raise ValueError("Language catalog command ids must be unique")
            self.language_catalog[command.name] = command.model_copy(deep=True)

    def language_command(self, instruction=None, instruction_id=None):
        if instruction_id is not None:
            command = self.language_catalog.get(instruction_id)
            if command is None or instruction is not None and instruction != command.instruction:
                raise ValueError("Unknown instruction_id or text differs from the registered command")
            return command
        command = next((c for c in self.language_catalog.values() if c.instruction == instruction), None)
        if command is None:
            raise ValueError("Select an exact instruction from language_catalog; free-form motion tutorials are disabled")
        return command

    def attempt_summary(self):
        step = self.observations[-1].step if self.observations else self.attempt_started_step
        return dict(id=self.attempt_id, number=self.attempt_number,
                    first_attempt=self.attempt_number == 1 and self.last_problem is None,
                    status=self.attempt_status,
                    started_step=self.attempt_started_step, elapsed_steps=step - self.attempt_started_step,
                    subtask=self.current.name, last_problem=self.last_problem)

    def begin_retry(self, reason, evidence=None):
        if self.attempt_status != "recovering":
            self.attempts.append(dict(**self.attempt_summary(), outcome="intervened", reason=reason,
                                      evidence=evidence))
        self.last_problem = dict(reason=reason, evidence=evidence, source="astra_assessment")
        self.attempt_status = "recovering"

    def resume_retry(self):
        if self.attempt_status != "recovering":
            return
        self.attempt_id = uuid.uuid4().hex
        self.attempt_number += 1
        self.attempt_started_step = self.observations[-1].step
        self.attempt_status = "running"
        self.rule_signals.clear()

    @property
    def current(self):
        return self.task.subtasks[self.subtask_index]

    def add(self, obs: Observation):
        if obs.episode_id != self.episode_id:
            raise ValueError("Cross-episode observation")
        if not self.observations or self.observations[-1].step < obs.step:
            self.observations.append(obs)

    def binding(self, epoch):
        obs = self.observations[-1]
        return Binding(
            episode_id=self.episode_id,
            subtask_version=self.subtask_version,
            instruction_version=self.instruction_version,
            control_epoch=epoch,
            observation_step=obs.step,
            observation_time=obs.timestamp,
        )

    def matches(self, binding, epoch):
        return (
            binding.episode_id == self.episode_id
            and binding.subtask_version == self.subtask_version
            and binding.instruction_version == self.instruction_version
            and binding.control_epoch == epoch
        )

    def inference_boundary(self):
        if self.pending_instruction is not None:
            self.instruction = self.pending_instruction
            self.instruction_version += 1
            self.pending_instruction = None

    def advance(self, verification=None):
        self.completed_subtasks.append(dict(subtask=self.current.model_dump(),
            subtask_version=self.subtask_version, verification=verification))
        if self.subtask_index + 1 >= len(self.task.subtasks):
            return False
        self.attempts.append(dict(**self.attempt_summary(), outcome="subtask_done"))
        self.subtask_index += 1
        self.subtask_version += 1
        self.pending_instruction = self.current.instruction
        self.attempt_id = uuid.uuid4().hex
        self.attempt_number = 1
        self.attempt_started_step = self.observations[-1].step if self.observations else 0
        self.attempt_status = "running"
        self.last_problem = None
        self.phases = PhaseMemory()
        self.groundings.clear()
        self.rule_signals.clear()
        self.code_assignment = None
        self.current_code = None
        self.executor = "unassigned"
        return True

    def replan(self, remaining: list[Subtask]):
        if not remaining:
            raise ValueError("Remaining plan must not be empty")
        same_goal = (remaining[0].instruction, remaining[0].completion_condition) == (
            self.current.instruction, self.current.completion_condition)
        self.task = self.task.model_copy(
            update={"subtasks": self.task.subtasks[: self.subtask_index] + remaining}
        )
        self.subtask_version += 1
        self.pending_instruction = self.current.instruction
        if not same_goal:
            self.attempts.append(dict(**self.attempt_summary(), outcome='replanned'))
            self.attempt_id = uuid.uuid4().hex
            self.attempt_number = 1
            self.attempt_started_step = self.observations[-1].step if self.observations else 0
            self.attempt_status = 'running'
            self.last_problem = None
            self.current_code = None
            self.phases = PhaseMemory()
            self.groundings.clear()
        self.rule_signals.clear()
        self.code_assignment = None
        self.executor = "vla"

    def summary(self):
        return dict(
            task=self.task.model_dump(),
            subtask_index=self.subtask_index,
            subtask_version=self.subtask_version,
            current_subtask=self.current.model_dump(),
            completed_subtasks=copy.deepcopy(self.completed_subtasks),
            instruction=self.instruction,
            instruction_version=self.instruction_version,
            pending_instruction=self.pending_instruction,
            recent_actions=list(self.action_history)[-30:],
            motion_windows=self.motion_windows(),
            phase_memory=self.phases.summary(),
            rule_signals=copy.deepcopy(self.rule_signals),
            executor=self.executor,
            code_assignment=self.code_assignment,
            groundings=list(self.groundings.values())[-4:],
            retrieved_experiences=self.retrieved_experiences,
            task_scope=self.task_scope,
            task_memory=copy.deepcopy(self.task_memory),
            interventions=list(self.interventions),
            notes=list(self.notes),
            current_code=self.current_code,
            attempt=self.attempt_summary(),
            attempts=list(self.attempts),
            reviews=list(self.reviews),
            reward_assessments=list(self.reward_assessments),
            language_catalog=[dict(instruction_id=key, **value.model_dump())
                              for key, value in self.language_catalog.items()],
        )

    def motion_windows(self):
        import numpy as np

        end = self.observations[-1].sim_time if self.observations else 0
        result = {}
        for seconds in (2, 10):
            rows = [r for r in self.action_history if r['step'] > self.attempt_started_step
                    and r.get('sim_time', float('-inf')) >= end - seconds]
            if not rows:
                continue
            states = np.array([r['measured_state'] for r in rows])
            result[str(seconds)] = dict(from_step=rows[0]['step'], to_step=rows[-1]['step'],
                samples=len(rows), observed_seconds=rows[-1]['sim_time']-rows[0]['sim_time'],
                eef_net_m=(states[-1, :3]-states[0, :3]).tolist(),
                eef_path_m=float(np.linalg.norm(np.diff(states[:, :3], axis=0), axis=1).sum()),
                base_net_m=(states[-1, 7:10]-states[0, 7:10]).tolist())
        return result
