"""Deterministic software fixture; these facts are NOT robot perception benchmarks."""

import asyncio
import time
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

from ..contracts import (
    Chunk,
    Decision,
    DecisionLabel,
    ExecutionReview,
    Intervention,
    Observation,
    Verification,
)


class MockEnvironment:
    verifier_conditions = ["object_lifted", "ready_to_resume", "object_placed", "task_success"]

    def __init__(self, seed=7):
        self.seed = seed
        self.episode_id = uuid.uuid4().hex
        self.step_id = 0
        self.pos = np.array([0.0, 0.0, 0.1])
        self.rotation = Rotation.identity()
        self.closed = False
        self.grasped = False
        self.lifted = False
        self.placed = False
        self.stopped = False
        self.history = []
        self.events = ()

    async def reset(self):
        return await self.observe()

    async def observe(self):
        state = np.r_[
            self.pos,
            self.rotation.as_quat(),
            [0, 0, 0],
            [0, 0, 0, 1],
            [0 if self.closed else 0.04, 0 if self.closed else 0.04],
        ]
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb[20:35, 20:35] = (40, 140, 230) if self.grasped else (220, 80, 30)
        return Observation(
            self.episode_id,
            self.step_id,
            time.monotonic(),
            self.step_id / 20,
            state,
            {k: rgb.copy() for k in ("left", "right", "wrist")},
            depth_m={"left": np.full((64, 64), 0.5)},
            calibration={
                "left": dict(
                    step=self.step_id,
                    frame="optical_to_osc_arm_base",
                    alignment_verified=True,
                    intrinsics=[[100, 0, 32], [0, 100, 32], [0, 0, 1]],
                    camera_to_base=np.eye(4).tolist(),
                )
            },
            facts=dict(
                fixture=True,
                grasped=self.grasped,
                lifted=self.lifted,
                placed=self.placed,
                empty_grasp=self.closed and not self.grasped,
                gripper_closed=self.closed,
            ),
            events=self.events,
            terminal=self.placed,
            success=self.placed,
            eef_control_pose=dict(
                step=self.step_id,
                frame="osc_arm_base",
                position=self.pos.tolist(),
                quaternion_xyzw=self.rotation.as_quat().tolist(),
            ),
            controller=dict(
                name="OSC_POSE",
                input_type="delta",
                input_ref_frame="base",
                impedance_mode="fixed",
                input_min=[-1.0] * 6,
                input_max=[1.0] * 6,
                output_min=[-0.02] * 3 + [-0.1] * 3,
                output_max=[0.02] * 3 + [0.1] * 3,
                max_position_delta=0.05,
                max_rotation_delta=0.35,
                max_mobile_command=0.25,
            ),
        )

    async def step(self, action):
        if self.stopped:
            raise RuntimeError("Fixture stopped")
        await asyncio.sleep(0)
        self.pos += np.asarray(action[:3]) * 0.02
        self.rotation = Rotation.from_rotvec(np.asarray(action[3:6]) * 0.1) * self.rotation
        was_closed = self.closed
        self.closed = action[6] >= 0.5
        self.events = ("gripper_changed",) if was_closed != self.closed else ()
        if self.closed and not was_closed and abs(self.pos[0] - 0.04) < 0.012:
            self.grasped = True
        if self.grasped and self.closed and self.pos[2] >= 0.14:
            self.lifted = True
        if not self.closed and self.grasped:
            self.placed = bool(self.lifted and self.pos[0] >= 0.1)
            self.grasped = False
        self.step_id += 1
        self.history.append(dict(lifted=self.lifted, grasped=self.grasped, step=self.step_id))
        return await self.observe()

    async def verify(self, condition):
        if condition in {"object_lifted", "ready_to_resume"}:
            passed = len(self.history) >= 2 and all(x["lifted"] and x["grasped"] for x in self.history[-2:])
        elif condition in {"object_placed", "task_success"}:
            passed = self.placed
        else:
            passed = None
        return Verification(
            condition=condition,
            passed=passed,
            observation_step=self.step_id,
            evidence=[f"fixture measurement at step {self.step_id}; condition={condition}"],
        )

    async def stop(self):
        self.stopped = True

    async def close(self):
        self.stopped = True


class MockVLA:
    async def infer(self, obs, instruction, binding):
        await asyncio.sleep(0)
        a = np.zeros((5, 12))
        a[:, 6] = 1
        if "place" in instruction.lower():
            a[:, 0] = 1
            if obs.state[0] >= 0.1:
                a[:, 0] = 0
                a[:, 6] = 0
        return Chunk(binding, a)

    async def close(self):
        pass


class MockDecision:
    async def decide(self, observations, memory_summary, binding, trigger):
        await asyncio.sleep(0)
        facts = observations[-1].facts
        if facts.get("empty_grasp"):
            label = DecisionLabel.REFINE
        elif facts.get("lifted") and memory_summary["subtask_index"] == 0:
            label = DecisionLabel.SWITCH
        else:
            label = DecisionLabel.KEEP
        return Decision(
            binding=binding,
            label=label,
            trigger=trigger,
            probabilities={x.value: float(x == label) for x in DecisionLabel},
        )

    async def close(self):
        pass


RECOVERY_CODE = """def run(ctx):
    ctx.note(facts=["fixture empty grasp"], hypotheses=["grasp alignment is offset"])
    ctx.stage("open and align")
    ctx.set_gripper(False)
    # 0.04 is a known coordinate in this deterministic software fixture only.
    dx = 0.04 - ctx.observe()["eef_control_pose"]["position"][0]
    ctx.move_relative([dx, 0, 0], False, steps=2)
    ctx.stage("grasp and lift")
    ctx.set_gripper(True)
    ctx.move_relative([0, 0, 0.045], True, steps=4)
    ctx.stage("verify")
    return "RECOVERED" if ctx.check("object_lifted")["passed"] else "UNRESOLVED"
"""


class MockPlanner:
    async def plan(self, decision, context, toolkit):
        await asyncio.sleep(0)
        if context.get("intent") == "select_next_executor":
            return Intervention(kind="continue", reason="Next complete subtask assigned to VLA")
        if context.get("intent") == "initialize_plan":
            return Intervention(kind="replan", reason="Initialize fixture plan", remaining_plan=toolkit.memory.task.subtasks)
        if context.get("intent") == "review_running":
            facts = context["observations"][-1].facts
            completed = facts.get("lifted") and context["memory"]["subtask_index"] == 0
            failed = facts.get("empty_grasp")
            return ExecutionReview(
                action="request_intervention" if completed or failed else "continue",
                assessment="subtask_complete" if completed else "failed_attempt" if failed else "progressing",
                phase="fixture execution", reason="Read the fixture measurements and attempt state",
                evidence=[dict(observation_step=context["binding"].observation_step,
                               fact="Fixture lift measured" if completed else "Fixture empty grasp measured")]
                if completed or failed else [],
            )
        if context.get("intent") == "confirm_intervention":
            facts = context["observations"][-1].facts
            if facts.get("lifted") and context["memory"]["subtask_index"] == 0:
                return Intervention(kind="switch", reason="Fresh fixture lift observed; verify")
            if not facts.get("empty_grasp"):
                return Intervention(kind="continue", reason="Earlier fixture problem no longer applies")
        if decision.label == DecisionLabel.SWITCH:
            return Intervention(kind="switch", reason="Fixture lift observed; verify before switching")
        return Intervention(
            kind="code",
            reason="Recover fixture grasp",
            code=RECOVERY_CODE,
            completion_condition="object_lifted",
            hypotheses=["misaligned grasp"],
        )

    async def close(self):
        pass


EXPLORATION_CODE = """# Deterministic mock physics only; these coordinates are fixture data.
LIFT_METERS = 0.01

def align(ctx):
    with ctx.stage("open and align"):
        raw = ctx.read_observation("left")
        assert raw["rgb"].shape[-1] == 3
        ctx.set_gripper(False)
        dx = 0.04 - ctx.observe()["eef_control_pose"]["position"][0]
        ctx.move_relative([dx, 0, 0], False, steps=2)

def grasp_and_lift(ctx):
    with ctx.stage("grasp and lift", completion_condition="object_lifted") as stage:
        ctx.set_gripper(True)
        measured = ctx.move_relative([0, 0, LIFT_METERS], True, steps=4)
        stage.record(position_error_m=measured["position_error_m"])

def run(ctx):
    align(ctx)
    grasp_and_lift(ctx)
    return "RECOVERED"
"""


class MockExplorer(MockPlanner):
    """Demonstrate an actual program revision through the same tools Astra uses."""

    async def plan(self, decision, context, toolkit):
        if context.get("intent") in {"initialize_plan", "review_running", "select_next_executor"}:
            return await super().plan(decision, context, toolkit)
        if decision.label == DecisionLabel.SWITCH:
            return await super().plan(decision, context, toolkit)
        if "recovery_feedback" not in context:
            version = await toolkit.call(
                "write_program",
                dict(
                    source=EXPLORATION_CODE,
                    hypothesis="fixture alignment then a small lift may suffice",
                    expected_effect="two successive fixture observations of a lifted object",
                    skill_name='align-grasp-lift-fixture',
                    when_to_use='Mock fixture requires alignment before grasp/lift',
                    reuse_reason='No procedure for this mock fixture is installed',
                ),
            )
        else:
            trace = await toolkit.call("read_recovery_trace", {})
            if trace["report"]["failed_stage"] != "grasp and lift":
                return Intervention(kind="stop", reason="Unexpected fixture failure; preserve evidence")
            version = await toolkit.call(
                "patch_program",
                dict(
                    base_version=trace["version"],
                    old_text="LIFT_METERS = 0.01",
                    new_text="LIFT_METERS = 0.045",
                    hypothesis="fixture lift distance was insufficient",
                    expected_effect="increase lift distance from the fresh measured pose",
                ),
            )
        checks = await toolkit.call("check_program", dict(version=version["version"]))
        if not checks["ok"]:
            return Intervention(kind="stop", reason="Fixture program failed static checks")
        return Intervention(
            kind="code",
            reason="Develop and verify fixture recovery",
            program_version=version["version"],
            completion_condition="object_lifted",
        )
