"""The only action executor. Epoch fencing also rejects late inference results."""

import asyncio
import time
from collections import deque

from .contracts import Lease, Source, validate_actions


class StaleControl(RuntimeError):
    pass


class Harness:
    def __init__(self, environment, recorder, step_limit=100000):
        self.env, self.log = environment, recorder
        self.step_limit = step_limit
        self.epoch = 0
        self.owner = None
        self.queue = deque()
        self.lock = asyncio.Lock()
        self.halted = False
        self.uncertain = False
        self.obs = None

    async def start(self):
        self.obs = await self.env.reset()
        self.log.observe(self.obs)
        return await self.claim(Source.VLA)

    async def claim(self, source):
        async with self.lock:
            if self.halted or self.uncertain:
                raise RuntimeError("Harness stopped or execution uncertain")
            discarded = len(self.queue)
            self.queue.clear()
            self.epoch += 1
            self.owner = source
            self.obs = await self.env.observe()
            self.log.event("control_claim", owner=source, epoch=self.epoch, discarded_steps=discarded)
            return Lease(self.obs.episode_id, self.epoch, source)

    def check(self, lease):
        if (
            self.halted
            or self.uncertain
            or lease.epoch != self.epoch
            or lease.source != self.owner
            or lease.episode_id != self.obs.episode_id
        ):
            raise StaleControl("Expired action authority")

    def enqueue(self, chunk, lease, memory, limit):
        self.check(lease)
        if not memory.matches(chunk.binding, self.epoch) or chunk.binding.observation_step != self.obs.step:
            raise StaleControl("Late VLA chunk")
        if self.queue:
            raise RuntimeError("Cannot replace an executing chunk")
        actions = validate_actions(chunk.actions)
        self.log.event(
            "vla_proposal", binding=chunk.binding, actions=actions, execute_steps=min(limit, len(actions))
        )
        self.queue.extend(actions[:limit])

    async def execute(self, action, lease, memory, intervention_id=None):
        async with self.lock:
            self.check(lease)
            if self.obs.terminal:
                raise RuntimeError("Environment already terminal")
            if self.obs.step >= self.step_limit:
                raise RuntimeError("Episode action budget exhausted")
            action = validate_actions([action], maximum=1)[0]
            before = self.log.observe(self.obs)
            command_id = self.log.event(
                "action_command",
                before=before,
                source=lease.source,
                control_epoch=self.epoch,
                action=action,
                instruction=memory.instruction,
                intervention_id=intervention_id,
            )
            try:
                environment_started = time.perf_counter()
                after = await self.env.step(action)
                environment_seconds = time.perf_counter() - environment_started
            except BaseException as error:
                self.uncertain = True
                self.queue.clear()
                self.epoch += 1
                self.log.event("execution_uncertain", command_id=command_id, error=type(error).__name__)
                raise
            self.obs = after
            memory.add(after)
            recording_started = time.perf_counter()
            after_path = self.log.observe(after)
            recording_seconds = time.perf_counter() - recording_started
            receipt = dict(
                command_id=command_id,
                before=before,
                after=after_path,
                source=lease.source,
                action=action,
                measured_state=after.state,
                measured_eef_control_pose=after.eef_control_pose,
                instruction=memory.instruction,
                instruction_version=memory.instruction_version,
                subtask_version=memory.subtask_version,
                intervention_id=intervention_id,
            )
            self.log.event("action_executed", **receipt)
            self.log.event(
                "action_performance", step=after.step, source=lease.source,
                environment_seconds=environment_seconds, recording_seconds=recording_seconds,
                **getattr(self.env, "last_step_timing", {}),
            )
            memory.action_history.append(
                dict(
                    step=after.step,
                    sim_time=after.sim_time,
                    source=lease.source,
                    command=action.tolist(),
                    measured_state=after.state.tolist(),
                )
            )
            return after

    async def next_step(self, lease, memory):
        self.check(lease)
        action = self.queue.popleft()
        return await self.execute(action, lease, memory)

    async def stop(self, reason):
        # Invalidate first, before waiting on environment I/O or the action lock.
        self.halted = True
        self.epoch += 1
        self.queue.clear()
        self.log.event("stop", reason=reason)
        await self.env.stop()

    async def verify(self, condition):
        self.obs = await self.env.observe()
        result = await self.env.verify(condition)
        if result.observation_step != self.obs.step:
            raise RuntimeError("Verifier returned stale evidence")
        self.log.event("verification", result=result, observation=self.log.observe(self.obs))
        return result
