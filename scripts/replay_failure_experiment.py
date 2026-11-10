"""Reproduce recorded VLA commands in real physics, then test live supervision/recovery.

This is a recovery-development replay, not a fresh VLA SR evaluation. Every replay
inference boundary must match the recorded measured state. No simulator object state
or future success label is exposed to the agent.
"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np

from vla_agent.adapters.pi05 import Pi05
from vla_agent.adapters.robocasa import RoboCasaEnvironment
from vla_agent.codex_planner import CodexPlanner
from vla_agent.config import load_config
from vla_agent.contracts import Chunk
from vla_agent.data import Recorder, read_events
from vla_agent.decision import Decision3
from vla_agent.runtime import AgentRuntime


class ReplayPrefix:
    def __init__(self, source, steps, live, recorder):
        self.source, self.steps, self.live, self.log = source, steps, live, recorder
        rows = read_events(source)
        actions = [r for r in rows if r["kind"] == "action_executed"]
        if any(r["source"] != "vla" for r in actions):
            raise ValueError("Replay source must be an uninterrupted VLA baseline")
        self.actions = np.asarray([r["action"] for r in actions])
        if steps > len(self.actions):
            raise ValueError("Replay prefix exceeds the recorded trajectory")
        self.log.event("replay_protocol", source=str(source.resolve()), prefix_steps=steps,
                       source_events_sha256=hashlib.sha256((source / "events.jsonl").read_bytes()).hexdigest(),
                       note="Recorded commands replayed in new physics; not live VLA inference before prefix end")

    async def infer(self, obs, instruction, binding):
        if obs.step >= self.steps:
            return await self.live.infer(obs, instruction, binding)
        reference = self.source / "observations" / f"{obs.step:07d}.npz"
        with np.load(reference) as arrays:
            error = float(np.max(np.abs(arrays["state"] - obs.state)))
            if error > 1e-5:
                raise ValueError(f"Replay state diverged at step {obs.step}: {error}")
            if obs.step == 0 and not np.array_equal(arrays["rgb/left"], obs.rgb["left"]):
                raise ValueError("Replay initial scene differs from the reference")
        actions = self.actions[obs.step:min(self.steps, obs.step + 5)]
        self.log.event("vla_replay", binding=binding, source=str(reference), state_error=error,
                       returned_steps=len(actions), instruction=instruction)
        return Chunk(binding, actions)

    async def close(self):
        await self.live.close()


async def main(args):
    config = load_config(args.config)
    config.environment.seed = args.seed
    config.environment.ignore_native_horizon = True
    config.runtime.experiment_label = "recorded-failure-prefix-recovery-development"
    config.decision.interval_steps = args.audit_step or args.prefix_steps
    config.runtime.audit_interval_steps = args.audit_step or args.prefix_steps
    config.runtime.stagnation_enabled = False
    log = Recorder(args.output.resolve(), config)
    try:
        env = RoboCasaEnvironment(config.environment, log.root)
        live = Pi05(config.vla, log)
        policy = ReplayPrefix(args.source, args.prefix_steps, live, log)
        decision = Decision3(config.decision, log) if config.decision.kind == 'decision3' else None
        result = await AgentRuntime(config, env, policy, decision,
                                    CodexPlanner(config.planner, log, config.runtime.allowed_interventions), log).run()
        print(json.dumps(result), flush=True)
    finally:
        log.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prefix-steps", type=int, default=800)
    p.add_argument("--audit-step", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.prefix_steps < 1 or args.prefix_steps % 5:
        p.error("Use a positive prefix length divisible by five")
    if args.audit_step is not None and args.audit_step < 1:
        p.error("Use a positive audit step")
    asyncio.run(main(args))
