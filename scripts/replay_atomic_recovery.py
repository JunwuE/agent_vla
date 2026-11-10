"""Replay an exact recorded VLA prefix, then execute a frozen recovery without model calls.

Development experiment, not an independent VLA/Agent success-rate evaluation.
Only recorded robot commands are replayed; all recovery geometry comes from new RGB-D.
"""

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from vla_agent.adapters.robocasa import RoboCasaEnvironment
from vla_agent.config import load_config
from vla_agent.contracts import Task
from vla_agent.data import Recorder, dumps
from vla_agent.harness import Harness
from vla_agent.memory import WorkingMemory
from vla_agent.recovery import RecoveryRunner
from vla_agent.skill_packages import execution_bundle
from vla_agent.skills import SkillLibrary


async def main(spec_path, output):
    spec = json.loads(spec_path.read_text())
    source = Path(spec["source"])
    cfg = load_config(spec["config"])
    cfg.runtime.experiment_label = "atomic-exact-prefix-code-development"
    cfg.runtime = type(cfg.runtime).model_validate(
        {**cfg.runtime.model_dump(), **spec.get("runtime_overrides", {})}
    )
    cfg.environment.seed = spec["seed"]
    rows = [json.loads(line) for line in (source / "events.jsonl").open()]
    actions = [e for e in rows if e["kind"] == "action_executed"][: spec["prefix_steps"]]
    assert len(actions) == spec["prefix_steps"] and all(e["source"] == "vla" for e in actions)
    library = SkillLibrary(spec["library"])
    skill = library.get(spec["skill_id"], spec["skill_version"], "PandaOmron")
    bundle = execution_bundle(skill["package"], skill["package_files"], library, "PandaOmron")
    log = Recorder(output, cfg)
    env = RoboCasaEnvironment(cfg.environment, output)
    harness = Harness(env, log, step_limit=cfg.runtime.max_steps)
    started = time.monotonic()
    result = dict(
        success=False,
        reason="infrastructure_error",
        astra_calls=0,
        flora_calls=0,
        astra_tokens={},
        control_steps=0,
        error=None,
        development_replay=True,
    )
    try:
        lease = await harness.start()
        task = Task(
            instruction=harness.obs.facts["task_instruction"],
            subtasks=[
                dict(
                    name="task",
                    instruction=harness.obs.facts["task_instruction"],
                    completion_condition="task_success",
                )
            ],
        )
        memory = WorkingMemory(task, harness.obs.episode_id)
        memory.add(harness.obs)
        log.event("task_initialized", task=task, source="environment")
        log.event(
            "replay_protocol",
            **spec,
            source_events_sha256=hashlib.sha256((source / "events.jsonl").read_bytes()).hexdigest(),
            model_calls=0,
            note="Live physics, recorded prefix, development recovery only; not a fresh Agent SR trial.",
        )
        max_error = 0.0
        for step in range(len(actions) + 1):
            if step % 5 == 0 or step == len(actions):
                with np.load(source / "observations" / f"{step:07d}.npz") as ref:
                    error = float(np.max(np.abs(ref["state"] - harness.obs.state)))
                    max_error = max(max_error, error)
                    if error > 1e-5:
                        raise RuntimeError(f"Replay state divergence at {step}: {error}")
                    if step in (0, len(actions)):
                        if not all(
                            np.array_equal(ref["rgb/" + c], harness.obs.rgb[c]) for c in harness.obs.rgb
                        ):
                            raise RuntimeError(f"Replay RGB differs at {step}")
                log.event(
                    "vla_replay",
                    binding=memory.binding(harness.epoch),
                    state_error=error,
                    returned_steps=min(5, len(actions) - step),
                    instruction=memory.instruction,
                )
            if step < len(actions):
                await harness.execute(np.asarray(actions[step]["action"]), lease, memory)
        prefix_seconds = time.monotonic() - started
        report = await RecoveryRunner(harness, memory, cfg.runtime, log).run(
            skill["code"],
            "task_success",
            "recovery-0",
            parameters=spec["parameters"],
            skill=skill,
            program_version=skill["version"],
            package_bundle=bundle,
        )
        result.update(
            success=harness.obs.success,
            reason="success" if harness.obs.success else "recovery_unresolved",
            control_steps=harness.obs.step,
            prefix_seconds=prefix_seconds,
            replay_max_state_error=max_error,
            recovery=report,
            skill_id=skill["id"],
            skill_version=skill["version"],
        )
    except Exception as error:
        result.update(
            error=f"{type(error).__name__}: {error}", control_steps=harness.obs.step if harness.obs else 0
        )
    finally:
        await env.close()
        result["wall_seconds"] = time.monotonic() - started
        log.event("episode_end", result=result)
        (output / "result.json").write_text(dumps(result) + "\n")
        log.close()
    print(dumps({k: v for k, v in result.items() if k != "recovery"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    asyncio.run(main(a.spec.resolve(), a.output.resolve()))
