"""Replay an uninterrupted, single-subtask baseline through Flora; never execute robot actions."""

import argparse
import asyncio
import os
from pathlib import Path

from vla_agent.config import load_config
from vla_agent.contracts import Task
from vla_agent.data import Recorder, dumps, read_events
from vla_agent.flora import FloraEvaluator, prepare_segment
from vla_agent.memory import WorkingMemory


async def run(args):
    source = args.episode.resolve()
    rows = read_events(source)
    tasks = [row["task"] for row in rows if row["kind"] == "task_initialized"]
    if len(tasks) != 1 or len(tasks[0]["subtasks"]) != 1:
        raise ValueError("Replay expects one initialized task with one subtask")
    if any(row["kind"] in {"intervention_plan", "attempt_interrupted", "initial_plan"} for row in rows):
        raise ValueError("Use an uninterrupted baseline; intervention/plan boundaries need separate reconstruction")
    observations = [row for row in rows if row["kind"] == "observation"]
    episodes = {row["episode_id"] for row in observations}
    if len(episodes) != 1 or not args.steps or min(args.steps) < 1:
        raise ValueError("Require one recorded episode and positive snapshot steps")
    config = load_config(args.config)
    config.flora.enabled = True
    log = Recorder(args.output.resolve(), config)
    client = FloraEvaluator(config.flora, log)
    report = dict(source=str(source), prepare_only=args.prepare_only, robot_actions_executed=0, results=[])
    try:
        for row in observations:
            log.observations[f"{row['episode_id']}:{row['step']}"] = os.path.relpath(source / row["path"], log.root)
        memory = WorkingMemory(Task.model_validate(tasks[0]), episodes.pop())
        for step in sorted(set(args.steps)):
            memory.add(log.read_observation(step, memory.episode_id, step))
            binding = memory.binding(0)
            if args.prepare_only:
                payload = await asyncio.to_thread(prepare_segment, log, memory, binding, config.flora)
                target = log.root / f"step-{step}.input.json"
                target.write_text(dumps(payload))
                result = dict(status="prepared_only", step=step, frames=len(payload["frames"]),
                              segment={k: v for k, v in payload["segment"].items() if k != "sources"},
                              input_path=target.name)
            else:
                result = await client.evaluate(memory, binding)
                if result["status"] == "scored":
                    cached = await client.evaluate(memory, binding)
                    result["repeat_cache_verified"] = cached.get("cache_hit") is True
            report["results"].append(result)
            print(dumps(result), flush=True)
        report["flora_calls"] = client.calls
        log.finish(report)
    finally:
        await client.close()
        log.close()
    return all(r["status"] in {"scored", "prepared_only"} for r in report["results"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/robocasa.yaml")
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--prepare-only", action="store_true", help="Validate/sample inputs without model inference")
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(run(args)) else 1)


if __name__ == "__main__":
    main()
