"""Run the fixed three-task development pilot; record all trials before selecting videos."""

import argparse
import asyncio
import json
import time
from pathlib import Path

from vla_agent.cli import run
from vla_agent.config import load_config

TASKS = {"OpenDrawer": 750, "PickPlaceCounterToStove": 600, "TurnOnSinkFaucet": 600}


async def main(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    mode_root = root / args.mode
    mode_root.mkdir(exist_ok=False)
    plan = dict(purpose="three atomic tasks development pilot, not full benchmark",
                tasks=TASKS, seeds=args.seeds, mode=args.mode, workers=args.workers,
                selection="all runs retained; visualization selection recorded separately",
                policy_rng="shared server sampling stream; simulator seeds fixed; no bitwise policy replay")
    (mode_root / "plan.json").write_text(json.dumps(plan, indent=2))
    records = []
    semaphore = asyncio.Semaphore(args.workers)

    async def case(task, seed):
        async with semaphore:
            c = load_config(args.config)
            c.environment.task_name, c.environment.seed = task, seed
            c.environment.task_group = "atomic_seen"
            c.environment.instruction_source = "environment"
            c.runtime.max_steps = TASKS[task]
            c.runtime.max_wall_seconds = 1800
            c.runtime.experiment_label = args.mode
            c.runtime.supervision_enabled = args.mode != "baseline"
            c.runtime.initial_planning = args.mode != "baseline"
            if args.mode == "supervised":
                # A transparent pilot budget, not a calibrated operating point.
                c.decision.interval_steps = 50
                c.decision.max_unreviewed_steps = 5
                c.decision.max_age_steps = 10
                c.planner.max_calls = 15
                c.runtime.allowed_interventions = ["continue", "refine", "replan", "observe", "stop"]
            output = mode_root / f"{task}_seed{seed}"
            print(json.dumps(dict(event="start", task=task, seed=seed, mode=args.mode)), flush=True)
            started = time.monotonic()
            result = await run(c, output)
            record = dict(task=task, seed=seed, path=str(output), result=result,
                          elapsed_seconds=time.monotonic() - started)
            records.append(record)
            (mode_root / "summary.json").write_text(json.dumps(records, indent=2))
            print(json.dumps(dict(event="finish", **record)), flush=True)

    await asyncio.gather(*(case(task, seed) for task in TASKS for seed in args.seeds))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/robocasa.yaml")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--mode", choices=("baseline", "supervised", "low_calls"), default="baseline")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()
    if args.workers < 1 or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        p.error("Use positive workers and distinct nonnegative seeds")
    asyncio.run(main(args))
