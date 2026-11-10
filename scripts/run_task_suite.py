"""Predeclare a 5+5+5 task suite; retain every attempted episode and its evidence."""

import argparse
import ast
import asyncio
import hashlib
import time
from pathlib import Path

from vla_agent.cli import run
from vla_agent.config import load_config
from vla_agent.data import dumps
from vla_agent.visualization import render_episode

TASKS = {
    "atomic_seen": ["OpenDrawer", "PickPlaceCounterToStove", "TurnOnSinkFaucet", "OpenCabinet", "CloseFridge"],
    "composite_seen": ["PrepareCoffee", "KettleBoiling", "LoadDishwasher", "RinseSinkBasin", "SetUpCuttingStation"],
    "composite_unseen": ["ArrangeTea", "BreadSelection", "GatherTableware", "HeatKebabSandwich", "WashFruitColander"],
}


def registry(source):
    """Read public constant task lists/horizons; the simulator independently checks membership at reset."""
    path = Path(source) / "robocasa/utils/dataset_registry.py"
    raw = path.read_bytes()
    groups, horizons = {}, {}
    for node in ast.parse(raw).body:
        if not isinstance(node, ast.Assign):
            continue
        names = {x.id for x in node.targets if isinstance(x, ast.Name)}
        if "TARGET_TASKS" in names:
            groups = {item.arg: ast.literal_eval(item.value) for item in node.value.keywords}
        if names & {"ATOMIC_TASK_DATASETS", "COMPOSITE_TASK_DATASETS"}:
            horizons.update({item.arg: next(ast.literal_eval(k.value) for k in item.value.keywords if k.arg == "horizon")
                             for item in node.value.keywords})
    for group, tasks in TASKS.items():
        if len(tasks) != 5 or len(set(tasks)) != 5 or not set(tasks) <= set(groups[group]):
            raise ValueError(f"Invalid task selection for {group}")
    return horizons, dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())


async def main(args):
    base = load_config(args.config)
    horizons, registry_source = registry(base.environment.source_path)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "episodes").mkdir(exist_ok=False)
    (root / "report").mkdir(exist_ok=False)
    cases = [dict(group=group, task=task, seed=args.seed, horizon=horizons[task])
             for group, tasks in TASKS.items() for task in tasks]
    plan = dict(purpose="15-task development evaluation, one preselected seed per task; not a benchmark estimate",
                tasks=cases, config=base.model_dump(), workers=args.workers,
                selection="Task names and seed fixed before observing results; no success-based replacement",
                registry=registry_source, memory_mode="frozen",
                policy_rng="Shared policy-server sampling stream; no bitwise action replay guarantee",
                verifiers="Current RoboCasa adapter exposes task_success only; finer reasoning is stored in notes",
                training_exposure="Official task-group labels; composite training exposure of base checkpoint not audited")
    with (root / "plan.json").open("x") as stream:
        stream.write(dumps(plan))
    records = []
    workers, renderers = asyncio.Semaphore(args.workers), asyncio.Semaphore(2)

    async def case(spec):
        name = spec["task"] + f"_seed{spec['seed']}"
        output = root / "episodes" / name
        async with workers:
            config = base.model_copy(deep=True)
            config.environment.task_name, config.environment.seed = spec["task"], spec["seed"]
            config.environment.task_group = spec["group"]
            config.environment.instruction_source = "environment"
            config.runtime.max_steps = spec["horizon"]
            config.runtime.memory_mode = "frozen"
            config.runtime.experiment_label = "compact-supervised-suite15"
            print(dumps(dict(event="start", **spec)), flush=True)
            start = time.monotonic()
            try:
                result = await run(config, output)
            except Exception as error:
                result = dict(success=False, reason="infrastructure_exception", error=f"{type(error).__name__}: {error}")
            record = dict(**spec, path=str(output.relative_to(root)), result=result,
                          wall_seconds=time.monotonic() - start)
            records.append(record)
            (root / "summary.json").write_text(dumps(records))
            print(dumps(dict(event="finish", **record)), flush=True)
        if args.render and (output / "result.json").is_file():
            async with renderers:
                try:
                    await asyncio.to_thread(render_episode, output, root / "report" / name)
                    record["report"] = f"report/{name}/index.html"
                except Exception as error:
                    record["render_error"] = f"{type(error).__name__}: {error}"
                (root / "summary.json").write_text(dumps(records))
    await asyncio.gather(*(case(spec) for spec in cases))
    print(dumps(dict(event="suite_complete", episodes=len(records),
                     successful=sum(r["result"].get("success", False) for r in records))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/suite15.yaml")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.seed < 0:
        parser.error("Require positive workers and a nonnegative seed")
    asyncio.run(main(args))
