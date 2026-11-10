"""Sequential, fully retained development trials. Extended-budget success is reported separately."""

import argparse
import asyncio
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

from vla_agent.cli import run
from vla_agent.config import load_config
from vla_agent.data import read_events


async def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    base = load_config(args.config)
    if base.environment.task_name != "CloseToasterOvenDoor":
        raise ValueError("This development runner is scoped to CloseToasterOvenDoor")
    files = sorted(Path("src").rglob("*.py"))
    (args.output / "source-hashes.json").write_text(json.dumps({
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
    }, indent=2))
    (args.output / "plan.json").write_text(json.dumps(dict(
        seeds=args.seeds, native_horizon=450, purpose="development, not held-out SR",
        config=base.model_dump(), policy_rng=("request seed from policy_seed+simulation_seed and observation step"
                                             if args.policy_seed is not None else "shared pi05 server sampling stream; not bitwise paired"),
    ), indent=2))
    records = []
    for seed in args.seeds:
        config = base.model_copy(deep=True)
        config.environment.seed = seed
        if args.policy_seed is not None:
            config.vla.sampling_seed = (args.policy_seed + seed) % (2**32)
        output = args.output / f"seed{seed}"
        (args.output / f"seed{seed}.yaml").write_text(yaml.safe_dump(config.model_dump(), sort_keys=False))
        print(json.dumps(dict(event="start", seed=seed, path=str(output))), flush=True)
        result = await run(config, output)
        events = read_events(output)
        recoveries = [r for r in events if r["kind"] == "recovery_result"]
        record = dict(seed=seed, path=str(output), **result,
                      success_within_native_horizon=result["success"] and result["control_steps"] <= 450,
                      recovery_attempts=len(recoveries),
                      verified_recoveries=sum(bool(r["verified"]) for r in recoveries),
                      decision_labels=dict(Counter(r["decision"]["label"] for r in events if r["kind"] == "decision")),
                      programs=[{k: r.get(k) for k in ("program_version", "sha256", "result", "error", "executed_steps")} for r in recoveries])
        records.append(record)
        (args.output / "summary.json").write_text(json.dumps(records, indent=2))
        print(json.dumps(dict(event="finish", **record)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 7])
    p.add_argument("--policy-seed", type=int, default=None)
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        p.error("Use distinct nonnegative seeds")
    asyncio.run(main(args))
