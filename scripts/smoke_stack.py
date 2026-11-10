"""Bounded integration check with real model services and simulation, not a benchmark."""

import argparse
import asyncio
import json
from pathlib import Path

from vla_agent.cli import run
from vla_agent.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/robocasa.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.steps <= 100:
        parser.error("Smoke steps must be between 1 and 100")
    config = load_config(args.config)
    if ((config.environment.kind, config.vla.kind, config.planner.kind) != ('robocasa', 'pi05', 'codex')
            or config.decision.kind not in {'decision3', 'flora'}):
        parser.error("This check requires real RoboCasa, pi05, Decision3/Flora and Codex backends")
    config.runtime.max_steps = args.steps
    config.runtime.max_wall_seconds = 600
    config.planner.max_calls = 4
    result = asyncio.run(run(config, args.output.resolve()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(1 if result["error"] else 0)


if __name__ == "__main__":
    main()
