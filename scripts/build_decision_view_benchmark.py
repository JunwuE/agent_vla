"""Create paired seven/fifteen-image requests from the same recorded causal baseline windows."""

import argparse
from pathlib import Path

from shadow_decision import baseline_memory

from vla_agent.config import load_config
from vla_agent.data import dumps, read_events
from vla_agent.decision import build_request
from vla_agent.perception import decision_window

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--config", default="configs/close_toaster_dev.yaml")
    parser.add_argument("--steps", type=int, nargs="+", default=[200, 400])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    rows = read_events(args.episode)
    requests = []
    for step in args.steps:
        memory = baseline_memory(args.episode, rows, step, config)
        selected = decision_window(memory, config.decision)
        for kind in ("seven", "fifteen"):
            variant = config.decision.model_copy(
                update=dict(
                    primary_camera="left" if kind == "seven" else None,
                    history_cameras=["left", "right", "wrist"],
                    latest_cameras=["wrist", "right"],
                )
            )
            payload, _ = build_request(variant, selected, memory.summary(), memory.binding(0))
            requests.append(
                dict(
                    id=f"{kind}-step{step}",
                    payload=payload,
                    source_episode=str(args.episode.resolve()),
                    observation_steps=[o.step for o in selected],
                )
            )
    with args.output.open("x") as handle:
        handle.write(dumps(requests))
    print("Saved", len(requests), "paired requests; predictions have no independent labels")
