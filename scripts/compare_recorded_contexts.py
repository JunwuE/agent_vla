"""Offline full/compact comparison at the last actually checked snapshot of each finished task."""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import httpx
import numpy as np

from vla_agent.config import load_config
from vla_agent.contracts import Binding
from vla_agent.data import dumps, read_events
from vla_agent.decision import CRITERIA, classify
from vla_agent.decision_context import compact_state
from vla_agent.perception import image_url


def main(args):
    config = load_config(args.config)
    report = dict(purpose="Counterfactual input comparison, not intervention accuracy labels",
                  robot_actions_executed=0, astra_calls=0, cases=[])
    root = args.root.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    with httpx.Client(timeout=60) as client:
        for episode in sorted((root / "episodes").iterdir()):
            if not (episode / "result.json").exists():
                continue
            rows = read_events(episode)
            windows = [r for r in rows if r["kind"] == "decision_window"]
            if not windows:
                continue
            window = windows[-1]
            binding = Binding.model_validate(window["binding"])
            sent = next(r for r in rows if r["kind"] == "decision_request"
                        and r["binding"]["request_id"] == binding.request_id)
            full = dict(memory=window["memory"], observations=window["observations"],
                        frame_map=sent["state"]["frame_map"], current_step=binding.observation_step)
            images = []
            for frame in full["frame_map"]:
                path = episode / "observations" / f"{frame['step']:07d}.npz"
                meta = json.loads(path.with_suffix(".json").read_text())
                if (meta["episode_id"] != binding.episode_id or frame["step"] > binding.observation_step
                        or hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]):
                    raise ValueError("Causal observation identity/checksum mismatch")
                with np.load(path, allow_pickle=False) as arrays:
                    images.append(image_url(arrays["rgb/" + frame["camera"]]))
            case = dict(task=episode.name, snapshot_step=binding.observation_step, results={})
            for mode in ("full", "compact"):
                state = full if mode == "full" else compact_state(full)
                if mode == "compact" and state != sent["state"]:
                    raise ValueError("Rebuilt compact context differs from the actual online request")
                # Match Decision3.decide's question, including its original instructions.
                payload = dict(model=config.decision.model, state=dumps(state), images=images,
                               questions=dict(route=dict(type="choice", criteria=copy.deepcopy(CRITERIA),
                                instructions="Classify the execution NOW, using only the supplied history. "
                                "Retreat before regrasp can be progress. Commands are not measured motion. "
                                "A closed gripper does not prove a grasp. "
                                "Only views listed in frame_map are attached; other camera names are metadata. "
                                "Occlusion or missing visual evidence does not establish failure. Select exactly one category.")))
                started = time.monotonic()
                response = client.post(config.decision.endpoint, json=payload)
                response.raise_for_status()
                result = response.json()
                if result["meta"]["revision"] != config.decision.revision:
                    raise ValueError("Unexpected model revision")
                decision = classify(result, config.decision, binding, "offline_context_comparison")
                case["results"][mode] = dict(response=result, route=decision.label,
                                            http_seconds=time.monotonic() - started)
                stem = args.output / (episode.name + "." + mode)
                stem.with_suffix(stem.suffix + ".request.json").write_text(dumps(payload))
                stem.with_suffix(stem.suffix + ".response.json").write_text(dumps(result))
            # Episode outcome is attached only to this analysis, never sent to the model.
            case["retrospective_episode_result"] = json.loads((episode / "result.json").read_text())
            report["cases"].append(case)
            (args.output / "comparison.json").write_text(dumps(report))
            print(dumps(dict(task=case["task"], step=case["snapshot_step"],
                             choices={m: r["response"]["answers"]["route"]["choice"] for m, r in case["results"].items()},
                             routes={m: r["route"] for m, r in case["results"].items()})), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--config", default="configs/suite15.yaml")
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
