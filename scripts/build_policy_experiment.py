"""Build a separate experimental config. Does not promote or change the default policy."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from vla_agent.config import load_config
from vla_agent.data import dumps
from vla_agent.policy_candidates import PolicyPatch, apply_experimental_patch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.candidate.read_text())
    digest = hashlib.sha256(dumps(record).encode()).hexdigest()
    if digest != args.candidate.stem or record["schema"] != "vla_agent.policy_candidate.v1":
        raise ValueError("Invalid candidate identity")
    config = apply_experimental_patch(load_config(args.config), PolicyPatch.model_validate(record["patch"]))
    config.runtime.experiment_label = "candidate:" + digest
    with args.output.open("x") as handle:
        yaml.safe_dump(config.model_dump(), handle, sort_keys=False)
    print(
        "Experimental config written; compare paired development/regression seeds before adoption:",
        args.output,
    )
