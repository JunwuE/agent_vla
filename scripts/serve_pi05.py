"""RoboCasa pi05 JAX inference without importing simulator-dependent training configs.

Select the pinned official RoboCasa recipe or the user's OmniSF atomic recipe.
Checkpoint and normalization statistics must be supplied explicitly.
"""

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="JAX checkpoint containing params/")
    parser.add_argument("--norm-stats", type=Path, required=True, help="Matching norm_stats.json")
    parser.add_argument("--normalization", choices=("zscore", "quantile"), default="zscore")
    parser.add_argument("--recipe", choices=("robocasa", "omnisf"), default="robocasa")
    parser.add_argument("--source", type=Path, help="OpenPI/OmniSF checkout; imported only in this process")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--action-horizon", type=int, default=50)
    args = parser.parse_args()
    if not (args.checkpoint / "params").is_dir() or not args.norm_stats.is_file():
        parser.error("Checkpoint params/ and matching norm_stats.json must exist")
    if args.source:
        sys.path[:0] = [str(args.source / "src"), str(args.source / "packages/openpi-client/src")]
    if args.recipe == "omnisf":
        import json

        marker = args.checkpoint / "_CHECKPOINT_METADATA"
        if not marker.is_file() or not json.loads(marker.read_text()).get("commit_timestamp_nsecs"):
            parser.error("OmniSF checkpoint must be fully committed")
        if args.normalization != "quantile":
            parser.error("The robocasa365_target_atomic OmniSF recipe requires quantile normalization")
    import jax.numpy as jnp
    from openpi import transforms as t
    from openpi.models import model as model_lib
    from openpi.models.pi0_config import Pi0Config
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.policies.policy import Policy
    if args.recipe == "omnisf":
        from openpi.policies.robocasa365_policy import RoboCasa365Inputs as RobocasaInputs
        from openpi.policies.robocasa365_policy import RoboCasa365Outputs as RobocasaOutputs
    else:
        from openpi.policies.robocasa_policy import RobocasaInputs, RobocasaOutputs
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.shared.normalize import deserialize_json

    config = Pi0Config(pi05=True, action_horizon=args.action_horizon, max_token_len=200)
    norm = deserialize_json(args.norm_stats.read_text())
    if not {"state", "actions"} <= set(norm):
        raise ValueError("State and actions normalization statistics are required")
    quantile = args.normalization == "quantile"
    model = config.load(model_lib.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16))
    policy = Policy(
        model,
        transforms=[
            RobocasaInputs(config.action_dim, config.model_type),
            t.Normalize(norm, use_quantiles=quantile),
            t.ResizeImages(224, 224),
            t.TokenizePrompt(PaligemmaTokenizer(config.max_token_len), discrete_state_input=True),
            t.PadStatesAndActions(config.action_dim),
        ],
        output_transforms=[t.Unnormalize(norm, use_quantiles=quantile), RobocasaOutputs()],
        metadata=dict(
            model="pi05",
            checkpoint=str(args.checkpoint.resolve()),
            model_source=str(args.source.resolve()) if args.source else None,
            contract="vla-agent.omnisf.mobile12.v1" if args.recipe == "omnisf" else "vla-agent.pi05.v1",
            training_config="robocasa365_target_atomic" if args.recipe == "omnisf" else "pi05_pretrain_human300",
            action_layout="base_first" if args.recipe == "omnisf" else "eef_first",
            state_layout="base_first" if args.recipe == "omnisf" else "eef_first",
            binary_encoding="signed" if args.recipe == "omnisf" else "zero_one",
            image_size=256 if args.recipe == "omnisf" else 224,
            normalization=args.normalization,
            norm_sha256=hashlib.sha256(args.norm_stats.read_bytes()).hexdigest(),
            action_dim=12,
            state_dim=16,
            action_horizon=args.action_horizon,
        ),
    )
    if args.warmup:
        import numpy as np

        size = policy.metadata["image_size"]
        sample = {f"observation/{key}": np.zeros((size, size, 3), dtype=np.uint8)
                  for key in ("image", "right_image", "wrist_image")}
        sample.update({"observation/state": np.zeros(16, dtype=np.float32), "prompt": "Open the drawer."})
        sample["observation/state"][[6, 13]] = 1
        actions = policy.infer(sample)["actions"]
        if actions.shape != (args.action_horizon, 12) or not np.isfinite(actions).all():
            raise RuntimeError("Policy warmup failed")
        logging.info("POLICY_READY shape=%s metadata=%s", actions.shape, policy.metadata)
    import jax

    from vla_agent.sampling import SeededPolicy

    serving_policy = SeededPolicy(policy, jax.random.key)
    WebsocketPolicyServer(serving_policy, host=args.host, port=args.port, metadata=serving_policy.metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
