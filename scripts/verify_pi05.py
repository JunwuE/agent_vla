"""Check the inference environment without a simulator or pretrained weights."""

import argparse
import importlib.metadata
import json
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-smoke", action="store_true", help="Full random pi05, short horizon; not task evaluation"
    )
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch
    from openpi.models.pi0_config import Pi0Config
    from openpi.models.tokenizer import PaligemmaTokenizer  # noqa: F401
    from openpi.policies.policy import Policy  # noqa: F401
    from openpi.policies.robocasa_policy import RobocasaInputs, RobocasaOutputs
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer  # noqa: F401

    report = dict(
        versions={
            name: importlib.metadata.version(name)
            for name in (
                "jax",
                "jaxlib",
                "flax",
                "torch",
                "transformers",
                "numpy",
                "lerobot",
                "openpi",
                "openpi-client",
            )
        },
        jax_devices=[str(x) for x in jax.devices()],
        torch_cuda=torch.cuda.is_available(),
        pretrained_weights_loaded=False,
        robocasa_installed=False,
    )
    if not any(x.platform == "gpu" for x in jax.devices()):
        raise RuntimeError("JAX GPU unavailable")
    report["jax_matmul"] = float(jax.jit(lambda x: x @ x)(jnp.ones((8, 8))).block_until_ready()[0, 0])
    report["torch_matmul"] = float(
        (torch.ones((8, 8), device="cuda") @ torch.ones((8, 8), device="cuda"))[0, 0]
    )
    c = Pi0Config(pi05=True)
    packet = {"observation/state": np.zeros(16), "prompt": "pick up the cup"}
    packet.update(
        {
            "observation/" + k: np.zeros((224, 224, 3), np.uint8)
            for k in ("image", "right_image", "wrist_image")
        }
    )
    inputs = RobocasaInputs(c.action_dim, c.model_type)(packet)
    assert inputs["state"].shape == (32,) and len(inputs["image"]) == 3
    assert RobocasaOutputs()({"actions": np.zeros((50, 32))})["actions"].shape == (50, 12)
    report["robocasa_transforms"] = "three RGB cameras, state16->32, action32->12"
    if args.model_smoke:
        print("Initializing full random pi05 on GPU; horizon=2, token_length=16", flush=True)
        started = time.monotonic()
        c = Pi0Config(pi05=True, action_horizon=2, max_token_len=16)
        model = c.create(jax.random.key(0))
        print("Model initialized; sampling one denoising step", flush=True)
        actions = model.sample_actions(jax.random.key(1), c.fake_obs(), num_steps=1)
        actions = np.asarray(actions.block_until_ready())
        assert actions.shape == (1, 2, 32) and np.isfinite(actions).all()
        report["random_model_smoke"] = dict(
            shape=list(actions.shape),
            finite=True,
            seconds=time.monotonic() - started,
            denoising_steps=1,
            trained=False,
        )
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
