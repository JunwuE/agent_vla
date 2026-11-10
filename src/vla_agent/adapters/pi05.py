import asyncio
import time

import numpy as np

from ..contracts import Chunk, validate_actions
from ..sampling import SAMPLING_CONTRACT, SEED_KEY, request_seed


def state_for_policy(state, layout):
    if layout == "eef_first":
        return state.copy()
    if layout == "base_first":
        return np.r_[state[7:14], state[:7], state[14:16]]
    raise ValueError("Unknown state layout")


def canonical_actions(actions, layout, binary_encoding="zero_one", clip=False):
    actions = np.asarray(actions, dtype=float)
    if actions.ndim != 2 or actions.shape[1] != 12 or not np.isfinite(actions).all():
        raise ValueError("Expected policy output with shape (N, 12)")
    if layout == "base_first":
        actions = np.concatenate([actions[:, 5:12], actions[:, :5]], axis=1)
    elif layout != "eef_first":
        raise ValueError("Unknown action layout")
    actions = actions.copy()
    if binary_encoding == "signed":
        actions[:, [6, 11]] = (actions[:, [6, 11]] + 1) / 2
    elif binary_encoding != "zero_one":
        raise ValueError("Unknown binary action encoding")
    if clip:
        actions = np.clip(actions, -1, 1)
        actions[:, [6, 11]] = np.clip(actions[:, [6, 11]], 0, 1)
    return validate_actions(actions)


def policy_input(obs, instruction, state_layout, image_size=224):
    from openpi_client import image_tools

    packet = {"prompt": instruction, "observation/state": state_for_policy(obs.state, state_layout)}
    for camera, key in (("left", "image"), ("right", "right_image"), ("wrist", "wrist_image")):
        packet[f"observation/{key}"] = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(np.ascontiguousarray(obs.rgb[camera]), image_size, image_size)
        )
    return packet


class Pi05:
    def __init__(self, config, recorder):
        self.config, self.log = config, recorder
        self.socket = None
        self.lock = asyncio.Lock()

    async def infer(self, obs, instruction, binding):
        from openpi_client import msgpack_numpy
        from websockets.asyncio.client import connect

        async with self.lock:
            started = time.monotonic()
            self.log.event("vla_inference_start", binding=binding, instruction=instruction)
            try:
                async with asyncio.timeout(self.config.timeout_seconds):
                    if self.socket is None:
                        self.socket = await connect(
                            self.config.endpoint, compression=None, max_size=64 << 20, ping_interval=None
                        )
                        metadata = msgpack_numpy.unpackb(await self.socket.recv())
                        for name in ("state_layout", "action_layout"):
                            if name in metadata and metadata[name] != getattr(self.config, name):
                                raise ValueError(f"Policy service {name} mismatch")
                        if self.config.expected_contract and metadata.get("contract") != self.config.expected_contract:
                            raise ValueError("Policy service contract mismatch")
                        if self.config.sampling_seed is not None and metadata.get("sampling_contract") != SAMPLING_CONTRACT:
                            raise ValueError("Policy service does not support deterministic sampling")
                        for name in ("binary_encoding", "normalization", "image_size"):
                            if name in metadata and metadata[name] != getattr(self.config, name):
                                raise ValueError(f"Policy service {name} mismatch")
                        if self.config.checkpoint and metadata.get("checkpoint") != self.config.checkpoint:
                            raise ValueError("Policy service checkpoint mismatch")
                        if self.config.norm_stats:
                            import hashlib
                            from pathlib import Path

                            expected_hash = hashlib.sha256(Path(self.config.norm_stats).read_bytes()).hexdigest()
                            if metadata.get("norm_sha256") != expected_hash:
                                raise ValueError("Policy service normalization statistics mismatch")
                        self.log.event(
                            "vla_service",
                            endpoint=self.config.endpoint,
                            metadata=metadata,
                            declared_config=self.config.training_config,
                            checkpoint_verified=bool(self.config.checkpoint),
                        )
                    packet = policy_input(obs, instruction, self.config.state_layout, self.config.image_size)
                    seed = None
                    if self.config.sampling_seed is not None:
                        seed = request_seed(self.config.sampling_seed, obs.step)
                        packet[SEED_KEY] = seed
                    await self.socket.send(msgpack_numpy.packb(packet))
                    raw = await self.socket.recv()
                    if isinstance(raw, str):
                        raise RuntimeError(f"pi05 service error: {raw[:500]}")
                    result = msgpack_numpy.unpackb(raw)
                    if seed is not None and result.get("sampling_seed") != seed:
                        raise ValueError("Policy response did not honor the sampling seed")
                    actions = canonical_actions(
                        result["actions"], self.config.action_layout,
                        self.config.binary_encoding, self.config.clip_actions,
                    )
                    self.log.event(
                        "policy_action_conversion", binding=binding, raw_actions=result["actions"],
                        canonical_actions=actions, layout=self.config.action_layout,
                        binary_encoding=self.config.binary_encoding, clip_enabled=self.config.clip_actions,
                    )
                    self.log.event("vla_inference_end", binding=binding, seconds=time.monotonic() - started,
                                   returned_steps=len(actions), error=None, sampling_seed=seed)
                    return Chunk(binding, actions)
            except BaseException as error:
                self.log.event("vla_inference_end", binding=binding, seconds=time.monotonic() - started,
                               error=type(error).__name__)
                await self.close()
                raise

    async def close(self):
        if self.socket:
            await self.socket.close()
            self.socket = None
