"""Optional request-local pi05 randomness for paired experiments, independent of server traffic."""

import hashlib

SAMPLING_CONTRACT = "vla-agent.seed-step.v1"
SEED_KEY = "vla_agent/sampling_seed"


def request_seed(seed, step):
    if type(seed) is not int or not 0 <= seed < 2**32 or type(step) is not int or step < 0:
        raise ValueError("Invalid sampling seed or observation step")
    return int.from_bytes(hashlib.sha256(f"{SAMPLING_CONTRACT}:{seed}:{step}".encode()).digest()[:4], "little")


class SeededPolicy:
    """The local websocket server calls infer synchronously; no shared RNG mutation survives a call."""

    def __init__(self, policy, key_factory):
        self.policy, self.key_factory = policy, key_factory

    @property
    def metadata(self):
        return {**self.policy.metadata, "sampling_contract": SAMPLING_CONTRACT}

    def infer(self, observation):
        packet = dict(observation)
        seed = packet.pop(SEED_KEY, None)
        if seed is None:
            return self.policy.infer(packet)
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Sampling seed must be a uint32 integer")
        previous = self.policy._rng
        try:
            self.policy._rng = self.key_factory(seed)
            result = self.policy.infer(packet)
        finally:
            self.policy._rng = previous
        return {**result, "sampling_seed": seed}
