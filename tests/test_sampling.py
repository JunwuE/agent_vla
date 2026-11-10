import pytest

from vla_agent.sampling import SEED_KEY, SeededPolicy, request_seed


def test_request_rng_is_order_independent_and_does_not_change_legacy_stream():
    class Policy:
        _rng = 10
        metadata = {}

        def infer(self, packet):
            assert SEED_KEY not in packet
            self._rng += 1
            if packet.get("fail"):
                raise RuntimeError("fixture")
            return {"sample": self._rng}

    p = Policy()
    wrapper = SeededPolicy(p, lambda seed: seed)
    seed = request_seed(42, 50)
    a = wrapper.infer({SEED_KEY: seed})
    wrapper.infer({SEED_KEY: request_seed(44, 90)})
    assert wrapper.infer({SEED_KEY: seed}) == a
    assert p._rng == 10
    assert wrapper.infer({})["sample"] == 11
    with pytest.raises(RuntimeError):
        wrapper.infer({SEED_KEY: seed, "fail": True})
    assert p._rng == 11
    with pytest.raises(ValueError):
        wrapper.infer({SEED_KEY: True})
    assert request_seed(42, 50) != request_seed(42, 55)
