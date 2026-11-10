import json

import numpy as np
import pytest
from openpi_client import msgpack_numpy
from test_models import Log, binding
from test_runtime import config
from websockets.asyncio.server import serve

from vla_agent.adapters.mock import MockEnvironment
from vla_agent.adapters.pi05 import Pi05
from vla_agent.cli import run
from vla_agent.config import VLAConfig
from vla_agent.data import export_decision_windows, read_events


@pytest.mark.parametrize('level', [1, 6, 9])
def test_rgbd_storage_is_numpy_compatible_and_bit_exact(tmp_path, level):
    from vla_agent.data import save_observation_arrays

    rng = np.random.default_rng(42)
    depth = rng.uniform(.001, 10, (32, 32)).astype(np.float64)
    depth[0, :4] = [np.nan, np.inf, -0., np.nextafter(1., 2.)]
    arrays = {'state': rng.normal(size=16), 'rgb/left': rng.integers(0, 256, (32, 32, 3), dtype=np.uint8),
              'depth_m/left': depth, 'depth_m/wrist': depth.astype(np.float32)}
    path = tmp_path / 'observation.npz'
    save_observation_arrays(path, arrays, level)
    with np.load(path, allow_pickle=False) as restored:
        assert set(restored.files) == set(arrays)
        for key, array in arrays.items():
            assert restored[key].dtype == array.dtype
            assert restored[key].shape == array.shape
            assert restored[key].tobytes() == array.tobytes()


async def test_pi05_websocket_contract():
    received = []

    async def handler(socket):
        await socket.send(msgpack_numpy.packb(dict(state_layout="eef_first", action_layout="eef_first")))
        packet = msgpack_numpy.unpackb(await socket.recv())
        received.append(packet)
        await socket.send(msgpack_numpy.packb({"actions": np.zeros((50, 12))}))

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = Pi05(VLAConfig(endpoint=f"ws://127.0.0.1:{port}"), Log())
        obs = await MockEnvironment().reset()
        chunk = await client.infer(obs, "new instruction", binding())
        assert chunk.actions.shape == (50, 12)
        assert received[0]["prompt"] == "new instruction"
        assert received[0]["observation/right_image"].shape == (224, 224, 3)
        np.testing.assert_array_equal(received[0]["observation/state"], obs.state)
        await client.close()


async def test_pi05_rejects_wrong_layout_before_inference():
    async def handler(socket):
        await socket.send(msgpack_numpy.packb(dict(action_layout="base_first")))
        await socket.wait_closed()

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = Pi05(VLAConfig(endpoint=f"ws://127.0.0.1:{port}"), Log())
        with pytest.raises(ValueError, match="mismatch"):
            await client.infer(await MockEnvironment().reset(), "test", binding())
        assert client.socket is None


async def test_decision_export_contains_no_future_observations(tmp_path):
    root = tmp_path / "run"
    await run(config(), root)
    output = tmp_path / "decisions.jsonl"
    count = export_decision_windows(root, output, keep_audit_rate=1)
    assert count > 0
    for row in map(json.loads, output.read_text().splitlines()):
        assert all(o["step"] <= row["binding"]["observation_step"] for o in row["observations"])
        assert row["online_teacher_label"] is None and row["hindsight_outcome"] is None
        assert row["review_requested"]
    events = read_events(root)
    assert all(r["id"] == i + 1 for i, r in enumerate(events))
