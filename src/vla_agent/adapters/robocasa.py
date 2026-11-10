"""Single-step subprocess bridge using an explicitly configured simulator interpreter."""

import asyncio
import base64
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..contracts import Observation, Verification


def encode(value):
    if isinstance(value, np.ndarray):
        return dict(
            __array__=True,
            dtype=value.dtype.str,
            shape=value.shape,
            data=base64.b64encode(value.tobytes()).decode(),
        )
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def decode(value):
    if value.get("__array__"):
        dtype = np.dtype(value["dtype"])
        if dtype.hasobject:
            raise ValueError("Object arrays forbidden")
        return np.frombuffer(base64.b64decode(value["data"]), dtype).reshape(value["shape"]).copy()
    return value


class RoboCasaEnvironment:
    verifier_conditions = ["task_success"]

    def __init__(self, config, output):
        if not config.source_path or not config.python:
            raise ValueError("RoboCasa is not configured: set environment.source_path and environment.python")
        if not Path(config.source_path).is_dir() or not Path(config.python).is_file():
            raise ValueError("Configured RoboCasa source/interpreter does not exist")
        self.config, self.output = config, Path(output)
        self.language_catalog = []
        self.verifier_conditions = ['task_success']
        self.subtask_boundary_checks = config.native_subtask_checks
        if config.native_subtask_checks:
            from .subtask_catalog import subtask_catalog

            self.language_catalog = subtask_catalog(config.task_name)
            self.verifier_conditions += sorted({condition for task in self.language_catalog
                for condition in [task.completion_condition, *task.preconditions]} - {'task_success'})
        self.process = None
        self.stderr = None
        self.lock = asyncio.Lock()
        self.cached = None
        self.request_id = 0
        self.last_step_timing = {}

    async def request(self, op, **args):
        async with self.lock:
            self.request_id += 1
            if not self.process or self.process.returncode is not None:
                raise RuntimeError("Simulator worker unavailable; commands are never replayed")
            packet = dict(id=self.request_id, op=op, **args)
            self.process.stdin.write((json.dumps(packet, default=encode) + "\n").encode())
            await self.process.stdin.drain()
            async with asyncio.timeout(self.config.rpc_timeout_seconds):
                line = await self.process.stdout.readline()
            if not line:
                raise RuntimeError("Simulator connection lost; execution may be uncertain")
            result = json.loads(line, object_hook=decode)
            if result.get("id") != self.request_id or "error" in result:
                raise RuntimeError(result.get("error", "Simulator reply identity mismatch"))
            if op == "step":
                self.last_step_timing = result.get("timing", {})
            return result["result"]

    async def reset(self):
        self.stderr = (self.output / "simulator.log").open("wb")
        env = os.environ.copy()
        paths = [str(Path(__file__).resolve().parents[2]), self.config.source_path]
        if self.config.robosuite_path:
            paths.append(self.config.robosuite_path)
        env["PYTHONPATH"] = os.pathsep.join(paths + self.config.extra_python_paths)
        if self.config.library_path:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                [self.config.library_path, env.get("LD_LIBRARY_PATH", "")]
            ).rstrip(os.pathsep)
        env["MUJOCO_GL"] = self.config.renderer
        env["PYOPENGL_PLATFORM"] = self.config.renderer
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.setdefault("OMP_NUM_THREADS", "4")
        self.process = await asyncio.create_subprocess_exec(
            self.config.python,
            "-m", "vla_agent.adapters.robocasa_worker",
            self.config.model_dump_json(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=self.stderr,
            env=env,
            limit=64 << 20,
        )
        self.cached = Observation(**await self.request("reset"))
        return self.cached

    async def observe(self):
        return self.cached

    async def step(self, action):
        self.cached = Observation(
            **await self.request(
                "step", action=action, episode_id=self.cached.episode_id, step=self.cached.step
            )
        )
        return self.cached

    async def verify(self, condition):
        return Verification(
            condition=condition,
            passed=self.cached.success if condition == "task_success" else
                   self.cached.facts.get('subtask_checks', {}).get(condition),
            observation_step=self.cached.step,
            evidence=[
                "RoboCasa native info.success"
                if condition == "task_success"
                else ('Simulator state boolean verifier (not visual inference); current observation only'
                      if condition in self.verifier_conditions else "No verifier installed for this condition")
            ],
        )

    async def stop(self):
        # Independent of the model loop and RPC lock. A killed in-flight step is uncertain.
        if self.process and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 5)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()

    async def close(self):
        await self.stop()
        if self.stderr:
            self.stderr.close()


def observation_wire(obs):
    return asdict(obs)
