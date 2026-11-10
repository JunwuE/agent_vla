"""Recovery process protocol. The host launches this inside the configured OS sandbox."""

import contextlib
import hashlib
import json
import os
import resource
import sys
import traceback
from pathlib import Path


class Stage:
    def __init__(self, ctx, stage_id):
        self.ctx, self.stage_id = ctx, stage_id
        self.measured = {}

    def __enter__(self):
        self.ctx.stage_stack.append(self.stage_id)
        return self

    def record(self, **measured):
        self.measured.update(measured)

    def __exit__(self, error_type, error, trace):
        if not self.ctx.stage_stack or self.ctx.stage_stack[-1] != self.stage_id:
            raise RuntimeError('Stage contexts must exit in nested order')
        try:
            result = self.ctx.request(
                "stage_end",
                stage_id=self.stage_id,
                error=str(error) if error is not None else None,
                measured=self.measured,
            )
        finally:
            self.ctx.stage_stack.pop()
        if error is None and not result["ok"]:
            raise RuntimeError(result["stage"]["error"])
        return False


class Context:
    def __init__(self, output):
        self.output = output
        self.stage_stack = []

    def request(self, op, **args):
        self.output.write(json.dumps(dict(op=op, args=args), allow_nan=False) + "\n")
        self.output.flush()
        line = sys.stdin.readline()
        if not line:
            raise RuntimeError("Recovery authority revoked")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result["result"]

    def observe(self):
        return self.request("observe")

    def call(self, tool, **arguments):
        """RAX-style supplied-context entry; all operations still use this attempt's broker."""
        return self.request(tool, **arguments)

    def read_observation(self, camera, step=None):
        """Read copied RGB / metric depth arrays; no robot/simulator access is exposed."""
        import numpy as np

        result = self.request("read_observation", camera=camera, step=step)
        path = Path(result["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != result["sha256"]:
            raise ValueError("Observation artifact checksum mismatch")
        with np.load(path, allow_pickle=False) as arrays:
            return dict(metadata=result["metadata"], **{key: arrays[key] for key in arrays.files})

    @property
    def parameters(self):
        return self.request("parameters")

    @property
    def completion_condition(self):
        return self.request("completion_condition")

    def capabilities(self):
        return self.request("capabilities")

    def move_eef(self, target, steps=1):
        return self.request("move_eef", target=target, steps=steps)

    def move_relative(self, delta_position, gripper_closed, delta_rotation_vector=(0, 0, 0), steps=1):
        return self.request(
            "move_relative",
            delta_position=delta_position,
            delta_rotation_vector=delta_rotation_vector,
            gripper_closed=gripper_closed,
            steps=steps,
        )

    def set_gripper(self, closed, steps=1):
        return self.request("set_gripper", closed=closed, steps=steps)

    def move_base(self, motion, gripper_closed, steps=1):
        return self.request("move_base", motion=motion, gripper_closed=gripper_closed, steps=steps)

    def stop(self, reason):
        return self.request("stop", reason=reason)

    def act(self, action):
        return self.request("act", action=action)

    def check(self, condition):
        return self.request("check", condition=condition)

    def geometry(self, camera, bbox):
        return self.request("geometry", camera=camera, bbox=bbox)

    def ground_target(self, target, camera, observation_step, bbox, confidence):
        return self.request('ground_target', target=target, camera=camera,
                            observation_step=observation_step, bbox=bbox, confidence=confidence)

    def query_geometry(self, grounding_id):
        return self.request('query_geometry', grounding_id=grounding_id)

    def stage(self, name, preconditions=(), completion_condition=None):
        result = self.request(
            "stage", name=name, preconditions=list(preconditions), completion_condition=completion_condition,
            parent_stage_id=self.stage_stack[-1] if self.stage_stack else None,
        )
        return Stage(self, result["stage_id"])

    def note(self, facts=None, hypotheses=None):
        return self.request("note", facts=facts or [], hypotheses=hypotheses or [])


def run_package(root, ctx, sdk_digest):
    """Load only host-frozen code, with one namespace per pinned dependency."""
    import importlib.util

    manifest = json.loads((root / 'bundle.json').read_text())
    expected = manifest.pop('sha256')
    actual = hashlib.sha256(json.dumps(manifest, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()).hexdigest()
    if actual != expected or manifest['runtime_sha256'] != sdk_digest:
        raise ValueError('Execution bundle or runtime checksum mismatch')
    sys.dont_write_bytecode = True
    for name, item in manifest['packages'].items():
        directory = root / 'packages' / name
        for relative, digest in item['files'].items():
            if hashlib.sha256((directory / relative).read_bytes()).hexdigest() != digest:
                raise ValueError('Frozen dependency file checksum mismatch')
        sys.path.insert(0, str(directory / 'code'))
    source = root / 'packages' / manifest['root'] / 'code/run.py'
    spec = importlib.util.spec_from_file_location('submitted_skill_entry', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main(ctx, **ctx.parameters)


def main():
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
    # Like RAX's runtime, isolate the protocol fd from Python AND native-library prints.
    output = os.fdopen(os.dup(sys.stdout.fileno()), 'w', buffering=1)
    os.dup2(sys.stderr.fileno(), 1)
    try:
        sdk_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if len(sys.argv) > 2:
            import importlib.util

            spec = importlib.util.spec_from_file_location('worker_isolation', sys.argv[2])
            isolation = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(isolation)
            isolation.confine(Path(sys.argv[1]).parent)
        with contextlib.redirect_stdout(sys.stderr):
            path = Path(sys.argv[1])
            if (path.parent / 'bundle.json').exists():
                result = run_package(path.parent, Context(output), sdk_digest)
            else:
                namespace = {"__name__": "recovery_policy"}
                source = path.read_text()
                exec(compile(source, str(path), "exec"), namespace)
                result = namespace["run"](Context(output))
        message = dict(op="result", result=result)
    except Exception as error:
        message = dict(
            op="failure", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()[-8000:]
        )
    output.write(json.dumps(message) + "\n")
    output.flush()


if __name__ == "__main__":
    main()
