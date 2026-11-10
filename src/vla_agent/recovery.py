import asyncio
import hashlib
import json
import os
import signal
import time
from pathlib import Path

import numpy as np

from .contracts import Source
from .control import (
    EEFRequest,
    GripperRequest,
    MobileRequest,
    OSCControl,
    RelativeRequest,
    execute_target,
    robot_capabilities,
)
from .data import dumps
from .grounding import GroundTarget, ground_target, grounded_geometry
from .perception import local_geometry
from .sandbox import recovery_command
from .skill_packages import export_bundle
from .stages import StageTracker


class RecoveryRunner:
    def __init__(self, harness, memory, config, recorder):
        self.harness, self.memory, self.config, self.log = harness, memory, config, recorder

    async def run(
        self,
        code,
        condition,
        intervention_id,
        *,
        parameters=None,
        skill=None,
        remaining_steps=None,
        remaining_seconds=None,
        program_version=None,
        package_bundle=None,
    ):
        if not code or len(code.encode()) > 64 * 1024:
            raise ValueError("Recovery source must be 1..64KiB")
        if package_bundle and package_bundle['packages'][package_bundle['manifest']['root']]['files']['code/run.py'] != code:
            raise ValueError('Execution bundle entry does not match the submitted source')
        directory = self.log.root / "recovery" / intervention_id
        directory.mkdir(parents=True, exist_ok=False)
        path = directory / "policy.py"
        path.write_text(code)
        if package_bundle:
            export_bundle(directory, package_bundle)
        digest = hashlib.sha256(code.encode()).hexdigest()
        self.memory.current_code = dict(
            sha256=digest, intervention_id=intervention_id, path=str(path.relative_to(self.log.root)),
            program_version=program_version,
            package_sha256=(package_bundle['packages'][package_bundle['manifest']['root']]['package']['sha256']
                            if package_bundle else None),
            bundle_sha256=package_bundle['sha256'] if package_bundle else None,
        )
        lease = await self.harness.claim(Source.CODE)
        self.log.event(
            "code_version",
            intervention_id=intervention_id,
            sha256=digest,
            path=str(path.relative_to(self.log.root)),
            completion_condition=condition,
            parameters=parameters or {},
            program_version=program_version,
            package_sha256=(package_bundle['packages'][package_bundle['manifest']['root']]['package']['sha256']
                            if package_bundle else None),
            bundle_sha256=package_bundle['sha256'] if package_bundle else None,
            dependency_lock=package_bundle['manifest']['packages'] if package_bundle else None,
            skill=({k: skill[k] for k in ("id", "version", "status", "sha256")} if skill else None),
        )
        worker = Path(__file__).with_name("recovery_worker.py")
        # No credentials inherited. The program communicates with its capability broker over stdio.
        env = {k: os.environ[k] for k in ("PATH", "LANG", "LD_LIBRARY_PATH") if k in os.environ}
        env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        stderr = (directory / "stderr.log").open("wb")
        process = None
        steps = 0
        attempted_steps = 0
        result = "UNRESOLVED"
        error = None
        traceback_text = None
        started = time.monotonic()
        step_budget = (
            min(self.config.recovery_steps, remaining_steps)
            if remaining_steps is not None
            else self.config.recovery_steps
        )
        time_budget = (
            min(self.config.recovery_seconds, remaining_seconds)
            if remaining_seconds is not None
            else self.config.recovery_seconds
        )
        stages = StageTracker(self.harness, intervention_id)
        raw_bytes = 0
        raw_paths = {}
        program_notes = []
        program_result = None
        exit_before_result = None

        async def execute(action):
            nonlocal attempted_steps, steps
            if steps >= step_budget:
                raise RuntimeError("Recovery action budget exhausted")
            attempted_steps += 1
            result = await self.harness.execute(action, lease, self.memory, intervention_id)
            steps += 1
            return result

        try:
            if step_budget <= 0 or time_budget <= 0:
                raise RuntimeError("Recovery session budget exhausted")
            compile(code, str(path), "exec")
            command = recovery_command(worker, path, directory, self.config.recovery_sandbox)
            self.log.event('recovery_isolation', mode=self.config.recovery_sandbox, intervention_id=intervention_id)
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=directory,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=stderr,
                start_new_session=True,
                limit=1 << 20,
            )
            async with asyncio.timeout(time_budget):
                for _ in range(self.config.recovery_tool_calls):
                    line = await process.stdout.readline()
                    if not line:
                        exit_before_result = await process.wait()
                        raise RuntimeError(f"Recovery worker exited without a result (returncode={exit_before_result})")
                    message = json.loads(line)
                    self.harness.check(lease)
                    op, args = message["op"], message.get("args", {})
                    tool_event = self.log.event(
                        "recovery_tool_request",
                        intervention_id=intervention_id,
                        operation=op,
                        arguments=args,
                        observation_step=self.harness.obs.step,
                    )
                    if op == "result":
                        result = message["result"]
                        if package_bundle:
                            if (not isinstance(result, dict) or type(result.get('ok')) is not bool
                                    or not isinstance(result.get('measured'), dict) or not isinstance(result.get('why'), str)):
                                raise ValueError('Package main must return {ok: bool, measured: object, why: str}')
                            dumps(result)
                            program_result = result
                            if not result['ok']:
                                error = result['why'] or 'Skill returned ok=false'
                            result = 'SUBTASK_DONE' if result['ok'] else 'UNRESOLVED'
                        if result not in {"RECOVERED", "SUBTASK_DONE", "UNRESOLVED"}:
                            raise ValueError("Unknown recovery result")
                        break
                    if op == "failure":
                        error = str(message.get("error", "Program failed"))
                        traceback_text = str(message.get("traceback", ""))[-8000:]
                        result = "UNRESOLVED"
                        break
                    if op == "observe":
                        value = (await self.harness.env.observe()).summary()
                    elif op == "read_observation":
                        camera = args["camera"]
                        step = self.harness.obs.step if args.get("step") is None else args["step"]
                        self.log.observe(self.harness.obs)
                        observation = self.log.read_observation(
                            step, self.memory.episode_id, self.harness.obs.step
                        )
                        if camera not in observation.rgb:
                            raise ValueError("Unknown observation camera")
                        arrays = dict(rgb=observation.rgb[camera], state=observation.state)
                        if camera in observation.depth_m:
                            arrays["depth_m"] = observation.depth_m[camera]
                        raw_dir = directory / "observations"
                        raw_dir.mkdir(exist_ok=True)
                        raw_path = (
                            raw_dir / f"{step:07d}-{hashlib.sha256(camera.encode()).hexdigest()[:12]}.npz"
                        )
                        if raw_path not in raw_paths:
                            raw_bytes += sum(array.nbytes for array in arrays.values())
                            if raw_bytes > self.config.recovery_observation_bytes:
                                raise ValueError("Recovery observation byte budget exhausted")
                            np.savez_compressed(raw_path, **arrays)
                            raw_paths[raw_path] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
                        if hashlib.sha256(raw_path.read_bytes()).hexdigest() != raw_paths[raw_path]:
                            raise ValueError("Exported observation artifact was modified")
                        metadata = dict(
                            **observation.summary(),
                            camera=camera,
                            calibration=observation.calibration.get(camera),
                            depth_units="meters" if "depth_m" in arrays else None,
                        )
                        raw_path.with_suffix(".json").write_text(dumps(metadata))
                        value = dict(
                            path=str(raw_path.resolve()),
                            sha256=raw_paths[raw_path],
                            metadata=metadata,
                        )
                    elif op == "act":
                        value = (await execute(args["action"])).summary()
                    elif op == "parameters":
                        value = parameters or {}
                    elif op == "completion_condition":
                        value = condition
                    elif op == "capabilities":
                        value = robot_capabilities(self.harness)
                    elif op in {"move_eef", "move_relative", "set_gripper", "move_base"}:
                        control = OSCControl(await self.harness.env.observe())
                        if op == "move_eef":
                            request = EEFRequest.model_validate(args)
                        elif op == "move_relative":
                            relative = RelativeRequest.model_validate(args)
                            request = EEFRequest(
                                target=control.relative_target(
                                    relative.delta_position,
                                    relative.delta_rotation_vector,
                                    relative.gripper_closed,
                                ),
                                steps=relative.steps,
                            )
                        elif op == "set_gripper":
                            gripper = GripperRequest.model_validate(args)
                            request = EEFRequest(
                                target=control.relative_target([0] * 3, [0] * 3, gripper.closed),
                                steps=gripper.steps,
                            )
                        else:
                            mobile = MobileRequest.model_validate(args)
                            target = control.relative_target([0] * 3, [0] * 3, mobile.gripper_closed)
                            target.control_mode, target.base_motion = "base", mobile.motion
                            request = EEFRequest(target=target, steps=mobile.steps)
                        value = await execute_target(
                            self.harness, self.memory, lease, request, intervention_id, step_executor=execute
                        )
                    elif op == "check":
                        value = await self.harness.verify(args["condition"])
                    elif op == "geometry":
                        value = local_geometry(await self.harness.env.observe(), **args)
                    elif op == 'ground_target':
                        value = ground_target(await self.harness.env.observe(), GroundTarget.model_validate(args),
                                              'code_selection')
                        self.memory.groundings[value['grounding_id']] = value
                        while len(self.memory.groundings) > 32:
                            self.memory.groundings.pop(next(iter(self.memory.groundings)))
                    elif op == 'query_geometry':
                        value = grounded_geometry(await self.harness.env.observe(),
                                                  self.memory.groundings[args['grounding_id']])
                    elif op == "stage":
                        value = await stages.start(args)
                    elif op == "stage_end":
                        value = await stages.finish(**args)
                    elif op == "note":
                        event_id = self.log.event("recovery_" + op, intervention_id=intervention_id, **args)
                        program_notes.append(dict(source="program_report", observation_step=self.harness.obs.step,
                                                  event_id=event_id, **args))
                        self.memory.notes.append(
                            dict(
                                event_id=event_id,
                                source="code_policy",
                                observation_step=self.harness.obs.step,
                                intervention_id=intervention_id,
                                **args,
                            )
                        )
                        value = {"recorded": True}
                    elif op == "stop":
                        await self.harness.stop(str(args["reason"]))
                        result, error = "UNRESOLVED", "Recovery requested stop: " + str(args["reason"])
                        break
                    else:
                        raise ValueError(f"Unsupported recovery operation: {op}")
                    self.log.event(
                        "recovery_tool_result",
                        request_event_id=tool_event,
                        intervention_id=intervention_id,
                        operation=op,
                        result=value,
                    )
                    process.stdin.write((dumps({"result": value}) + "\n").encode())
                    await process.stdin.drain()
                else:
                    raise RuntimeError("Recovery tool budget exhausted")
        except Exception as exc:
            result, error = "UNRESOLVED", f"{type(exc).__name__}: {exc}"
        finally:
            # Fence the worker before cleanup, including cancellation of this coroutine.
            self.harness.epoch += 1
            self.harness.queue.clear()
            if process and process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()
            stderr.close()
            stages.close_all(
                error
                or ("Program returned UNRESOLVED or was interrupted" if result == "UNRESOLVED" else None)
            )
        verified = False
        proof = None
        if result != "UNRESOLVED":
            proof = await self.harness.verify(condition)
            verified = proof.passed is True
            if not verified:
                result = "UNRESOLVED"
                error = "Final completion condition not verified"
        report = dict(
            result=result,
            verified=verified,
            evidence=proof,
            error=error,
            executed_steps=steps,
            attempted_steps=attempted_steps,
            sha256=digest,
            program_version=program_version,
            program_result=program_result,
            package_sha256=(package_bundle['packages'][package_bundle['manifest']['root']]['package']['sha256']
                            if package_bundle else None),
            bundle_sha256=package_bundle['sha256'] if package_bundle else None,
            stages=stages.rows,
            failed_stage=stages.failed_stage or ("final_verification" if proof and not verified else None),
            traceback=traceback_text,
            worker_exit_before_result=exit_before_result,
            program_notes=program_notes[-12:],
            seconds=time.monotonic() - started,
            artifacts=dict(
                source=str(path.relative_to(self.log.root)),
                stderr=str((directory / "stderr.log").relative_to(self.log.root)),
                bundle=str((directory / 'bundle.json').relative_to(self.log.root)) if package_bundle else None,
            ),
        )
        self.log.event("recovery_result", intervention_id=intervention_id, **report)
        return report
