"""Two bounded simulation steps through the real adapter and isolated Code Policy worker."""

import argparse
import asyncio
from pathlib import Path

from vla_agent.adapters.robocasa import RoboCasaEnvironment
from vla_agent.config import load_config
from vla_agent.data import Recorder, dumps
from vla_agent.harness import Harness
from vla_agent.memory import WorkingMemory
from vla_agent.recovery import RecoveryRunner

CODE = """def run(ctx):
    observation = ctx.observe()
    raw = ctx.read_observation('left')
    assert raw['depth_m'].ndim == 2
    target = ctx.ground_target('visible surface patch for geometry smoke test', 'left',
        observation['step'], [120,120,136,136], 0.5)
    geometry = ctx.query_geometry(target['grounding_id'])
    assert geometry['reliable'] and geometry['frame'] == 'osc_arm_base'
    with ctx.stage('bounded motion smoke test'):
        ctx.move_relative([0.001,0,0], False, steps=2)
    ctx.note(facts=['real RGB-D query and bounded command completed'], hypotheses=[])
    return 'UNRESOLVED'
"""


async def run(args):
    config = load_config(args.config)
    config.runtime.recovery_steps = 2
    log = Recorder(args.output.resolve(), config)
    env = RoboCasaEnvironment(config.environment, log.root)
    harness = Harness(env, log, 2)
    try:
        await harness.start()
        memory = WorkingMemory(config.task, harness.obs.episode_id)
        memory.add(harness.obs)
        report = await RecoveryRunner(harness, memory, config.runtime, log).run(CODE, "task_success", "smoke")
        assert report["error"] is None and report["executed_steps"] == 2, report
        assert not report["verified"]
        log.finish(dict(report=report, note="Interface smoke test, not task success evaluation"))
        print(dumps(report))
    finally:
        await harness.stop("smoke complete")
        await env.close()
        log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/close_toaster_dev.yaml")
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
