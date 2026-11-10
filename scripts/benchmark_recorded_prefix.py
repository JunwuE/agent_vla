"""Replay recorded commands with live simulation to time lossless recording changes.

No model calls, new planning, or fresh success-rate claims. Every stored observation
is compared with the recorded reference, including metric depth and all cameras.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np

from vla_agent.adapters.robocasa import RoboCasaEnvironment
from vla_agent.config import load_config
from vla_agent.contracts import Task
from vla_agent.data import Recorder
from vla_agent.harness import Harness
from vla_agent.memory import WorkingMemory


async def main(args):
    source = args.episode.resolve()
    cfg = load_config(args.config)
    cfg.environment.seed = args.seed
    cfg.runtime.observation_compression_level = args.level
    rows = [json.loads(line) for line in (source / 'events.jsonl').open()]
    actions = [r for r in rows if r['kind'] == 'action_executed'][:args.steps]
    if len(actions) != args.steps or any(r['source'] != 'vla' for r in actions):
        raise ValueError('Expected a complete VLA prefix')
    log = Recorder(args.output, cfg)
    env = RoboCasaEnvironment(cfg.environment, args.output)
    harness = Harness(env, log, step_limit=args.steps)
    result = dict(success=False, reason='recorded_prefix', control_steps=0, astra_calls=0,
                  flora_calls=0, astra_tokens={}, model_calls=0, evaluation_kind='timing_replay_only',
                  compression_level=args.level, error=None)
    started = time.perf_counter()
    reference_read_seconds = 0.
    try:
        lease = await harness.start()
        task = Task(instruction=harness.obs.facts['task_instruction'], subtasks=[dict(
            name='task', instruction=harness.obs.facts['task_instruction'], completion_condition='task_success')])
        memory = WorkingMemory(task, harness.obs.episode_id)
        memory.add(harness.obs)
        prefix_started = time.perf_counter()
        for step in range(args.steps + 1):
            check_started = time.perf_counter()
            with np.load(source / 'observations' / f'{step:07d}.npz', allow_pickle=False) as reference:
                current = {'state': harness.obs.state,
                           **{'rgb/' + k: v for k, v in harness.obs.rgb.items()},
                           **{'depth_m/' + k: v for k, v in harness.obs.depth_m.items()}}
                if set(current) != set(reference.files) or not all(
                    np.array_equal(a, reference[k], equal_nan=True) for k, a in current.items()
                ):
                    raise RuntimeError(f'Observation differs at step {step}')
            reference_read_seconds += time.perf_counter() - check_started
            if step < args.steps:
                await harness.execute(np.asarray(actions[step]['action']), lease, memory)
        result.update(all_observations_identical=True,
                      prefix_seconds=time.perf_counter() - prefix_started,
                      reference_read_seconds=reference_read_seconds,
                      success=harness.obs.success, control_steps=harness.obs.step)
    except Exception as error:
        result.update(reason='error', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        result['wall_seconds'] = time.perf_counter() - started
        await env.close()
        log.finish(result)
        log.close()
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--level', type=int, choices=(1, 6), required=True)
    asyncio.run(main(parser.parse_args()))
