"""Compare renderer/RPC/recording time on fixed actions, without model calls."""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import numpy as np

from vla_agent.adapters.robocasa import RoboCasaEnvironment
from vla_agent.config import load_config
from vla_agent.data import Recorder


async def main(config, root):
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    for backend in ('egl', 'osmesa'):
        cfg = config.model_copy(deep=True)
        cfg.environment.renderer = backend
        cfg.environment.rpc_timeout_seconds = 90
        out = root / backend
        log = Recorder(out, cfg)
        env = RoboCasaEnvironment(cfg.environment, out)
        result = dict(renderer=backend, steps=0,
            purpose='20 fixed zero-action steps; no VLA/Astra; not a task success experiment')
        try:
            start = time.monotonic()
            obs = await env.reset()
            result['reset_seconds'] = time.monotonic() - start
            log.observe(obs)
            rpc, record = [], []
            for _ in range(20):
                start = time.monotonic()
                obs = await env.step(np.zeros(12, dtype=np.float64))
                rpc.append(time.monotonic() - start)
                start = time.monotonic()
                log.observe(obs)
                record.append(time.monotonic() - start)
            result.update(steps=20, rpc_seconds=rpc, record_seconds=record,
                          rpc_mean=statistics.mean(rpc), record_mean=statistics.mean(record))
        except Exception as error:
            result['error'] = f'{type(error).__name__}: {error}'
        finally:
            await env.close()
            log.close()
        rows.append(result)
        (root / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(main(load_config(args.config), args.output.resolve()))
