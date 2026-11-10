"""Replay causal baseline windows through Decision; never treat predictions as labels."""

import argparse
import asyncio
import json
import time
from collections import Counter
from dataclasses import fields
from pathlib import Path

import numpy as np

from vla_agent.config import load_config
from vla_agent.contracts import Observation, Task
from vla_agent.data import dumps, read_events
from vla_agent.decision import Decision3, checked_decision
from vla_agent.memory import WorkingMemory
from vla_agent.perception import decision_window
from vla_agent.supervision_rules import low_motion_signals


class Log:
    def __init__(self, path):
        self.file = path.open("x", buffering=1)
        self.sequence = 0

    def event(self, kind, **kwargs):
        self.sequence += 1
        self.file.write(dumps(dict(id=self.sequence, time_ns=time.time_ns(), kind=kind, **kwargs)) + "\n")


def observation(root, row):
    meta = json.loads((root / row["path"]).with_suffix(".json").read_text())
    with np.load(root / row["path"], allow_pickle=False) as arrays:
        params = {f.name: meta[f.name] for f in fields(Observation) if f.name in meta}
        params.update(state=arrays["state"], rgb={k[4:]: arrays[k] for k in arrays.files if k.startswith("rgb/")},
                      depth_m={k[8:]: arrays[k] for k in arrays.files if k.startswith("depth_m/")})
        return Observation(**params)


def baseline_memory(root, rows, end, config):
    if any(r['kind'] in {'intervention_plan', 'attempt_interrupted'} for r in rows):
        raise ValueError('Use uninterrupted baselines; intervention boundaries need separate reconstruction')
    tasks = [r['task'] for r in rows if r['kind'] == 'task_initialized']
    if len(tasks) != 1 or len(tasks[0]['subtasks']) != 1:
        raise ValueError('Expected one initialized baseline task')
    obs = {r['step']: r for r in rows if r['kind'] == 'observation'}
    if end not in obs:
        raise ValueError('Requested endpoint was not recorded')
    memory = WorkingMemory(Task.model_validate(tasks[0]), obs[end]['episode_id'])
    for step in range(max(0, end-319), end+1):
        memory.add(observation(root, obs[step]))
    times = {o.step: o.sim_time for o in memory.observations}
    for r in rows:
        if r['kind'] == 'action_executed':
            step = int(Path(r['after']).stem)
            if step in times:
                memory.action_history.append(dict(step=step, sim_time=times[step], source=r['source'],
                    command=r['action'], measured_state=r['measured_state']))
    memory.rule_signals = low_motion_signals(memory, config.runtime)
    return memory


async def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    c = load_config(args.config)
    log = Log(output / "events.jsonl")
    backend = Decision3(c.decision, log)
    predictions = []
    try:
        with (output / "predictions.jsonl").open("x", buffering=1) as stream:
            for root in sorted(args.baseline.glob("*")):
                if not (root / "result.json").exists():
                    continue
                rows = read_events(root)
                obs = {r["step"]: r for r in rows if r["kind"] == "observation"}
                last = max(obs)
                ends = sorted(set(np.linspace(min(20, last), last, args.windows_per_episode, dtype=int)))
                result = json.loads((root / "result.json").read_text())
                manifest = json.loads((root / "manifest.json").read_text())
                for index, end in enumerate(ends):
                    memory = baseline_memory(root, rows, int(end), c)
                    window = decision_window(memory, c.decision)
                    chosen = [o.step for o in window]
                    started = time.monotonic()
                    decision = await checked_decision(backend, c.decision, window, memory.summary(),
                                                      memory.binding(0), "offline_shadow", log)
                    value = dict(id=f"{root.name}:{int(end)}", source_episode=str(root.resolve()),
                                 task=manifest["config"]["environment"]["task_name"],
                                 seed=manifest["config"]["environment"]["seed"],
                                 observation_steps=chosen, window_end=int(end), prediction=decision,
                                 seconds=time.monotonic() - started, independent_label=None,
                                 independent_evidence=None,
                                 retrospective_only=dict(episode_success=result["success"],
                                                         note="Not sent to Decision; never use as an online label"))
                    stream.write(dumps(value) + "\n")
                    predictions.append(value)
                print(json.dumps(dict(episode=root.name, windows=len(ends))), flush=True)
    finally:
        await backend.close()
        log.file.close()
    summary = dict(purpose="offline causal shadow predictions; independent labels still required",
                   count=len(predictions), labels=dict(Counter(p["prediction"].label for p in predictions)),
                   mean_seconds=float(np.mean([p["seconds"] for p in predictions])),
                   thresholds=c.decision.model_dump(), calibrated=False, false_positive_rate=None,
                   false_negative_rate=None, reason="No independent human-reviewed ground truth yet")
    (output / "summary.json").write_text(dumps(summary))
    print(dumps(summary), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--config", default="configs/robocasa.yaml")
    p.add_argument("--windows-per-episode", type=int, default=20)
    args = p.parse_args()
    if args.windows_per_episode < 1:
        p.error("windows-per-episode must be positive")
    asyncio.run(main(args))
