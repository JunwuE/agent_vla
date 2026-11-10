"""Replay causal numeric trigger candidates on retained VLA prefixes; never run models or actions."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vla_agent.config import Config, load_config
from vla_agent.memory import WorkingMemory
from vla_agent.supervision_rules import low_motion_signals


def replay(run, runtime):
    saved = Config.model_validate(json.loads((run / 'manifest.json').read_text())['config'])
    memory, first_time = None, None
    last_trigger_time = float('-inf')
    matches, triggers = 0, []
    for line in (run / 'events.jsonl').open():
        event = json.loads(line)
        if event['kind'] == 'observation':
            if memory is None:
                memory = WorkingMemory(saved.task, event['episode_id'])
                first_time = event['time_ns']
            memory.add(SimpleNamespace(episode_id=event['episode_id'], step=event['step'],
                                      sim_time=event['sim_time'], state=np.asarray(event['state']),
                                      terminal=event['terminal']))
        elif event['kind'] == 'memory' and memory is not None:
            state = event['state']
            memory.phases.started_step = state.get('phase_memory', {}).get('started_step', 0)
            memory.attempt_started_step = state.get('attempt', {}).get('started_step', 0)
        elif event['kind'] == 'action_executed':
            if event['source'] != 'vla':
                break  # The question concerns the uninterrupted first VLA prefix.
            obs = memory.observations[-1]
            memory.action_history.append(dict(step=obs.step, sim_time=obs.sim_time, source='vla',
                                              command=event['action'], measured_state=event['measured_state']))
            if obs.terminal:
                continue
            signals = low_motion_signals(memory, runtime)
            if not signals:
                continue
            matches += 1
            if obs.sim_time - last_trigger_time >= runtime.stagnation_cooldown_seconds:
                last_trigger_time = obs.sim_time
                triggers.append(dict(step=obs.step, wall_seconds=round((event['time_ns'] - first_time) / 1e9, 2),
                                     signal=signals[0]))
    return dict(run=str(run), matching_windows=matches, candidates_after_rule_cooldown=triggers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/close_toaster_explore.yaml')
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    old = config.runtime.model_copy(update=dict(stagnation_scope='whole_robot', stagnation_seconds=4.0,
                                                stagnation_position_m=0.002, stagnation_rotation_rad=0.02,
                                                stagnation_gripper_m=0.001))
    profiles = {'old_whole_robot': old, 'current_eef_translation': config.runtime}
    records = []
    for run in args.runs:
        for name, runtime in profiles.items():
            row = dict(profile=name, **replay(run, runtime))
            records.append(row)
            print(json.dumps(dict(profile=name, run=run.name + '/' + run.parent.name,
                                  matches=row['matching_windows'],
                                  candidate_steps=[x['step'] for x in row['candidates_after_rule_cooldown']]),
                             ensure_ascii=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        scope='Causal numeric evidence replay only; no model or simulator execution. Candidate triggers '
              'are not confirmed failures or counterfactual Astra calls. Pending reviews may coalesce them.',
        profiles={name: {k: v for k, v in runtime.model_dump().items() if k.startswith('stagnation_')}
                  for name, runtime in profiles.items()}, results=records), ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
