"""One diagnostic baseline worker for seeds where code actually took ownership."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml
from run_seed_sweep import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    prior = root.parent/'toaster-cross-seed-20261030'
    known = {5:prior/'round11-paired-vla/seed5', 6:prior/'round11-paired-vla/seed6',
             8:prior/'round11-paired-vla/seed8', 9:prior/'round14-paired-vla/seed9',
             10:prior/'round14-paired-vla/seed10'}
    known = {seed:path for seed,path in known.items() if (path/'result.json').exists()}
    completed = {}
    active = None
    handle = None
    while True:
        if active and active['process'].poll() is not None:
            handle.close()
            if not (active['path']/'result.json').exists():
                raise RuntimeError(f'Control seed {active["seed"]} exited without a result; logs retained')
            completed[active['seed']] = str(active['path'])
            print(json.dumps(dict(event='finished', seed=active['seed'],
                 result=json.loads((active['path']/'result.json').read_text()))), flush=True)
            active = None
        targets = []
        for episode in sorted((root/'episodes').glob('seed[0-9][0-9]')):
            if not episode.is_dir() or not any(episode.glob('recovery/*/policy.py')):
                continue
            seed = int(episode.name[4:])
            targets.append(seed)
            own = root/'controls'/f'seed{seed:02d}'
            if (own/'result.json').exists():
                completed[seed] = str(own)
            elif seed in known:
                baseline_config = json.loads((known[seed]/'manifest.json').read_text())['config']
                agent_config = json.loads((episode/'manifest.json').read_text())['config']
                for key in ('checkpoint', 'norm_stats', 'sampling_seed', 'action_layout', 'state_layout'):
                    if baseline_config['vla'][key] != agent_config['vla'][key]:
                        raise ValueError(f'Prior baseline contract mismatch: seed {seed}, {key}')
                assert baseline_config['runtime']['max_steps'] == agent_config['runtime']['max_steps']
                completed[seed] = str(known[seed])
        pending = [seed for seed in targets if seed not in completed and (not active or seed != active['seed'])]
        if active is None and pending:
            seed = pending[0]
            path = root/'controls'/f'seed{seed:02d}'
            if path.exists():
                raise RuntimeError(f'Existing unfinished control {path}; refusing overwrite')
            config = yaml.safe_load((root/'episodes'/f'seed{seed:02d}.yaml').read_text())
            config['runtime']['supervision_enabled'] = False
            config['runtime']['initial_planning'] = False
            config['runtime']['memory_mode'] = 'frozen'
            config['runtime']['experiment_label'] = 'handoff-diagnostic-matched-vla'
            # The Flora decision constructor validates this even in VLA-only mode.
            # supervision_enabled=False prevents all monitor/planner calls, as in the proven baseline profile.
            config['flora']['enabled'] = True
            launch = json.loads((root/'episodes'/f'seed{seed:02d}-launch.json').read_text())
            config_path = root/'controls'/f'seed{seed:02d}.yaml'
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            env = os.environ.copy()
            env['PYTHONPATH'] = launch['source']
            env['OMP_NUM_THREADS'] = '4'
            env['OPENBLAS_NUM_THREADS'] = '1'
            handle = (root/'controls'/f'seed{seed:02d}.log').open('wb')
            process = subprocess.Popen([sys.executable, '-m', 'vla_agent.cli', 'run', '--config',str(config_path),
                                        '--output',str(path)],env=env,stdout=handle,stderr=subprocess.STDOUT)
            active = dict(seed=seed, path=path, process=process)
            write_json(root/'controls'/f'seed{seed:02d}-launch.json',dict(pid=process.pid, source=launch['source']))
            print(json.dumps(dict(event='start', seed=seed)),flush=True)
        sweep = json.loads((root/'status.json').read_text())
        done = sweep['state']=='completed' and not pending and active is None
        write_json(root/'controls/index.json', dict(state='completed' if done else 'running',
            targets=targets, completed=completed, pending=pending,
            active_seed=active['seed'] if active else None,
            note='Matched baselines only for code handoffs; selection is not an estimate of full baseline SR.'))
        if done:
            break
        time.sleep(5)


if __name__ == '__main__':
    main()
