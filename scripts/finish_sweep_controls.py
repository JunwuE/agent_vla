"""Finish selected matched controls with bounded concurrency and live-worker adoption."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml
from run_concurrent_sweep import alive
from run_seed_sweep import write_json


def read(path):
    return json.loads(path.read_text())


def targets_for(root):
    selected = set()
    for episode in (root/'episodes').glob('seed[0-9][0-9]'):
        if episode.is_dir() and any(episode.glob('recovery/*/policy.py')):
            selected.add(int(episode.name[4:]))
    for path in (root/'analysis').glob('seed*.json'):
        record = read(path)
        if record.get('physical_intervention_proposals'):
            selected.add(record['seed'])
    return sorted(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--workers', type=int, default=2, choices=(1,2,3))
    args = parser.parse_args()
    root = args.root.resolve()
    previous = read(root/'controls/index.json')
    completed = {int(k):v for k,v in previous['completed'].items()}
    active = {}
    while True:
        targets = targets_for(root)
        for seed,item in list(active.items()):
            finished = item['process'].poll() is not None if item['process'] else not alive(item['pid'])
            if finished:
                if item.get('log'):
                    item['log'].close()
                path = root/'controls'/f'seed{seed:02d}'
                if not (path/'result.json').exists():
                    raise RuntimeError(f'Control {seed} exited without result; evidence retained')
                completed[seed] = str(path)
                del active[seed]
                print(json.dumps(dict(event='finished',seed=seed,result=read(path/'result.json'))),flush=True)
        # Adopt only a live process whose command line names this exact output.
        # This also handles the old scheduler stopping while its child continues.
        for seed in targets:
            path = root/'controls'/f'seed{seed:02d}'
            if (path/'result.json').exists() and seed not in active:
                completed[seed] = str(path)
            if seed in completed or seed in active or not path.exists():
                continue
            launch = read(path.with_name(path.name+'-launch.json'))
            pid = launch['pid']
            cmd = (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if alive(pid) else []
            if str(path).encode() not in cmd or b'vla_agent.cli' not in cmd:
                raise RuntimeError(f'Cannot safely adopt unfinished control {seed}: pid {pid}')
            active[seed] = dict(pid=pid,process=None)
            print(json.dumps(dict(event='adopt',seed=seed,pid=pid)),flush=True)
        pending = [s for s in targets if s not in completed and s not in active]
        while pending and len(active) < args.workers:
            seed = pending.pop(0)
            path = root/'controls'/f'seed{seed:02d}'
            if path.exists():
                raise RuntimeError(f'Refusing to overwrite {path}')
            config = yaml.safe_load((root/'episodes'/f'seed{seed:02d}.yaml').read_text())
            config['runtime'].update(supervision_enabled=False,initial_planning=False,memory_mode='frozen',
                                     experiment_label='matched-vla-selected-interventions')
            config['flora']['enabled'] = True
            source = read(root/'episodes'/f'seed{seed:02d}-launch.json')['source']
            config_path = path.with_suffix('.yaml')
            config_path.write_text(yaml.safe_dump(config,sort_keys=False))
            env = os.environ.copy()
            env.update(PYTHONPATH=source,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
            log = path.with_suffix('.log').open('wb')
            process = subprocess.Popen([sys.executable,'-m','vla_agent.cli','run','--config',str(config_path),
                                        '--output',str(path)],env=env,stdout=log,stderr=subprocess.STDOUT)
            active[seed] = dict(pid=process.pid,process=process,log=log)
            write_json(path.with_name(path.name+'-launch.json'),dict(pid=process.pid,source=source,
                started_ns=time.time_ns(),control_concurrency_limit=args.workers))
            print(json.dumps(dict(event='start',seed=seed,pid=process.pid)),flush=True)
        done = read(root/'status.json')['state']=='completed' and not pending and not active
        write_json(root/'controls/index.json',dict(state='completed' if done else 'running',targets=targets,
            completed=completed,pending=pending,active_seeds=sorted(active),workers=args.workers,
            note='Controls selected after code handoffs or read-only pauses, not a full baseline SR estimate. '
                 'Diagnostic concurrency increased as main workers finished; see launch/transition records.'))
        if done:
            break
        time.sleep(5)


if __name__ == '__main__':
    main()
