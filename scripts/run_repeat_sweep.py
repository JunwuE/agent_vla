"""Repeat a seed sweep with one frozen runtime and identical initial memory snapshots."""

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import yaml
from run_concurrent_sweep import alive
from run_seed_sweep import analyze, episode_report, write_json


def checked_hashes(directory, hashes):
    return all((directory / name).is_file()
               and hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
               for name, digest in hashes.items())


def wait_services(root):
    services = json.loads((root / 'services/processes.json').read_text())
    started = time.monotonic()
    while True:
        ready = {}
        for item, port, marker in zip(services, [8010, 8030], ['POLICY_READY', 'FLORA_READY'], strict=True):
            if not alive(item['pid']):
                raise RuntimeError('Service exited during startup: ' + item['component'])
            content = (root / 'services' / (item['component'] + '.log')).read_text(errors='replace')
            try:
                with socket.create_connection(('127.0.0.1', port), timeout=.5):
                    listening = True
            except OSError:
                listening = False
            ready[item['component']] = marker in content and listening
        write_json(root / 'startup-status.json', dict(ready=ready, updated_ns=time.time_ns()))
        if all(ready.values()):
            return
        if time.monotonic() - started > 900:
            raise RuntimeError('Model startup deadline; no new episodes started')
        time.sleep(2)


def summarize(root, plan):
    rows = []
    for path in sorted((root / 'analysis').glob('seed*.json')):
        record = json.loads(path.read_text())
        result = record['result']
        rows.append(dict(seed=record['seed'], success=result['success'], steps=result['control_steps'],
                         wall_seconds=result['wall_seconds'], astra_turns=result['astra_calls'],
                         attempts=record['program_attempts'], source_edit_calls=record['source_edit_calls'],
                         direct_calls=record['direct_skill_calls'], verified_direct_calls=record['verified_direct_calls'],
                         code_steps=record['source_steps'].get('code_policy', 0),
                         native_450_success=record['native_success']))
    write_json(root / 'summary.json', dict(completed=len(rows), requested=len(plan['seeds']),
               successes=sum(r['success'] for r in rows), within_450=sum(r['native_450_success'] for r in rows), rows=rows))


def collect(root, seed, profile, plan):
    episode = root / 'episodes' / f'seed{seed:02d}'
    if not (episode / 'result.json').exists():
        raise RuntimeError(f'Seed {seed} exited without result; preserve partial evidence before resuming')
    data = analyze(episode)
    # Stream rather than retaining a second copy of image-heavy planner contexts.
    invocations, outputs, packages = [], [], []
    with (episode / 'events.jsonl').open() as stream:
        for line in stream:
            event = json.loads(line)
            if event['kind'] == 'skill_invocation':
                invocations.append(event)
            elif event['kind'] == 'recovery_result':
                outputs.append(event)
            elif event['kind'] == 'code_version' and event.get('package_sha256'):
                packages.append({k: event.get(k) for k in
                                 ('intervention_id', 'program_version', 'package_sha256', 'bundle_sha256')})
    data['release'] = profile['release']
    data['sweep_concurrency_limit'] = plan['workers']
    data['initial_memory_policy'] = plan['memory_policy']
    data['source_edit_calls'] = sum(data['tool_counts'].get(name, 0) for name in (
        'write_program', 'patch_program', 'patch_program_batch', 'write_skill_package', 'patch_skill_package', 'edit_program'))
    data['verified_direct_calls'] = sum(
        next((r['verified'] for r in outputs if r['id'] > invocation['id']), False) for invocation in invocations)
    data['package_executions'] = packages
    episode_report(root, data)
    summarize(root, plan)
    print(json.dumps(dict(event='completed', seed=seed, success=data['result']['success'],
                          seconds=data['result']['wall_seconds'], attempts=data['program_attempts'],
                          edits=data['source_edit_calls'], direct=data['direct_skill_calls']), ensure_ascii=False), flush=True)


def main(root):
    root = root.resolve()
    plan = json.loads((root / 'plan.json').read_text())
    profile = plan['source_profile']
    source = Path(profile['source'])
    hashes = json.loads((source.parent / 'source-hashes.json').read_text())
    initial_hashes = json.loads((root / 'initial-hashes.json').read_text())
    if not checked_hashes(source, hashes) or not checked_hashes(root / 'initial', initial_hashes):
        raise RuntimeError('Frozen source or initial memory changed')
    memory_digest = hashlib.sha256(json.dumps(initial_hashes, sort_keys=True).encode()).hexdigest()
    (root / 'inputs').mkdir(exist_ok=True)
    completed = {int(p.stem[4:]) for p in (root / 'analysis').glob('seed*.json')}
    active, handles = {}, {}
    # A controller restart adopts exact existing worker commands and never replaces results.
    for seed in plan['seeds']:
        if seed in completed:
            continue
        episode = root / 'episodes' / f'seed{seed:02d}'
        launch_path = root / 'episodes' / f'seed{seed:02d}-launch.json'
        if launch_path.exists():
            launch = json.loads(launch_path.read_text())
            if alive(launch['pid']):
                command = (Path('/proc') / str(launch['pid']) / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
                if str(episode) not in command or 'vla_agent.cli' not in command:
                    raise RuntimeError('Cannot adopt a reused PID')
                active[seed] = dict(pid=launch['pid'], process=None, started_ns=launch['started_ns'])
            else:
                collect(root, seed, profile, plan)
                completed.add(seed)
        elif episode.exists():
            raise RuntimeError(f'Untracked episode {seed}; refusing overwrite')
    remaining = [seed for seed in plan['seeds'] if seed not in completed and seed not in active]
    wait_services(root)
    ready_ns = time.time_ns()
    if not (root / 'batch-start.json').exists():
        services = json.loads((root / 'services/processes.json').read_text())
        write_json(root / 'batch-start.json', dict(started_ns=ready_ns,
                   cold_start_seconds=(ready_ns - min(s['started_ns'] for s in services)) / 1e9))
    config = yaml.safe_load(Path(profile['config']).read_text())
    while active or remaining:
        for seed, item in list(active.items()):
            finished = item['process'].poll() is not None if item['process'] else not alive(item['pid'])
            if not finished:
                continue
            if seed in handles:
                handles.pop(seed).close()
            collect(root, seed, profile, plan)
            completed.add(seed)
            del active[seed]
        while remaining and len(active) < plan['workers']:
            seed = remaining.pop(0)
            episode = root / 'episodes' / f'seed{seed:02d}'
            inputs = root / 'inputs' / f'seed{seed:02d}'
            if episode.exists() or inputs.exists():
                raise RuntimeError(f'Untracked output for {seed}; refusing overwrite')
            shutil.copytree(root / 'initial', inputs)
            if not checked_hashes(inputs, initial_hashes):
                raise RuntimeError('Worker input snapshot differs from initial memory')
            current = json.loads(json.dumps(config))
            current['environment']['seed'] = seed
            current['vla']['sampling_seed'] = 42 + seed
            current['runtime'].update(skill_library=str(inputs / 'skills'),
                                      experience_library=str(inputs / 'experiences'),
                                      task_memory_library=str(inputs / 'tasks'))
            config_path = root / 'episodes' / f'seed{seed:02d}.yaml'
            config_path.write_text(yaml.safe_dump(current, sort_keys=False))
            env = os.environ.copy()
            env.update(PYTHONPATH=profile['source'], OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
            log = (root / 'episodes' / f'seed{seed:02d}.log').open('wb')
            command = [sys.executable, '-m', 'vla_agent.cli', 'run', '--config', str(config_path), '--output', str(episode)]
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            started = time.time_ns()
            handles[seed] = log
            active[seed] = dict(pid=process.pid, process=process, started_ns=started)
            write_json(root / 'episodes' / f'seed{seed:02d}-launch.json', dict(
                **profile, pid=process.pid, started_ns=started, command=command,
                completed_before_launch=sorted(completed), concurrency_limit=plan['workers'],
                initial_memory_sha256=memory_digest, identical_initial_inputs=True))
            print(json.dumps(dict(event='start', seed=seed, release=profile['release'])), flush=True)
        write_json(root / 'status.json', dict(state='running' if active else 'completed',
            completed=sorted(completed), concurrency_limit=plan['workers'],
            active=[dict(seed=seed, pid=item['pid'], started_ns=item['started_ns'], release=profile['release'])
                    for seed, item in active.items()], updated_ns=time.time_ns()))
        if active:
            time.sleep(2)
    if not checked_hashes(source, hashes) or not checked_hashes(root / 'initial', initial_hashes):
        raise RuntimeError('Frozen source or initial memory changed during sweep')
    write_json(root / 'batch-end.json', dict(finished_ns=time.time_ns(), completed=sorted(completed),
                                           pinned_source_and_initial_memory_unchanged=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
