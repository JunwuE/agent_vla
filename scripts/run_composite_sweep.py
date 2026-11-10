"""Resume a predeclared task sweep with frozen source and private memory."""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import yaml
from run_concurrent_sweep import alive
from run_repeat_sweep import checked_hashes, wait_services
from run_seed_sweep import analyze, write_json


def context_chars(value):
    return len(json.dumps(value, ensure_ascii=False))


def summarize(root, plan):
    rows = [json.loads(p.read_text()) for p in sorted((root / 'analysis').glob('*.json'))]
    groups = {}
    for task in plan['tasks']:
        selected = [r for r in rows if r['task'] == task['task']]
        groups[task['task']] = dict(
            completed=len(selected), requested=len(plan['seeds']),
            successes=sum(bool(r['result'].get('success')) for r in selected),
            mean_seconds=statistics.mean(r['result'].get('wall_seconds', 0) for r in selected)
            if selected else None,
        )
    write_json(root / 'summary.json', dict(completed=len(rows), requested=len(plan['cases']), groups=groups))
    lines = [f"# 机器人任务 {len(plan['seeds'])}-seed 测试", '',
             '所有预定 seed 均保留。细阶段为 Astra 信念，整任务成功以仿真器原生验证为准。', '',
             '| 任务 | seed | 成功 | 结束原因 | 步数 | 分钟 | Astra turns | 代码尝试 |',
             '|---|---:|---|---|---:|---:|---:|---:|']
    for row in rows:
        r = row['result']
        lines.append(f"| {row['task']} | {row['seed']} | {r.get('success', False)} | {r.get('reason')} | "
                     f"{r.get('control_steps', 0)} | {r.get('wall_seconds', 0)/60:.2f} | "
                     f"{r.get('astra_calls', 0)} | {row.get('program_attempts', 0)} |")
    (root / 'results.md').write_text('\n'.join(lines) + '\n')


def collect(root, spec, plan, *, refresh_summary=True):
    episode = root / 'episodes' / spec['key']
    if (episode / 'result.json').exists():
        data = analyze(episode)
        data.pop('native_success', None)  # The reused analyzer's 450 is toaster-specific.
        data.update(task=spec['task'], group=spec['group'], declared_action_budget=spec['horizon'])
        data['within_declared_budget_success'] = (
            data['result']['success'] and data['result']['control_steps'] <= spec['horizon'])
        phase_events, instructions, accepted_phase_memory, accepted_updates = [], [], None, []
        vla_instructions, refinements = Counter(), []
        observations, commands, receipts, context_sizes, stops = [], set(), [], [], []
        flora_statuses = Counter()
        with (episode / 'events.jsonl').open() as stream:
            for line in stream:
                event = json.loads(line)
                if event['kind'] == 'observation':
                    observations.append((event['step'], event['success']))
                if event['kind'] == 'action_command':
                    commands.add(event['id'])
                if event['kind'] == 'action_executed':
                    receipts.append(event['command_id'])
                if event['kind'] == 'stop':
                    stops.append(dict(event_id=event['id'], reason=event['reason']))
                if event['kind'] == 'flora_result':
                    flora_statuses[event['result']['status']] += 1
                if event['kind'] == 'codex_turn_start':
                    context = event['context']
                    context_sizes.append(dict(
                        event_id=event['id'], intent=context['intent'], total_chars=context_chars(context),
                        top_fields={k: context_chars(v) for k, v in context.items()},
                        memory_fields={k: context_chars(v) for k, v in context.get('memory', {}).items()}))
                if event['kind'] in {'phase_plan', 'phase_assessment', 'review_phase_update'}:
                    phase_events.append(event)
                if event['kind'] == 'task_initialized':
                    instructions.append(event['task'])
                if event['kind'] == 'action_command' and event['source'] == 'vla':
                    vla_instructions[event['instruction']] += 1
                if event['kind'] == 'intervention_plan' and event['plan']['kind'] == 'refine':
                    refinements.append(dict(step=event['binding']['observation_step'], plan=event['plan']))
                if event['kind'] == 'memory':
                    current = event['state'].get('phase_memory')
                    if current and current != accepted_phase_memory:
                        accepted_updates.append(dict(event_id=event['id'], phase_memory=current))
                    accepted_phase_memory = current
        data['phase_events'] = phase_events
        data['last_accepted_phase_memory'] = accepted_phase_memory
        data['accepted_phase_updates'] = accepted_updates
        data['native_task'] = instructions
        data['vla_instruction_counts'] = dict(vla_instructions)
        data['refine_proposals'] = refinements
        data['stop_events'] = stops
        data['context_sizes'] = context_sizes
        data['flora_statuses'] = dict(flora_statuses)
        data['native_success_first_step'] = next((step for step, success in observations if success), None)
        data['execution_audit'] = dict(
            contiguous_observations=[s for s, _ in observations] == list(range(data['result']['control_steps'] + 1)),
            all_commands_acknowledged=commands == set(receipts),
            no_duplicate_receipts=len(receipts) == len(set(receipts)),
            receipt_count_matches_result=len(receipts) == data['result']['control_steps'],
            result_matches_native_observation=bool(observations) and observations[-1][1] == data['result']['success'],
            result_matches_any_native_success=any(success for _, success in observations) == data['result']['success'],
            within_declared_action_budget=data['result']['control_steps'] <= spec['horizon'])
    else:
        launch = json.loads((root / 'episodes' / (spec['key'] + '-launch.json')).read_text())
        data = dict(task=spec['task'], group=spec['group'], seed=spec['seed'], program_attempts=0,
                    result=dict(success=False, reason='infrastructure_no_result',
                                wall_seconds=(time.time_ns()-launch['started_ns'])/1e9),
                    evidence=str(episode), note='Retained failure; no replacement seed or silent retry.')
    write_json(root / 'analysis' / (spec['key'] + '.json'), data)
    if refresh_summary:
        summarize(root, plan)
    print(json.dumps(dict(event='completed', key=spec['key'], result=data['result']), ensure_ascii=False), flush=True)


def main(root):
    root = root.resolve()
    plan = json.loads((root / 'plan.json').read_text())
    source = Path(plan['source'])
    hashes = json.loads((source.parent / 'source-hashes.json').read_text())
    initial_hashes = json.loads((root / 'initial-hashes.json').read_text())
    if not checked_hashes(source, hashes) or not checked_hashes(root / 'initial', initial_hashes):
        raise RuntimeError('Frozen source or starting memory changed')
    active, handles = {}, {}
    completed = {p.stem for p in (root / 'analysis').glob('*.json')}
    for spec in plan['cases']:
        if spec['key'] in completed:
            continue
        launch_path = root / 'episodes' / (spec['key'] + '-launch.json')
        if launch_path.exists():
            launch = json.loads(launch_path.read_text())
            if alive(launch['pid']):
                command = (Path('/proc') / str(launch['pid']) / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
                if str(root / 'episodes' / spec['key']) not in command or 'vla_agent.cli' not in command:
                    raise RuntimeError('Worker PID changed; refusing to adopt')
                active[spec['key']] = dict(pid=launch['pid'], process=None, spec=spec,
                                          started_ns=launch['started_ns'])
            else:
                collect(root, spec, plan)
                completed.add(spec['key'])
    remaining = [s for s in plan['cases'] if s['key'] not in completed and s['key'] not in active]
    if active or remaining:
        wait_services(root)
    elif (root / 'batch-end.json').exists():
        summarize(root, plan)
        return  # A completed sweep needs no model services; preserve its original completion time.
    if not (root / 'batch-start.json').exists():
        write_json(root / 'batch-start.json', dict(started_ns=time.time_ns()))
    while active or remaining:
        for key, item in list(active.items()):
            finished = item['process'].poll() is not None if item['process'] else not alive(item['pid'])
            if finished:
                if key in handles:
                    handles.pop(key).close()
                collect(root, item['spec'], plan)
                completed.add(key)
                del active[key]
        admission_directory = plan['config']['planner'].get('admission_directory')
        admission_state = Path(admission_directory) / 'state.json' if admission_directory else None
        quota_blocked = bool(admission_state and admission_state.exists()
                             and json.loads(admission_state.read_text()).get('quota_blocked'))
        if quota_blocked and remaining and not active:
            value = dict(state='blocked_provider_quota', completed=sorted(completed), active=[],
                         remaining=len(remaining), updated_ns=time.time_ns(),
                         note='No new episodes launched; confirm refreshed provider quota before resuming.')
            write_json(root / 'status.json', value)
            write_json(root / 'batch-blocked.json', value)
            return
        while remaining and len(active) < plan['workers'] and not quota_blocked:
            spec = remaining.pop(0)
            key = spec['key']
            episode, inputs = root / 'episodes' / key, root / 'inputs' / key
            if episode.exists() or inputs.exists():
                raise RuntimeError(f'Untracked output {key}; refusing overwrite')
            shutil.copytree(root / 'initial', inputs)
            if not checked_hashes(inputs, initial_hashes):
                raise RuntimeError('Worker starting memory does not match')
            config = json.loads(json.dumps(plan['config']))
            config['environment'].update(task_name=spec['task'], task_group=spec['group'], seed=spec['seed'])
            config['vla']['sampling_seed'] = 42 + spec['seed']
            config['runtime'].update(max_steps=spec['horizon'],
                                      skill_library=str(inputs / 'skills'),
                                      experience_library=str(inputs / 'experiences'),
                                      task_memory_library=str(inputs / 'tasks'))
            config_path = root / 'episodes' / (key + '.yaml')
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            env = os.environ.copy()
            env.update(PYTHONPATH=str(source), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
            log = (root / 'episodes' / (key + '.log')).open('wb')
            command = [sys.executable, '-m', 'vla_agent.cli', 'run', '--config', str(config_path),
                       '--output', str(episode)]
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            started_ns = time.time_ns()
            active[key] = dict(pid=process.pid, process=process, spec=spec, started_ns=started_ns)
            handles[key] = log
            write_json(root / 'episodes' / (key + '-launch.json'), dict(
                pid=process.pid, started_ns=started_ns, command=command, source=str(source),
                spec=spec, initial_memory_sha256=plan['initial_memory_sha256']))
            print(json.dumps(dict(event='start', **spec)), flush=True)
        write_json(root / 'status.json', dict(
            state='running' if active else 'completed', completed=sorted(completed),
            active=[dict(key=k, pid=v['pid'], started_ns=v['started_ns']) for k, v in active.items()],
            remaining=len(remaining), updated_ns=time.time_ns()))
        if active:
            time.sleep(2)
    if not checked_hashes(source, hashes) or not checked_hashes(root / 'initial', initial_hashes):
        raise RuntimeError('Frozen source/memory changed during batch')
    write_json(root / 'batch-end.json', dict(completed=sorted(completed), finished_ns=time.time_ns(),
                                           pinned_source_and_initial_memory_unchanged=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
