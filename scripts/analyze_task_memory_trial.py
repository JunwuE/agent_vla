"""Analyze a pinned trial and compare its VLA prefix with a saved paired baseline."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from run_seed_sweep import analyze

from vla_agent.data import read_events


def compare_prefix(episode, baseline):
    a, b = read_events(episode), read_events(baseline)
    first_a = next(r for r in a if r['kind'] == 'observation')
    first_b = next(r for r in b if r['kind'] == 'observation')
    with np.load(episode / first_a['path']) as x, np.load(baseline / first_b['path']) as y:
        same_initial = set(x.files) == set(y.files) and all(np.array_equal(x[k], y[k]) for k in x.files)
    prefix = []
    for event in a:
        if event['kind'] == 'action_executed':
            if event['source'] != 'vla':
                break
            prefix.append(event)
    actions_b = [r for r in b if r['kind'] == 'action_executed']
    count = min(len(prefix), len(actions_b))
    differences = [i + 1 for i, (x, y) in enumerate(zip(prefix[:count], actions_b[:count], strict=True))
                   if x['action'] != y['action'] or x['measured_state'] != y['measured_state']]
    cfg_a = json.loads((episode / 'manifest.json').read_text())['config']
    cfg_b = json.loads((baseline / 'manifest.json').read_text())['config']
    fields = dict(environment=['task_name', 'seed', 'scene_split', 'camera_size', 'ignore_native_horizon'],
                  vla=['checkpoint', 'norm_stats', 'sampling_seed', 'action_layout', 'state_layout', 'normalization'],
                  runtime=['max_steps', 'chunk_steps'])
    contract = {f'{section}.{key}': cfg_a[section][key] == cfg_b[section][key]
                for section, keys in fields.items() for key in keys}
    return dict(baseline=str(baseline), initial_arrays_identical=same_initial, matching_config=contract,
                compared_steps=count, vla_prefix_steps=len(prefix), first_divergence_step=next(iter(differences), None),
                commands_and_states_identical=not differences and count == len(prefix))


def main(args):
    data = analyze(args.episode)
    rows = read_events(args.episode)
    task_memory = next((r for r in rows if r['kind'] == 'task_memory_snapshot'), None)
    data['task_memory'] = dict(version=task_memory['memory'].get('version'), scope=task_memory['task_scope']) if task_memory else None
    data['package_executions'] = [dict(intervention_id=r['intervention_id'], program_version=r.get('program_version'),
        package_sha256=r.get('package_sha256'), bundle_sha256=r.get('bundle_sha256'), skill=r.get('skill'))
        for r in rows if r['kind'] == 'code_version']
    starts = {r['call_id']: r for r in rows if r['kind'] == 'codex_tool_start'}
    durations, errors, output_chars = defaultdict(list), Counter(), Counter()
    for r in rows:
        if r['kind'] != 'codex_tool':
            continue
        if r['call_id'] in starts:
            durations[r['name']].append((r['time_ns'] - starts[r['call_id']]['time_ns']) / 1e9)
        output_chars[r['name']] += len(json.dumps(r.get('result'), ensure_ascii=False))
        if isinstance(r.get('result'), dict) and r['result'].get('error'):
            errors[r['name']] += 1
    data['tool_timings'] = {name: dict(count=len(values), seconds=sum(values), max_seconds=max(values),
                                     errors=errors[name], output_chars=output_chars[name]) for name, values in durations.items()}
    data['source_edit_calls'] = sum(data['tool_counts'].get(k, 0) for k in (
        'write_program', 'patch_program', 'patch_program_batch', 'write_skill_package', 'patch_skill_package'))
    data['skill_invocations'] = [r for r in rows if r['kind'] == 'skill_invocation']
    data['phase_plan_events'] = [r for r in rows if r['kind'] == 'phase_plan']
    if args.baseline:
        data['paired_prefix'] = compare_prefix(args.episode, args.baseline)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'analysis.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    comparison = []
    for name, episode in [('historical_agent', args.previous), ('new_agent', args.episode)]:
        if episode is None:
            continue
        r = data if episode == args.episode else analyze(episode)
        comparison.append(dict(name=name, path=str(episode), success=r['result']['success'],
            steps=r['result']['control_steps'], wall_seconds=r['result']['wall_seconds'],
            source_steps=r['source_steps'], astra_calls=r['result']['astra_calls'],
            attempts=r['program_attempts'], versions=r['generated_versions'], direct_calls=r['direct_skill_calls'],
            zero_action_attempts=r['zero_action_attempts'], wall_components=r['wall_components_seconds'],
            source_edit_calls=sum(r['tool_counts'].get(k, 0) for k in (
                'write_program', 'patch_program', 'patch_program_batch', 'write_skill_package', 'patch_skill_package'))))
    (args.output / 'comparison.json').write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
