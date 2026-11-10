"""Verify atomic evaluation protocol, native outcomes, action receipts and frozen artifacts."""

import argparse
import json
from pathlib import Path

import yaml
from run_repeat_sweep import checked_hashes


def audit(root, output=None):
    plan = json.loads((root / 'plan.json').read_text())
    end = json.loads((root / 'batch-end.json').read_text())
    metrics = json.loads((root / 'atomic-metrics.json').read_text())
    by_seed = {r['seed']: r for r in metrics['rows']}
    details = {r['key']: r for r in metrics['details']}
    source = Path(plan['source'])
    per_seed = []
    for case in plan['cases']:
        key = case['key']
        episode = root / 'episodes' / key
        config = yaml.safe_load((root / 'episodes' / (key + '.yaml')).read_text())
        native = None
        instructions, services, horizons = set(), [], set()
        for line in (episode / 'events.jsonl').open():
            e = json.loads(line)
            if e['kind'] == 'task_initialized':
                native = e['task']['instruction']
            elif e['kind'] == 'action_command' and e['source'] == 'vla':
                instructions.add(e['instruction'])
            elif e['kind'] == 'vla_service':
                services.append(e)
            elif e['kind'] == 'observation':
                value = e.get('facts', {}).get('native_horizon')
                if value is not None:
                    horizons.add(value)
        # Observation events use a compact record; the first stored snapshot is authoritative.
        initial = json.loads((episode / 'observations/0000000.json').read_text())
        horizons.add(initial['facts']['native_horizon'])
        pid = json.loads((root / 'episodes' / (key + '-launch.json')).read_text())['pid']
        path = Path('/proc') / str(pid) / 'cmdline'
        command = path.read_bytes().replace(b'\0', b' ').decode() if path.exists() else ''
        data = details[key]
        checks = dict(execution=bool(data['execution_audit']) and all(data['execution_audit'].values()),
            budgets=data['budget_audit']['turns_within_limit'] and data['budget_audit']['tools_within_limit'],
            original_instruction=bool(instructions) and instructions == {native},
            checkpoint=bool(services) and all(s['checkpoint_verified'] and s['metadata']['checkpoint']
                == plan['config']['vla']['checkpoint'] for s in services),
            sampling_seed=config['vla']['sampling_seed'] == 42 + case['seed'],
            native_horizon=horizons == {plan['native_horizon']},
            worker_finished=not ('vla_agent.cli' in command and str(episode) in command))
        per_seed.append(dict(seed=case['seed'], checks=checks))
    checks = dict(all_declared_cases_completed=set(end['completed']) == {c['key'] for c in plan['cases']},
        atomic_only=all(c['group'].startswith('atomic') for c in plan['cases']),
        source_unchanged=checked_hashes(source, json.loads((source.parent / 'source-hashes.json').read_text())),
        initial_memory_unchanged=checked_hashes(root / 'initial', json.loads((root / 'initial-hashes.json').read_text())),
        all_per_seed_checks=all(all(x['checks'].values()) for x in per_seed))
    start = json.loads((root / 'batch-start.json').read_text())
    value = dict(checks=checks, per_seed=per_seed,
        batch_wall_seconds=(end['finished_ns'] - start['started_ns']) / 1e9,
        reported_token_threshold_exceeded_seeds=[s for s, r in by_seed.items()
            if r['reported_tokens'] > plan['config']['planner']['max_total_tokens']],
        note='Token limit is an asynchronous stop threshold; in-flight usage may exceed it. Frozen hashes exclude bytecode caches.')
    (output or root / 'final-audit.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(checks))
    if not all(checks.values()):
        raise SystemExit('Evaluation audit failed; inspect final-audit.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    audit(args.root.resolve(), args.output)
