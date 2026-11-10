"""Prepare reproducible coffee controls or a bounded microwave speed pilot."""

import argparse
import json
import shutil
import time
from pathlib import Path

import yaml
from prepare_composite_pilot import hashes, save

from vla_agent.config import Config


def main(root, variant):
    repo = Path(__file__).resolve().parents[1]
    coffee = repo / 'runs/coffee-seen-budgeted-20261103'
    prior = repo / 'runs/waffle-subtasks-20261102'
    template = coffee if variant == 'coffee-whole' else prior
    old = json.loads((template / 'plan.json').read_text())
    config = json.loads(json.dumps(old['config']))
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    for folder in ('episodes', 'analysis', 'inputs', 'report'):
        (root / folder).mkdir()
    shutil.copytree(template / 'initial', root / 'initial')
    initial_hashes = hashes(root / 'initial')
    save(root / 'initial-hashes.json', initial_hashes)
    if variant == 'coffee-whole':
        config['runtime'].update(supervision_enabled=False, initial_planning=False,
                                 experiment_label='coffee-native-instruction-control')
        config['flora']['enabled'] = False
        config['decision']['kind'] = 'mock'
        config['planner']['kind'] = 'mock'
        seeds, workers = list(range(5)), 2
        comparison = 'Same coffee seeds/checkpoint/action budget/initial arrays; native whole instruction, no Astra/Flora.'
    else:
        config['planner'].update(max_calls=12, max_tool_calls=16, max_turns_per_intervention=1,
            max_total_tokens=2_000_000, timeout_seconds=300, thread_max_turns=4,
            historical_memory_chars=4000, working_memory_chars=8000)
        config['runtime'].update(max_reviews=12, max_interventions=12, recovery_attempts=3,
            recovery_steps=400, recovery_seconds=240, recovery_total_steps=1200,
            recovery_total_seconds=1200, recovery_program_versions=24,
            experiment_label='waffle-code-tools-v2-noop-refine-budgeted')
        config['decision']['flora']['max_calls'] = 80
        config['flora']['max_calls'] = 4
        seeds, workers = [0], 1
        comparison = ('Same WaffleReheat seed 0/checkpoint/action budget and original initial memory. '
                      'New tools/no-op refine handling and smaller inference/recovery budgets; '
                      'compare verified milestones, not just earlier termination.')
        if variant == 'waffle-stall':
            config['planner'].update(max_calls=8, max_total_tokens=1_500_000)
            config['runtime'].update(max_reviews=8, max_interventions=8, recovery_attempts=2,
                experiment_label='waffle-stalled-policy-review-v3-budgeted')
            comparison += (' Follow-up: recognize sustained inactive commands with no phase progress; '
                           'maximum 8 turns, 1.5M reported tokens, two code attempts. No fixed step interruption.')
    config['runtime'].update(skill_library=str(root / 'initial/skills'),
        experience_library=str(root / 'initial/experiences'), task_memory_library=str(root / 'initial/tasks'))
    Config.model_validate(config)
    (root / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    source = root / 'profile/src'
    shutil.copytree(repo / 'src', source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    save(source.parent / 'source-hashes.json', hashes(source))
    task = old['tasks'][0]
    plan = dict(tasks=[task], seeds=seeds, workers=workers, source=str(source), config=config,
        cases=[dict(task, seed=s, key=f"{task['task']}-seed{s:02d}") for s in seeds],
        initial_memory_sha256=old['initial_memory_sha256'], comparison_root=str(template),
        comparison_note=comparison, prepared_ns=time.time_ns(), variant=variant)
    save(root / 'plan.json', plan)
    (root / 'services').symlink_to((coffee / 'services').resolve(), target_is_directory=True)
    (root / 'README.md').write_text(f'# {variant}\n\n{comparison}\n\n'
        'All declared seeds retained. Frozen source and identical per-seed initial memory. '
        'Native simulator success is authoritative. No automatic replacement/retry.\n')
    print(json.dumps(dict(root=str(root), cases=len(seeds), variant=variant)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--variant', choices=('coffee-whole', 'waffle-bounded', 'waffle-stall'), required=True)
    args = parser.parse_args()
    main(args.root, args.variant)
