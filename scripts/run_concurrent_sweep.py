"""Bounded simulation workers; each episode owns a private library, merged only after completion."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml
from run_seed_sweep import analyze, episode_report, report, write_json

from vla_agent.experience import ExperienceStore
from vla_agent.skills import SkillLibrary


def merge_libraries(source_skills, source_experiences, target_skills, target_experiences):
    """Called by one scheduler after a worker exits; never modifies an in-flight library."""
    SkillLibrary(source_skills)
    ExperienceStore(source_experiences)
    source_catalog, target_catalog = source_skills/'catalog.json', target_skills/'catalog.json'
    if source_catalog.exists() and target_catalog.exists():
        if json.loads(source_catalog.read_text()) != json.loads(target_catalog.read_text()):
            raise ValueError('Skill catalogs differ; explicitly reconcile semantic identities before merging')
    for source in sorted(source_skills.glob('*/*/manifest.json')):
        target = target_skills/source.relative_to(source_skills)
        if not target.exists():
            shutil.copytree(source.parent, target.parent)
            continue
        if source.with_name('policy.py').read_bytes() != target.with_name('policy.py').read_bytes():
            raise ValueError('Skill identity collision while merging completed episode')
        if json.loads(source.read_text()).get('package') != json.loads(target.read_text()).get('package'):
            raise ValueError('Skill package/dependency contract collision while merging completed episode')
        src_runs, dst_runs = source.with_name('runs.jsonl'), target.with_name('runs.jsonl')
        if src_runs.exists():
            existing = dst_runs.read_text().splitlines() if dst_runs.exists() else []
            records = {json.dumps(json.loads(line), sort_keys=True): line for line in existing}
            for line in src_runs.read_text().splitlines():
                records.setdefault(json.dumps(json.loads(line), sort_keys=True), line)
            temporary = dst_runs.with_suffix('.tmp')
            temporary.write_text('\n'.join(records.values())+'\n')
            temporary.replace(dst_runs)
    for source in source_experiences.glob('*.json'):
        target = target_experiences/source.name
        if target.exists():
            if target.read_bytes() != source.read_bytes():
                raise ValueError('Experience identity collision')
        else:
            shutil.copy2(source, target)
    if source_catalog.exists() and not target_catalog.exists():
        shutil.copy2(source_catalog, target_catalog)
        if (source_skills/'INDEX.md').exists():
            shutil.copy2(source_skills/'INDEX.md', target_skills/'INDEX.md')
    SkillLibrary(target_skills)
    ExperienceStore(target_experiences)


def alive(pid):
    path = Path('/proc')/str(pid)/'stat'
    return path.exists() and path.read_text().split(') ', 1)[1].split()[0] != 'Z'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--adopt-seed-zero', type=int)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error('Use 1..4 workers')
    root = args.root.resolve()
    shared = root/'shared'
    if not shared.exists():
        shared.mkdir()
        shutil.copytree('skills', shared/'skills')
        shutil.copytree('memory/close_toaster/experiences', shared/'experiences')
    (root/'inputs').mkdir(exist_ok=True)
    completed = {int(p.stem[4:]) for p in (root/'analysis').glob('seed*.json')}
    active = {}
    handles = {}
    if args.adopt_seed_zero and 0 not in completed:
        previous = json.loads((root/'episodes/seed00-launch.json').read_text())
        active[0] = dict(pid=args.adopt_seed_zero, process=None, profile=previous,
                         started_ns=json.loads((root/'status.json').read_text())['started_ns'],
                         skills=root/'skills', experiences=root/'experiences')
    remaining = [seed for seed in range(50) if seed not in completed and seed not in active]
    while active or remaining:
        for seed, item in list(active.items()):
            finished = item['process'].poll() is not None if item['process'] else not alive(item['pid'])
            if not finished:
                continue
            if seed in handles:
                handles.pop(seed).close()
            episode = root/'episodes'/f'seed{seed:02d}'
            if not (episode/'result.json').exists():
                raise RuntimeError(f'Worker seed {seed} exited without result; partial evidence retained')
            data = analyze(episode)
            data['release'] = item['profile']['release']
            data['sweep_concurrency_limit'] = args.workers
            episode_report(root, data)
            merge_libraries(item['skills'], item['experiences'], shared/'skills', shared/'experiences')
            completed.add(seed)
            del active[seed]
            report(root)
            print(json.dumps(dict(event='completed', seed=seed, result=data['result'],
                                  attempts=data['program_attempts'], release=data['release']), ensure_ascii=False), flush=True)
        while remaining and len(active) < args.workers:
            seed = remaining.pop(0)
            episode = root/'episodes'/f'seed{seed:02d}'
            if episode.exists():
                raise RuntimeError(f'Unadopted existing seed {seed}; refusing overwrite')
            profile = json.loads((root/'active-profile.json').read_text())
            inputs = root/'inputs'/f'seed{seed:02d}'
            inputs.mkdir(exist_ok=False)
            shutil.copytree(shared/'skills', inputs/'skills')
            shutil.copytree(shared/'experiences', inputs/'experiences')
            config = yaml.safe_load(Path(profile['config']).read_text())
            # Task guidance is pinned for this worker and is never merged from online proposals.
            task_library = Path(config['runtime'].get('task_memory_library') or
                                Path(config['runtime'].get('experience_library', 'memory/experiences')).parent/'tasks')
            if task_library.exists():
                shutil.copytree(task_library, inputs/'tasks')
            else:
                (inputs/'tasks').mkdir()
            # The new runtime derives this from inputs/experiences; omit the optional key
            # so archived runtime versions can still consume their original config schema.
            config['runtime'].pop('task_memory_library', None)
            config['environment']['seed'] = seed
            config['vla']['sampling_seed'] = 42+seed
            config['runtime']['skill_library'] = str(inputs/'skills')
            config['runtime']['experience_library'] = str(inputs/'experiences')
            config_path = root/'episodes'/f'seed{seed:02d}.yaml'
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            env = os.environ.copy()
            env['PYTHONPATH'] = profile['source']
            env['OMP_NUM_THREADS'] = '4'
            env['OPENBLAS_NUM_THREADS'] = '1'
            log = (root/'episodes'/f'seed{seed:02d}.log').open('wb')
            process = subprocess.Popen([sys.executable, '-m', 'vla_agent.cli', 'run', '--config', str(config_path),
                                        '--output', str(episode)], env=env, stdout=log, stderr=subprocess.STDOUT)
            started = time.time_ns()
            handles[seed] = log
            active[seed] = dict(pid=process.pid, process=process, started_ns=started, profile=profile,
                                skills=inputs/'skills', experiences=inputs/'experiences')
            write_json(root/'episodes'/f'seed{seed:02d}-launch.json', dict(**profile, pid=process.pid,
                        started_ns=started, completed_before_launch=sorted(completed), concurrency_limit=args.workers))
            print(json.dumps(dict(event='start', seed=seed, release=profile['release'],
                                  completed_before_launch=sorted(completed))), flush=True)
        write_json(root/'status.json', dict(state='running' if active else 'completed', completed=sorted(completed),
            concurrency_limit=args.workers, active=[dict(seed=seed, pid=item['pid'], started_ns=item['started_ns'],
                release=item['profile']['release']) for seed,item in active.items()], updated_ns=time.time_ns()))
        if active:
            time.sleep(2)


if __name__ == '__main__':
    main()
