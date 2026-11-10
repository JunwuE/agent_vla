"""Freeze an OpenDrawer development retest with bounded, evidence-linked prior skills."""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import yaml
from prepare_composite_pilot import hashes, save

from vla_agent.config import load_config
from vla_agent.experience import ExperienceStore
from vla_agent.skill_packages import execution_bundle
from vla_agent.skills import SkillLibrary
from vla_agent.task_memory import TaskMemoryStore, TaskProfile


def main(args):
    repo = Path(__file__).resolve().parents[1]
    baseline, round_root, root = args.baseline.resolve(), args.round.resolve(), args.output.resolve()
    cases = json.loads((round_root / 'plan.json').read_text())['cases']
    results = {}
    for case in cases:
        candidates = ([Path(case['validation_episode'])] if case.get('validation_episode') else []) + [round_root / 'episodes' / f"OpenDrawer-seed{case['seed']:02d}",
                      round_root / 'episodes' / f"seed{case['seed']:02d}"]
        episode = next(p for p in candidates if (p / 'result.json').exists())
        result = json.loads((episode / 'result.json').read_text())
        if result.get('replay_max_state_error') != 0 or result.get('error'):
            raise ValueError(f'Invalid recovery replay: {episode}')
        if result['skill_version'] != case['skill_version']:
            raise ValueError(f'Replay evidence does not match selected skill: {episode}')
        results[case['seed']] = (episode, result)
    root.mkdir(parents=True, exist_ok=False)
    for name in ('episodes', 'inputs', 'analysis', 'report', 'initial/skills', 'initial/tasks'):
        (root / name).mkdir(parents=True)
    library = SkillLibrary(round_root / 'skills')
    selected = [(c['skill_id'], c['skill_version']) for c in cases]
    copied = set()

    def include(skill_id, version):
        if (skill_id, version) in copied:
            return
        skill = library.get(skill_id, version, 'PandaOmron')
        shutil.copytree(library.root / skill.get('storage_id', skill['id']) / version,
                        root / 'initial/skills' / skill.get('storage_id', skill['id']) / version)
        copied.add((skill_id, version))
        for dependency in (skill.get('package') or {}).get('uses', []):
            include(dependency['skill_id'], dependency['version'])

    for ref in selected:
        include(*ref)
    candidate_library = SkillLibrary(root / 'initial/skills', mode='accumulate')
    for case in cases:
        skill = candidate_library.get(case['skill_id'], case['skill_version'], 'PandaOmron')
        episode, result = results[case['seed']]
        candidate_library.record_candidate(
            skill['code'], 'PandaOmron', 'task_success',
            dict(kind='exact_prefix_development_replay', seed=case['seed'], evidence_root=str(episode),
                 report=result['recovery'], replay_max_state_error=0.0,
                 note='Same recorded entry state; not held-out validation or fresh Agent success rate.'),
            skill['layout'], parent_skill=skill, package=skill['package'], package_files=skill['package_files'])
    # Expose two operation families, retaining separately pinned entry-condition variants.
    families = []
    for family, group in [('drawer-pull-from-handle-hold', [c for c in cases if c['seed'] == 5]),
                          ('drawer-regrasp-and-open', [c for c in cases if c['seed'] != 5])]:
        families.append(dict(id=family, robot='PandaOmron',
            description='OpenDrawer | Reuse measured geometry with the matching entry condition; candidate evidence only.',
            versions=[dict(storage_id=c['skill_id'], version=c['skill_version'],
                when_to_use=library.get(c['skill_id'], c['skill_version'], 'PandaOmron')['when_to_use']
                    + ' Development replay outcome: ' + str(results[c['seed']][1]['recovery'].get('error')
                        or results[c['seed']][1]['recovery']['result'])) for c in group]))
    save(root / 'initial/skills/catalog.json', dict(schema_version=1, skills=families))
    candidate_library = SkillLibrary(root / 'initial/skills')
    for case in cases:
        skill = candidate_library.get(case['skill_id'], case['skill_version'], 'PandaOmron')
        execution_bundle(skill['package'], skill['package_files'], candidate_library, 'PandaOmron')
    shutil.copytree(baseline / 'retained/OpenDrawer/experiences', root / 'initial/experiences')
    experiences = ExperienceStore(root / 'initial/experiences')
    records = [json.loads(p.read_text()) for p in (root / 'initial/experiences').glob('*.json')]
    by_seed = {r['conditions']['environment']['seed']: r for r in records}

    def ref(seed, role='context'):
        r = by_seed[seed]
        return dict(episode_id=r['episode_id'], evidence_sha256=r['evidence_sha256'], role=role)

    held = next(c for c in cases if c['seed'] == 5)
    profile = TaskProfile(
        scope=dict(robot='PandaOmron', environment_kind='robocasa', task_name='OpenDrawer'),
        summary='Preserve effective first VLA execution. When a real repeated mismatch is established, reuse a measured handle-hold or regrasp procedure with fresh inputs.',
        applicability=['Atomic-only checkpoint 59999; native left/right drawer instruction remains active.',
                       'Development experience on seeds 0–9, not held-out evidence. All procedures remain candidates.'],
        rules=[
            dict(id='preserve_progress', when=['First VLA attempt or current phase still shows meaningful progress.'],
                recommendation='continue_vla', status='observed_pattern',
                rationale='Five original episodes succeeded without code recovery. Use brief running reviews; detailed recovery design belongs after a confirmed takeover.',
                avoid=['Do not infer failure solely from a low reward or small EEF translation.'], evidence=[ref(0, 'support'), ref(1, 'support')]),
            dict(id='reuse_handle_hold', when=['Current measured fingers and drawer geometry support an existing handle hold.',
                                              'Repeated VLA execution has not produced outward drawer motion.'],
                recommendation='consider_skill', status='observed_pattern', tools=['read_skill', 'measure_targets', 'check_skill'],
                skills=[dict(skill_id='drawer-pull-from-handle-hold', version=held['skill_version'])],
                rationale='Seed 5 succeeded with this controller family. The 0.30 m travel cap caused a redundant retry; a 0.45 m candidate completed its exact-prefix replay. Confirm current entry conditions; closure alone is insufficient.',
                avoid=['Do not copy historical pixel boxes or observation_step.'], evidence=[ref(5, 'support')]),
            dict(id='reuse_regrasp_with_caveats', when=['Fresh evidence establishes open or missed grasp, requiring a whole-task recovery.'],
                recommendation='inspect', status='hypothesis', tools=['list_skill_versions', 'read_skill', 'measure_targets', 'import_skill_program'],
                skills=[dict(skill_id='drawer-regrasp-and-open', version=c['skill_version'])
                        for c in cases if c['seed'] != 5 and results[c['seed']][1]['success']],
                rationale='Inspect drawer-regrasp-and-open entry variants and latest replay evidence before editing. Listed versions completed their development replay, not arbitrary new entries. Original episodes and replays exposed 3-pixel geometry below the 4-valid-pixel API minimum, persistent pose residual, fixed-window tracking loss and mixed U-shaped-handle geometry.',
                avoid=['Do not rerun a known failed branch without addressing its measured failure.',
                       'Keep task_success and geometric validity checks; do not weaken completion to force a pass.'],
                evidence=[ref(4), ref(6), ref(7), ref(8)])],
        open_questions=['Clearance, handle selection and tracking have only development-state tests; fresh entries may differ.',
                        'Replay evidence attached to skill versions is more recent than the original episode guidance.'])
    TaskMemoryStore(root / 'initial/tasks').write_profile(profile, experiences)
    config = yaml.safe_load((baseline / 'config.yaml').read_text())
    config['runtime'].update(experiment_label='opendrawer-recovery-reuse-development',
        progress_review_cooldown_steps=200, recovery_steps=600, recovery_seconds=360,
        skill_library=str(root / 'initial/skills'), experience_library=str(root / 'initial/experiences'),
        task_memory_library=str(root / 'initial/tasks'))
    (root / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    load_config(root / 'config.yaml')
    source = root / 'profile/src'
    shutil.copytree(repo / 'src', source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    save(source.parent / 'source-hashes.json', hashes(source))
    ih = hashes(root / 'initial')
    save(root / 'initial-hashes.json', ih)
    task = dict(task='OpenDrawer', group='atomic_seen', horizon=1500)
    save(root / 'plan.json', dict(tasks=[task], seeds=args.seeds,
        cases=[dict(task, seed=s, key=f'OpenDrawer-seed{s:02d}') for s in args.seeds], workers=5,
        source=str(source), config=config, native_horizon=750, declared_horizon_multiplier=2,
        initial_memory_sha256=hashlib.sha256(json.dumps(ih, sort_keys=True).encode()).hexdigest(),
        memory_policy='Identical private copies of bounded original experience, three task rules and two candidate recovery families; no cross-episode updates within this batch.',
        changes='Brief running review prompt; accepted-progress duplicate review cooldown 200 steps; explicit 4-valid-depth-pixel contract; candidate recovery revisions and memory; per-code budget 600 steps/360 s (previously 400/240), episode still 1500 steps.',
        evaluation_kind='same_seed_development_retest_not_held_out', baseline=str(baseline),
        recovery_development_source=str(round_root), prepared_ns=time.time_ns()))
    (root / 'services').symlink_to((baseline / 'services').resolve(), target_is_directory=True)
    (root / 'README.md').write_text(f'''# OpenDrawer {len(args.seeds)}-seed 开发复测

固定 seed {', '.join(map(str, args.seeds))}、atomic checkpoint 59999、采样 seed 42+seed、原生语言、750/1500 步双口径。
每 episode 最多 8 Astra turns、每 turn 24 tools，累计上报 token 1,500,000 时停止；
最后在途请求可能超过阈值，token 不能直接当实际账单。Flora 仍每 50 步观测。

新增同阶段已接受的“正常推进”审核去重（200 步），不自动接管 VLA。
载入三个精简任务规则、原先 10 条经验、两个语义技能族；最新回放证据随技能版本保存。
每个 seed 的初始库相同且独立，批内不互传经验，源码和初始库冻结。
所有技能仍是 candidate；新像素参数必须从当时观测计算。

单次恢复预算从 400 步/240 秒调整为 600 步/360 秒，总恢复预算仍 1200 步/1200 秒，
episode 仍 1500 步、无墙钟限时。报告明确区分预算变化、局部恢复回放和完整 Agent 复测。
这是已参与开发的同一批 seed，不是独立泛化评测；没有新增纯 VLA 配对组。
''')
    print(json.dumps(dict(root=str(root), skills=len(candidate_library.index), families=len(families)), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--round', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    main(parser.parse_args())
