"""Freeze a ten-seed OpenDrawer evaluation with bounded model use."""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import yaml
from prepare_composite_pilot import hashes, save

from vla_agent.config import load_config
from vla_agent.skills import SkillLibrary


def main(root, services):
    repo = Path(__file__).resolve().parents[1]
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name in ('episodes', 'inputs', 'analysis', 'report', 'initial/skills',
                 'initial/experiences', 'initial/tasks'):
        (root / name).mkdir(parents=True)
    config = yaml.safe_load((repo / 'configs/waffle_subtasks.yaml').read_text())
    config['environment'].update(task_name='OpenDrawer', task_group='atomic_seen', seed=0,
                                 native_subtask_checks=False, ignore_native_horizon=True)
    config['planner'].update(max_calls=8, max_tool_calls=24, max_turns_per_intervention=1,
        max_total_tokens=1_500_000, timeout_seconds=300, thread_max_turns=4,
        compact_context=True, historical_memory_chars=4000, working_memory_chars=8000,
        preserve_initial_instruction=True)
    config['decision']['flora']['max_calls'] = 40
    config['flora']['max_calls'] = 4
    config['runtime'].update(experiment_label='opendrawer-atomic10-bounded', max_steps=1500,
        max_wall_seconds=None, max_reviews=8, max_interventions=8,
        recovery_attempts=3, recovery_steps=400, recovery_seconds=240,
        recovery_total_steps=1200, recovery_total_seconds=1200, recovery_program_versions=24)
    initial = root / 'initial'
    for name in ('rgbd-surface-geometry', 'eef-feedback-control'):
        shutil.copytree(repo / 'skills' / name, initial / 'skills' / name)
    config['runtime'].update(skill_library=str(initial / 'skills'),
        experience_library=str(initial / 'experiences'), task_memory_library=str(initial / 'tasks'))
    (root / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    load_config(root / 'config.yaml')
    library = SkillLibrary(initial / 'skills')
    assert len(library.index) == 2 and all(x['status'] == 'candidate' for x in library.index)
    source = root / 'profile/src'
    shutil.copytree(repo / 'src', source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    save(source.parent / 'source-hashes.json', hashes(source))
    ih = hashes(initial)
    save(root / 'initial-hashes.json', ih)
    memory_digest = hashlib.sha256(json.dumps(ih, sort_keys=True).encode()).hexdigest()
    task = dict(task='OpenDrawer', group='atomic_seen', horizon=1500)
    plan = dict(tasks=[task], seeds=list(range(10)),
        cases=[dict(task, seed=seed, key=f'OpenDrawer-seed{seed:02d}') for seed in range(10)],
        workers=5, source=str(source), config=config, initial_memory_sha256=memory_digest,
        memory_policy='Identical private copies; two common candidate packages; no task-specific history. '
                      'New skills retained privately and archived only after all ten episodes finish.',
        native_horizon=750, declared_horizon_multiplier=2,
        changes='Atomic-only checkpoint evaluation; current code tools and stall review; compact Astra context; '
                'native task instruction preserved; bounded calls/tokens; no composite subtask verifiers.',
        selection_evidence=[
            'runs/context-suite15-20261029: OpenDrawer seed0 failed; the other four atomic tasks passed.',
            'docs/atomic-pilot.md: OpenDrawer pure VLA 2/3; older supervised seed0 attempts failed. '
            'Small and repeated historical samples, not a statistically established task ranking.'],
        prepared_ns=time.time_ns())
    save(root / 'plan.json', plan)
    (root / 'services').symlink_to(services.resolve(), target_is_directory=True)
    (root / 'README.md').write_text('''# OpenDrawer：Atomic-Seen，10 seeds

仅测试 atomic；使用 OmniSF atomic checkpoint 59999、π0.5、Astra high、Flora 每 50 步。
固定 seed 0–9，策略采样 seed=42+环境 seed。源码和初始库冻结，最多五个并发 worker。
每个 seed 独立复制两个公共候选包（RGB-D 几何、末端反馈控制），不传递本批新经验。
保留原生任务语言；Astra 可以记录观察阶段，但初始规划不能替换为细粒度运动指令。
不启用 composite 专用子任务验证器；整任务成功只采用 OpenDrawer 原生判据。

每 episode 最多 8 个 Codex turn，每 turn 最多 24 个工具调用；模型上报累计 token
达到 1,500,000 后停止。用量异步上报，最后在途请求可能超出阈值，并非精确计费硬上限。
每个代码目标最多 3 次尝试；每次 400 动作步/240 秒，总计 1200 步/1200 秒。
线程每 4 turns 重建；Working Memory 8000 字符、历史索引 4000 字符；
Astra 使用 compact_context，保留历史窗口首尾两个时刻及三视角和紧凑数值摘要。
Flora 监测最多 40 次，Astra 可额外请求最多 4 次 Flora 工具评价。

最大动作预算 1500 步，无 episode 墙钟限时。单独报告原生建议 horizon 750 步内成功率，
以及延长至 1500 步的成功率；不可把延长预算的结果直接与旧 750 步结果比较。
本轮是固定版本的 Agent 测试，没有新跑纯 VLA 配对组，不能据此声称 Agent 带来成功率提升。
seed 0–2 曾用于历史开发。任务选择依据见 plan.json；旧样本少且有重复，不是稳定成功率排名。

运行：`.venv/bin/python scripts/run_task_sweep.py <本目录>`。
保留全部失败，不替换 seed，不在本批改源码。新增程序仅归档为候选，不自动成为默认技能。
''')
    print(json.dumps(dict(root=str(root), source=str(source),
                         skills=[x['id'] for x in library.index], seeds=plan['seeds']), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--services', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.services)
