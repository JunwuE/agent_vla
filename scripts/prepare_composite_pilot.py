"""Freeze the current runtime and a small common skill library for a five-seed pilot."""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import yaml

from vla_agent.config import Config
from vla_agent.skills import SkillLibrary


def hashes(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main(root, services):
    repo = Path(__file__).resolve().parents[1]
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name in ('episodes', 'inputs', 'analysis', 'report', 'initial/skills', 'initial/experiences', 'initial/tasks'):
        (root / name).mkdir(parents=True)
    config = yaml.safe_load((repo / 'configs/waffle_subtasks.yaml').read_text())
    config['environment'].update(task_name='PrepareCoffee', task_group='composite_seen', seed=0,
                                 native_subtask_checks=True)
    config['planner'].update(max_calls=12, max_tool_calls=16, max_turns_per_intervention=1,
                             max_total_tokens=2_000_000, timeout_seconds=300,
                             thread_max_turns=4, historical_memory_chars=4000, working_memory_chars=8000)
    config['decision']['flora']['max_calls'] = 80
    config['flora']['max_calls'] = 4
    config['runtime'].update(experiment_label='coffee-seen-code-tools-v2-budgeted', max_steps=3600,
        max_wall_seconds=None, max_reviews=12, max_interventions=12,
        recovery_attempts=3, recovery_steps=400, recovery_seconds=240,
        recovery_total_steps=1200, recovery_total_seconds=1200, recovery_program_versions=24)
    initial = root / 'initial'
    for name in ('rgbd-surface-geometry', 'eef-feedback-control'):
        shutil.copytree(repo / 'skills' / name, initial / 'skills' / name)
    config['runtime'].update(skill_library=str(initial/'skills'), experience_library=str(initial/'experiences'),
                             task_memory_library=str(initial/'tasks'))
    Config.model_validate(config)
    library = SkillLibrary(initial/'skills')
    assert len(library.index) == 2 and all(x['status'] == 'candidate' for x in library.index)
    (root/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    source = root / 'profile/src'
    shutil.copytree(repo / 'src', source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    save(source.parent/'source-hashes.json', hashes(source))
    ih = hashes(initial)
    save(root/'initial-hashes.json', ih)
    memory_digest = hashlib.sha256(json.dumps(ih, sort_keys=True).encode()).hexdigest()
    task = dict(task='PrepareCoffee', group='composite_seen', horizon=3600)
    plan = dict(tasks=[task], seeds=list(range(5)),
        cases=[dict(task, seed=seed, key=f'PrepareCoffee-seed{seed:02d}') for seed in range(5)],
        workers=5, source=str(source), config=config, initial_memory_sha256=memory_digest,
        memory_policy='Identical private copies; two common candidate packages, no prior coffee-task history. '
                      'New code retained privately, merged only after all five episodes finish.',
        native_horizon=1800, declared_horizon_multiplier=2,
        changes='Code tools v2; PrepareCoffee atomic language catalog and boolean verifiers; bounded Astra calls/tokens.',
        prepared_ns=time.time_ns())
    save(root/'plan.json', plan)
    (root/'services').symlink_to(services.resolve(), target_is_directory=True)
    (root/'README.md').write_text('''# PrepareCoffee：Composite-Seen，5 seeds，新代码工具架构

固定 seed 0–4、同一 OmniSF atomic checkpoint 59999、π0.5、Astra high、Flora 每 50 步。
当前源码已冻结；每个 seed 使用相同私有初始库，只包含两个公共候选包，无咖啡任务历史。
本轮不在 seed 之间传递新经验；结果是受预算约束的迁移开发测试，不是训练或纯 VLA 对照。

每个 episode 最多 12 个 Codex turn，每 turn 最多 16 个工具调用，不追加预算续轮。
每个 Code 子任务最多 3 次程序尝试。上报的输入+输出 token 达到 2,000,000 时取消模型调用并停止；
用量是异步上报，最后一个在途请求可能超过阈值，不能当作服务端精确计费硬上限。
线程每 4 turns 重建，Working Memory 索引 8000 字符，历史索引 4000 字符。
最大 3600 动作步（原生建议 horizon 1800 的两倍），无 Episode 墙钟限时；单调用/程序仍有预算。

任务目录按 RoboCasa 原子语言提供柜到台面、台面到咖啡机、启动三个目标，由 Astra 选择计划。
局部完成使用仿真只读布尔验证器，不提供物体坐标；属于验证器辅助评测。
整任务成功仅采用 PrepareCoffee 原生判据，不以 Flora 分数或程序声明代替。

运行入口：`scripts/run_composite_sweep.py <本目录>`。
`status.json` 记录进度；`episodes/` 保留原始日志；`analysis/`、`results.md` 保存结果。
所有 seed 均保留，不用新 seed 替换失败，不在同一批中修改源码。
''')
    print(json.dumps(dict(root=str(root), source=str(source), skills=[x['id'] for x in library.index],
                         max_turns_per_episode=12, max_tools_per_turn=16), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--services', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.services)
