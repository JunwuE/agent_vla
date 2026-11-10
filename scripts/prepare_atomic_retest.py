"""Freeze fresh atomic seeds with byte-identical starting memory from a completed sweep."""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import yaml
from prepare_composite_pilot import hashes, save

from vla_agent.config import load_config


def main(args):
    previous, root = args.previous.resolve(), args.output.resolve()
    plan = json.loads((previous / 'plan.json').read_text())
    seeds = args.seeds
    if len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError('Seeds must be unique and nonnegative')
    if len(plan['tasks']) != 1 or not plan['tasks'][0]['group'].startswith('atomic'):
        raise ValueError('Expected one atomic task')
    if set(seeds) & set(plan['seeds']) and not args.allow_seen_seeds:
        raise ValueError('Fresh-seed retest cannot overlap previous seeds')
    if not (previous / 'batch-end.json').exists():
        raise ValueError('Previous sweep must be completed')
    initial_hashes = json.loads((previous / 'initial-hashes.json').read_text())
    if hashes(previous / 'initial') != initial_hashes:
        raise ValueError('Previous starting memory changed')
    root.mkdir(parents=True, exist_ok=False)
    for directory in ('episodes', 'inputs', 'analysis', 'report'):
        (root / directory).mkdir()
    shutil.copytree(previous / 'initial', root / 'initial')
    save(root / 'initial-hashes.json', initial_hashes)
    config = yaml.safe_load((args.config or previous / 'config.yaml').read_text())
    if args.mode == 'vla':
        config['runtime'].update(supervision_enabled=False, initial_planning=False)
        config['decision']['kind'] = 'mock'
        config['planner']['kind'] = 'mock'
        config['flora']['enabled'] = False
    config['runtime'].update(experiment_label='atomic-fresh-seeds-performance',
        skill_library=str(root / 'initial/skills'), experience_library=str(root / 'initial/experiences'),
        task_memory_library=str(root / 'initial/tasks'))
    (root / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    load_config(root / 'config.yaml')
    repo = Path(__file__).resolve().parents[1]
    source = root / 'profile/src'
    shutil.copytree(repo / 'src', source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    save(source.parent / 'source-hashes.json', hashes(source))
    task = plan['tasks'][0]
    plan.update(seeds=seeds, cases=[dict(task, seed=s, key=f"{task['task']}-seed{s:02d}") for s in seeds],
        source=str(source), config=config, workers=args.workers,
        initial_memory_sha256=hashlib.sha256(json.dumps(initial_hashes, sort_keys=True).encode()).hexdigest(),
        previous_sweep=str(previous), evaluation_kind='paired_seed_development' if args.allow_seen_seeds else 'fresh_seed_fixed_memory_development_evaluation',
        mode=args.mode, changes=args.note,
        prepared_ns=time.time_ns())
    save(root / 'plan.json', plan)
    (root / 'services').symlink_to((previous / 'services').resolve(), target_is_directory=True)
    (root / 'README.md').write_text(f'''# {task['task']} 固定 seeds 速度与成功率测试：{args.mode}

预先指定 seeds：{seeds}；并发 {args.workers}。每个 seed 使用相同、独立的初始 Memory。
初始库与 `{previous}` 字节一致；本轮说明：{args.note}
750/1500 步分别统计，保持原生语言和 atomic checkpoint 59999；采样 seed = 42 + 环境 seed。
每 episode 最多 8 Astra turns / 每 turn 24 tools / 上报 token 1,500,000 停止阈值。
最后在途调用可超过阈值；无 episode 墙钟限时。所有结果均保留，不替换失败 seed。
允许重复已见 seeds：{args.allow_seen_seeds}。这是同一任务的开发评测，不是新任务泛化。
vla 模式关闭监督、规划、Flora 与代码恢复；不发生 Astra/Flora 调用。配对组见 campaign 报告。
统计 simulator（含内部渲染）、snapshot、RPC/序列化及调度剩余时间、观测存盘。
计时采用 monotonic clock；运行时计时不提供给 Astra/VLA/Flora。
''')
    print(json.dumps(dict(root=str(root), seeds=seeds, workers=args.workers)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--workers', type=int, default=5, choices=range(1, 9))
    parser.add_argument('--allow-seen-seeds', action='store_true')
    parser.add_argument('--mode', choices=('agent', 'vla'), default='agent')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--note', default='Current source frozen; unchanged initial memory and budget.')
    main(parser.parse_args())
