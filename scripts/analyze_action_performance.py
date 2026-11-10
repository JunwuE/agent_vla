"""Separate measured action latency from model/planning latency, including old logs."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def stats(values):
    values = sorted(values)
    if not values:
        return dict(count=0)
    index = (len(values) - 1) * .9
    lower = int(index)
    p90 = values[lower] + (index - lower) * (values[min(lower + 1, len(values) - 1)] - values[lower])
    return dict(count=len(values), total=sum(values), mean=statistics.mean(values),
                median=statistics.median(values), p90=p90, maximum=max(values))


def analyze(root):
    rows = []
    aggregate = defaultdict(list)
    for episode in sorted((root / 'episodes').glob('*/result.json')):
        episode = episode.parent
        pending, durations = {}, defaultdict(list)
        for line in (episode / 'events.jsonl').open():
            event = json.loads(line)
            if event['kind'] == 'action_command':
                pending[event['id']] = event
            elif event['kind'] == 'action_executed':
                command = pending.pop(event['command_id'])
                seconds = (event['time_ns'] - command['time_ns']) / 1e9
                durations['action_seconds'].append(seconds)
                durations[command['source'] + '_action_seconds'].append(seconds)
            elif event['kind'] == 'action_performance':
                for key in ('environment_seconds', 'recording_seconds', 'simulator_seconds', 'snapshot_seconds'):
                    if key in event:
                        durations[key].append(event[key])
                if 'simulator_seconds' in event and 'snapshot_seconds' in event:
                    durations['transport_serialization_scheduling_seconds'].append(
                        event['environment_seconds'] - event['simulator_seconds'] - event['snapshot_seconds'])
        result = json.loads((episode / 'result.json').read_text())
        for key, values in durations.items():
            aggregate[key].extend(values)
        rows.append(dict(episode=episode.name, success=result['success'],
            wall_seconds=result['wall_seconds'], steps=result['control_steps'],
            timing={k: stats(v) for k, v in durations.items()}))
    report = dict(episodes=rows, timing={k: stats(v) for k, v in aggregate.items()}, notes=[
        'Simulator time includes physics, controllers and internal camera rendering; not pure physics.',
        'Transport residual includes worker JSON/base64 encoding, IPC, client decoding and scheduling.',
        'Recording includes lossless npz compression, writes, hashing and observation metadata.',
        'Action timings exclude Astra reasoning, VLA inference and code tool logic between actions.',
        'Old logs only measure command-to-receipt latency; finer attribution cannot be recovered retrospectively.',
        'Background model calls overlap actions; their durations cannot be added to action wall time.',
    ])
    (root / 'action-performance.json').write_text(json.dumps(report, indent=2) + '\n')
    columns = ['simulator_seconds', 'snapshot_seconds', 'transport_serialization_scheduling_seconds', 'recording_seconds']
    lines = ['# 动作执行耗时拆解', '', '| episode | 成功 | 步数 | 总秒 | 动作均秒 | 动作 P90 | 仿真均秒 | 几何快照均秒 | 通信/调度均秒 | 存盘均秒 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        timing = row['timing']
        extras = [f"{timing[k]['mean']:.4f}" if k in timing else '未计时' for k in columns]
        action = timing.get('action_seconds', {})
        lines.append(f"| {row['episode']} | {row['success']} | {row['steps']} | {row['wall_seconds']:.1f} | "
            f"{action.get('mean', 0):.4f} | {action.get('p90', 0):.4f} | " + ' | '.join(extras) + ' |')
    lines += ['', '仿真项包含控制器、物理步进及内部相机渲染。通信剩余项也包含序列化和进程调度，不能视为纯网络延迟。',
              '动作计时不包含 Astra、VLA 推理及动作之间的代码几何计算。旧日志没有内部细项，不能事后精确分摊。']
    (root / 'action-performance.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(episodes=len(rows), timing=report['timing'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    analyze(parser.parse_args().root)
