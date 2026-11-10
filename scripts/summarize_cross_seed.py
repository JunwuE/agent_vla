"""Audit paired prefixes, model calls and recovery costs in the cross-seed development runs."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from summarize_toaster_iterations import summarize

from vla_agent.data import read_events


def audit(episode):
    rows = read_events(episode)
    manifest = json.loads((episode / 'manifest.json').read_text())['config']
    result = json.loads((episode / 'result.json').read_text())
    commands = [r for r in rows if r['kind'] == 'action_executed']
    first_observation = next((r for r in rows if r['kind'] == 'observation'), None)
    starts = {r['turn_id']: r for r in rows if r['kind'] == 'codex_turn_start'}
    elapsed = Counter()
    finished = set()
    for row in rows:
        if row['kind'] == 'codex_turn_end':
            start = starts[row['turn_id']]
            context = start['context']
            label = ('revision' if 'recovery_feedback' in context else
                     'initialization' if context['intent'] == 'initialize_plan' else
                     'review' if context['intent'] == 'review_running' else 'paused_planning')
            elapsed[label] += (row['time_ns'] - start['time_ns']) / 1e9
            finished.add(row['turn_id'])
    prefix = []
    for action in commands:
        if action['source'] != 'vla':
            break
        prefix.append(action)
    prep = [r for r in rows if r['kind'] == 'flora_result' and r['result']['status'] == 'scored']
    recovery = [r for r in rows if r['kind'] == 'recovery_result']
    imported = [r for r in rows if r['kind'] == 'skill_program_imported']
    proposals = [r for r in rows if r['kind'] == 'intervention_plan']
    return dict(path=str(episode), seed=manifest['environment']['seed'], result=result,
                supervision=manifest['runtime']['supervision_enabled'],
                max_steps=manifest['runtime']['max_steps'], policy_seed=manifest['vla'].get('sampling_seed'),
                native_success=result['success'] and result['control_steps'] <= 450,
                source_steps=dict(Counter(r['source'] for r in commands)),
                astra_completed_turn_seconds=dict(elapsed),
                astra_turns_without_completion_event=len(set(starts) - finished),
                astra_context_compactions=sum(r['kind'] == 'codex_item'
                                               and r['item'].get('type') == 'contextCompaction' for r in rows),
                program_seconds=sum(r['seconds'] for r in recovery),
                program_attempts=len(recovery), imports=imported,
                generated_versions=len(list(episode.glob('workspaces/*/versions/*/policy.py'))),
                direct_skill_calls=sum(r['kind'] == 'skill_invocation' for r in rows),
                tools=dict(Counter(r['name'] for r in rows if r['kind'] == 'codex_tool')),
                physical_intervention_proposals=[r['plan']['kind'] for r in proposals
                                                 if r['binding']['observation_step'] > 0],
                flora_client_seconds=[r['result']['seconds'] for r in prep],
                flora_scores=[dict(step=r['result']['binding']['observation_step'],
                                   score=r['result']['score'], goal=r['result']['task']) for r in prep],
                motion_wall_seconds=((commands[-1]['time_ns'] - next(r['time_ns'] for r in rows
                                      if r['kind'] == 'action_command')) / 1e9 if commands else 0),
                prefix=prefix, initial=first_observation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root
    summarize(root)
    runs = [audit(p.parent) for p in sorted(root.glob('round*/seed*/result.json'))]
    pairs = []
    for run in runs:
        if not run['supervision'] or run['policy_seed'] is None or run['initial'] is None:
            continue
        baseline = next((r for r in runs if not r['supervision'] and r['seed'] == run['seed']
                         and r['max_steps'] == run['max_steps'] and r['policy_seed'] == run['policy_seed']
                         and r['initial'] is not None), None)
        if baseline is None:
            continue
        a, b = baseline['prefix'], run['prefix']
        count = min(len(a), len(b))
        with np.load(Path(baseline['path']) / baseline['initial']['path']) as x, \
                np.load(Path(run['path']) / run['initial']['path']) as y:
            initial = set(x.files) == set(y.files) and all(np.array_equal(x[k], y[k]) for k in x.files)
        pairs.append(dict(seed=run['seed'], baseline=baseline['path'], agent=run['path'],
                          initial_arrays_identical=initial, prefix_steps=count,
                          commands_identical=all(x['action'] == y['action'] for x, y in zip(a[:count], b[:count])),
                          measured_states_identical=all(x['measured_state'] == y['measured_state']
                                                       for x, y in zip(a[:count], b[:count])),
                          matching_configured_step_budget=True))
    for row in runs:
        row.pop('prefix')
        row.pop('initial')
    (root / 'timing-and-pairing-audit.json').write_text(json.dumps(dict(runs=runs, pairs=pairs),
                                                                ensure_ascii=False, indent=2) + '\n')
    lines = ['# CloseToasterOvenDoor 跨 seed 结果', '',
             '开发数据；所有运行保留。总耗时包含初始化，Astra 审核和 Flora 与 VLA 异步重叠，组件耗时不能直接相加。', '',
             '| 运行 | 成功 | 步数 VLA/代码 | 总秒数 | 动作执行区间秒数 | Astra turns | 代码尝试/导入/直接调用 |',
             '|---|---|---|---:|---:|---:|---|']
    for r in runs:
        path = Path(r['path']).relative_to(root)
        lines.append(f"| [{path}]({path}/result.json) | {r['result']['success']} | "
                     f"{r['source_steps'].get('vla', 0)}/{r['source_steps'].get('code_policy', 0)} | "
                     f"{r['result']['wall_seconds']:.1f} | {r['motion_wall_seconds']:.1f} | "
                     f"{r['result']['astra_calls']} | {r['program_attempts']}/{len(r['imports'])}/{r['direct_skill_calls']} |")
    lines += ['', '## 固定采样配对', '']
    for p in pairs:
        lines.append(f"- seed {p['seed']}：相同 2000 步预算={p['matching_configured_step_budget']}，"
                     f"初始数组一致={p['initial_arrays_identical']}，比较 {p['prefix_steps']} 步；"
                     f"命令一致={p['commands_identical']}，实测状态一致={p['measured_states_identical']}。")
    lines += ['', '## 各运行推理与程序耗时', '']
    for r in runs:
        lines += [f"### {Path(r['path']).relative_to(root)}", '',
                  '已完成 Astra turns 秒数：`' + json.dumps(r['astra_completed_turn_seconds']) + '`。',
                  f"没有完成事件的 turn 数：{r['astra_turns_without_completion_event']}（可能取消、超时或错误，不能视作零成本）。",
                  f"程序执行共 {r['program_seconds']:.2f} 秒，生成 {r['generated_versions']} 个版本。", '',
                  'Astra 工具：`' + json.dumps(r['tools'], ensure_ascii=False) + '`。', '']
    lines += ['', '逐程序及独立验证：[results.md](results.md)；完整计时与配对：[审计 JSON](timing-and-pairing-audit.json)。',
              '低运动、Flora 评分和阶段信念只请求审核，不能充当成功标签。跨 seed 技能复用效果以实际执行和原生验证为准。']
    (root / 'cross-seed-results.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(episodes=len(runs), pairs=len(pairs)), ensure_ascii=False))


if __name__ == '__main__':
    main()
