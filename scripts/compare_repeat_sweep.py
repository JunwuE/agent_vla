"""Compare completed paired seeds without counting early failures as efficiency gains."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from reconcile_sweep_timing import reconcile


def metrics(records):
    times = [r['result']['wall_seconds'] for r in records]
    successes = sum(r['result']['success'] for r in records)
    edits = ('write_program', 'patch_program', 'patch_program_batch', 'write_skill_package', 'patch_skill_package')
    return dict(
        episodes=len(records), successes=successes,
        within_450=sum(r['result']['success'] and r['result']['control_steps'] <= 450 for r in records),
        sum_seconds=sum(times), mean_seconds=float(np.mean(times)) if times else None,
        median_seconds=float(np.median(times)) if times else None,
        p90_seconds=float(np.quantile(times, .9)) if times else None,
        episode_seconds_per_observed_success=sum(times) / successes if successes else None,
        astra_turns=sum(r['result']['astra_calls'] for r in records),
        astra_revision_seconds=sum(r['astra_completed_turn_seconds'].get('revision', 0)
                                  + r.get('timing_reconciliation', {}).get('reclassified_seconds', 0)
                                  for r in records),
        attempts=sum(r['program_attempts'] for r in records),
        source_edit_calls=sum(sum(r['tool_counts'].get(k, 0) for k in edits) for r in records),
        imports=sum(len(r['imports']) for r in records),
        direct_calls=sum(r['direct_skill_calls'] for r in records),
        code_episodes=sum(bool(r['program_attempts']) for r in records),
        code_episode_successes=sum(bool(r['program_attempts']) and r['result']['success'] for r in records),
        no_code_episode_successes=sum(not r['program_attempts'] and r['result']['success'] for r in records),
        zero_action_attempts=sum(r['zero_action_attempts'] for r in records),
        explicitly_timed_out_planning_seconds=sum(r.get('timing_reconciliation', {}).get('reclassified_seconds', 0)
                                                  for r in records),
        code_steps=sum(r['source_steps'].get('code_policy', 0) for r in records),
        flora_calls=sum(r['result']['flora_calls'] for r in records),
        wall_components={k: sum(r['wall_components_seconds'].get(k, 0) for r in records) for k in (
            'astra_paused', 'code_execution', 'vla_actions', 'vla_inference', 'other_and_unattributed')})


def main(root):
    root = root.resolve()
    plan = json.loads((root / 'plan.json').read_text())
    previous = Path(plan['baseline'])
    old = {int(p.stem[4:]): reconcile(json.loads(p.read_text())) for p in (previous / 'analysis').glob('seed*.json')}
    new = {int(p.stem[4:]): reconcile(json.loads(p.read_text())) for p in (root / 'analysis').glob('seed*.json')}
    seeds = sorted(set(new) & set(old))
    rows = []
    categories = dict(both_success=[], recovered_previous_failure=[], lost_previous_success=[], both_failure=[])
    edits = ('write_program', 'patch_program', 'patch_program_batch', 'write_skill_package', 'patch_skill_package')
    for seed in seeds:
        a, b = old[seed], new[seed]
        ar, br = a['result'], b['result']
        category = ('both_success' if ar['success'] and br['success'] else
                    'recovered_previous_failure' if br['success'] else
                    'lost_previous_success' if ar['success'] else 'both_failure')
        categories[category].append(seed)
        row = dict(seed=seed, category=category)
        for prefix, record in [('old', a), ('new', b)]:
            result = record['result']
            row.update({prefix + '_' + k: v for k, v in dict(
                success=result['success'], seconds=result['wall_seconds'], steps=result['control_steps'],
                astra=result['astra_calls'], code_steps=record['source_steps'].get('code_policy', 0),
                attempts=record['program_attempts'], edits=sum(record['tool_counts'].get(k, 0) for k in edits),
                direct=record['direct_skill_calls'], paused_astra=record['wall_components_seconds']['astra_paused'],
                reason=result['reason']).items()})
        row['seconds_saved'] = row['old_seconds'] - row['new_seconds']
        row['relative_time_change_percent'] = 100 * (row['new_seconds'] / row['old_seconds'] - 1)
        row['speed_comparable_both_success'] = category == 'both_success'
        rows.append(row)
    subsets = dict(completed=seeds, both_success=categories['both_success'],
                   previous_code_attempts=[s for s in seeds if old[s]['program_attempts']],
                   previous_no_code=[s for s in seeds if not old[s]['program_attempts']],
                   both_success_previous_code=[s for s in categories['both_success'] if old[s]['program_attempts']],
                   both_success_previous_no_code=[s for s in categories['both_success'] if not old[s]['program_attempts']])
    aggregates = {key: dict(seeds=values, old=metrics([old[s] for s in values]),
                           new=metrics([new[s] for s in values])) for key, values in subsets.items()}
    total = aggregates['completed']
    result = dict(completed=len(seeds), requested=len(plan['seeds']), categories=categories,
                  aggregates=aggregates, per_seed=rows,
                  note='Paired by seed, policy sampling and task budget. Historical software, memory and service load differ. '
                       'Failures remain failures; speed among jointly successful pairs is reported separately. '
                       'Existing training/evaluation seed exposure prevents a held-out generalization interpretation.')
    if (root / 'comparability.json').exists():
        result['inference_comparability'] = json.loads((root / 'comparability.json').read_text())
    (root / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    if rows:
        with (root / 'comparison.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ['# 第二批 50-seed：已有 Memory 与技能复测', '',
             f'当前完成 **{len(seeds)}/50**；比较仅使用这些相同 seed 的旧结果。', '',
             '固定当前源码和起始 Memory，seed 0–49，策略 sampling seed=42+seed，2000 步，无 Episode 墙钟上限，3 条并发。'
             '每条允许局部代码探索，但本批新经验不进入其他 Episode 的输入。', '',
             '这是已见 seed 的开发复测。旧批次含框架更新、经验积累及不同并发负载，不能把差异单独归因于记忆。'
             '标准 450 步结果单列；失败提前退出不算提速。', '',
             '| 指标（已完成 seed 的配对集合） | 上一批 | 本轮 |', '|---|---:|---:|']
    for label, key in [('成功数', 'successes'), ('450 步内成功数', 'within_450'), ('均值秒', 'mean_seconds'),
                       ('中位秒', 'median_seconds'), ('P90 秒', 'p90_seconds'), ('Astra turns', 'astra_turns'),
                       ('累计 Episode 秒/成功数（含失败开销）', 'episode_seconds_per_observed_success'),
                       ('有明确事件边界的规划超时秒', 'explicitly_timed_out_planning_seconds'),
                       ('代码修订阶段等待秒（含选技能、参数与超时）', 'astra_revision_seconds'),
                       ('代码尝试', 'attempts'), ('源码修改工具调用', 'source_edit_calls'),
                       ('直接技能调用', 'direct_calls'), ('旧程序导入', 'imports'), ('代码执行轨迹数', 'code_episodes')]:
        values = [total[k][key] for k in ('old', 'new')]
        values = [f'{x:.2f}' if isinstance(x, float) else str(x) for x in values]
        lines.append(f'| {label} | {values[0]} | {values[1]} |')
    labels = dict(both_success='双方成功', recovered_previous_failure='旧失败→本轮成功',
                  lost_previous_success='旧成功→本轮失败', both_failure='双方失败')
    lines += ['', '## 结果变化', '']
    lines += [f'- {labels[key]}：{values}。' for key, values in categories.items()]
    both = aggregates['both_success']
    if both['seeds']:
        change = 100 * (both['new']['sum_seconds'] / both['old']['sum_seconds'] - 1)
        lines += ['', f'双方成功的 {len(both["seeds"])} 个配对 seed：总 Episode 时间变化 **{change:+.1f}%**；'
                  f'均值 {both["old"]["mean_seconds"]:.2f} → {both["new"]["mean_seconds"]:.2f} 秒。'
                  '该子集存在结果选择效应，需连同成功数变化阅读。']
    lines += ['', '## 按上一批是否需要代码恢复分组', '',
              '分组依据在本轮结果出来前已经确定。组内均值包含失败，必须同时查看成功数；'
              '下面另列双方都成功的子集。', '',
              '| 组别 | 数量 | 成功旧→新 | 平均分钟旧→新 | 修改次数旧→新 |',
              '|---|---:|---:|---:|---:|']
    for key, label in [
        ('previous_code_attempts', '上一批需要代码'),
        ('previous_no_code', '上一批无需代码'),
        ('both_success_previous_code', '上一批需要代码且双方成功'),
        ('both_success_previous_no_code', '上一批无需代码且双方成功'),
    ]:
        group = aggregates[key]
        if not group['seeds']:
            continue
        a, b = group['old'], group['new']
        lines.append(f'| {label} | {len(group["seeds"])} | {a["successes"]} → {b["successes"]} | '
                     f'{a["mean_seconds"]/60:.2f} → {b["mean_seconds"]/60:.2f} | '
                     f'{a["source_edit_calls"]} → {b["source_edit_calls"]} |')
    lines += ['', '## 逐 seed 结果', '',
              '| seed | 旧→新结果 | 旧→新分钟 | 时间差秒（旧减新） | Astra 旧→新 | 尝试 旧→新 | 改码 旧→新 | 直接调用 旧→新 |',
              '|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| [{r["seed"]}](analysis/seed{r["seed"]:02d}.md) | {labels[r["category"]]} | '
                     f'{r["old_seconds"]/60:.2f} → {r["new_seconds"]/60:.2f} | {r["seconds_saved"]:+.1f} | '
                     f'{r["old_astra"]} → {r["new_astra"]} | {r["old_attempts"]} → {r["new_attempts"]} | '
                     f'{r["old_edits"]} → {r["new_edits"]} | {r["old_direct"]} → {r["new_direct"]} |')
    lines += ['', '## 主链路时间累计', '',
              '这些是 Episode 时间之和，不是并发整批墙钟。Flora 和后台审核与动作重叠，不能额外累加。', '',
              '| 分项 | 旧秒 | 新秒 |', '|---|---:|---:|']
    for key, label in [('astra_paused', '暂停 Astra'), ('code_execution', '代码程序执行'),
                       ('vla_actions', 'VLA 动作确认'), ('vla_inference', 'VLA 推理'),
                       ('other_and_unattributed', '其余')]:
        lines.append(f'| {label} | {total["old"]["wall_components"][key]:.2f} | {total["new"]["wall_components"][key]:.2f} |')
    lines += ['', '[机器可读比较](comparison.json) · [CSV](comparison.csv) · [运行计划](plan.json)。'
              '源码、输入 Memory、每条结果、失败程序与回执保留；不以重跑覆盖首次结果。']
    if 'inference_comparability' in result:
        lines += ['', '**跨服务复现限制：** 初始 RGB-D/状态、权重、指令、采样种子一致，'
                  '当前服务同输入重复推理一致，但抽查首个动作 Chunk 与历史服务存在小幅数值差异。'
                  '因此不能声称当前与历史轨迹拥有完全一致的动作前缀，不能将全部变化归因于 Memory。'
                  '原因尚未证实；见 [复现检查](comparability.json)。']
    (root / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(completed=len(seeds), old_success=total['old']['successes'],
                          new_success=total['new']['successes'], categories=categories), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
