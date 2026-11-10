"""Aggregate composite episodes without treating phase beliefs as success labels."""

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from run_seed_sweep import write_json


def wilson(successes, total):
    if not total:
        return None
    z = 1.96
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [max(0.0, center-radius), min(1.0, center+radius)]


def episode_details(root, data):
    key = f"{data['task']}-seed{data['seed']:02d}"
    outcome = data['result']
    lines = [f'# {key}', '',
             f"原生整任务成功：{outcome.get('success', False)}；结束原因：{outcome.get('reason')}；"
             f"{outcome.get('control_steps', 0)} 步；{outcome.get('wall_seconds', 0)/60:.2f} 分钟。", '',
             '## 时间与输入', '',
             '主链路分项（秒）：`'+json.dumps(data.get('wall_components_seconds'), ensure_ascii=False)+'`。', '',
             'Flora 与后台审核可和动作重叠，不额外加到主链路分项。未结束的 Astra turn 不计入已完成推理时长，'
             '其时间可能位于 other；原始总时间不变。', '',
             '执行回执审计：`'+json.dumps(data.get('execution_audit'), ensure_ascii=False)+'`。', '',
             '实际 VLA 指令及对应动作数：`'+json.dumps(data.get('vla_instruction_counts'), ensure_ascii=False)+'`。', '',
             '## 程序尝试', '',
             '| 版本 | 动作步 | 执行秒 | 上次结果至本次启动秒 | 独立验证 | 程序停止理由 |',
             '|---|---:|---:|---:|---|---|']
    for item in data.get('recoveries', []):
        gap = item.get('revision_and_setup_seconds')
        gap_text = f'{gap:.2f}' if gap is not None else '—'
        reason = str(item.get('error') or item.get('last_program_facts') or '').replace('|', '/').replace('\n', ' ')
        lines.append(f"| [{item['program_version']}](../episodes/{key}/{item['policy_path']}) | "
                     f"{item['steps']} | {item['seconds']:.2f} | {gap_text} | {item['verified']} | {reason} |")
    lines += ['', '这些停止理由是程序/模型报告，不自动认定为物理根因。部分 checkpoint 表示局部推进后请求新观测。', '',
              '## 工具与最终判断', '',
              'Astra 工具次数：`'+json.dumps(data.get('tool_counts'), ensure_ascii=False)+'`。', '',
              '工具错误：`'+json.dumps(data.get('tool_errors'), ensure_ascii=False)+'`。', '',
              '最终停止记录：']
    for stop in data.get('stop_events', [])[-3:]:
        lines += ['', f"- event {stop['event_id']}：{stop['reason']}"]
    lines += ['', '阶段完成记录均为 Astra 信念，不是原生子任务成功标签。', '',
              f'[完整分析 JSON]({key}.json)；[原始结果](../episodes/{key}/result.json)；'
              f'[原始事件](../episodes/{key}/events.jsonl)。']
    (root/'analysis'/f'{key}.md').write_text('\n'.join(lines)+'\n')


def skill_inventory(root, records):
    """Index immutable evidence, without promoting or injecting new candidates."""
    inventory = []
    initial = root/'initial'/'skills'
    for data in records:
        key = f"{data['task']}-seed{data['seed']:02d}"
        library = root/'inputs'/key/'skills'
        for path in sorted(library.glob('*/*/manifest.json')):
            if (initial/path.relative_to(library)).exists():
                continue
            manifest = json.loads(path.read_text())
            inventory.append(dict(task=data['task'], seed=data['seed'], id=manifest['id'],
                                  version=manifest['version'], status=manifest['status'],
                                  manifest=str(path.relative_to(root)),
                                  contract=str(path.with_name('SKILL.md').relative_to(root))
                                  if path.with_name('SKILL.md').exists() else str(path.relative_to(root)),
                                  code=str(path.with_name('policy.py').relative_to(root)),
                                  package_files={name: str((path.parent/name).relative_to(root))
                                                 for name in (manifest.get('package') or {}).get('files', {})},
                                  sha256=manifest['sha256'],
                                  full_task_verified=any(e.get('report', {}).get('verified') is True
                                      for e in manifest.get('evidence', []))))
    write_json(root/'skill-inventory.json', dict(records=inventory, promoted_to_default=False,
        note='New candidate versions only; original packages and evidence stay in private libraries. '
             'Task failure does not prove every local manipulation failed. No stage-level promotion inferred.'))
    lines = ['# 本批新增代码索引', '',
             '仅索引已结束 Episode 的新增候选版本，不自动写入默认技能库；原始前置条件、程序、依赖和运行证据仍在各私有库。', '',
             '| 任务 / seed | 技能 | 版本 | 状态 | 整任务验证 |', '|---|---|---|---|---|']
    for item in inventory:
        lines.append(f"| {item['task']} / {item['seed']} | [{item['id']}]({item['contract']}) | "
                     f"[{item['version']}]({item['code']}) | {item['status']} | "
                     f"[{item['full_task_verified']}]({item['manifest']}) |")
    (root/'skill-inventory.md').write_text('\n'.join(lines)+'\n')
    return inventory


def build(root):
    plan = json.loads((root/'plan.json').read_text())
    records = [json.loads(p.read_text()) for p in sorted((root/'analysis').glob('*.json'))]
    groups, rows = {}, []
    for data in records:
        episode_details(root, data)
        result = data['result']
        phases = data.get('last_accepted_phase_memory') or {}
        observed_complete = set()
        for update in data.get('accepted_phase_updates', []):
            for assessment in update['phase_memory'].get('recent_assessments', []):
                if assessment['status'] == 'complete':
                    observed_complete.add(assessment['phase_id'])
        rows.append(dict(
            task=data['task'], group=data['group'], seed=data['seed'],
            success=result.get('success', False), reason=result.get('reason'),
            steps=result.get('control_steps', 0), seconds=result.get('wall_seconds', 0),
            astra_turns=result.get('astra_calls', 0),
            code_attempts=data.get('program_attempts', 0),
            direct_skill_calls=data.get('direct_skill_calls', 0),
            source_edit_calls=sum(data.get('tool_counts', {}).get(name, 0) for name in
                             ('write_program', 'patch_program', 'patch_program_batch',
                              'write_skill_package', 'patch_skill_package', 'edit_program')),
            vla_steps=data.get('source_steps', {}).get('vla', 0),
            code_steps=data.get('source_steps', {}).get('code_policy', 0),
            astra_paused_seconds=data.get('wall_components_seconds', {}).get('astra_paused', 0),
            code_seconds=data.get('program_seconds', 0),
            flora_completed=data.get('flora_latency_seconds', {}).get('count', 0),
            last_believed_phase=phases.get('active_phase'),
            phases_ever_believed_complete=len(observed_complete),
            ever_believed_complete_ids=sorted(observed_complete),
            discarded_vla_steps=data.get('discarded_vla_queue_steps', 0),
            distinct_vla_instructions=len(data.get('vla_instruction_counts', {})),
            refine_proposals=len(data.get('refine_proposals', [])),
            unfinished_astra_turns=data.get('astra_turns_without_completion_event', 0),
            context_compactions=data.get('astra_context_compactions', 0),
            max_context_chars=max((s['total_chars'] for s in data.get('context_sizes', [])), default=0),
            tool_errors=len(data.get('tool_errors', [])),
            initialization_seconds=data.get('astra_completed_turn_seconds', {}).get('initialization', 0),
            revision_seconds=data.get('astra_completed_turn_seconds', {}).get('revision', 0),
            mean_flora_seconds=data.get('flora_latency_seconds', {}).get('mean', 0),
            mean_vla_seconds=data.get('vla_latency_seconds', {}).get('mean', 0),
            reported_input_tokens=result.get('astra_tokens', {}).get('input_tokens', 0),
            reported_output_tokens=result.get('astra_tokens', {}).get('output_tokens', 0),
            audit_all_passed=all(data.get('execution_audit', {}).values()) if data.get('execution_audit') else None,
        ))
    for spec in plan['tasks']:
        selected = [r for r in rows if r['task'] == spec['task']]
        successes = sum(r['success'] for r in selected)
        groups[spec['task']] = dict(
            group=spec['group'], completed=len(selected), requested=len(plan['seeds']),
            successes=successes, success_rate=successes/len(selected) if selected else None,
            wilson95=wilson(successes, len(selected)),
            mean_seconds=statistics.mean(r['seconds'] for r in selected) if selected else None,
            median_seconds=statistics.median(r['seconds'] for r in selected) if selected else None,
            total_astra_turns=sum(r['astra_turns'] for r in selected),
            code_episodes=sum(r['code_attempts'] > 0 for r in selected),
            reasons=dict(Counter(r['reason'] for r in selected)),
            active_phase_beliefs=dict(Counter(r['last_believed_phase'] for r in selected)),
            mean_astra_turns=statistics.mean(r['astra_turns'] for r in selected) if selected else None,
            mean_code_attempts=statistics.mean(r['code_attempts'] for r in selected) if selected else None,
            astra_paused_fraction=sum(r['astra_paused_seconds'] for r in selected)/sum(r['seconds'] for r in selected)
            if selected and sum(r['seconds'] for r in selected) else None,
            code_fraction=sum(r['code_seconds'] for r in selected)/sum(r['seconds'] for r in selected)
            if selected and sum(r['seconds'] for r in selected) else None,
        )
    inventory = skill_inventory(root, records)
    batch_seconds = None
    if (root/'batch-end.json').exists():
        finish = json.loads((root/'batch-end.json').read_text())
        begin = json.loads((root/'batch-start.json').read_text())
        batch_seconds = (finish['finished_ns']-begin['started_ns'])/1e9
    write_json(root/'aggregate.json', dict(groups=groups, rows=rows, batch_wall_seconds=batch_seconds,
        reported_stop_counts=dict(Counter(a.get('error') or 'no_error_message'
                                          for d in records for a in d.get('recoveries', []))),
        source_steps=dict(sum((Counter(d.get('source_steps', {})) for d in records), Counter())),
        versioned_skill_direct_calls=sum(d.get('direct_skill_calls', 0) for d in records),
        versioned_skill_imports=sum(len(d.get('imports', [])) for d in records),
        limitations=['Two tasks, 20 seeds each; not a whole benchmark score.',
                     'No paired pure-VLA batch; cannot attribute improvement to the agent or compaction.',
                     'Phase beliefs are not independent subtask verifications.',
                     'Atomic checkpoint; official seen/unseen labels do not audit its training exposure.',
                     'Identical starting memory; this is not an online cross-seed learning curve.']))
    if rows:
        with (root/'metrics.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ['# 长程任务评测结果', '',
             f"已完成 {len(rows)}/{len(plan['cases'])} 条预先指定的 Episode。", '',
             '| 任务 | 类别 | 原生成功 | 平均分钟 | 中位分钟 | 平均 Astra turns | 平均代码尝试 | 使用代码的 Episode |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for task, group in groups.items():
        lines.append(f"| {task} | {group['group']} | {group['successes']}/{group['completed']} | "
                     f"{(group['mean_seconds'] or 0)/60:.2f} | {(group['median_seconds'] or 0)/60:.2f} | "
                     f"{(group['mean_astra_turns'] or 0):.2f} | {(group['mean_code_attempts'] or 0):.2f} | "
                     f"{group['code_episodes']} |")
    lines += ['', '主要发现与改进方向见 [详细分析](interpretation.md)，完整性检查见 [执行审计](audit.json)。',
              '', '全部单条结果见 [results.md](results.md)，可分析数据见 [metrics.csv](metrics.csv) 与 [aggregate.json](aggregate.json)。',
              '', '## 条件与边界', '',
              '- 同一 OmniSF atomic checkpoint 59999，π0.5；Flora 每 50 个 VLA 步监督，Astra high。',
              '- SteamInMicrowave / WaffleReheat 的动作预算为 2100 / 4050；无整条 Episode 墙钟超时。',
              '- 每个 seed 的起始 Memory 与源码一致；新经验留在私有库，不传给本批其他 seed。',
              '- 调度先在无完成结果时从 4 路扩为 8 路，完成 7 条后扩为 16 路；原有场景原样续跑。'
              '运行策略不变，但共享服务负载和异步审核返回时机可能变化，耗时不能视为独占硬件延迟或学习曲线。',
              '- 原生整任务 instruction 输入 VLA，Astra 的细阶段用于观察与诊断。当前只有 task_success 独立验证器。',
              '- 阶段 believed complete、Flora 高分、程序自报成功均不能代替原生任务成功。',
              '- 平均时间包含失败与提前停止；更短的失败 Episode 不代表完成任务更快。',
              '- Token 为模型报告的累计输入/输出，包含多轮反复上下文；未分解缓存命中，不能直接换算实际计费。',
              '- Astra 时间统计完整规划 turn，包含工具往返与等待；不是模型单次前向延迟。',
              '- 本批没有同服务纯 VLA 对照，不据此声明 Agent 增益或 Memory 压缩带来的成功率变化。',
              '- 官方 Seen / Unseen 表示任务划分，不代表已审计当前 checkpoint 的全部训练数据。',
              '', '## Memory 压缩', '',
              '旧 50 条首次输入的同输入离线回放：历史 52,619→5,937 字符，用户上下文 57,668.56→11,304.56 字符。',
              '保留原始经验、反例、代码及证据；使用索引与按需读取，没有新增总结模型。该结果不等于延迟或任务能力提升。',
              '见 [压缩对照](memory-context-comparison.json) 与 [实现说明](../../docs/memory-compaction.md)。']
    lines += ['', '## 可视化', '',
              '- [SteamInMicrowave seed 0 回放](report/SteamInMicrowave-seed00-overview/index.html) '
              '／[MP4](report/SteamInMicrowave-seed00-overview/replay.mp4)',
              '- [WaffleReheat seed 0 回放](report/WaffleReheat-seed00-overview/index.html) '
              '／[MP4](report/WaffleReheat-seed00-overview/replay.mp4)', '',
              '回放保留全部物理动作帧；推理等待压缩、重复底层工具卡片折叠，原始墙钟时间仍在画面显示。'
              '阶段标题是日志执行阶段，不是独立成功标签。', '',
              '![逐 seed 主链路耗时](timing.png)', '',
              '[调用次数图](calls.png)；[耗时 PDF](timing.pdf)；[调用 PDF](calls.pdf)。']
    lines += ['', '## 代码与逐条证据', '',
              f'保存新增候选版本记录 {len(inventory)} 条；见 [代码索引](skill-inventory.md)。'
              '没有将局部推进自动晋升为整任务成功技能。', '',
              '每条分析包含程序版本、动作步数、修订间隔、停止理由和执行审计：', '']
    for spec in plan['tasks']:
        retained = root/'retained'/spec['task']
        if (retained/'retention.json').exists():
            receipt = json.loads((retained/'retention.json').read_text())
            lines.append(f"- [{spec['task']} 候选库](retained/{spec['task']}/INDEX.md)："
                         f"合并后 {receipt['new_skill_versions']} 个新增候选版本，"
                         f"{receipt['new_experiences']} 条新经验；未改变默认库，尚未进行合并后复用评测。")
    lines.append('')
    for data in records:
        key = f"{data['task']}-seed{data['seed']:02d}"
        lines.append(f'- [{key}](analysis/{key}.md)')
    (root/'findings.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(groups, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    build(parser.parse_args().root.resolve())
