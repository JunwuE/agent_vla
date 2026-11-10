"""Report native-horizon and extended-horizon outcomes for a completed atomic sweep."""

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

from report_bounded_comparison import read_episode


def main(root):
    plan = json.loads((root / 'plan.json').read_text())
    end = json.loads((root / 'batch-end.json').read_text())
    expected = {case['key'] for case in plan['cases']}
    if set(end['completed']) != expected or not end['pinned_source_and_initial_memory_unchanged']:
        raise RuntimeError('Require every declared case and unchanged source/initial memory')
    if any(not c['group'].startswith('atomic') for c in plan['cases']):
        raise ValueError('This report is for atomic tasks only')
    horizon = plan['native_horizon']
    budget = plan['config']['planner']
    rows, details = [], []
    for case in plan['cases']:
        data = read_episode(root, case)
        if data is None:
            analysis = json.loads((root / 'analysis' / (case['key'] + '.json')).read_text())
            data = dict(key=case['key'], task=case['task'], seed=case['seed'],
                        result=analysis['result'], first_true={}, source_steps={}, tools={},
                        code_attempts=[], execution_audit={}, max_tools_in_turn=0)
        result = data['result']
        first = data['first_true'].get('task_success', {})
        data['budget_audit'] = dict(
            turns_within_limit=result.get('astra_calls', 0) <= budget['max_calls'],
            tools_within_limit=data['max_tools_in_turn'] <= budget['max_tool_calls'],
            token_stop_threshold=budget['max_total_tokens'],
            reported_tokens=sum(result.get('astra_tokens', {}).values()))
        row = dict(seed=case['seed'], task=case['task'],
            native_success=bool(first and first['step'] <= horizon), success=bool(result.get('success')),
            first_success_step=first.get('step'), reason=result.get('reason'),
            steps=result.get('control_steps', 0), wall_seconds=result.get('wall_seconds', 0),
            astra=result.get('astra_calls', 0), flora=result.get('flora_calls', 0),
            code_attempts=len(data['code_attempts']), code_steps=data['source_steps'].get('code_policy', 0),
            reported_tokens=sum(result.get('astra_tokens', {}).values()))
        rows.append(row)
        details.append(data)
    total_tokens = sum(r['reported_tokens'] for r in rows)
    summary = dict(requested=len(plan['cases']), completed=len(rows), native_horizon=horizon,
        native_successes=sum(r['native_success'] for r in rows),
        extended_successes=sum(r['success'] for r in rows),
        mean_seconds=statistics.mean(r['wall_seconds'] for r in rows),
        median_seconds=statistics.median(r['wall_seconds'] for r in rows),
        astra_turns=sum(r['astra'] for r in rows), flora_calls=sum(r['flora'] for r in rows),
        code_attempts=sum(r['code_attempts'] for r in rows), reported_tokens=total_tokens,
        termination_reasons=dict(Counter(r['reason'] for r in rows)),
        source_and_initial_memory_unchanged=end['pinned_source_and_initial_memory_unchanged'])
    (root / 'atomic-metrics.json').write_text(json.dumps(dict(summary=summary, rows=rows, details=details),
                                                      ensure_ascii=False, indent=2) + '\n')
    with (root / 'per-episode.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tasks = ', '.join(sorted({r['task'] for r in rows}))
    lines = [f"# Atomic {len(rows)}-seed 结果：{tasks}", '',
        f"原生 {horizon} 步内成功 **{summary['native_successes']}/{len(rows)}**；"
        f"延长至 1500 步成功 **{summary['extended_successes']}/{len(rows)}**。",
        f"平均 episode {summary['mean_seconds']/60:.2f} 分钟，中位数 {summary['median_seconds']/60:.2f} 分钟。"
        f"Astra 共 {summary['astra_turns']} turns，Flora {summary['flora_calls']} 次，"
        f"代码尝试 {summary['code_attempts']} 次，上报输入+输出 token {total_tokens:,}。", '',
        'Astra turn 可以包含多次模型/工具往返。上报 token 包含重复上下文，缺少缓存计费拆分，'
        '不能当作实际费用；异步上报也可能超出停止阈值。失败提前结束不是完成任务提速。', '',
        '| seed | 750 步成功 | 最终成功 | 结束原因 | 步数 | 分钟 | Astra | Flora | 代码尝试 | token |',
        '|---:|---|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['seed']} | {row['native_success']} | {row['success']} | {row['reason']} | "
            f"{row['steps']} | {row['wall_seconds']/60:.2f} | {row['astra']} | {row['flora']} | "
            f"{row['code_attempts']} | {row['reported_tokens']:,} |")
    lines += ['', '仅使用 atomic checkpoint；保留原生语言，固定源码和每个 seed 相同初始库。'
        '本批不互传经验，不重试替换失败 seed。组间配对结论请见上级 campaign 报告；本表只汇总本组。',
        f"评测模式：{plan.get('mode', 'agent')}；任务选择来自历史开发结果，不是独立泛化测试。", '',
        '[配置与预算](README.md) · [逐条 CSV](per-episode.csv) · '
        '[完整指标与执行核验](atomic-metrics.json) · [回放索引](report/README.md)。']
    (root / 'atomic-results.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root.resolve())
