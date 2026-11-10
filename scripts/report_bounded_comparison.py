"""Summarize actual costs and verified milestones across bounded comparison runs."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def read_episode(root, case):
    episode = root / 'episodes' / case['key']
    result_path = episode / 'result.json'
    if not result_path.is_file():
        return None
    result = json.loads(result_path.read_text())
    analysis_path = root / 'analysis' / (case['key'] + '.json')
    analysis = json.loads(analysis_path.read_text()) if analysis_path.exists() else {}
    first_true, last_checks, tools, turn_tools, transitions = {}, {}, Counter(), Counter(), []
    first_obs, last_obs, initial_plan = None, None, None
    failures, turns, refinements = [], {}, []
    sources, settings = Counter(), set()
    for line in (episode / 'events.jsonl').open():
        e = json.loads(line)
        kind = e['kind']
        if kind == 'observation':
            first_obs = first_obs or e
            last_obs = e
            last_checks = dict(e.get('facts', {}).get('subtask_checks', {}), task_success=e['success'])
            for key, value in last_checks.items():
                if value is True and key not in first_true:
                    first_true[key] = dict(step=e['step'], seconds=(e['time_ns']-first_obs['time_ns'])/1e9)
        elif kind == 'action_executed':
            sources[e['source']] += 1
        elif kind == 'codex_thread':
            settings.add(f"{e['model']}:{e['reasoning_effort']}")
        elif kind == 'codex_turn_start':
            turns[e['turn_id']] = dict(intent=e['context']['intent'], start_ns=e['time_ns'])
        elif kind == 'codex_turn_end':
            turns[e['turn_id']].update(seconds=(e['time_ns']-turns[e['turn_id']]['start_ns'])/1e9,
                                      status=e['status'])
        elif kind == 'codex_tool_start':
            turn_tools[e['turn_id']] += 1
        elif kind == 'codex_tool':
            tools[e['name']] += 1
        elif kind == 'subtask_transition':
            transitions.append(dict(completed=e['completed']['name'],
                                    step=e['verification']['observation_step']))
        elif kind == 'intervention_plan':
            if initial_plan is None and e['plan'].get('remaining_plan'):
                initial_plan = e['plan']['remaining_plan']
            if e['plan']['kind'] == 'refine':
                refinements.append(dict(step=e['binding']['observation_step'], plan=e['plan']))
        elif kind == 'recovery_result':
            failures.append({k: e.get(k) for k in ('program_version', 'executed_steps', 'seconds',
                                                  'verified', 'error', 'failed_stage')})
    return dict(key=case['key'], root=str(root), episode=str(episode), task=case['task'], seed=case['seed'],
        result=result, source_steps=dict(sources), first_true=first_true, final_checks=last_checks,
        completed_subtasks=transitions, initial_plan=initial_plan, refinements=refinements,
        code_attempts=failures, tools=dict(tools), max_tools_in_turn=max(turn_tools.values(), default=0),
        threads=sorted(settings), turns=list(turns.values()),
        first_observation=first_obs, last_observation=last_obs,
        wall_components_seconds=analysis.get('wall_components_seconds', {}),
        execution_audit=analysis.get('execution_audit', {}))


def main(output, roots):
    output.mkdir(parents=True, exist_ok=True)
    groups, rows = {}, []
    for name in roots:
        root = name.resolve()
        plan = json.loads((root / 'plan.json').read_text())
        selected = [r for c in plan['cases'] if (r := read_episode(root, c)) is not None]
        budget = plan['config']['planner']
        for r in selected:
            r['group'] = root.name
            r['budget_audit'] = dict(turns_within_limit=r['result']['astra_calls'] <= budget['max_calls'],
                tools_within_limit=r['max_tools_in_turn'] <= budget['max_tool_calls'],
                reported_token_threshold=budget.get('max_total_tokens'),
                reported_tokens=sum(r['result'].get('astra_tokens', {}).values()))
        groups[root.name] = dict(root=str(root), requested=len(plan['cases']), completed=len(selected),
            successes=sum(r['result']['success'] for r in selected),
            astra_turns=sum(r['result']['astra_calls'] for r in selected),
            reported_tokens=sum(sum(r['result'].get('astra_tokens', {}).values()) for r in selected),
            mean_seconds=sum(r['result']['wall_seconds'] for r in selected)/len(selected) if selected else None)
        rows.extend(selected)
    pairing = []
    for a in rows:
        if not a['first_observation']:
            continue
        for b in rows:
            if a['group'] >= b['group'] or (a['task'], a['seed']) != (b['task'], b['seed']):
                continue
            if not b['first_observation']:
                continue
            with np.load(Path(a['episode']) / a['first_observation']['path']) as x, \
                    np.load(Path(b['episode']) / b['first_observation']['path']) as y:
                match = set(x.files) == set(y.files) and all(np.array_equal(x[k], y[k]) for k in x.files)
            pairing.append(dict(task=a['task'], seed=a['seed'], groups=[a['group'], b['group']],
                                initial_arrays_identical=match))
    (output / 'metrics.json').write_text(json.dumps(dict(groups=groups, rows=rows, pairing=pairing),
                                                   ensure_ascii=False, indent=2)+'\n')
    table = []
    for r in rows:
        result = r['result']
        table.append(dict(group=r['group'], task=r['task'], seed=r['seed'], success=result['success'],
            reason=result['reason'], steps=result['control_steps'], minutes=result['wall_seconds']/60,
            astra=result['astra_calls'], flora=result['flora_calls'], code_attempts=len(r['code_attempts']),
            tokens=sum(result.get('astra_tokens', {}).values()),
            completed_subtasks=','.join(x['completed'] for x in r['completed_subtasks']),
            first_true=json.dumps(r['first_true'], ensure_ascii=False),
            final_checks=json.dumps(r['final_checks'], ensure_ascii=False)))
    if table:
        with (output / 'per-episode.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    lines = ['# 有预算的微波炉与 Composite-Seen 对照', '',
        '成功以仿真原生判据为准；子任务布尔判据是仿真验证辅助。耗时含初始化。',
        'Astra 次数是 Codex turn；每个 turn 可以有多次模型/工具往返。Token 为上报输入+输出，'
        '包含重复上下文，缺少缓存计费拆分，不等于费用。预算提前结束不能当作完成任务提速。', '',
        '| 组 | 完成 | 成功 | 平均分钟 | Astra | 上报 token |', '|---|---:|---:|---:|---:|---:|']
    for name, g in groups.items():
        mean = f"{g['mean_seconds']/60:.2f}" if g['mean_seconds'] is not None else '—'
        lines.append(f"| {name} | {g['completed']}/{g['requested']} | {g['successes']} | {mean} | "
                     f"{g['astra_turns']} | {g['reported_tokens']:,} |")
    lines += ['', '| 组 / seed | 成功 / 结束原因 | 步数 | 分钟 | Astra / Flora | 代码尝试 | 完成子任务 |',
              '|---|---|---:|---:|---:|---:|---|']
    for r in table:
        lines.append(f"| {r['group']} / {r['seed']} | {r['success']} / {r['reason']} | {r['steps']} | "
            f"{r['minutes']:.2f} | {r['astra']} / {r['flora']} | {r['code_attempts']} | {r['completed_subtasks'] or '无'} |")
    lines += ['', '初始 RGB、深度和机器人状态数组配对：`'+json.dumps(pairing, ensure_ascii=False)+'`。', '',
        '[完整指标与阶段证据](metrics.json) · [逐轨迹 CSV](per-episode.csv)。', '',
        '咖啡原主指令对照移除了规划、监督和恢复，不能仅凭组间差异归因于规划。'
        '微波炉新旧配置的预算不同，应比较相同已验证阶段，不能直接解释为架构的净速度收益。']
    (output / 'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(groups, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('roots', type=Path, nargs='+')
    args = parser.parse_args()
    main(args.output.resolve(), args.roots)
