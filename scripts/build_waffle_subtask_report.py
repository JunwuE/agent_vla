"""Report actual executable goals, local handoffs, context sizes and native success."""

import argparse
import csv
import hashlib
import html
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
from run_composite_sweep import collect, summarize
from run_seed_sweep import write_json


def mean(values):
    return statistics.mean(values) if values else None


def inspect_episode(root, spec, reference):
    path = root/'episodes'/spec['key']
    base = json.loads((root/'analysis'/f"{spec['key']}.json").read_text())
    plans, transitions, contexts, stops, contracts, code_done, rotations = [], [], [], [], [], [], []
    first_checks, code_goals, first_obs = {}, Counter(), None
    observed_checks, flora_scores = {}, []
    sessions = {}
    model_errors = []
    with (path/'events.jsonl').open() as stream:
        for line in stream:
            event = json.loads(line)
            kind = event['kind']
            if kind == 'observation':
                first_obs = first_obs or event
                observed_checks[event['step']] = dict(event.get('facts', {}).get('subtask_checks', {}),
                                                      task_success=event['success'])
                for condition, passed in event.get('facts', {}).get('subtask_checks', {}).items():
                    if passed is True:
                        first_checks.setdefault(condition, event['step'])
            elif kind == 'intervention_plan' and event['plan']['kind'] == 'replan':
                plans.append(dict(step=event['binding']['observation_step'], plan=event['plan']))
            elif kind == 'subtask_transition':
                transitions.append(event)
            elif kind == 'working_context_index':
                contexts.append(event)
            elif kind == 'codex_thread_rotated':
                rotations.append(event)
            elif kind == 'code_subtask_complete':
                code_done.append(event)
                if event['session_id'] in sessions:
                    sessions[event['session_id']].update(verified=True, end_ns=event['time_ns'])
            elif kind == 'skill_contract_adapted':
                contracts.append(event)
            elif kind == 'stop':
                stops.append(event)
            elif kind == 'codex_error':
                model_errors.append(event)
            elif kind == 'code_version':
                code_goals[event['completion_condition']] += 1
                identity = event['intervention_id'].rsplit('-', 1)[0]
                if identity in sessions:
                    sessions[identity]['attempts'] += 1
            elif kind == 'executor_assignment' and event.get('executor') == 'code':
                assigned = event['assignment']
                sessions[assigned['session_id']] = dict(
                    goal=assigned['subtask']['completion_condition'], start_step=assigned['start_step'],
                    start_ns=event['time_ns'], end_ns=event['time_ns'], steps=0, attempts=0, verified=False)
            elif kind == 'recovery_result':
                identity = event['intervention_id'].rsplit('-', 1)[0]
                if identity in sessions:
                    sessions[identity]['steps'] += event['executed_steps']
                    sessions[identity]['end_ns'] = event['time_ns']
            elif kind == 'flora_result':
                scored = event['result']
                if scored.get('status') == 'scored' and scored.get('scope', {}).get('kind') == 'subtask':
                    at = scored['binding']['observation_step']
                    condition = scored.get('completion_condition')
                    flora_scores.append(dict(step=at, score=scored['score'], completion_condition=condition,
                        completion_at_snapshot=observed_checks.get(at, {}).get(condition),
                        request_id=scored.get('request_id')))
    old_path = reference/'episodes'/spec['key']
    old = json.loads((reference/'analysis'/f"{spec['key']}.json").read_text())
    old_first = None
    with (old_path/'events.jsonl').open() as stream:
        for line in stream:
            event = json.loads(line)
            if event['kind'] == 'observation':
                old_first = event
                break
    # Compare arrays, not timestamp/episode identity or newly added boolean facts.
    def observation_arrays(directory, event):
        key = next((event[k] for k in ('path', 'file', 'artifact') if k in event), None)
        if key is None:
            key = next(p.relative_to(directory) for p in (directory/'observations').glob('*.npz')
                       if json.loads(p.with_suffix('.json').read_text())['step'] == 0)
        return np.load(directory/key)
    comparison = {}
    if first_obs and old_first:
        with observation_arrays(path, first_obs) as new, observation_arrays(old_path, old_first) as previous:
            comparison = {k: bool(np.array_equal(new[k], previous[k])) for k in new.files if k in previous.files}
    result = base['result']
    quota_interrupted = any('usageLimitExceeded' in json.dumps(e) for e in model_errors)
    termination_class = ('quota_interrupted' if quota_interrupted else
                         'runtime_error' if result['reason'] == 'runtime_error' else
                         'success' if result['success'] else 'policy_failure')
    for session in sessions.values():
        session['minutes'] = (session['end_ns']-session['start_ns'])/60e9
    opening_handoffs = []
    for i, transition in enumerate(transitions):
        if transition['completed']['completion_condition'] != 'microwave_open':
            continue
        start = transition['verification']['observation_step']
        end = (transitions[i + 1]['verification']['observation_step']
               if i + 1 < len(transitions) else result['control_steps'])
        lost_at = next((step for step, checks in sorted(observed_checks.items())
                        if start < step <= end and checks.get('microwave_open') is False), None)
        opening_handoffs.append(dict(verified_step=start, next_transition_or_end_step=end,
            first_subsequent_open_check_false=lost_at,
            note='Boolean regression, not proof of full closure or the physical cause.'))
    row = dict(seed=spec['seed'], success=result['success'], reason=result['reason'],
        termination_class=termination_class,
        native_success_first_step=base['native_success_first_step'], steps=result['control_steps'],
        minutes=result['wall_seconds']/60, astra_turns=result['astra_calls'],
        astra_input_tokens=result.get('astra_tokens', {}).get('input_tokens', 0),
        astra_output_tokens=result.get('astra_tokens', {}).get('output_tokens', 0),
        tool_calls=sum(base['tool_counts'].values()),
        code_attempts=base['program_attempts'], local_transitions=len(transitions),
        generated_versions=base['generated_versions'],
        unique_completed_subtasks=len({x['completed']['completion_condition'] for x in transitions}),
        code_completed_subtasks=len(code_done), vla_instructions=len(base['vla_instruction_counts']),
        max_context_chars=max((x['total_chars'] for x in base['context_sizes']), default=0),
        mean_context_chars=mean([x['total_chars'] for x in base['context_sizes']]),
        mean_before_working_index=mean([x['original_chars'] for x in contexts]),
        mean_after_working_index=mean([x['indexed_chars'] for x in contexts]),
        thread_rotations=len(rotations), imports=len(base.get('imports', [])),
        direct_skill_calls=base.get('direct_skill_calls', 0), adapted_contracts=len(contracts),
        high_flora_without_completion=sum(x['score'] >= 4 and x['completion_at_snapshot'] is False
                                          for x in flora_scores),
        astra_paused_minutes=base.get('wall_components_seconds', {}).get('astra_paused', 0)/60,
        vla_steps=base['source_steps'].get('vla', 0),
        code_steps=base['source_steps'].get('code_policy', 0),
        zero_action_attempts=base['zero_action_attempts'],
        flora_calls=result.get('flora_calls', sum(base['flora_statuses'].values())),
        success_within_previous_4050_steps=bool(result['success'] and
            base['native_success_first_step'] is not None and base['native_success_first_step'] <= 4050),
        old_success=old['result']['success'], old_minutes=old['result']['wall_seconds']/60,
        old_astra_turns=old['result']['astra_calls'], old_code_attempts=old['program_attempts'])
    details = dict(row=row, plans=plans, transitions=transitions, code_subtask_completions=code_done,
        completed_subtask_counts=dict(Counter(x['completed']['completion_condition'] for x in transitions)),
        code_sessions=sessions,
        opening_handoff_checks=opening_handoffs,
        local_checks_first_true=first_checks, actual_vla_instructions=base['vla_instruction_counts'],
        code_attempts_by_condition=dict(code_goals), stop_events=stops, model_errors=model_errors,
        initial_array_comparison=comparison,
        execution_audit=base['execution_audit'], contract_adaptations=contracts,
        wall_components_seconds=base['wall_components_seconds'],
        tool_counts=base['tool_counts'], recoveries=base['recoveries'],
        decision_labels=base['decision_labels'], flora_statuses=base['flora_statuses'],
        program_reported_stops=dict(Counter(r.get('error') or 'no_error_reported'
                                            for r in base['recoveries'])),
        flora_subtask_scores=flora_scores, tool_errors=base.get('tool_errors', []),
        working_context_samples=contexts, old_max_context_chars=max(
            (x['total_chars'] for x in old.get('context_sizes', [])), default=0))
    write_json(root/'subtask-analysis'/f"{spec['key']}.json", details)
    return details


def build(root):
    plan = json.loads((root/'plan.json').read_text())
    # Keep supplemental derived JSON outside the scheduler's analysis/*.json set.
    supplementary = root/'subtask-analysis'
    supplementary.mkdir(exist_ok=True)
    records = []
    for spec in plan['cases']:
        if not (root/'episodes'/spec['key']/'result.json').exists():
            continue
        collect(root, spec, plan, refresh_summary=False)
        details = inspect_episode(root, spec, Path(plan['reference_batch']))
        records.append(details)
    summarize(root, plan)
    rows = [d['row'] for d in records]
    if rows:
        with (root/'metrics.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    audit = dict(completed=len(rows), requested=len(plan['cases']),
                 all_execution_checks_passed=all(all(d['execution_audit'].values()) for d in records),
                 native_successes=sum(r['success'] for r in rows),
                 quota_interrupted=sum(r['termination_class'] == 'quota_interrupted' for r in rows),
                 policy_failures=sum(r['termination_class'] == 'policy_failure' for r in rows),
                 source_unchanged=all(hashlib.sha256((Path(plan['source'])/p).read_bytes()).hexdigest()==digest
                     for p,digest in json.loads((Path(plan['source']).parent/'source-hashes.json').read_text()).items()),
                 initial_memory_unchanged=all(hashlib.sha256((root/'initial'/p).read_bytes()).hexdigest()==digest
                     for p,digest in json.loads((root/'initial-hashes.json').read_text()).items()))
    write_json(root/'audit.json', audit)
    batch_minutes = None
    if (root/'batch-end.json').exists():
        batch_minutes = (json.loads((root/'batch-end.json').read_text())['finished_ns'] -
                         json.loads((root/'batch-start.json').read_text())['started_ns'])/60e9
    write_json(root/'aggregate.json', dict(rows=rows, audit=audit, limitations=plan['limitations'],
        batch_minutes=batch_minutes, total_astra_turns=sum(r['astra_turns'] for r in rows),
        total_code_attempts=sum(r['code_attempts'] for r in rows),
        total_astra_input_tokens=sum(r['astra_input_tokens'] for r in rows),
        total_astra_output_tokens=sum(r['astra_output_tokens'] for r in rows),
        total_flora_calls=sum(r['flora_calls'] for r in rows),
        total_tool_calls=sum(r['tool_calls'] for r in rows),
        zero_action_attempts=sum(r['zero_action_attempts'] for r in rows),
        mean_minutes=mean([r['minutes'] for r in rows]),
        old_mean_minutes=mean([r['old_minutes'] for r in rows]),
        mean_astra_turns=mean([r['astra_turns'] for r in rows]),
        mean_code_attempts=mean([r['code_attempts'] for r in rows])))
    lines = ['# WaffleReheat：可执行子任务，seed 0–4', '',
        f"已结束 {len(rows)}/5；观测到的原生整任务成功 **{sum(r['success'] for r in rows)}/{len(rows)}**。"
        f"其中 {audit['quota_interrupted']} 条被用量上限中断，{audit['policy_failures']} 条正常停止失败；"
        '额度中断不计为完整策略评测，不以重试替换这些原始结果。', '',
        '| seed | 原生成功 | 动作步数 | 分钟 | Astra turns | 代码尝试 | 局部切换 | Code 完成局部任务 | VLA 指令数 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['seed']} | {row['success']} | {row['steps']} | {row['minutes']:.2f} | "
                     f"{row['astra_turns']} | {row['code_attempts']} | {row['local_transitions']} | "
                     f"{row['code_completed_subtasks']} | {row['vla_instructions']} |")
    lines += ['', '## 每条任务的实际消耗', '',
        '| seed | 结束分类 | Astra 输入 token | 输出 token | Flora 请求 | 工具调用 | 暂停等待 Astra（分钟） |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['seed']} | {row['termination_class']} | {row['astra_input_tokens']:,} | "
                     f"{row['astra_output_tokens']:,} | {row['flora_calls']} | {row['tool_calls']} | "
                     f"{row['astra_paused_minutes']:.2f} |")
    lines += ['', 'Token 来自 Codex 各线程累计用量的最大值，再跨线程相加；没有重复累加用量通知。'
        '输入包含多轮对话/工具结果被再次送入模型的 token，不是独立文本量。'
        '当前日志没有保留 cached input 拆分或实际账单，不能将输入总量直接换算成实付费用。'
        '工具调用发生在 Astra turn 内，一个 turn 可能包含多个内部模型请求，不能将 turns 等同于 API 请求数。',
        '', '## 实验条件与边界', '',
        '- 同一 OmniSF atomic checkpoint 59999；Flora 每 50 VLA 步监督，Astra high，5 路并行。',
        '- 动作预算 4050→8100；每个代码子任务尝试 12→32；单程序 600 步/360 秒；无 Episode 墙钟超时。',
        '- 局部验证使用仿真只读布尔状态，不提供物体真值坐标；属于验证器辅助评测，不能当作纯视觉判断能力。',
        '- 每个 seed 使用相同的上一轮 Waffle 归档（已含 seed 0–19 经验），新结果不在本批 seed 间共享。',
        '- 任务拆分、局部检查、预算和起始 Memory 同时变化；这是开发回测，没有纯 VLA 配对对照。',
        '- 整任务成功只按原生 task_success；局部检查/阶段推进与程序自报结果分别记录。',
        '- Astra turn 包含内部模型、工具调用和等待；索引字符不等于 token 或收费。',
        '- Episode 耗时不含批次模型服务预热、初始库复制及事后视频生成；五路并行，不能把各条时长相加当作批次墙钟时长。',
        '', '[实现说明](../../docs/executable-subtasks.md) · [CSV](metrics.csv) · [审计](audit.json)', '',
        '## 每条轨迹', '']
    for spec, details in [(s, d) for s in plan['cases'] for d in records if s['seed']==d['row']['seed']]:
        row = details['row']
        lines += [f"### seed {row['seed']}", '',
            f"停止原因：{row['reason']}。上轮 {row['old_minutes']:.2f} 分钟、{row['old_astra_turns']} turns；"
            f"本轮 {row['minutes']:.2f} 分钟、{row['astra_turns']} turns。失败时长不能解读为完成速度。", '',
            '实际 VLA 指令：', '']
        for text, steps in details['actual_vla_instructions'].items():
            lines.append(f'- {text}：{steps} 步')
        lines += ['', '局部验证首次为真：`'+json.dumps(details['local_checks_first_true'],ensure_ascii=False)+'`。',
                  '', '代码尝试对应目标：`'+json.dumps(details['code_attempts_by_condition'],ensure_ascii=False)+'`。',
                  '', '通过局部验证的目标及次数（重复完成不代表不同子任务）：`'+
                  json.dumps(details['completed_subtask_counts'],ensure_ascii=False)+'`。',
                  '', '开门交接后判据变化（false 不等于门完全关闭，也不单独证明原因）：`'+
                  json.dumps(details['opening_handoff_checks'],ensure_ascii=False)+'`。',
                  '', f"上下文峰值 {row['max_context_chars']} 字符；线程重建 {row['thread_rotations']} 次；"
                  f"旧技能导入 {row['imports']} 次，契约适配 {row['adapted_contracts']} 次，直接调用 {row['direct_skill_calls']} 次。", '',
                  f"Flora 子任务评分 ≥4 而同快照完成条件仍为 false 的次数：{row['high_flora_without_completion']}。"
                  '这是高分与完成条件的对照，不将执行质量高分直接当作任务完成。', '',
                  '原始停止记录：`'+json.dumps(details['stop_events'],ensure_ascii=False)+'`。', '',
                  '主链路耗时（秒，Flora 与运行中审核重叠，不再相加）：`'+
                  json.dumps(details['wall_components_seconds'],ensure_ascii=False)+'`。', '',
                  '工具调用计数：`'+json.dumps(details['tool_counts'],ensure_ascii=False)+'`。', '',
                  '代码会话（从首次执行前接管起到末次程序结果，包含会话内修订；不含首次准备与最终停止审核）：', '',
                  '| 目标 | 起始步 | 动作步 | 尝试 | 分钟 | 通过局部验证 |',
                  '|---|---:|---:|---:|---:|---|']
        for session in details['code_sessions'].values():
            lines.append(f"| {session['goal']} | {session['start_step']} | {session['steps']} | "
                         f"{session['attempts']} | {session['minutes']:.2f} | {session['verified']} |")
        lines += ['',
                  '程序报告与执行证据（错误文本是程序报告，不等于已确认的物理原因；检查点也计入尝试）：', '',
                  '| 尝试 | 版本 | 步数 | 执行秒 | 修订与准备秒 | 通过验证 | 停止报告 |',
                  '|---:|---|---:|---:|---:|---|---|']
        for index, attempt in enumerate(details['recoveries'], 1):
            setup = attempt['revision_and_setup_seconds']
            setup_text = f'{setup:.1f}' if setup is not None else '—'
            error = (attempt.get('error') or '—').replace('|', '\\|').replace('\n', ' ')
            lines.append(f"| {index} | {attempt['program_version']} | {attempt['steps']} | "
                         f"{attempt['seconds']:.1f} | {setup_text} | {attempt['verified']} | {error} |")
        lines += ['',
                  f"[完整阶段证据](subtask-analysis/{spec['key']}.json) · "
                  f"[回放页面](report/{spec['key']}/index.html) · [MP4](report/{spec['key']}/replay.mp4)", '']
    (root/'findings.md').write_text('\n'.join(lines)+'\n')
    video_root = root/'report'
    video_root.mkdir(exist_ok=True)
    cards = []
    for d in records:
        r = d['row']
        key = f"WaffleReheat-seed{r['seed']:02}"
        cards.append(f'<article><h2>Seed {r["seed"]}</h2>'
            f'<p>整任务成功：{r["success"]} · {r["minutes"]:.2f} 分钟 · '
            f'{r["steps"]} 步 · Astra {r["astra_turns"]} turns · 代码 {r["code_attempts"]} 次</p>'
            f'<p>局部切换 {r["local_transitions"]} 次，涉及 {r["unique_completed_subtasks"]} 种目标。'
            f'停止：{html.escape(r["reason"])}</p>'
            f'<a href="{key}/index.html">阶段回放页面</a> · '
            f'<a href="{key}/replay.mp4">MP4</a></article>')
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>WaffleReheat · seeds 0–4</title><style>'
        'body{font:16px/1.6 system-ui,sans-serif;max-width:960px;margin:40px auto;padding:0 20px;'
        'color:#172033;background:#f5f7fb}article{background:white;border:1px solid #dbe1eb;'
        'padding:20px;margin:16px 0;border-radius:10px}h2{margin:0}a{color:#1456b8}</style>'
        '<h1>WaffleReheat · seeds 0–4</h1>'
        '<p>固定源码；同一初始 Memory；扩展执行预算。局部验证使用仿真布尔状态，整任务仍按原生成功判据。</p>'
        '<p>视频保留全部物理动作帧，折叠重复工具详情与等待；以叠加的墙钟时间和原始日志查看真实耗时。</p>'
        '<p><a href="../findings.md">详细结果与逐次代码尝试</a> · '
        '<a href="../metrics.csv">CSV</a> · <a href="../audit.json">审计</a></p>'
        + ''.join(cards) + '</html>')
    (video_root/'index.html').write_text(page)
    print(json.dumps(dict(audit=audit,rows=rows),ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    build(parser.parse_args().root.resolve())
