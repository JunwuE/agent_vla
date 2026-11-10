"""Resumable sequential development sweep with immutable per-episode software releases."""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from summarize_cross_seed import audit

from vla_agent.data import read_events


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def analyze(episode):
    data = audit(episode)
    data.pop('prefix')
    data.pop('initial')
    rows = read_events(episode)
    def durations(values):
        return dict(count=len(values), total=sum(values), mean=float(np.mean(values)) if values else 0,
                    p95=float(np.quantile(values, .95)) if values else 0, maximum=max(values, default=0))

    commands = {r['id']: r for r in rows if r['kind'] == 'action_command'}
    data['vla_latency_seconds'] = durations([r['seconds'] for r in rows
                                           if r['kind'] == 'vla_inference_end' and not r.get('error')])
    data['flora_latency_seconds'] = durations(data['flora_client_seconds'])
    data['action_latency_seconds'] = durations([(r['time_ns']-commands[r['command_id']]['time_ns'])/1e9
                                                for r in rows if r['kind'] == 'action_executed'])
    data['action_latency_by_source'] = {
        source: durations([(r['time_ns']-commands[r['command_id']]['time_ns'])/1e9
                           for r in rows if r['kind']=='action_executed' and r['source']==source])
        for source in data['source_steps']}
    astra = data['astra_completed_turn_seconds']
    components = dict(vla_inference=data['vla_latency_seconds']['total'],
                      vla_actions=data['action_latency_by_source'].get('vla', {}).get('total', 0),
                      code_execution=data['program_seconds'],
                      astra_paused=sum(astra.get(k,0) for k in ('initialization','paused_planning','revision')))
    components['other_and_unattributed'] = data['result']['wall_seconds']-sum(components.values())
    data['wall_components_seconds'] = components
    data['wall_accounting_note'] = ('Flora and running reviews overlap execution and are reported separately. '
        'Other includes simulator startup, scheduling, cleanup and any uncompleted inference; '
        'component timestamps are measured boundaries, not an isolated hardware benchmark.')
    first = next((r for r in rows if r['kind'] == 'action_command'), None)
    data['before_first_action_seconds'] = (first['time_ns']-rows[0]['time_ns'])/1e9 if first else None
    data['tool_errors'] = [dict(name=r['name'], error=r['result']['error']) for r in rows
                           if r['kind'] == 'codex_tool' and isinstance(r.get('result'), dict)
                           and r['result'].get('error')]
    data['recoveries'] = [dict(program_version=r.get('program_version'), seconds=r['seconds'],
                               steps=r['executed_steps'], verified=r['verified'], error=r.get('error'),
                               failed_stage=r.get('failed_stage'),
                               last_program_facts=[n.get('facts', []) for n in r.get('program_notes', [])][-2:],
                               stages=[s['name'] for s in r.get('stages', [])]) for r in rows
                          if r['kind'] == 'recovery_result']
    code_starts = {r['intervention_id']:r for r in rows if r['kind']=='code_version'}
    previous = None
    for item, result in zip(data['recoveries'], (r for r in rows if r['kind']=='recovery_result')):
        start = code_starts[result['intervention_id']]
        item['policy_path'] = start['path']
        item['sha256'] = start['sha256']
        same_session = previous is not None and (
            previous['intervention_id'].rsplit('-',1)[0] == result['intervention_id'].rsplit('-',1)[0])
        item['revision_and_setup_seconds'] = ((start['time_ns']-previous['time_ns'])/1e9
                                              if same_session else None)
        item['tools_since_previous_attempt'] = dict(Counter(
            r['name'] for r in rows if same_session and previous['id'] < r['id'] < start['id']
            and r['kind']=='codex_tool'))
        previous = result
    starts = {r['turn_id']: r for r in rows if r['kind'] == 'codex_turn_start'}
    data['turns'] = [dict(turn_id=r['turn_id'], seconds=(r['time_ns']-starts[r['turn_id']]['time_ns'])/1e9,
                         intent=starts[r['turn_id']]['context']['intent'],
                         context_chars=len(json.dumps(starts[r['turn_id']]['context'])))
                     for r in rows if r['kind'] == 'codex_turn_end']
    data['tool_counts'] = dict(Counter(r['name'] for r in rows if r['kind'] == 'codex_tool'))
    data['decision_labels'] = dict(Counter(r['decision']['label'] for r in rows if r['kind']=='decision'))
    data['review_triggers'] = dict(Counter(r['decision']['trigger'] for r in rows if r['kind']=='review_begin'))
    data['discarded_vla_queue_steps'] = sum(r['discarded_steps'] for r in rows if r['kind']=='control_claim')
    data['chunk_boundary_deferrals'] = [dict(step=r['current_step'], remaining_steps=r['remaining_steps'],
                                             assessment=r['assessment']) for r in rows
                                        if r['kind']=='review_waiting_for_chunk_boundary']
    data['reviews'] = [dict(snapshot_step=r['binding']['observation_step'],
                            return_step=r['current_step'], seconds=r['seconds'], **r['result'])
                       for r in rows if r['kind'] == 'review_end' and isinstance(r.get('result'), dict)]
    data['handoffs'] = [dict(step=r['binding']['observation_step'], kind=r['plan']['kind'],
                             reason=r['plan']['reason']) for r in rows if r['kind']=='intervention_plan'
                        and r['binding']['observation_step']>0]
    data['zero_action_attempts'] = sum(r['steps']==0 for r in data['recoveries'])
    data['skill_queries'] = [dict(query=r['arguments'].get('query'), candidates=[
        dict(id=skill['id'], version=skill['version'], status=skill['status'],
             success_evidence=any(e.get('report', {}).get('verified') is True
                 for e in (skill.get('evidence', []) if isinstance(skill.get('evidence'), list)
                           else [skill.get('evidence', {})]) if isinstance(e, dict)))
        for skill in r['result']]) for r in rows if r['kind']=='codex_tool'
        and r['name']=='search_skills' and isinstance(r.get('result'), list)]
    elapsed = data['astra_completed_turn_seconds']
    data['diagnosis'] = (
        '代码修订推理为主要耗时；检查重复失败阶段及可参数化的修改'
        if elapsed.get('revision', 0) > data['result']['wall_seconds'] * .4 else
        '恢复执行为主要耗时；检查阶段提前退出与重复移动'
        if data['program_seconds'] > data['result']['wall_seconds'] * .4 else
        '无代码接管；检查监督开销与首次审核时机'
        if not data['program_attempts'] else '混合耗时；逐项查看规划、程序和工具错误')
    return data


def report(root):
    records = [json.loads(p.read_text()) for p in sorted((root/'analysis').glob('seed*.json'))]
    fields = ['seed', 'release', 'success', 'reason', 'steps', 'wall_seconds', 'vla_steps', 'code_steps',
              'astra_turns', 'flora_calls', 'attempts', 'versions', 'imports', 'direct_calls',
              'initialization_seconds', 'review_seconds', 'planning_seconds', 'revision_seconds',
              'code_seconds', 'tool_errors', 'native_450_success', 'vla_inference_seconds',
              'vla_action_seconds', 'other_seconds', 'zero_action_attempts', 'discarded_queue_steps']
    table = []
    for r in records:
        result, duration = r['result'], r['astra_completed_turn_seconds']
        table.append(dict(seed=r['seed'], release=r['release'], success=result['success'], reason=result['reason'],
                          steps=result['control_steps'], wall_seconds=result['wall_seconds'],
                          vla_steps=r['source_steps'].get('vla', 0), code_steps=r['source_steps'].get('code_policy', 0),
                          astra_turns=result['astra_calls'], flora_calls=result['flora_calls'],
                          attempts=r['program_attempts'], versions=r['generated_versions'], imports=len(r['imports']),
                          direct_calls=r['direct_skill_calls'], initialization_seconds=duration.get('initialization', 0),
                          review_seconds=duration.get('review', 0), planning_seconds=duration.get('paused_planning', 0),
                          revision_seconds=duration.get('revision', 0), code_seconds=r['program_seconds'],
                          tool_errors=len(r['tool_errors']), native_450_success=r['native_success'],
                          vla_inference_seconds=r.get('wall_components_seconds',{}).get('vla_inference'),
                          vla_action_seconds=r.get('wall_components_seconds',{}).get('vla_actions'),
                          other_seconds=r.get('wall_components_seconds',{}).get('other_and_unattributed'),
                          zero_action_attempts=r.get('zero_action_attempts'),
                          discarded_queue_steps=r.get('discarded_vla_queue_steps')))
    with (root/'per-seed.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    write_json(root/'summary.json', dict(completed=len(table), requested=50,
                                        successes=sum(r['success'] for r in table), rows=table))
    lines = ['# CloseToasterOvenDoor · 50 seeds 持续开发', '',
             f'已记录 {len(table)}/50 条，成功 {sum(r["success"] for r in table)} 条。', '',
             'seed 0–49；2000 步预算；每条独立 Memory 快照，经验按完成顺序积累；不是冻结模型/技能的独立泛化评测。',
             '耗时包含初始化。异步审核、Flora 与 VLA 重叠，分项时间不能简单相加；失败与基础设施错误均保留。', '',
             '| seed | 版本 | 成功 | VLA/代码步 | 总秒 | Astra | Flora | 尝试/版本 | 导入/直接调用 | 修订秒 |',
             '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in table:
        lines.append(f'| [{r["seed"]}](analysis/seed{r["seed"]:02d}.md) | {r["release"]} | '
                     f'{r["success"]} | {r["vla_steps"]}/{r["code_steps"]} | {r["wall_seconds"]:.1f} | '
                     f'{r["astra_turns"]} | {r["flora_calls"]} | {r["attempts"]}/{r["versions"]} | '
                     f'{r["imports"]}/{r["direct_calls"]} | {r["revision_seconds"]:.1f} |')
    lines += ['', '[逐轨迹 CSV](per-seed.csv)；[优化记录](changes.md)。每条分析链接包含工具次数、修订阶段和失败证据。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def episode_report(root, data):
    seed = data['seed']
    write_json(root/'analysis'/f'seed{seed:02d}.json', data)
    lines = [f'# seed {seed} · {data["release"]}', '', data['diagnosis'], '',
             '结果：`'+json.dumps(data['result'], ensure_ascii=False)+'`', '',
             'Astra 已完成调用耗时：`'+json.dumps(data['astra_completed_turn_seconds'])+'`', '',
             '不重叠主链路耗时（秒）：`'+json.dumps(data.get('wall_components_seconds'))+'`', '',
             'VLA 推理延迟（秒）：`'+json.dumps(data.get('vla_latency_seconds'))+'`', '',
             'Flora 请求延迟（秒，含准备/请求）：`'+json.dumps(data.get('flora_latency_seconds'))+'`', '',
             '动作确认延迟（秒，包含仿真/传输/记录）：`'+json.dumps(data.get('action_latency_seconds'))+'`', '',
             '工具次数：`'+json.dumps(data['tool_counts'])+'`', '',
             'Decision 类别：`'+json.dumps(data.get('decision_labels'))+'`；后台审核触发来源：`'+
             json.dumps(data.get('review_triggers'))+'`。', '',
             f'控制切换丢弃 VLA 队列动作：{data.get("discarded_vla_queue_steps", 0)}；'
             '等待 Chunk 边界的审核：`'+json.dumps(data.get('chunk_boundary_deferrals'))+'`。', '',
             f'零动作恢复尝试：{data.get("zero_action_attempts", 0)}。', '',
             '## 审核与接管', '']
    for review in data.get('reviews', []):
        lines.append(f'- 观测 step {review["snapshot_step"]} → 返回 step {review["return_step"]}，'
                     f'{review["seconds"]:.1f} 秒，{review.get("action")} / {review.get("assessment")}：'
                     f'{review.get("reason", "")}')
    for handoff in data.get('handoffs', []):
        lines += ['', f'实际暂停后计划（step {handoff["step"]}，{handoff["kind"]}）：{handoff["reason"]}']
    lines += ['', '这些是 Astra 的判断记录，不自动视作已证实的失败原因。', '',
             '## 程序执行', '',
             '修订/准备间隔为同一恢复会话中，上次结果返回至本次程序启动的墙钟时间，包含模型修订、工具和准备，'
             '不等同于纯模型推理延迟。', '',
             '| 程序 | 步数 | 执行秒 | 修订/准备秒 | 验证 | 错误 |', '|---|---:|---:|---:|---|---|']
    for r in data['recoveries']:
        error = str('原生验证成功' if r['verified'] else
                    r['error'] or r.get('last_program_facts') or '未报告原因').replace('|', '/').replace('\n', ' ')
        gap = r.get('revision_and_setup_seconds')
        gap_text = f'{gap:.2f}' if gap is not None else '—'
        program = f'[{r["program_version"]}](../episodes/seed{seed:02d}/{r["policy_path"]})'
        lines.append(f'| {program} | {r["steps"]} | {r["seconds"]:.2f} | {gap_text} | {r["verified"]} | {error} |')
    lines += ['', '工具错误：`'+json.dumps(data['tool_errors'],ensure_ascii=False)+'`', '',
              '技能检索：`'+json.dumps(data.get('skill_queries', []),ensure_ascii=False)+'`', '',
              '程序 facts 是程序记录的观测/停止理由，不能自动当成已证实根因。', '',
              f'[完整计时和调用 JSON](seed{seed:02d}.json)；[原始结果](../episodes/seed{seed:02d}/result.json)。']
    (root/'analysis'/f'seed{seed:02d}.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(50)))
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    (root/'episodes').mkdir(exist_ok=True)
    (root/'analysis').mkdir(exist_ok=True)
    if args.report_only:
        for path in sorted((root/'episodes').glob('seed*/result.json')):
            data = analyze(path.parent)
            launch = json.loads(path.parent.with_name(path.parent.name+'-launch.json').read_text())
            data['release'] = launch['release']
            episode_report(root, data)
        report(root)
        return
    for seed in args.seeds:
        episode = root/'episodes'/f'seed{seed:02d}'
        record = root/'analysis'/f'seed{seed:02d}.json'
        if record.exists():
            continue
        profile = json.loads((root/'active-profile.json').read_text())
        config = yaml.safe_load(Path(profile['config']).read_text())
        config['environment']['seed'] = seed
        config['vla']['sampling_seed'] = 42 + seed
        config_path = root/'episodes'/f'seed{seed:02d}.yaml'
        if not episode.exists():
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            write_json(root/'status.json', dict(state='running', seed=seed, release=profile['release'],
                                               started_ns=time.time_ns()))
            print(json.dumps(dict(event='start', seed=seed, release=profile['release'])), flush=True)
            env = os.environ.copy()
            env['PYTHONPATH'] = profile['source']
            with (root/'episodes'/f'seed{seed:02d}.log').open('wb') as log:
                process = subprocess.Popen([sys.executable, '-m', 'vla_agent.cli', 'run', '--config', str(config_path),
                                            '--output', str(episode)], env=env, stdout=log, stderr=subprocess.STDOUT)
                write_json(root/'episodes'/f'seed{seed:02d}-launch.json', dict(**profile, pid=process.pid))
                returncode = process.wait()
            print(json.dumps(dict(event='exit', seed=seed, returncode=returncode)), flush=True)
        if not (episode/'result.json').exists():
            write_json(root/'status.json', dict(state='infrastructure_error', seed=seed,
                                               reason='missing result; retain partial logs and repair before resume'))
            raise RuntimeError(f'seed {seed} has no result; refusing to overwrite its evidence')
        launch = json.loads((root/'episodes'/f'seed{seed:02d}-launch.json').read_text())
        data = analyze(episode)
        data['release'] = launch['release']
        episode_report(root, data)
        report(root)
        print(json.dumps(dict(event='analyzed', seed=seed, result=data['result'],
                              diagnosis=data['diagnosis']), ensure_ascii=False), flush=True)
    write_json(root/'status.json', dict(state='completed', seeds=args.seeds, finished_ns=time.time_ns()))


if __name__ == '__main__':
    main()
