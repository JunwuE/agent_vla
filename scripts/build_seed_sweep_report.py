"""Standalone sortable per-seed dashboard and aggregate metrics from retained sweep analyses."""

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def statistics(records):
    times = [r['result']['wall_seconds'] for r in records]
    success_times = [r['result']['wall_seconds'] for r in records if r['result']['success']]
    decisions, triggers = Counter(), Counter()
    for record in records:
        decisions.update(record.get('decision_labels', {}))
        triggers.update(record.get('review_triggers', {}))
    return dict(episodes=len(records), successes=sum(r['result']['success'] for r in records),
                within_450=sum(r['native_success'] for r in records),
                total_episode_seconds=sum(times), mean_seconds=float(np.mean(times)) if times else None,
                median_seconds=float(np.median(times)) if times else None,
                p90_seconds=float(np.quantile(times, .9)) if times else None,
                successful_mean_seconds=float(np.mean(success_times)) if success_times else None,
                successful_median_seconds=float(np.median(success_times)) if success_times else None,
                astra_turns=sum(r['result']['astra_calls'] for r in records),
                code_attempts=sum(r['program_attempts'] for r in records),
                zero_action_attempts=sum(r.get('zero_action_attempts',0) for r in records),
                code_program_seconds=sum(r['program_seconds'] for r in records),
                astra_revision_seconds=sum(r['astra_completed_turn_seconds'].get('revision', 0)
                                           + r.get('timing_reconciliation', {}).get('reclassified_seconds', 0)
                                           for r in records),
                explicitly_timed_out_planning_seconds=sum(
                    r.get('timing_reconciliation', {}).get('reclassified_seconds', 0) for r in records),
                flora_calls=sum(r['result']['flora_calls'] for r in records),
                decision_labels=dict(decisions), review_triggers=dict(triggers),
                direct_action_steps=sum(r['source_steps'].get('astra_local',0) for r in records),
                direct_skill_calls=sum(r['direct_skill_calls'] for r in records),
                imported_programs=sum(len(r['imports']) for r in records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root
    records = [json.loads(p.read_text()) for p in sorted((root/'analysis').glob('seed*.json'))]
    grouped = defaultdict(list)
    for record in records:
        grouped[record['release']].append(record)
    aggregates = dict(all=statistics(records), by_release={k:statistics(v) for k,v in grouped.items()},
                      with_code=statistics([r for r in records if r['program_attempts']]),
                      without_code=statistics([r for r in records if not r['program_attempts']]),
                      without_code_with_paused_plan=statistics([
                          r for r in records if not r['program_attempts'] and r['physical_intervention_proposals']]))
    (root/'aggregate.json').write_text(json.dumps(aggregates,ensure_ascii=False,indent=2)+'\n')
    rows = []
    maximum = max((r['result']['wall_seconds'] for r in records), default=1)
    palette = {'vla_inference':('#4592dc','VLA 推理'), 'vla_actions':('#71869d','VLA 动作确认'),
               'code_execution':('#329879','代码执行'), 'astra_paused':('#db8943','Astra 暂停规划/修订'),
               'other_and_unattributed':('#d8dee5','初始化/清理及其他')}
    for r in records:
        seed, result = r['seed'],r['result']
        seconds = result['wall_seconds']
        source = r['source_steps']
        segments = ''.join(f'<span style="width:{max(0,value)/seconds*100:.2f}%;background:{palette[key][0]}" '
                           f'title="{palette[key][1]}: {value:.2f} 秒"></span>'
                           for key,value in r.get('wall_components_seconds',{}).items())
        bar = f'<div class="bar" style="width:{seconds/maximum*100:.1f}%">{segments}</div>'
        cells = [f'<a href="analysis/seed{seed:02d}.md">{seed}</a>', html.escape(r['release']),
                 '成功' if result['success'] else html.escape(result['reason']),
                 f'{seconds/60:.2f}'+bar,
                 f'{source.get("vla",0)}/{source.get("code_policy",0)}', str(result['astra_calls']),
                 str(result['flora_calls']), f'{r["program_attempts"]}/{r["generated_versions"]}',
                 f'{len(r["imports"])}/{r["direct_skill_calls"]}']
        detail = json.dumps(dict(astra_seconds=r['astra_completed_turn_seconds'],
                                 wall_seconds=r.get('wall_components_seconds'),
                                 tools=r['tools'], recoveries=r['recoveries'],
                                 decision_labels=r.get('decision_labels'),
                                 review_triggers=r.get('review_triggers'),
                                 discarded_queue_steps=r.get('discarded_vla_queue_steps'),
                                 chunk_boundary_deferrals=r.get('chunk_boundary_deferrals'),
                                 timing_reconciliation=r.get('timing_reconciliation'),
                                 errors=r.get('tool_errors',[])),ensure_ascii=False,indent=2)
        cells.append('<details><summary>耗时与调用</summary><pre>'+html.escape(detail)+'</pre></details>')
        rows.append(f'<tr data-seed="{seed}" data-seconds="{seconds}" data-code="{r["program_attempts"]}" '
                    f'data-failure="{int(not result["success"])}">'+''.join(f'<td>{c}</td>' for c in cells)+'</tr>')
    total=aggregates['all']
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CloseToasterOvenDoor · 50 seeds</title><style>
body{font:15px system-ui,sans-serif;margin:30px auto;max-width:1500px;padding:0 20px;color:#172c3b;background:#f6f8fa}
h1{font-size:26px}p{line-height:1.7}.cards{display:flex;gap:20px;flex-wrap:wrap}.card{padding:18px 24px;background:white;border-radius:10px}
.card strong{display:block;font-size:26px;color:#176985}table{width:100%;border-collapse:collapse;background:white}
th,td{padding:10px;text-align:left;border-bottom:1px solid #e5eaef;vertical-align:top}th{background:#e8f0f5;white-space:nowrap}
select{padding:7px;margin:8px}a{color:#176985}.bar{height:7px;display:flex;margin-top:5px;min-width:1px}
.bar span{height:7px}.legend{display:inline-block;padding:5px 10px;background:white;margin:4px;border-left:7px solid}
pre{white-space:pre-wrap;max-width:600px;max-height:420px;overflow:auto;font-size:12px}summary{cursor:pointer}
</style><h1>CloseToasterOvenDoor · 50 seeds 持续开发</h1>
<p>相同 π0.5 权重，seed 0–49，2000 步预算；报告包含失败。每条使用冻结的软件与私有技能快照，完成后合并经验。
最多 3 条主批仿真并发，诊断对照与复测单列；模型排队会影响延迟。共享模型服务已预热，单条计时不含一次性模型加载。
此批含在线积累和版本更新，不能当作冻结系统的独立成功率评测。</p>
<div class="cards">__CARDS__</div>
<p>Astra 次数包含初始化、后台审核、暂停确认与代码修订。导入不代表执行成功；直接技能调用也可能失败。
Flora 与后台审核和 VLA 异步重叠，不能把所有分项时间简单相加。视频时长也不是 Episode 墙钟耗时。
未执行代码的轨迹也可能有暂停后继续；不能据此认定其动作和纯 VLA 完全相同。</p>
<p><a href="changes.md">每轮修改与诊断发现</a> · <a href="per-seed.csv">CSV</a> · <a href="aggregate.json">汇总 JSON</a> ·
<a href="README.md">设置与复现</a></p>
<label>排序<select id="sort"><option value="seed">seed</option><option value="seconds">耗时从高到低</option><option value="code">代码尝试从多到少</option></select></label>
<label>筛选<select id="filter"><option value="all">全部</option><option value="code">执行了代码</option><option value="failure">未成功</option></select></label>
<p>主链路耗时分解（悬停查看秒数）：__LEGEND__</p>
<table><thead><tr><th>seed</th><th>版本</th><th>结果</th><th>分钟</th><th>VLA/代码步</th><th>Astra</th><th>Flora</th><th>尝试/版本</th><th>导入/直接调用</th><th>证据</th></tr></thead>
<tbody>__ROWS__</tbody></table><script>
const body=document.querySelector('tbody'),rows=[...body.children];
function update(){const key=document.querySelector('#sort').value,filter=document.querySelector('#filter').value;
rows.sort((a,b)=>key==='seed'?Number(a.dataset[key])-Number(b.dataset[key]):Number(b.dataset[key])-Number(a.dataset[key]));
for(const row of rows){row.hidden=filter!=='all'&&!Number(row.dataset[filter]);body.appendChild(row)}}
document.querySelectorAll('select').forEach(x=>x.addEventListener('change',update));
</script></html>'''
    plan_path = root / 'plan.json'
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
    if plan.get('initial_task_memory_version') and plan.get('memory_policy', '').startswith('Same initial snapshot'):
        page = page.replace('50 seeds 持续开发', '50 seeds · 固定已有经验复测')
        begin = page.index('<p>相同 π0.5 权重')
        end = page.index('</p>', begin) + len('</p>')
        page = page[:begin] + '''<p>相同 π0.5 权重，seed 0–49，2000 步预算，无 Episode 墙钟上限；失败保留。
本轮使用同一份冻结源码，每条从相同的已有技能和 Memory 快照开始；本批新经验仅在全部结束后合并。
最多 3 条仿真并发，单条计时不含一次性模型加载。本轮是已见 seed 的开发复测。
历史批次的软件、Memory、负载不同；抽查还发现当前与历史模型服务首个动作存在小幅数值差异，
因此速度和成功率变化不能单独归因于 Memory，也不是严格配对的纯 VLA 因果对照。</p>''' + page[end:]
        page = page.replace('href="changes.md">每轮修改与诊断发现', 'href="comparison.md">与上一批比较')
        page = page.replace('href="per-seed.csv"', 'href="comparison.csv"')
    cards = [(f'{len(records)}/50','已完成'), (str(total['successes']),'完整任务成功'),
             (str(total['within_450']),'450 步内成功'),(str(total['code_attempts']),'代码执行尝试'),
             (str(total['astra_turns']),'Astra turns')]
    page=page.replace('__CARDS__',''.join(f'<div class="card"><strong>{v}</strong>{k}</div>' for v,k in cards))
    page=page.replace('__ROWS__','\n'.join(rows))
    page=page.replace('__LEGEND__',''.join(f'<span class="legend" style="border-color:{color}">{label}</span>'
                                          for color,label in palette.values()))
    (root/'dashboard.html').write_text(page)
    print(json.dumps(aggregates,ensure_ascii=False))


if __name__ == '__main__':
    main()
