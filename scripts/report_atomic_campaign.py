"""Compare complete, fixed-memory paired atomic groups and audit deterministic VLA prefixes."""

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def execution(episode):
    observations, actions = {}, []
    for line in (episode / 'events.jsonl').open():
        event = json.loads(line)
        if event['kind'] == 'observation':
            observations[event['step']] = event
        elif event['kind'] == 'action_command':
            actions.append(event)
    return observations, actions


def arrays_equal(a, b):
    with np.load(a) as x, np.load(b) as y:
        return set(x.files) == set(y.files) and all(
            np.array_equal(x[key], y[key], equal_nan=True) for key in x.files)


def prefix_pair(baseline, agent):
    a_obs, a_actions = execution(baseline)
    b_obs, b_actions = execution(agent)
    prefix = 0
    for a, b in zip(a_actions, b_actions):
        if (a['source'] != 'vla' or b['source'] != 'vla'
                or a['action'] != b['action'] or a['instruction'] != b['instruction']):
            break
        prefix += 1
    samples = sorted({0, prefix // 2, prefix})
    return dict(common_vla_command_prefix=prefix,
        baseline_actions=len(a_actions), agent_actions=len(b_actions),
        measured_states_identical=all(a_obs[s]['state'] == b_obs[s]['state'] for s in range(prefix + 1)),
        initial_arrays_identical=arrays_equal(baseline / a_obs[0]['path'], agent / b_obs[0]['path']),
        rgbd_checks=[dict(step=s, identical=arrays_equal(baseline / a_obs[s]['path'], agent / b_obs[s]['path']))
                    for s in samples],
        note='Only the common action prefix is compared. Later divergence after intervention is expected.')


def compare(a, b):
    common = [s for s in a if a[s]['success'] and b[s]['success']]
    return dict(
        gains=[s for s in a if not a[s]['success'] and b[s]['success']],
        losses=[s for s in a if a[s]['success'] and not b[s]['success']],
        still_failed=[s for s in a if not a[s]['success'] and not b[s]['success']],
        common_success_seeds=common,
        common_success_mean_seconds_a=mean(a[s]['wall_seconds'] for s in common),
        common_success_mean_seconds_b=mean(b[s]['wall_seconds'] for s in common),
        common_success_mean_paired_seconds_delta=mean(b[s]['wall_seconds'] - a[s]['wall_seconds'] for s in common),
        note='Elapsed time includes initialization. Common-success timing excludes unmatched rescued/failing episodes.')


def infrastructure_exceptions(path, audit):
    """Include zero-action service failures honestly, without certifying missing execution."""
    failed = {key for key, passed in audit['checks'].items() if not passed}
    if failed - {'all_per_seed_checks'}:
        raise ValueError(f'Protocol/source failure cannot be waived: {path.name}: {failed}')
    exceptions = []
    for entry in audit['per_seed']:
        missing = {key for key, passed in entry['checks'].items() if not passed}
        if not missing:
            continue
        if missing - {'original_instruction', 'checkpoint'}:
            raise ValueError(f'Non-service audit failure: {path.name}: {entry}')
        episode = path / 'episodes' / f"OpenDrawer-seed{entry['seed']:02d}"
        result = read(episode / 'result.json')
        events = [json.loads(line) for line in (episode / 'events.jsonl').open()]
        errors = [e for e in events if e['kind'] in {'codex_error', 'planner_service_failure'}]
        if (result['success'] or result['control_steps'] != 0 or not errors
                or any(e['kind'] in {'action_command', 'action_executed', 'vla_service'} for e in events)):
            raise ValueError(f'Cannot verify zero-action infrastructure failure: {episode}')
        exceptions.append(dict(seed=entry['seed'], unavailable_execution_checks=sorted(missing),
            result=result, error_evidence=errors))
    return exceptions


def main(root, names, include_infrastructure_failures=False):
    groups, rows, tables, plans, prefix = {}, [], {}, {}, []
    for name in names:
        path = root / name
        audit = read(path / 'final-audit.json')
        routing = read(path / 'routing-audit.json')
        if not routing['summary']['all_checks_passed']:
            raise ValueError(f'Incomplete/failed protocol audit: {name}')
        exceptions = []
        if not all(audit['checks'].values()):
            if not include_infrastructure_failures:
                raise ValueError(f'Incomplete/failed protocol audit: {name}')
            exceptions = infrastructure_exceptions(path, audit)
        plan, metrics = read(path / 'plan.json'), read(path / 'atomic-metrics.json')
        runtime = read(path / 'agent-runtime-audit.json')
        expected = set(plan['seeds'])
        by_seed = {r['seed']: r for r in metrics['rows']}
        if set(by_seed) != expected:
            raise ValueError(f'Missing/extra seeds: {name}')
        plans[name], tables[name] = plan, by_seed
        summary = dict(metrics['summary'])
        summary.update(protocol_audit_passed=all(audit['checks'].values()),
            infrastructure_failures=exceptions,
            mean_success_seconds=mean(r['wall_seconds'] for r in by_seed.values() if r['success']),
            mean_failure_seconds=mean(r['wall_seconds'] for r in by_seed.values() if not r['success']),
            total_code_steps=sum(r['code_steps'] for r in by_seed.values()),
            source_steps=dict(sum((Counter(d['source_steps']) for d in metrics['details']), Counter())),
            batch_wall_seconds=audit['batch_wall_seconds'],
            token_threshold_exceeded_seeds=audit['reported_token_threshold_exceeded_seeds'],
            routing=routing['summary'], latency=runtime['latency'], intent_timing=runtime['intents'],
            review_deferred=runtime['counts'].get('review_deferred', 0),
            budget_proposals_released=runtime['counts'].get('codex_budget_proposal_released', 0))
        groups[name] = summary
        for r in metrics['rows']:
            rows.append(dict(group=name, **r))
    baseline = names[0]
    for name in names[1:]:
        for key in ('seeds', 'native_horizon', 'initial_memory_sha256'):
            if plans[baseline][key] != plans[name][key]:
                raise ValueError(f'Pairing mismatch: {key}')
        for case in plans[name]['cases']:
            pair = prefix_pair(root / baseline / 'episodes' / case['key'], root / name / 'episodes' / case['key'])
            prefix.append(dict(seed=case['seed'], group=name, **pair))
    if not all(p['initial_arrays_identical'] for p in prefix):
        raise ValueError('Initial state/observation mismatch across paired groups')
    pairs = {f'{a}_to_{b}': compare(tables[a], tables[b])
             for i, a in enumerate(names) for b in names[i + 1:]}
    data = dict(groups=groups, paired_comparisons=pairs, rows=rows, prefix_equivalence=prefix,
        protocol=dict(seeds=plans[baseline]['seeds'], fixed_memory=True,
            native_horizon=plans[baseline]['native_horizon'], max_steps=plans[baseline]['config']['runtime']['max_steps']),
        limitations=['Fixed seen seeds used for development; no held-out generalization estimate.',
            'One realization per group/seed; asynchronous scheduling and model responses are not deterministic.',
            'Reported tokens include repeated context; no authoritative cached-token billing or currency total.',
            'Background Flora/review durations overlap actions and cannot be added to episode wall time.',
            'No alternate reasoner was tested; pipeline gains cannot establish Astra as the best model.'])
    (root / 'comparison.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    with (root / 'per-episode.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# OpenDrawer seeds 0–19 配对实验', '',
        '各组配置相同初始状态、原生 instruction、π0.5 atomic 权重和独立初始 Memory。'
        '750 步与延长 1500 步分开统计；失败没有替换成其他 seed 的成功结果。', '',
        '| 组 | 750 步成功 | 最终成功 | 平均分钟 | 中位分钟 | Astra turns | Flora | Code 尝试 / 步数 | 上报 token |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, g in groups.items():
        lines.append(f"| {name} | {g['native_successes']}/{g['requested']} | {g['extended_successes']}/{g['requested']} | "
            f"{g['mean_seconds']/60:.2f} | {g['median_seconds']/60:.2f} | {g['astra_turns']} | "
            f"{g['flora_calls']} | {g['code_attempts']} / {g['total_code_steps']} | {g['reported_tokens']:,} |")
    lines += ['', 'Astra turn 可含多次工具往返。token 为上报输入+输出，包含重复上下文，缺少缓存计费拆分，'
        '不能换算成真实账单。最后在途调用可能超出停止阈值。', '', '## 配对变化', '']
    for name, g in groups.items():
        if g['infrastructure_failures']:
            seeds = [x['seed'] for x in g['infrastructure_failures']]
            lines += [f"**{name} 的 seeds {seeds} 在首次动作前因模型服务错误结束，仍计入失败和全部 20 条耗时。**"
                '这些条目没有 VLA 动作/服务记录，因此 instruction 与 checkpoint 的实际执行检查缺证；'
                '原始 final-audit 保持失败状态，未改成通过。其余协议检查与路由检查通过。'
                '平均耗时包含早退，不能解释为完成任务变快。', '']
    for name, pair in pairs.items():
        lines.append(f"- {name}：新增成功 {pair['gains']}；成功退化 {pair['losses']}；仍失败 {pair['still_failed']}。")
        if pair['common_success_seeds']:
            lines.append(f"  两组都成功的 {len(pair['common_success_seeds'])} 条平均耗时 "
                f"{pair['common_success_mean_seconds_a']/60:.2f} → {pair['common_success_mean_seconds_b']/60:.2f} 分钟。")
    lines += ['', '## 逐 seed 结果', '', '| seed | ' + ' | '.join(names) + ' |',
              '|---:|' + '---|' * len(names)]
    for seed in plans[baseline]['seeds']:
        values = []
        for name in names:
            r = tables[name][seed]
            values.append(f"{'成功' if r['success'] else '失败'} / {r['steps']} 步 / {r['wall_seconds']/60:.2f} 分 / A{r['astra']} F{r['flora']} C{r['code_attempts']}")
        lines.append(f"| {seed} | " + ' | '.join(values) + ' |')
    lines += ['', 'A= Astra turns；F= Flora 调用；C= Code 尝试。', '', '## 执行与路由核验', '',
        f"{len(prefix)} 组配对的初始 RGB-D 与状态逐值一致；共同 VLA 命令前缀与实测状态检查保存在 comparison.json。",
        '每组的 routing-audit.json 保留所有分数、转交原因、审核结论、控制权、代码结果和原生成功核验。', '',
        '## 解读边界', '',
        '这是固定已见 seeds 的开发评测。异步到达时机及模型回复存在波动；不能把一次差值全部归因于单个改动。'
        '预算提前失败不能算完成任务提速。当前没有其他 reasoning 模型的配对实验，不能声称 Astra 最优。', '',
        '[逐条 CSV](per-episode.csv) · [完整配对指标](comparison.json) · [实验协议](README.md)。']
    (root / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(groups={k: {x: v[x] for x in ('native_successes', 'extended_successes', 'mean_seconds', 'astra_turns', 'reported_tokens')}
                                 for k, v in groups.items()}, paired=pairs), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--groups', nargs='+', default=['vla20', 'agent-r1', 'agent-r2-fast-review'])
    parser.add_argument('--include-infrastructure-failures', action='store_true',
        help='Report verified zero-action service failures as failures, retaining unavailable execution checks.')
    args = parser.parse_args()
    main(args.root.resolve(), args.groups, args.include_infrastructure_failures)
