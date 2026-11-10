"""Audit causal routing and control authority without treating reward scores as failure labels."""

import argparse
import json
from collections import Counter
from pathlib import Path


def audit(root):
    config = json.loads((root / 'plan.json').read_text())['config']
    review_limit = config['planner'].get('review_tool_limit')
    review_tools = {'observe', 'view_observation_image', 'read_execution', 'read_context',
                    'assess_phase', 'record_note', 'evaluate_subtask', 'submit_review'}
    episodes = []
    counts = Counter()
    for path in sorted((root / 'episodes').glob('*/result.json')):
        episode = path.parent
        result = json.loads(path.read_text())
        route, scores, reviews, plans, recovery, turns = [], [], [], [], [], []
        decisions, routed, commands, decision_events = {}, {}, {}, {}
        actual_sources = Counter()
        executed_ids = set()
        last_action_event = -1
        profiles, turn_profiles, tool_counts = {}, {}, Counter()
        checks = dict(causal_flora_windows=True, action_owner_matches=True,
                      receipt_matches_command=True, keep_never_dispatches_astra=True,
                      nonkeep_has_route=True, model_claims_never_count_as_success=True,
                      quick_review_tool_limit=True, quick_review_authority=True)
        owner, epoch, step = None, None, 0
        native_success = False
        for line in (episode / 'events.jsonl').open():
            event = json.loads(line)
            kind = event['kind']
            if kind == 'observation':
                step, native_success = event['step'], event['success']
            if kind == 'control_claim':
                owner, epoch = event['owner'], event['epoch']
            elif kind == 'action_command':
                last_action_event = event['id']
                checks['action_owner_matches'] &= event['source'] == owner and event['control_epoch'] == epoch
                commands[event['id']] = (event['source'], event['action'])
            elif kind == 'action_executed':
                checks['receipt_matches_command'] &= commands.get(event['command_id']) == (event['source'], event['action'])
                checks['receipt_matches_command'] &= event['command_id'] not in executed_ids
                executed_ids.add(event['command_id'])
                actual_sources[event['source']] += 1
            elif kind == 'flora_request':
                segment = event['segment']
                end = event['binding']['observation_step']
                samples = segment['sampled_steps']
                checks['causal_flora_windows'] &= (segment['end_step'] == end <= step
                    and segment['start_step'] <= samples[0] <= samples[-1] <= end
                    and all(a < b for a, b in zip(samples, samples[1:])))
            elif kind == 'flora_monitor_result':
                value = event['result']
                scores.append(dict(event_id=event['id'], snapshot_step=event['binding']['observation_step'],
                    arrival_step=step, status=value.get('status'), score=value.get('score'),
                    goal=value.get('task'), scope=value.get('scope'), request_id=value.get('request_id')))
            elif kind == 'decision':
                decision = event['decision']
                decisions[decision['binding']['request_id']] = decision
                decision_events[decision['binding']['request_id']] = event['id']
            elif kind in {'review_coalesced', 'review_begin', 'intervention_begin', 'review_deferred'}:
                decision = event.get('decision')
                if decision:
                    identity = decision['binding']['request_id']
                    routed[identity] = kind
                    checks['keep_never_dispatches_astra'] &= decision['label'] != 'KEEP'
                    route.append(dict(event_id=event['id'], step=step, kind=kind,
                        trigger=decision['trigger'], score_reason=decision.get('fallback_reason'),
                        outcome=event.get('reason'), request_id=identity))
            elif kind in {'review_end', 'review_discarded', 'review_failed', 'review_cancelled'}:
                value = event.get('result', {})
                reviews.append(dict(event_id=event['id'], step=step, kind=kind,
                    snapshot_step=event.get('binding', {}).get('observation_step'),
                    action=value.get('action'), assessment=value.get('assessment'), reason=value.get('reason'),
                    seconds=event.get('seconds'), rejection_reason=event.get('reason')))
            elif kind == 'codex_thread':
                profiles[event['thread_id']] = event.get('tool_profile', 'full')
            elif kind == 'codex_turn_start':
                turn_profiles[event['turn_id']] = profiles.get(event['thread_id'], 'full')
                turns.append(dict(event_id=event['id'], step=step, intent=event['context']['intent']))
            elif kind == 'codex_tool_start':
                tool_counts[event['turn_id']] += 1
                if turn_profiles.get(event['turn_id']) == 'review':
                    checks['quick_review_tool_limit'] &= review_limit is not None and tool_counts[event['turn_id']] <= review_limit
            elif kind == 'codex_tool' and turn_profiles.get(event['turn_id']) == 'review':
                checks['quick_review_authority'] &= event['name'] in review_tools or bool(event['result'].get('error'))
            elif kind == 'intervention_plan':
                plans.append(dict(event_id=event['id'], step=step, plan=event['plan']))
            elif kind == 'recovery_result':
                recovery.append({k: event.get(k) for k in ('id', 'verified', 'error', 'executed_steps', 'seconds')})
            elif kind == 'budget_exhausted' and event.get('outcome') == 'stop_without_further_actions':
                # Older profiles raise at the routing budget guard instead of emitting
                # review_deferred. Count only the final decision with no intervening action;
                # do not hide earlier missing routes behind eventual episode termination.
                if decision_events:
                    identity = max(decision_events, key=decision_events.get)
                    decision = decisions[identity]
                    if (identity not in routed and decision['label'] != 'KEEP'
                            and decision_events[identity] > last_action_event):
                        routed[identity] = 'episode_stopped_on_budget'
                        route.append(dict(event_id=event['id'], step=step, kind='episode_stopped_on_budget',
                            trigger=decision['trigger'], score_reason=decision.get('fallback_reason'),
                            outcome=event['budget'], request_id=identity,
                            association='derived from last non-KEEP decision with no intervening action'))
        unresolved = [d for key, d in decisions.items() if d['label'] != 'KEEP' and key not in routed]
        checks['nonkeep_has_route'] = not unresolved
        checks['model_claims_never_count_as_success'] = native_success == result['success']
        checks['receipt_matches_command'] &= executed_ids == set(commands)
        row = dict(episode=episode.name, result=result, checks=checks, scores=scores, route=route,
            reviews=reviews, plans=plans, recoveries=recovery, turns=turns,
            decision_labels=dict(Counter(d['label'] for d in decisions.values())),
            decision_reasons=dict(Counter(d.get('fallback_reason') for d in decisions.values())),
            route_outcomes=dict(Counter(r['kind'] + ':' + str(r.get('outcome')) for r in route)),
            review_actions=dict(Counter(str(r['action']) + ':' + str(r['assessment']) for r in reviews if r['kind'] == 'review_end')),
            executed_sources=dict(actual_sources), unhandled_decisions=unresolved)
        for name, passed in checks.items():
            counts[name + '_passed'] += passed
        episodes.append(row)
    successful_vla = [r for r in episodes if r['result']['success'] and set(r['executed_sources']) == {'vla'}]
    values = [s for r in successful_vla for s in r['scores'] if s['status'] == 'scored']
    summary = dict(episodes=len(episodes), all_checks_passed=all(all(r['checks'].values()) for r in episodes),
        score_histogram=dict(Counter(s['score'] for r in episodes for s in r['scores'] if s['status'] == 'scored')),
        successful_vla_only_episodes=len(successful_vla), successful_vla_only_scored_windows=len(values),
        low_score_windows_in_successful_vla_only=sum(s['score'] <= 2 for s in values),
        review_outcomes=dict(sum((Counter(r['review_actions']) for r in episodes), Counter())),
        check_pass_counts=dict(counts))
    notes = ['Low scores on a later-successful trajectory do not by themselves establish a false positive at that earlier time.',
             'Astra review labels are judgments, not independent ground truth.',
             'Phase-specific high scores cannot be evaluated against whole-task success without separate phase labels.',
             'A control lease is distinct from actual action execution; planning may own a lease without issuing commands.']
    (root / 'routing-audit.json').write_text(json.dumps(dict(summary=summary, episodes=episodes, notes=notes), ensure_ascii=False, indent=2) + '\n')
    lines = ['# 监督路由核验', '', '| episode | 成功 | Flora | 低分≤2 | Astra | 实际动作来源 | 检查 |',
             '|---|---|---:|---:|---:|---|---|']
    for row in episodes:
        lines.append(f"| {row['episode']} | {row['result']['success']} | {len(row['scores'])} | "
            f"{sum(s.get('score', 6) <= 2 for s in row['scores'] if s['status'] == 'scored')} | "
            f"{len(row['turns'])} | {row['executed_sources']} | {all(row['checks'].values())} |")
    lines += ['', f"在 {len(successful_vla)} 条最终仅靠 VLA 动作完成的轨迹中，{len(values)} 个评分窗口有 "
              f"{summary['low_score_windows_in_successful_vla_only']} 个低分。低分并不等同于需要代码纠正；这不是正式误报率。",
              '', '完整逐事件路由与检查：[routing-audit.json](routing-audit.json)。']
    (root / 'routing-audit.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, ensure_ascii=False))
    if not summary['all_checks_passed']:
        raise SystemExit('Routing audit failed; inspect per-episode checks')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
