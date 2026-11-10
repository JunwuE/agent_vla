"""Read retained events to audit score routing and planner latency; makes no model calls."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def stats(values):
    values = list(values)
    if not values:
        return dict(count=0)
    return dict(count=len(values), total=sum(values), mean=float(np.mean(values)),
                median=float(np.median(values)), p90=float(np.quantile(values, .9)),
                maximum=max(values))


def chars(value):
    # Match vla_agent.data.dumps after the event JSON has been decoded.
    return len(json.dumps(value, ensure_ascii=False))


def main(root):
    counters = defaultdict(Counter)
    numbers = defaultdict(list)
    reviews, turns, initializations, handoffs, diagnostics = [], [], [], [], []
    for episode in sorted((root / 'episodes').glob('seed[0-9][0-9]')):
        seed = int(episode.name[4:])
        rows = [json.loads(line) for line in (episode / 'events.jsonl').open()]
        outcome = json.loads((episode / 'result.json').read_text())
        launch = json.loads(episode.with_name(episode.name + '-launch.json').read_text())
        turnmap, reviewmap, tool_starts = {}, {}, {}
        monitor_bindings, request_kinds = set(), {}
        first = next(e for e in rows if e['kind'] == 'observation')
        first_action = next((e for e in rows if e['kind'] == 'action_command'), None)
        last_review, active_intervention = None, None
        for e in rows:
            kind = e['kind']
            if kind.startswith('flora_'):
                counters['flora_event_counts'][kind] += 1
            if kind == 'decision_window' and e.get('backend') == 'flora':
                monitor_bindings.add(e['binding']['request_id'])
            if kind == 'flora_request':
                request_kind = ('monitor' if e['binding']['request_id'] in monitor_bindings else 'diagnostic')
                request_kinds[e['request_id']] = request_kind
                counters['flora_request_kinds'][request_kind] += 1
                numbers['flora_sampled_frames'].append(len(e['segment']['sampled_steps']))
            if kind == 'flora_result':
                result = e['result']
                request_kind = request_kinds[result['request_id']]
                numbers['flora_' + request_kind + '_client_seconds'].append(result['seconds'])
                response_path = episode / 'flora' / (result['request_id'] + '.response.json')
                response = json.loads(response_path.read_text())
                numbers['flora_' + request_kind + '_server_seconds'].append(response['seconds'])
            if kind == 'flora_monitor_result':
                counters['monitor_acceptance']['accepted' if e['accepted'] else 'stale'] += 1
                counters['monitor_status'][e['result'].get('status')] += 1
                if e['accepted']:
                    counters['accepted_scores'][str(e['result'].get('score'))] += 1
            if kind == 'decision' and e.get('backend') == 'flora':
                counters['flora_decisions'][e['decision']['label']] += 1
                counters['flora_reasons'][e['decision']['fallback_reason']] += 1
            if kind == 'review_coalesced':
                counters['coalesced'][e['decision']['trigger'] + ':' + e['reason']] += 1
            if kind == 'review_begin':
                reviewmap[e['review']] = dict(seed=seed, review=e['review'],
                    trigger=e['decision']['trigger'], reason=e['decision']['fallback_reason'],
                    snapshot_step=e['binding']['observation_step'], start_ns=e['time_ns'],
                    request_id=e['binding']['request_id'], status='unfinished',
                    episode_success=outcome['success'])
            if kind in ('review_end', 'review_discarded', 'review_cancelled', 'review_failed'):
                r = reviewmap[e['review']]
                r['status'] = kind.removeprefix('review_')
                r['seconds'] = (e['time_ns'] - r['start_ns']) / 1e9
                if 'result' in e:
                    result = e['result']
                    r.update(action=result['action'], assessment=result['assessment'],
                             rationale=result['reason'], phase=result['phase'])
                if kind == 'review_end':
                    r['return_step'] = e['current_step']
                    last_review = r
            if kind == 'intervention_begin':
                active_intervention = None
                if e['decision']['trigger'] == 'astra_review_request':
                    assert last_review and last_review['action'] == 'request_intervention'
                    active_intervention = dict(seed=seed, review=last_review['review'],
                        trigger=last_review['trigger'], reason=last_review['reason'],
                        step=e['context_binding']['observation_step'], episode_success=outcome['success'])
            if kind == 'intervention_plan' and active_intervention is not None:
                active_intervention['kind'] = e['plan']['kind']
                handoffs.append(active_intervention)
                active_intervention = None
            if kind == 'codex_turn_start':
                c = e['context']
                memory = c.get('memory', {})
                t = dict(seed=seed, turn_id=e['turn_id'], intent=c['intent'],
                    start_ns=e['time_ns'], context_chars=chars(c), image_count=len(c['frames']),
                    memory_chars={k: chars(v) for k, v in memory.items()},
                    tools=[], tool_seconds=0., item_types=Counter(), status='unfinished')
                turnmap[e['turn_id']] = t
            if kind == 'codex_item' and e['turn_id'] in turnmap:
                turnmap[e['turn_id']]['item_types'][e['item'].get('type', 'unknown')] += 1
            if kind == 'codex_tool_start':
                tool_starts[e['call_id']] = e['time_ns']
                t = turnmap[e['turn_id']]
                t.setdefault('first_tool_seconds', (e['time_ns'] - t['start_ns']) / 1e9)
            if kind == 'codex_tool':
                t = turnmap[e['turn_id']]
                t['tools'].append(e['name'])
                t['tool_seconds'] += (e['time_ns'] - tool_starts[e['call_id']]) / 1e9
                if e['name'] == 'evaluate_subtask':
                    diagnostics.append(dict(seed=seed, intent=t['intent'], result=e['result']))
                if e['name'] in ('submit_intervention', 'submit_review'):
                    t['proposal_seconds'] = (e['time_ns'] - t['start_ns']) / 1e9
                    t['proposal_kind'] = e['arguments'].get('kind', e['arguments'].get('action'))
            if kind == 'codex_turn_end' and e['turn_id'] in turnmap:
                t = turnmap[e['turn_id']]
                t['status'] = e['status']
                t['seconds'] = (e['time_ns'] - t['start_ns']) / 1e9
                if 'proposal_seconds' in t:
                    t['after_proposal_seconds'] = t['seconds'] - t['proposal_seconds']
        for r in reviewmap.values():
            counters['review_by_trigger'][r['trigger']] += 1
            counters['review_status'][r['trigger'] + ':' + r['status']] += 1
            if r['status'] == 'end':
                counters['review_actions'][r['trigger'] + ':' + r['action']] += 1
                counters['review_assessments'][r['trigger'] + ':' + r['assessment']] += 1
            # Phase updates are staged during a review; retain only accepted reviews.
            phase_updates = []
            for t in turnmap.values():
                if t['intent'] == 'review_running' and t['start_ns'] >= r['start_ns']:
                    if t['start_ns'] < r['start_ns'] + r.get('seconds', 0) * 1e9:
                        phase_updates = [e['arguments'] for e in rows if e['kind'] == 'codex_tool'
                                         and e['turn_id'] == t['turn_id'] and e['name'] == 'assess_phase'
                                         and not e['result'].get('error')]
                        break
            r['phase_updates'] = phase_updates if r['status'] == 'end' else []
            reviews.append(r)
        turns.extend(turnmap.values())
        t = next(v for v in turnmap.values() if v['intent'] == 'initialize_plan')
        init = dict(seed=seed, launch_to_first_observation_seconds=(first['time_ns'] - launch['started_ns']) / 1e9,
            first_observation_to_turn_seconds=(t['start_ns'] - first['time_ns']) / 1e9,
            initial_turn_seconds=t['seconds'], context_chars=t['context_chars'],
            memory_chars=t['memory_chars'], tool_names=t['tools'], image_count=t['image_count'],
            tool_schema_chars=len((episode / 'codex/tools.json').read_text()),
            instruction_chars=len((episode / 'codex/instructions.md').read_text()),
            tool_count=len(json.loads((episode / 'codex/tools.json').read_text())))
        if first_action:
            init['launch_to_first_action_seconds'] = (first_action['time_ns'] - launch['started_ns']) / 1e9
            init['observation_to_first_action_seconds'] = (first_action['time_ns'] - first['time_ns']) / 1e9
        initializations.append(init)
    for h in handoffs:
        counters['handoffs'][h['trigger'] + ':' + h['kind']] += 1
    for r in reviews:
        if r['status'] == 'end' and any(p.get('next_phase') for p in r['phase_updates']):
            counters['reviews_advancing_phase'][r['trigger']] += 1
    by_intent = {}
    for intent in sorted({t['intent'] for t in turns}):
        group = [t for t in turns if t['intent'] == intent and 'seconds' in t]
        by_intent[intent] = {key: stats(t[key] for t in group if key in t)
                             for key in ('seconds', 'first_tool_seconds', 'tool_seconds',
                                         'after_proposal_seconds', 'context_chars', 'image_count')}
        by_intent[intent]['tool_calls'] = dict(Counter(name for t in group for name in t['tools']))
    data = dict(counters={k: dict(v) for k, v in counters.items()},
        measurements={k: stats(v) for k, v in numbers.items()}, turn_statistics=by_intent,
        initialization_statistics={key: stats(x[key] for x in initializations)
                                   for key in initializations[0] if key not in ('seed', 'memory_chars', 'tool_names')},
        initialization_memory_chars={k: stats(x['memory_chars'].get(k, 0) for x in initializations)
                                     for k in initializations[0]['memory_chars']},
        reviews=reviews, handoffs=handoffs, turns=turns, initializations=initializations,
        diagnostics=diagnostics,
        notes=['Astra continue is not ground-truth proof of a false positive; phase updates may be useful.',
               'Success after a triggered intervention is not a causal benefit attributable to Flora.',
               'Turn time includes provider/network/model/tool gaps, not isolated GPU reasoning time.',
               'Context size is serialized character count, not tokens; accumulated thread context is larger.',
               'Completed-turn latency statistics exclude cancelled turns and explicit planning timeouts.'])
    (root / 'supervision-efficiency.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(counters=data['counters'], turn_statistics=by_intent,
                         initialization_statistics=data['initialization_statistics']), ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    main(p.parse_args().root)
