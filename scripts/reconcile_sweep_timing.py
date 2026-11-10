"""Attribute explicitly bounded, unfinished planning turns without changing episode results."""

import argparse
import json
from pathlib import Path


def reconcile(data):
    if not data.get('astra_turns_without_completion_event'):
        return data
    completed, boundaries, latest = set(), [], None
    with (Path(data['path']) / 'events.jsonl').open() as stream:
        for line in stream:
            event = json.loads(line)
            if event['kind'] == 'codex_turn_start':
                latest = event
            elif event['kind'] == 'codex_turn_end':
                completed.add(event['turn_id'])
            elif (event['kind'] == 'recovery_session_budget' and event.get('reason') == 'revision_deadline'
                  and latest and latest['context']['intent'] == 'revise_code'):
                boundaries.append(dict(turn_id=latest['turn_id'], start_event=latest['id'],
                                       end_event=event['id'], boundary='revision_deadline',
                                       seconds=(event['time_ns'] - latest['time_ns']) / 1e9))
    boundaries = [b for b in boundaries if b['turn_id'] not in completed]
    if not boundaries:
        return data
    seconds = sum(b['seconds'] for b in boundaries)
    original = dict(data.get('timing_reconciliation', {}).get('original_components', data['wall_components_seconds']))
    if seconds < 0 or seconds > original['other_and_unattributed'] + 1e-6:
        raise ValueError('Explicit incomplete planning span exceeds available unattributed wall time')
    data['wall_components_seconds'] = dict(original)
    data['wall_components_seconds']['astra_paused'] += seconds
    data['wall_components_seconds']['other_and_unattributed'] -= seconds
    data['timing_reconciliation'] = dict(original_components=original, reclassified_seconds=seconds,
                                        explicit_timeout_spans=boundaries,
                                        note='Moved an observed planning interval from other to astra_paused. '
                                             'No completed-turn event was invented; result, wall time and calls are unchanged.')
    note = ' Explicit revision_deadline spans without turn-end events are attributed to paused Astra.'
    if note not in data['wall_accounting_note']:
        data['wall_accounting_note'] += note
    return data


def main(root):
    from run_seed_sweep import episode_report

    changes = []
    for path in sorted((root / 'analysis').glob('seed*.json')):
        data = reconcile(json.loads(path.read_text()))
        if 'timing_reconciliation' in data:
            episode_report(root, data)
            report_path = root / 'analysis' / f'seed{data["seed"]:02d}.md'
            with report_path.open('a') as stream:
                stream.write('\n## 超时计时修正\n\n'
                             f'有明确结束事件的规划超时合计 {data["timing_reconciliation"]["reclassified_seconds"]:.3f} 秒，'
                             '从其余耗时移入暂停 Astra。完整 Episode 时间、结果、调用数保持不变；'
                             '“已完成调用耗时”仍只统计有完成事件的调用。\n')
            changes.append(dict(seed=data['seed'], **data['timing_reconciliation']))
    (root / 'timing-reconciliation.json').write_text(json.dumps(changes, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(episodes=len(changes), seconds=sum(c['reclassified_seconds'] for c in changes))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root.resolve())
