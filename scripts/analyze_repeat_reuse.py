"""Measure executed source reuse and planning time from completed repeat episodes."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

from reconcile_sweep_timing import reconcile

EDIT_TOOLS = {'write_program', 'patch_program', 'patch_program_batch', 'write_skill_package', 'patch_skill_package'}


def main(root):
    root = root.resolve()
    initial = defaultdict(list)
    success_layouts = defaultdict(set)
    for path in (root / 'initial/skills').glob('*/*/policy.py'):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        initial[digest].append(str(path.parent.relative_to(root / 'initial/skills')))
        runs = path.with_name('runs.jsonl')
        if runs.exists():
            for line in runs.read_text().splitlines():
                run = json.loads(line)
                report = run.get('evidence', {}).get('report', {})
                if report.get('verified') and run.get('layout') is not None:
                    success_layouts[digest].add(str(run['layout']))
    episodes, all_attempts, turn_groups = [], [], defaultdict(lambda: {'count': 0, 'seconds': 0.0})
    tool_times = defaultdict(lambda: {'count': 0, 'seconds': 0.0})
    finished_turns, failed_episodes = [], []
    for path in sorted((root / 'analysis').glob('seed*.json')):
        analysis = reconcile(json.loads(path.read_text()))
        seed = analysis['seed']
        turns, attempts = {}, []
        tool_starts = defaultdict(deque)
        pending_invocation = None
        stop_reasons = []
        with (Path(analysis['path']) / 'events.jsonl').open() as stream:
            for line in stream:
                event = json.loads(line)
                kind = event['kind']
                if kind == 'codex_turn_start':
                    turns[event['turn_id']] = dict(intent=event['context']['intent'],
                                                  start=event['time_ns'], tools=Counter())
                elif kind == 'codex_tool_start':
                    tool_starts[(event['turn_id'], event['call_id'], event['name'])].append(event['time_ns'])
                elif kind == 'codex_tool':
                    if event['turn_id'] in turns:
                        turns[event['turn_id']]['tools'][event['name']] += 1
                    queue = tool_starts[(event['turn_id'], event['call_id'], event['name'])]
                    if queue:
                        elapsed = (event['time_ns'] - queue.popleft()) / 1e9
                        tool_times[event['name']]['count'] += 1
                        tool_times[event['name']]['seconds'] += elapsed
                elif kind == 'codex_turn_end' and event['turn_id'] in turns:
                    turn = turns[event['turn_id']]
                    turn['seconds'] = (event['time_ns'] - turn['start']) / 1e9
                elif kind == 'skill_invocation':
                    pending_invocation = {k: event[k] for k in ['skill_id', 'version']}
                elif kind == 'stop' and event.get('reason') not in {'success', 'stopped'}:
                    stop_reasons.append(event.get('reason'))
                elif kind == 'recovery_result':
                    sha = event['sha256']
                    row = dict(seed=seed, program_version=event['program_version'], sha256=sha,
                               steps=event['executed_steps'], seconds=event['seconds'],
                               verified=event['verified'], result=event['result'],
                               exact_initial_source=sha in initial, initial_versions=initial.get(sha, []),
                               source_seen_earlier_in_episode=any(a['sha256'] == sha for a in attempts),
                               initial_success_layouts=sorted(success_layouts.get(sha, [])),
                               direct_invocation=pending_invocation, error=event.get('error'))
                    attempts.append(row)
                    pending_invocation = None
        for span in analysis.get('timing_reconciliation', {}).get('explicit_timeout_spans', []):
            if span['turn_id'] in turns:
                turns[span['turn_id']]['seconds'] = span['seconds']
                turns[span['turn_id']]['timed_out'] = True
        for turn in turns.values():
            if 'seconds' not in turn:
                continue
            group = turn['intent']
            if group == 'revise_code':
                group += '_with_source_edits' if any(turn['tools'][k] for k in EDIT_TOOLS) else '_without_source_edits'
            if turn.get('timed_out'):
                group += '_timed_out'
            turn_groups[group]['count'] += 1
            turn_groups[group]['seconds'] += turn['seconds']
            finished_turns.append(dict(seed=seed, group=group, seconds=turn['seconds'], tools=dict(turn['tools'])))
        if not analysis['result']['success']:
            failed_episodes.append(dict(seed=seed, result=analysis['result'],
                                        source_steps=analysis['source_steps'], stop_reasons=stop_reasons,
                                        tool_errors=analysis['tool_errors'],
                                        last_program_facts=analysis['recoveries'][-1]['last_program_facts']
                                        if analysis['recoveries'] else []))
        all_attempts.extend(attempts)
        episodes.append(dict(seed=seed, attempts=len(attempts),
                             exact_initial_source_attempts=sum(a['exact_initial_source'] for a in attempts),
                             in_episode_repeat_source_attempts=sum(a['source_seen_earlier_in_episode'] for a in attempts),
                             reused_source_attempts=sum(a['exact_initial_source'] or a['source_seen_earlier_in_episode']
                                                        for a in attempts),
                             direct_invocations=sum(a['direct_invocation'] is not None for a in attempts),
                             verified_direct_invocations=sum(a['direct_invocation'] is not None and a['verified'] for a in attempts),
                             verified_initial_source_attempts=sum(a['exact_initial_source'] and a['verified'] for a in attempts)))
    total = {key: sum(e[key] for e in episodes) for key in (
        'attempts', 'exact_initial_source_attempts', 'in_episode_repeat_source_attempts',
        'reused_source_attempts', 'direct_invocations',
        'verified_direct_invocations', 'verified_initial_source_attempts')}
    total['completed_episodes'] = len(episodes)
    total['code_episodes'] = sum(e['attempts'] > 0 for e in episodes)
    result = dict(total=total, turn_groups=dict(turn_groups), astra_tool_times=dict(tool_times),
                  episodes=episodes, attempts=all_attempts, failed_episodes=failed_episodes,
                  longest_turns=sorted(finished_turns, key=lambda t: -t['seconds'])[:20],
                  note='Exact source equality is a reuse metric, not proof of compatible parameters or success. '
                       'Revision turns without source edits still include geometry queries, selection and parameter changes. '
                       'Tool intervals measure tool execution, not the model reasoning that chooses geometry parameters. '
                       'Explicit revision deadlines are included with a timed_out label. '
                       'Turn durations include tool time; asynchronous reviews overlap actions and must not be summed into wall time.')
    (root / 'reuse-analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(total=total, turn_groups=dict(turn_groups)), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
