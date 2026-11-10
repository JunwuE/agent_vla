"""Read live sweep progress; never modify a running episode."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path


def main(root, brief=False):
    status = json.loads((root/'status.json').read_text())
    print('completed', len(status['completed']), 'active', len(status['active']), 'queued', status['remaining'])
    if (root/'summary.json').exists():
        print(json.dumps(json.loads((root/'summary.json').read_text())['groups'], ensure_ascii=False))
    samples = []
    for item in status['active']:
        episode = root/'episodes'/item['key']
        counts, last, phase, step, error = Counter(), {}, None, 0, None
        path = episode/'events.jsonl'
        if path.exists():
            with path.open() as stream:
                for line in stream:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        # A writer may be midway through the final line.
                        continue
                    last = event
                    counts[event['kind']] += 1
                    if event['kind'] == 'observation':
                        step = event['step']
                    if event['kind'] == 'memory':
                        phase = event['state'].get('phase_memory', {}).get('active_phase')
                    if event['kind'] == 'recovery_result':
                        error = event.get('error')
        row = dict(case=item['key'], step=step, astra=counts['codex_turn_start'],
                   attempts=counts['recovery_result'], phase=phase, last=last.get('kind'),
                   last_event_age_seconds=round((time.time_ns()-last.get('time_ns', time.time_ns()))/1e9),
                   last_program_stop=str(error)[:100])
        samples.append(row)
        if not brief:
            print(json.dumps(row, ensure_ascii=False))
    if brief:
        print(json.dumps(dict(active_steps={r['case']:r['step'] for r in samples},
                              oldest_event_seconds=max((r['last_event_age_seconds'] for r in samples), default=0),
                              quiet_workers=[r for r in samples if r['last_event_age_seconds'] > 180]),
                         ensure_ascii=False))
    with (root/'progress-samples.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(time_ns=time.time_ns(), completed=len(status['completed']), active=samples))+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--brief', action='store_true')
    args = parser.parse_args()
    main(args.root, args.brief)
