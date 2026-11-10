"""Small live status for a seed sweep; never print images or complete model contexts."""

import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root
    status = json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
    summary = json.loads((root/'summary.json').read_text()) if (root/'summary.json').exists() else {}
    output = dict(state=status.get('state'), completed=summary.get('completed', 0), successes=summary.get('successes', 0))
    if summary.get('rows'):
        output['last_finished'] = [{k:r[k] for k in ('seed','success','steps','wall_seconds','astra_turns','attempts')}
                                   for r in summary['rows'][-3:]]
    output['active'] = []
    for active in status.get('active', [status] if status.get('state') == 'running' else []):
        current = {k: active[k] for k in ('seed', 'release')}
        current['elapsed_seconds'] = round((time.time_ns()-active['started_ns'])/1e9, 1)
        output['active'].append(current)
        episode = root/'episodes'/f'seed{active["seed"]:02d}'
        path = episode/'events.jsonl'
        if path.exists():
            with path.open('rb') as handle:
                size = path.stat().st_size
                handle.seek(max(0, size-2_000_000))
                raw = handle.read().splitlines()
            rows = []
            for line in raw:
                try:
                    rows.append(json.loads(line))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
            steps = [r.get('step', r.get('observation_step')) for r in rows if r['kind'] in ('observation', 'action_executed')]
            current['latest_step'] = next((v for v in reversed(steps) if v is not None), None)
            current['latest_events'] = []
            for row in rows:
                if row['kind'] in ('codex_tool', 'recovery_result', 'review_end', 'codex_turn_end', 'codex_turn_start'):
                    item = {k: row[k] for k in ('kind', 'name', 'verified', 'executed_steps', 'error', 'seconds') if k in row}
                    if row['kind'] == 'codex_turn_start':
                        item['intent'] = row['context']['intent']
                    current['latest_events'].append(item)
            current['latest_events'] = current['latest_events'][-3:]
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
