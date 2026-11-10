"""Replay historical planner inputs through the new index, without model calls."""

import argparse
import json
import statistics
from pathlib import Path

from vla_agent.data import dumps
from vla_agent.memory_context import compact_historical_memory


def compare(source, destination):
    records = []
    for path in sorted(source.glob('episodes/*/events.jsonl')):
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['kind'] != 'codex_turn_start':
                    continue
                before = row['context']
                if before.get('intent') != 'initialize_plan':
                    continue
                after = dict(before, memory=compact_historical_memory(before['memory']))
                metrics = after['memory']['historical_context']
                records.append(dict(episode=path.parent.name, before_chars=len(dumps(before)),
                                    after_chars=len(dumps(after)), **metrics))
                break
    result = dict(episodes=len(records), mean_before_chars=statistics.mean(r['before_chars'] for r in records),
                  mean_after_chars=statistics.mean(r['after_chars'] for r in records),
                  mean_history_before_chars=statistics.mean(r['original_chars'] for r in records),
                  mean_history_after_chars=statistics.mean(r['sent_chars'] for r in records),
                  note='Characters, not tokens. Offline same-input transformation; no model latency/quality claim. '
                       'Images, current state and actions unchanged. Tools and instructions excluded.', records=records)
    destination.write_text(dumps(result))
    print(dumps({k: v for k, v in result.items() if k != 'records'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    compare(args.source, args.destination)
