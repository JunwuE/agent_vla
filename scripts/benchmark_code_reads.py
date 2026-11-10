"""Replay recorded read_program responses through the new bounded presentation.

No model or robot calls. Measures payload characters, not tokens or end-to-end
latency. All pages of the originally requested module are counted and checked.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from vla_agent.code_tools import code_view


def size(value):
    return len(json.dumps(value, ensure_ascii=False))


def main(root, output):
    rows = []
    for episode in sorted((root / 'episodes').iterdir()):
        if not (episode / 'result.json').is_file():
            continue
        with (episode / 'events.jsonl').open() as stream:
            for line in stream:
                event = json.loads(line)
                if event['kind'] != 'codex_tool' or event['name'] != 'read_program':
                    continue
                old, args = event['result'], event['arguments']
                path = args.get('path')
                # Paired comparisons require the full original requested module.
                if not path or not isinstance(old, dict) or old.get('file', {}).get('path') != path:
                    continue
                source = old['file']['source']
                value = dict(old, package_files={path: source})
                offset, fragments, responses = 0, [], []
                while offset is not None:
                    new = code_view(value, path, offset=offset)
                    responses.append(new)
                    fragments.append(new['file']['source'])
                    offset = new['file']['next_offset']
                assert ''.join(fragments) == source
                rows.append(dict(episode=episode.name, event_id=event['id'], version=old.get('version'),
                    path=path, old_chars=size(old), new_chars=sum(size(r) for r in responses),
                    new_pages=len(responses), source_chars=len(source), exact_source_preserved=True))
    if not rows:
        raise ValueError('No historical selected-module reads found')
    old = sum(r['old_chars'] for r in rows)
    new = sum(r['new_chars'] for r in rows)
    result = dict(root=str(root), matched_reads=len(rows), old_chars=old, new_chars_all_pages=new,
                  reduction_percent=100 * (1 - new / old), total_pages=sum(r['new_pages'] for r in rows),
                  old_mean_chars=old / len(rows), new_mean_chars_all_pages=new / len(rows),
                  new_p90_chars=float(np.quantile([r['new_chars'] for r in rows], .9)),
                  all_source_preserved=all(r['exact_source_preserved'] for r in rows), rows=rows,
                  notes=['Paired selected-file reads only; interface-only calls excluded.',
                         'All pages included, not just the shorter first page.',
                         'Duplicate docs/contracts omitted; separately retrievable through explicit views.',
                         'No model tokens, physical task outcomes or wall-time savings measured.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.root.resolve(), args.output.resolve())
