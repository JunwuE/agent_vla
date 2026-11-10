"""Merge completed frozen-snapshot experiments after the full batch, preserving candidates."""

import argparse
import hashlib
import json
import shutil
import time
from collections import Counter
from pathlib import Path

from run_concurrent_sweep import merge_libraries


def inventory(root):
    manifests = list((root / 'skills').glob('*/*/manifest.json'))
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(root.rglob('*')) if p.is_file()}
    return dict(
        skill_versions=len(manifests),
        distinct_policy_source_hashes=len({hashlib.sha256(p.with_name('policy.py').read_bytes()).hexdigest()
                                          for p in manifests}),
        statuses=dict(Counter(json.loads(p.read_text()).get('status') for p in manifests)),
        experiences=len(list((root / 'experiences').glob('*.json'))),
        run_records=sum(len(p.read_text().splitlines()) for p in (root / 'skills').glob('*/*/runs.jsonl')),
        hashes=hashes,
    )


def main(root, target):
    root, target = root.resolve(), target.resolve()
    plan = json.loads((root / 'plan.json').read_text())
    end = json.loads((root / 'batch-end.json').read_text())
    if sorted(end['completed']) != sorted(plan['seeds']) or not end['pinned_source_and_initial_memory_unchanged']:
        raise RuntimeError('Full unchanged batch must finish before retaining evidence')
    for seed in plan['seeds']:
        if not (root / 'analysis' / f'seed{seed:02d}.json').exists():
            raise RuntimeError(f'Missing analysis for seed {seed}')
    receipt = root / 'retained-evidence.json'
    if receipt.exists():
        raise RuntimeError('Retention already recorded; inspect receipt instead of rerunning')
    before = inventory(target)
    shared = root / 'shared'
    # Preparation may have made an empty shared/ directory, and an interrupted
    # merge may have populated only skills/. Initialize each component without
    # overwriting retained source or run evidence, then collision-check the seed.
    for name in ('skills', 'experiences', 'tasks'):
        (shared / name).mkdir(parents=True, exist_ok=True)
    for source in (root / 'initial' / 'tasks').rglob('*'):
        if not source.is_file():
            continue
        destination = shared / 'tasks' / source.relative_to(root / 'initial' / 'tasks')
        if destination.exists():
            if destination.read_bytes() != source.read_bytes():
                raise RuntimeError('Shared Task Memory differs from frozen initial rules')
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    merge_libraries(root / 'initial' / 'skills', root / 'initial' / 'experiences',
                    shared / 'skills', shared / 'experiences')
    for seed in plan['seeds']:
        source = root / 'inputs' / f'seed{seed:02d}'
        merge_libraries(source / 'skills', source / 'experiences', shared / 'skills', shared / 'experiences')
    # Version identities and existing source are collision-checked by merge_libraries.
    # Runs are unioned by content, without promoting candidate status or changing task rules.
    merge_libraries(shared / 'skills', shared / 'experiences', target / 'skills', target / 'experiences')
    after = inventory(target)
    data = dict(time_ns=time.time_ns(), target=str(target), shared=str(shared),
                before=before, after=after, merged=inventory(shared),
                note='Appended candidates and execution evidence after all seeds finished. '
                     'No status promotion, task-memory rule update, or in-batch input mutation.')
    receipt.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: {n: v for n, v in data[k].items() if n != 'hashes'}
                      for k in ['before', 'after', 'merged']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('target', type=Path)
    args = parser.parse_args()
    main(args.root, args.target)
