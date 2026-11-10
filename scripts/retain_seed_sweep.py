"""Append completed, audited development skills/experience to the project libraries."""

import argparse
import hashlib
import json
from pathlib import Path

from run_concurrent_sweep import merge_libraries


def inventory(skills, experiences):
    immutable = {}
    for name in ('manifest.json', 'policy.py'):
        for path in skills.glob(f'*/*/{name}'):
            immutable[str(path.relative_to(skills))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(skill_versions=len(list(skills.glob('*/*/manifest.json'))),
                experiences=len(list(experiences.glob('*.json'))), immutable=immutable)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--skills', type=Path, default=Path('skills'))
    parser.add_argument('--experiences', type=Path, default=Path('memory/close_toaster/experiences'))
    args = parser.parse_args()
    root = args.root.resolve()
    status = json.loads((root/'status.json').read_text())
    audit = json.loads((root/'execution-audit.json').read_text())
    if status['state'] != 'completed' or audit['episodes'] != 50 or not audit['all_execution_audits_pass']:
        raise ValueError('Complete and audit all 50 episodes before promoting their storage to project libraries')
    before = inventory(args.skills, args.experiences)
    merge_libraries(root/'shared/skills', root/'shared/experiences', args.skills, args.experiences)
    after = inventory(args.skills, args.experiences)
    if any(after['immutable'].get(name) != digest for name,digest in before['immutable'].items()):
        raise RuntimeError('An existing manifest/code changed; inspect retained audit before further runs')
    receipt = dict(source=str(root/'shared'), skills=str(args.skills.resolve()),
                   experiences=str(args.experiences.resolve()), before=before, after=after,
                   old_manifests_and_code_unchanged=True,
                   note='Append-only storage merge, deduplicated run evidence; candidate statuses are not promoted.')
    (root/'retained-libraries.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:{n:v for n,v in x.items() if n!='immutable'} for k,x in [('before',before),('after',after)]}))


if __name__ == '__main__':
    main()
