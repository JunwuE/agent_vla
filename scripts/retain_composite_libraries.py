"""Merge finished private libraries by task without changing any evaluation input."""

import argparse
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from run_concurrent_sweep import merge_libraries
from run_seed_sweep import write_json


def main(root):
    plan = json.loads((root/'plan.json').read_text())
    end = json.loads((root/'batch-end.json').read_text())
    expected = {s['key'] for s in plan['cases']}
    if set(end['completed']) != expected or not end['pinned_source_and_initial_memory_unchanged']:
        raise RuntimeError('Require the complete frozen-source sweep before merging')
    retained = root/'retained'
    retained.mkdir(exist_ok=True)
    for task in plan['tasks']:
        cases = [s for s in plan['cases'] if s['task'] == task['task']]
        target = retained/task['task']
        if target.exists():
            receipt = json.loads((target/'retention.json').read_text())
            if receipt['cases'] != [s['key'] for s in cases]:
                raise RuntimeError('Existing retained library belongs to another case set')
            print(json.dumps(receipt, ensure_ascii=False), flush=True)
            continue
        staging = Path(tempfile.mkdtemp(prefix=task['task']+'-', dir=retained))
        shutil.copytree(root/'initial', staging, dirs_exist_ok=True)
        for spec in cases:
            source = root/'inputs'/spec['key']
            merge_libraries(source/'skills', source/'experiences', staging/'skills', staging/'experiences')
        new = []
        for path in sorted((staging/'skills').glob('*/*/manifest.json')):
            if (root/'initial'/path.relative_to(staging)).exists():
                continue
            data = json.loads(path.read_text())
            new.append(dict(id=data['id'], version=data['version'], status=data['status'],
                            contract=str(path.with_name('SKILL.md').relative_to(staging))
                            if path.with_name('SKILL.md').exists() else str(path.relative_to(staging)),
                            manifest=str(path.relative_to(staging)), code=str(path.with_name('policy.py').relative_to(staging))))
        groups = defaultdict(list)
        for item in new:
            groups[item['id']].append(item)
        index = ['# '+task['task']+'：候选代码索引', '',
                 '这是一份独立归档，不是默认运行库。包含冻结的原始技能依赖和本任务新增候选。'
                 '完整证据保留；未做跨布局技能晋升或默认版本选择。', '',
                 '| 技能 | 候选版本数 | 状态 |', '|---|---:|---|']
        (staging/'versions').mkdir(exist_ok=True)
        for name, versions in sorted(groups.items()):
            page = ['# '+name, '', '版本不代表质量排名；使用前检查完整契约和当前几何。', '',
                    '| 版本 | 状态 | 契约 | 证据 |', '|---|---|---|---|']
            for item in versions:
                page.append(f"| [{item['version']}](../{item['code']}) | {item['status']} | "
                            f"[SKILL](../{item['contract']}) | [manifest](../{item['manifest']}) |")
            (staging/'versions'/f'{name}.md').write_text('\n'.join(page)+'\n')
            index.append(f"| [{name}](versions/{name}.md) | {len(versions)} | "
                         f"{', '.join(sorted({v['status'] for v in versions}))} |")
        (staging/'INDEX.md').write_text('\n'.join(index)+'\n')
        receipt = dict(task=task['task'], cases=[s['key'] for s in cases],
                       new_skill_versions=len(new), skill_identities=len(groups),
                       new_status_counts=dict(Counter(v['status'] for v in new)),
                       new_experiences=sum(not (root/'initial/experiences'/p.name).exists()
                                           for p in (staging/'experiences').glob('*.json')),
                       initial_memory_sha256=plan['initial_memory_sha256'],
                       default_library_changed=False, tested_with_merged_memory=False,
                       guidance='Retained candidates, not independently validated reusable skills. '
                                'Per-seed task-memory proposals, if any, remain in original inputs directories.')
        write_json(staging/'retention.json', receipt)
        staging.rename(target)
        print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root.resolve())
