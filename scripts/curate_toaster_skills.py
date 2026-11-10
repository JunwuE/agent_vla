"""Build a semantic view of reviewed toaster procedures without changing historical code/evidence."""

import argparse
import hashlib
import json
from pathlib import Path

from vla_agent.skills import SkillCatalog, SkillLibrary

FAMILIES = [
    ('close-toaster-door-from-upright',
     'Close the toaster oven door from a measured nearly upright state: release, clear, approach and seat.', [
        ('6e0569a16adc', 'Nearly upright door; downward wrist; visible right-camera rim, roof and fixed lip. '
         'Arc-reference tracking; all image boxes and observation_step are fresh parameters. '
         'Full seating observed on seed 9; this does not cover grasping a lowered door.'),
        ('bd071b31b22d', 'Nearly upright door; current fingerpad geometry; right-camera roof and rim. '
         'Cartesian tracking supports a tilted wrist. Seed 25 completed during release/clearance; '
         'later seating branches were not exercised by that success. Subsequent failures remain recorded.'),
        ('2ff9c7e5cdd3', 'Same nearly upright procedure with independently selectable oven_camera. '
         'Use a fresh unobscured roof region when the right roof view is blocked. Seed 44 completed '
         'during clearance; no evidence of all subsequent branches working.')]),
    ('close-toaster-door-from-open-clearance',
     'Close the toaster oven door after open fingers have already cleared the raised door.', [
        ('727ef0dcbde5', 'Open fingers with measured upper clearance already achieved; retain height, '
         'withdraw, close and seat. Fresh right-camera rim, roof and fixed-lip parameters. '
         'Do not restart a blocked upward clearance stage.')]),
    ('close-toaster-door-from-closed-clearance',
     'Close the toaster oven door from measured closed-finger exterior clearance and stalled alignment.', [
        ('8c814dfd1409', 'High exterior clearance and closed fingers achieved. Bounded integral position '
         'feedback and latched angular targets during alignment. camera, arc_reference and resume flags '
         'are parameters; resume flags require current measured prerequisites.'),
        ('e0ddab6a9e8b', 'Closed fingers at high exterior clearance; stalled wrist alignment. '
         'Measured outward relocation before alignment; current camera, fixed lip and '
         'alignment_withdrawal_m inputs. Checks require actual clearance and movement.'),
        ('1d523dff2778', 'Closed exterior pose; height restoration blocked before alignment. '
         'Bounded outward relocation precedes height restoration. Cartesian tracking, current camera, '
         'clearance_above_roof_m and alignment_withdrawal_m; do not assume contact from resume flags.')]),
    ('seat-toaster-door-from-exterior',
     'Finish closing the toaster oven door from an already achieved exterior approach/contact pose.', [
        ('40d558b2ca66', 'Current downward exterior alignment and approach already achieved; fresh paired '
         'upper-rim measurements and current lateral pose. Final seating with independent tracking. '
         'No initial grasp or clearance procedure is established by this variant.'),
        ('34bebce885e2', 'Achieved exterior contact approach after withdrawal/lowering. Paired thin upper-lip '
         'tracking; current camera, resume and tracking flags, bounded clearance radius. '
         'Select flags only when current checks establish the corresponding entry state.')]),
]


def curate(root):
    root = Path(root)
    before = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.glob('*/*/*') if p.is_file()}
    library = SkillLibrary(root)
    families = []
    for name, description, entries in FAMILIES:
        versions = []
        for prefix, when in entries:
            matches = [x for x in library.index if x.get('storage_id', x['id']) == 'recovery-' + prefix]
            if len(matches) != 1:
                raise ValueError(f'Expected exactly one preserved source for {prefix}')
            item = matches[0]
            versions.append(dict(storage_id=item.get('storage_id', item['id']),
                                 version=item['version'], when_to_use=when))
        families.append(dict(id=name, robot='PandaOmron', description=description, versions=versions))
    catalog = SkillCatalog(archive_unlisted_recoveries=True, skills=families)
    path = root / 'catalog.json'
    rendered = catalog.model_dump_json(indent=2) + '\n'
    if path.exists() and path.read_text() != rendered:
        raise ValueError('Existing catalog changed; review and merge it explicitly instead of overwriting')
    path.write_text(rendered)
    library = SkillLibrary(root)
    lines = ['# 操作技能目录', '',
             '技能名称描述操作与起始状态；哈希仅标识不可变版本。旧 ID 仍可读取，原代码、manifest 和运行证据不迁移。', '',
             '以下均为 candidate，成功记录不等于跨场景验证通过。当前没有经过验证的通用抓取技能。', '',
             '调用：`search_skills → list_skill_versions → read_skill → check_skill → submit_intervention`。',
             '换场景先更新 `parameters`；参数重试不导入、不 patch，不产生新技能或代码版本。', '',
             '| 技能 | 已收录版本 | 用途 |', '|---|---:|---|']
    for f in families:
        packages = [x for x in library.index if x['id'] == f['id'] and x.get('package')]
        count = str(len(f['versions'])) + (f" 个历史版本 + {len(packages)} 个模块化版本" if packages else '')
        lines.append(f"| `{f['id']}` | {count} | {f['description']} |")
    for f in families:
        lines.extend(['', '## ' + f['id'], ''])
        for item in library.index:
            if item['id'] == f['id'] and item.get('package'):
                ref = f"{item['id']}/{item['version']}"
                lines.append(f"- 模块化 `{item['version']}`（{item['status']}）：[SKILL.md]({ref}/SKILL.md)，"
                             f"{item['when_to_use']}；验证范围参见 [运行证据]({ref}/runs.jsonl)。")
        for v in f['versions']:
            ref = f"{v['storage_id']}/{v['version']}"
            lines.append(f"- `{v['version']}` — {v['when_to_use']} [源码]({ref}/policy.py) · [原始证据]({ref}/manifest.json)")
    helpers = [x for x in library.index if x.get('package') and x['id'] not in {f['id'] for f in families}]
    if helpers:
        lines.extend(['', '## 共享模块', '', '| 包 | 固定版本 | 状态 |', '|---|---|---|'])
        for item in helpers:
            lines.append(f"| [{item['id']}]({item['id']}/{item['version']}/SKILL.md) | `{item['version']}` | {item['status']} |")
        lines.extend(['', '被导入不等于已独立验证。组合规则见 [RAX Code Policy](../docs/rax-code-policy.md)。'])
    archived = sum(x['archived'] for x in library.index)
    lines.extend(['', f'{archived} 个未整理的历史恢复版本保留在原目录，通过旧 ID 或 '
                  '`search_skills(include_candidates=true, include_archived=true)` 查阅；不占默认检索位置。',
                  '模板 `eef_offset`、`gripper_hold`、`mobile_nudge` 仍需显式 include_templates。', ''])
    (root / 'INDEX.md').write_text('\n'.join(lines))
    after = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in root.glob('*/*/*') if p.is_file()}
    if before != after:
        raise AssertionError('Historical library files changed')
    return dict(procedures=len(families), curated_versions=sum(len(f['versions']) for f in families),
                archived_versions=archived, preserved_files=len(before))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('skills'))
    print(json.dumps(curate(parser.parse_args().root)))
