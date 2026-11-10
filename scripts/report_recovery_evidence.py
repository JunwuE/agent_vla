"""Index actual code attempts, reused versions and measured convergence evidence."""

import argparse
import json
from pathlib import Path

import numpy as np


def main(root, groups):
    rows = []
    for group in groups:
        for path in sorted((root / group / 'episodes').glob('*/events.jsonl')):
            if not (path.parent / 'result.json').exists():
                continue
            events = [json.loads(line) for line in path.open()]
            versions = {e['intervention_id']: e for e in events if e['kind'] == 'code_version'}
            invocations = {e['intervention_id']: e for e in events if e['kind'] == 'skill_invocation'}
            origins = {e['result']['version']: e['result']['origin_skill'] for e in events
                       if e['kind'] == 'codex_tool' and isinstance(e.get('result'), dict)
                       and e['result'].get('origin_skill') and e['result'].get('version')}
            for event in events:
                if event['kind'] != 'recovery_result':
                    continue
                identity = event['intervention_id']
                version = versions[identity]
                tracking = [e for e in events if e['kind'] == 'eef_tracking' and e['intervention_id'] == identity]
                errors = [np.linalg.norm(e['error'][:3]) for e in tracking[-10:]]
                angles = [np.linalg.norm(e['error'][3:]) for e in tracking[-10:]]
                # Relative error is to the submitted short bounded segment, not necessarily the final goal.
                row = dict(group=group, episode=path.parent.name, event_id=event['id'],
                    intervention_id=identity, result=event['result'], verified=event['verified'],
                    error=event.get('error'), failed_stage=event.get('failed_stage'),
                    steps=event['executed_steps'], seconds=event.get('seconds'),
                    program_version=event['program_version'], package_sha256=event.get('package_sha256'),
                    reuse=invocations.get(identity.rsplit('-', 1)[0]),
                    origin_skill=origins.get(event['program_version']),
                    parameters=version['parameters'], dependencies=version.get('dependency_lock', {}),
                    program=str((path.parent / version['path']).relative_to(root)),
                    stages=event.get('stages', []), verification=event.get('evidence'),
                    program_measurements=event.get('program_result'),
                    final_tracking_position_errors_m=[float(x) for x in errors],
                    final_tracking_orientation_errors_rad=[float(x) for x in angles])
                rows.append(row)
    report = dict(attempts=rows, notes=[
        'Candidate reuse is recorded even if the catalog has not promoted a version to validated status.',
        'Failed convergence alone does not identify collision, reachability or perception as the cause.',
        'Tracking errors refer to bounded segment targets and must not be confused with object pose errors.',
        'A successful reuse is evidence for the measured entry conditions, not universal skill validity.'])
    (root / 'code-recovery-evidence.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Code Policy 执行证据', '',
        '只统计实际运行的程序。未执行的编辑、被拒绝的提案仍保存在原始 workspace，不计恢复成功。', '',
        '| 组 / seed | 程序来源 | 步数 / 秒 | 核验 | 失败阶段 / 错误 |', '|---|---|---:|---|---|']
    for row in rows:
        reuse = row['reuse']
        source = f"复用 {reuse['skill_id']}@{reuse['version']}" if reuse else f"程序 {row['program_version']}"
        if not reuse and row['origin_skill']:
            origin = row['origin_skill']
            source = f"复用并修改 {origin['id']}@{origin['version']} → {row['program_version']}"
        lines.append(f"| {row['group']} / {row['episode']} | [{source}]({row['program']}) | "
            f"{row['steps']} / {row['seconds']:.1f} | {row['result']} / {row['verified']} | "
            f"{row['failed_stage'] or ''} / {row['error'] or ''} |")
    lines += ['', '成功由独立 task_success 判据核验。位姿不收敛只说明程序没达到其检查条件，'
        '不能仅凭此确定碰撞、不可达或深度误差。已保存每次尝试最后 10 条短段跟踪误差及阶段记录。', '',
        '[完整参数、依赖版本、阶段与验证证据](code-recovery-evidence.json)。']
    (root / 'code-recovery-evidence.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(attempts=len(rows), verified=sum(r['verified'] for r in rows))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--groups', nargs='+', required=True)
    args = parser.parse_args()
    main(args.root.resolve(), args.groups)
