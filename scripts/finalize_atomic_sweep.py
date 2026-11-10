"""Audit a completed atomic sweep, preserve candidates and render every episode."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main(root, wait, render_only=False):
    root = root.resolve()
    while wait and not (root / 'batch-end.json').exists():
        time.sleep(2)
    if not (root / 'batch-end.json').exists():
        raise ValueError('Sweep has not completed')
    repo = Path(__file__).resolve().parents[1]
    plan = json.loads((root / 'plan.json').read_text())
    label = f"{plan['tasks'][0]['task']} {root.name}"
    stages = ('report_atomic_sweep', 'audit_agent_runtime', 'audit_atomic_sweep',
              'retain_composite_libraries', 'analyze_action_performance')
    if render_only:
        required = ('atomic-metrics.json', 'agent-runtime-audit.json', 'final-audit.json',
                    'action-performance.json')
        if any(not (root / name).is_file() for name in required):
            raise ValueError('Render-only requires completed reports and audits')
        if not all(json.loads((root / 'final-audit.json').read_text())['checks'].values()):
            raise ValueError('Render-only requires a passed protocol audit')
        stages = ()
    for name in stages:
        with (root / (name + '.log')).open('w') as log:
            subprocess.run([sys.executable, str(repo / 'scripts' / (name + '.py')), str(root)],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
        print('completed ' + name, flush=True)
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')

    def render(result_path):
        episode = result_path.parent
        output = root / 'report' / episode.name
        if not (output / 'video-check.json').exists():
            with output.with_suffix('.log').open('w') as log:
                subprocess.run([sys.executable, str(repo / 'scripts/render_episode.py'), str(episode),
                                '--output', str(output), '--overview', '--label', label],
                               check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
        timeline = json.loads((output / 'timeline.json').read_text())
        probe = json.loads((output / 'video-check.json').read_text())
        assert int(probe['nb_read_frames']) == sum(c['duration_frames'] for c in timeline['timeline'])
        assert timeline['source_events_sha256'] == hashlib.sha256((episode / 'events.jsonl').read_bytes()).hexdigest()
        result = json.loads(result_path.read_text())
        print('rendered ' + episode.name, flush=True)
        return dict(episode=episode.name, success=result['success'], verified=True,
                    frames=int(probe['nb_read_frames']))

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        receipts = list(pool.map(render, sorted((root / 'episodes').glob('*/result.json'))))
    (root / 'replay-audit.json').write_text(json.dumps(receipts, indent=2) + '\n')
    lines = ['# OpenDrawer 回放', '', '保留全部物理帧和阶段，概览展示关键工具调用；完整调用见原始 events.jsonl。推理等待在视频中压缩，真实耗时见结果表。', '',
             '| seed | 成功 | 阶段与工具 | 视频 |', '|---|---|---|---|']
    for row in receipts:
        name = row['episode']
        lines.append(f"| {name} | {row['success']} | [页面]({name}/index.html) | [MP4]({name}/replay.mp4) |")
    (root / 'report/README.md').write_text('\n'.join(lines) + '\n')
    (root / 'finalization.json').write_text(json.dumps(dict(completed=True, videos=len(receipts),
                                                          finished_ns=time.time_ns()), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--render-only', action='store_true', help='Use completed reports/audits; only render and verify videos')
    args = parser.parse_args()
    main(args.root, args.wait, args.render_only)
