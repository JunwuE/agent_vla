"""Summarize exact-prefix recovery development separately from fresh Agent evaluations."""

import argparse
import csv
import json
from pathlib import Path


def main(root):
    rows = []
    for round_root in sorted(root.glob('round[0-9]*')):
        if not (round_root / 'plan.json').exists():
            continue
        for case in json.loads((round_root / 'plan.json').read_text())['cases']:
            candidates = [round_root / 'episodes' / f"seed{case['seed']:02d}",
                          round_root / 'episodes' / f"OpenDrawer-seed{case['seed']:02d}"]
            episode = next((e for e in candidates if (e / 'result.json').exists()), None)
            if episode is None:
                continue
            result = json.loads((episode / 'result.json').read_text())
            report = result.get('recovery', {})
            kinds = {}
            first_success = None
            for line in (episode / 'events.jsonl').open():
                event = json.loads(line)
                kinds[event['kind']] = kinds.get(event['kind'], 0) + 1
                if event['kind'] == 'observation' and event['success'] and first_success is None:
                    first_success = event['step']
            native_result_agrees = bool(first_success is not None) == bool(result['success'])
            rows.append(dict(round=round_root.name, seed=case['seed'], prefix_steps=case['prefix_steps'],
                success=result['success'], final_steps=result['control_steps'], first_success_step=first_success,
                code_steps=report.get('executed_steps', 0), code_seconds=report.get('seconds'),
                prefix_seconds=result.get('prefix_seconds'), wall_seconds=result['wall_seconds'],
                error=result.get('error') or report.get('error'), failed_stage=report.get('failed_stage'),
                exact_state_error=result.get('replay_max_state_error'),
                native_result_agrees=native_result_agrees,
                model_calls=sum(kinds.get(k, 0) for k in ['codex_turn_start', 'vla_inference_start', 'flora_request']),
                skill_id=case['skill_id'], skill_version=case['skill_version'], change=case['change'],
                evidence=str(episode.resolve())))
    (root / 'replay-metrics.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
    with (root / 'replay-metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# 原轨迹接管状态的代码恢复开发测试', '',
        '每次创建新仿真器，执行原日志中的 VLA 命令前缀；恢复期间只使用当时实际观测。'
        '每 5 步核对机器人状态，起点和接管点核对三视角图像；不能把这些测试算作新的 Agent 成功率。',
        '完整重放耗时包含场景初始化、前缀执行、日志落盘；这里主要比较代码步数和失败位置。'
        '各轮的并发与线程设置可能影响墙钟时间。', '',
        '| 轮次 | seed | 成功 | 前缀步数 | 代码步数 | 代码秒数 | 失败阶段 / 原因 |',
        '|---|---:|---|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['round']} | {r['seed']} | {r['success']} | {r['prefix_steps']} | {r['code_steps']} | "
                     f"{(r['code_seconds'] or 0):.2f} | {r['failed_stage'] or '-'} / {r['error'] or '-'} |")
    lines += ['', 'round1 使用 400 步/240 秒单次代码预算；round2 起使用 600 步/360 秒，'
                   'episode 仍不超过 1500 步。程序自己的更小上限仍生效，未降低成功或几何检查门槛。',
              '所有修改是本地开发修订，过程未调用 Astra、VLA 或 Flora 模型；原有 VLA 前缀来自上一轮实验。']
    (root / 'replay-results.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(completed=len(rows), successes=sum(r['success'] for r in rows),
                         exact_replay=all(r['exact_state_error'] == 0 for r in rows),
                         native_results_agree=all(r['native_result_agrees'] for r in rows),
                         model_calls=sum(r['model_calls'] for r in rows))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root.resolve())
