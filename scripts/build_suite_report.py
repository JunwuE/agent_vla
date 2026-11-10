"""Summarize every predeclared task, including failed/unfinished runs, from recorded evidence."""

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path

import numpy as np

from vla_agent.data import read_events


def stats(values):
    return dict(count=len(values), median=float(np.median(values)), p95=float(np.percentile(values, 95)),
                maximum=max(values)) if values else dict(count=0, median=None, p95=None, maximum=None)


def build(root):
    root = root.resolve()
    plan = json.loads((root / "plan.json").read_text())
    summaries = json.loads((root / "summary.json").read_text()) if (root / "summary.json").exists() else []
    records = []
    for spec in plan["tasks"]:
        trial = next((r for r in summaries if r["task"] == spec["task"] and r["seed"] == spec["seed"]), None)
        path = root / "episodes" / f"{spec['task']}_seed{spec['seed']}"
        rows = read_events(path) if (path / "events.jsonl").exists() else []
        result = json.loads((path / "result.json").read_text()) if (path / "result.json").exists() else (
            trial["result"] if trial else {})
        counts = Counter(row["kind"] for row in rows)
        decisions = [r for r in rows if r["kind"] == "decision"]
        executed = [r for r in rows if r["kind"] == "action_executed"]
        observations = [r for r in rows if r["kind"] == "observation"]
        final = observations[-1] if observations else None
        initialized = next((r["task"] for r in rows if r["kind"] == "task_initialized"), None)
        tokens = [r["response"].get("usage", {}).get("input_tokens") for r in rows if r["kind"] == "decision_response"]
        report = root / "report" / path.name / "index.html"
        records.append(dict(
            **spec, status="completed" if result else "running" if rows else "pending",
            result=result, instruction=initialized["instruction"] if initialized else None,
            final_native_success=final["success"] if final else None,
            command_count=counts["action_command"], executed_count=len(executed),
            all_commands_acknowledged=({r["id"] for r in rows if r["kind"] == "action_command"}
                                       == {r["command_id"] for r in executed}),
            decision_routes=dict(Counter(r["decision"]["label"] for r in decisions)),
            decision_triggers=dict(Counter(r["decision"]["trigger"] for r in decisions)),
            decision_seconds=stats([r["seconds"] for r in decisions]),
            decision_tokens=stats([v for v in tokens if v is not None]),
            stale_decisions=counts["stale_decision"], discarded_reviews=counts["review_discarded"],
            failed_reviews=counts["review_failed"],
            failed_episode_without_astra_review=(bool(result) and result.get("reason") == "horizon"
                                                and bool(decisions) and counts["review_begin"] == 0),
            execution_takeovers=sum(r["kind"] == "intervention_begin" and
                                    r["decision"]["trigger"] != "initialization" for r in rows),
            intervention_kinds=dict(Counter(r["plan"]["kind"] for r in rows if r["kind"] == "intervention_plan")),
            tool_calls=dict(Counter(r["name"] for r in rows if r["kind"] == "codex_tool")),
            code_attempts=counts["recovery_result"], event_counts=dict(counts),
            report=str(report.relative_to(root)) if report.exists() else None,
            render_error=trial.get("render_error") if trial else None,
        ))
    groups = {}
    for group in dict.fromkeys(r["group"] for r in records):
        cases = [r for r in records if r["group"] == group]
        groups[group] = dict(planned=len(cases), completed=sum(r["status"] == "completed" for r in cases),
                             successes=sum(r["result"].get("success", False) for r in cases),
                             reasons=dict(Counter(r["result"].get("reason", r["status"]) for r in cases)),
                             astra_turns=sum(r["result"].get("astra_calls", 0) for r in cases),
                             execution_takeovers=sum(r["execution_takeovers"] for r in cases))
    report = dict(groups=groups, records=records, limitations=[
        "One preselected seed per task, no matched VLA-only control: cannot establish improvement in success rate.",
        "Official task splits; base-model training exposure not fully audited; frozen skill retrieval.",
        "Flora disabled. Only native task_success verifier installed; fine-grained subtask switching not validated.",
        "No independent online intervention labels: no false-positive/false-negative rate claimed.",
        "Three episodes share GPU services; online latency includes contention and client overhead.",
    ])
    comparison_path = root / "context-comparison/comparison.json"
    comparison = json.loads(comparison_path.read_text()) if comparison_path.exists() else None
    if comparison:
        report["context_comparison"] = [dict(
            task=c["task"], snapshot_step=c["snapshot_step"],
            choices={m: r["response"]["answers"]["route"]["choice"] for m, r in c["results"].items()},
            routes={m: r["route"] for m, r in c["results"].items()},
            retrospective_success=c["retrospective_episode_result"]["success"],
        ) for c in comparison["cases"]]
    (root / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    lines = ["# 数值历史精简与 15 任务实测", "", "固定 seed 0；所有任务均在运行前选定。", "",
             "| 划分 | 已完成 | 成功 | Astra turns | 执行中接管 |", "|---|---:|---:|---:|---:|"]
    for group, data in groups.items():
        lines.append(f"| {group} | {data['completed']}/{data['planned']} | {data['successes']} | "
                     f"{data['astra_turns']} | {data['execution_takeovers']} |")
    lines += ["", "Astra turns 包含初始化、只读审核与暂停复核；执行中接管不包含初始化。", ""]
    benchmark = root / "benchmark/results.json"
    if benchmark.exists():
        measurements = json.loads(benchmark.read_text())["cases"]
        pairs = {}
        for sample in measurements:
            name, mode = sample["id"].rsplit(":", 1)
            pairs.setdefault(name, {})[mode] = sample
        lines += ["## 相同图片的数值上下文对照", "",
                  "| 历史窗口 | token 原始→摘要 | 热调用秒数 原始→摘要 | 原始类别变化 |",
                  "|---|---:|---:|---|"]
        for name, pair in pairs.items():
            a, b = pair["full"], pair["compact"]
            labels = [s["responses"][-1]["answers"]["route"]["choice"] for s in (a, b)]
            lines.append(f"| {name} | {a['input_tokens']} → {b['input_tokens']} | "
                         f"{a['median_seconds']:.3f} → {b['median_seconds']:.3f} | {labels[0]} → {labels[1]} |")
        lines += ["", "两组使用相同九张图，只替换数值上下文；服务端热调用排除客户端与 HTTP。",
                  "精简是有损变换，类别变化不是准确率提高的证据。[原始测量](benchmark/results.json)。", ""]
    lines += ["## 各任务结果", "",
              "| 任务 | 结果/停止原因 | 步数 | Astra turns | Decision 次数 | 轨迹 |",
              "|---|---|---:|---:|---:|---|"]
    table = []
    for r in records:
        v = r["result"]
        status = "成功" if v.get("success") else v.get("reason", r["status"])
        link = f"[回放]({r['report']})" if r["report"] else "待生成"
        lines.append(f"| {r['task']} | {status} | {v.get('control_steps', 0)}/{r['horizon']} | "
                     f"{v.get('astra_calls', 0)} | {v.get('decision_checks', 0)} | {link} |")
        href = r["report"].removeprefix("report/") if r["report"] else None
        replay = f'<a href="{html.escape(href)}">视频与工具时间线</a>' if href else "待生成"
        table.append(f"<tr><td>{html.escape(r['group'])}</td><td>{html.escape(r['task'])}</td>"
                     f"<td>{html.escape(status)}</td><td>{v.get('control_steps', 0)}/{r['horizon']}</td>"
                     f"<td>{v.get('astra_calls', 0)}</td><td>{r['execution_takeovers']}</td><td>{replay}</td></tr>")
    total_routes = sum((Counter(r["decision_routes"]) for r in records), Counter())
    total_triggers = sum((Counter(r["decision_triggers"]) for r in records), Counter())
    unreviewed = [r["task"] for r in records if r["failed_episode_without_astra_review"]]
    lines += ["", "完整指标见 analysis.json；每条轨迹有独立 manifest、原始日志与观测。",
              "[原始指令与 Astra 规划笔记](instructions.md)；[执行与视频完整性检查](integrity-check.json)。", "",
              "## 监督链路观察", "",
              f"Decision 路由次数：`{dict(total_routes)}`；触发来源：`{dict(total_triggers)}`。",
              f"跑满控制预算且未触发 Astra 执行中审核的任务：{', '.join(unreviewed) or '无'}。",
              "这些任务可用于后续逐段标注；最终失败不能直接作为先前每次 KEEP 的错误标签。"]
    if comparison:
        lines += ["", "## 各任务末次检查的同观测重放", "",
                  "保留原五张图、指令、模型与阈值，分别发送完整和精简上下文。此对照不执行机器人动作。",
                  "事后结果仅写入分析，未发送给模型；完整上下文的判断也不是正确性标签。", "",
                  "| 任务 | 检查步数 | 原始类别 完整→精简 | 实际路由 完整→精简 |",
                  "|---|---:|---|---|"]
        for c in report["context_comparison"]:
            lines.append(f"| {c['task']} | {c['snapshot_step']} | "
                         f"{c['choices']['full']} → {c['choices']['compact']} | "
                         f"{c['routes']['full']} → {c['routes']['compact']} |")
        lines += ["", "[原始对照记录](context-comparison/comparison.json)。"]
    lines += ["",
              "## 解释边界", "", "- 每任务一条轨迹，未做同条件纯 VLA 对照，不能归因成功率变化。",
              "- Flora 未启用；真实环境只有 task_success 验证器，细分计划保存在 Astra 笔记中。",
              "- 没有独立人工干预标签，不能报告 Decision 漏判率或误判率。",
              "- 三个 Episode 共享模型服务，在线延迟包含排队与图像编码，不能与独占 GPU 热调用直接混比。",
              "- 官方 seen/unseen 划分不等于已经审计基座模型的全部训练数据暴露。"]
    (root / "README.md").write_text("\n".join(lines) + "\n")
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>15 任务实测</title>'
    page += '<style>body{font:16px system-ui;background:#101724;color:#e6edf7;padding:32px}table{border-collapse:collapse;width:100%}th,td{padding:12px;border-bottom:1px solid #304055;text-align:left}a{color:#7dd3fc}</style>'
    page += '<h1>当前架构 · 15 任务实测</h1><p>固定 seed 0；包含失败。Astra turns 包含初始化，执行中接管排除初始化。Flora 本轮关闭。</p>'
    page += '<table><thead><tr><th>划分</th><th>任务</th><th>结果</th><th>步数</th><th>Astra turns</th><th>执行中接管</th><th>回放</th></tr></thead><tbody>'
    page += "".join(table) + '</tbody></table><p>每任务一条轨迹，当前只有整体 task_success 验证器；不能据此报告稳定成功率或 Decision 判别准确率。</p></html>'
    (root / "report/index.html").write_text(page)
    with (root / "metrics.csv").open("w") as stream:
        columns = ["group", "task", "success", "reason", "control_steps", "astra_calls", "decision_checks", "execution_takeovers"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for r in records:
            writer.writerow(dict(group=r["group"], task=r["task"], execution_takeovers=r["execution_takeovers"],
                                 **{key: r["result"].get(key) for key in columns[2:-1]}))
    return groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.root), ensure_ascii=False))
