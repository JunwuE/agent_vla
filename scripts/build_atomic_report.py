"""Build a local, shareable report from all pilot trials and fixed seed-zero replays."""

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np

from vla_agent.data import read_events
from vla_agent.visualization import episode_summary, render_episode, write_page

TASKS = {"OpenDrawer": "打开抽屉", "PickPlaceCounterToStove": "台面物体搬到炉灶", "TurnOnSinkFaucet": "打开水龙头"}


def initial_differences(a, b):
    if not a or not b:
        return None
    differences = {}
    with np.load(Path(a["source"]) / "observations/0000000.npz", allow_pickle=False) as x, \
         np.load(Path(b["source"]) / "observations/0000000.npz", allow_pickle=False) as y:
        for key in x.files:
            if not np.array_equal(x[key], y[key]):
                delta = np.abs(x[key].astype(float) - y[key].astype(float))
                differences[key] = dict(changed_values=int(np.count_nonzero(delta)), values=delta.size,
                                        max_abs=float(delta.max()), mean_abs=float(delta.mean()))
    return differences


def build(root, render=False):
    root = root.resolve()
    output = root / "report"
    output.mkdir(exist_ok=True)
    records, selected = [], []
    for mode in ("baseline", "supervised", "low_calls"):
        for task in TASKS:
            for path in sorted((root / mode).glob(task + "_seed*")):
                if not (path / "result.json").exists():
                    continue
                summary = episode_summary(path, read_events(path))
                records.append(summary)
                if summary["seed"] == 0:
                    target = output / f"{task}_{mode}"
                    if render and not (target / "index.html").exists():
                        render_episode(path, target)
                    if (target / "timeline.json").exists():
                        write_page(target, json.loads((target / "timeline.json").read_text()))
                    selected.append(dict(task=task, mode=mode, seed=0, source=str(path),
                                         replay=str(target.relative_to(output)),
                                         available=(target / "index.html").exists()))
    pairs = []
    for task in TASKS:
        a = next((r for r in records if r["task"] == task and r["seed"] == 0 and r["mode"] == "baseline"), None)
        b = next((r for r in records if r["task"] == task and r["seed"] == 0 and r["mode"] == "supervised"), None)
        c = next((r for r in records if r["task"] == task and r["seed"] == 0 and r["mode"] == "low_calls"), None)
        pairs.append(dict(task=task, seed=0, initial_observations_identical=(
            a["initial_fingerprint"] == b["initial_fingerprint"] if a and b else None),
            low_calls_initial_observations_identical=(
                a["initial_fingerprint"] == c["initial_fingerprint"] if a and c else None),
            initial_differences={"supervised": initial_differences(a, b), "low_calls": initial_differences(a, c)},
            baseline=a, supervised=b, low_calls=c))
    shadow = root / "decision-shadow/summary.json"
    sensitivity = root / "decision-shadow/threshold-sensitivity.json"
    report = dict(purpose="three-task development pilot, not a full benchmark", trials=records,
                  selected=selected, selection_rule="Seed 0 fixed before supervised execution; never select by success",
                  paired_initial_conditions=pairs,
                  decision_shadow=json.loads(shadow.read_text()) if shadow.exists() else None,
                  threshold_sensitivity=json.loads(sensitivity.read_text()) if sensitivity.exists() else None,
                  limits=["Three seeds per baseline; one original and one low-call supervised seed per task",
                          "Decision thresholds and review intervals are uncalibrated pilot parameters",
                          "Low-call run changes routing, review interval, initial wording and context together; no single-factor attribution",
                          "Only language/plan interventions enabled in both supervised variants",
                          "Only native whole-task success verifier installed; no geometric recovery evaluation",
                          "Phases are actual runtime phases, not independent semantic subtask labels"])
    totals = {}
    for mode in ("supervised", "low_calls"):
        group = [r["result"] for r in records if r["mode"] == mode]
        totals[mode] = dict(episodes=len(group), successes=sum(r["success"] for r in group),
                            astra_turns=sum(r["astra_calls"] for r in group),
                            input_tokens=sum(r["astra_tokens"].get("input_tokens", 0) for r in group),
                            output_tokens=sum(r["astra_tokens"].get("output_tokens", 0) for r in group),
                            wall_seconds=sum(r["wall_seconds"] for r in group))
    report["supervision_totals"] = totals
    (output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (output / "selection.json").write_text(json.dumps(dict(rule=report["selection_rule"], cases=selected), indent=2))
    with (output / "trials.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["task", "mode", "seed", "success", "reason", "steps", "wall_seconds",
                                                  "vla_calls", "decision_checks", "astra_turns", "input_tokens", "output_tokens", "error"])
        writer.writeheader()
        for r in records:
            v = r["result"]
            writer.writerow(dict(task=r["task"], mode=r["mode"], seed=r["seed"], success=v["success"],
                                 reason=v["reason"], steps=v["control_steps"], wall_seconds=round(v["wall_seconds"], 2),
                                 vla_calls=r["vla_calls"], decision_checks=v["decision_checks"],
                                 astra_turns=v["astra_calls"],
                                 input_tokens=v["astra_tokens"].get("input_tokens", 0),
                                 output_tokens=v["astra_tokens"].get("output_tokens", 0), error=v["error"]))

    def result(r):
        if r is None:
            return "等待运行"
        v = r["result"]
        return f"{'成功' if v['success'] else '未完成'} · {v['control_steps']} 步 · {html.escape(v['reason'])}"

    original, low = totals["supervised"], totals["low_calls"]
    savings = (f"同一组三任务的 Astra 调用：{original['astra_turns']} → {low['astra_turns']} turns；"
               f"原始监督成功 {original['successes']}/{original['episodes']}，"
               f"低调用监督成功 {low['successes']}/{low['episodes']}。"
               "次数包含开场规划；一个 turn 内可以调用多个工具，不等于底层 API 请求数。")
    profile = ("低调用配置：周期检查从 50 改为 100 控制步；取消未经校准的 KEEP ≥ 0.8 门槛，"
               "保留前两类分差小于 0.05 时的 ESCALATE；每条轨迹最多 4 次 Astra 审核。"
               "开场规划保留原生任务指令，细分推理写入笔记；缩短每次传入的动作历史与图像窗口。"
               "非 KEEP 仍交给 Astra，预算用尽则停止。阈值只用于本轮试验，尚未完成准确率校准。")
    tokens = (f"Codex 报告的累计输入 token：{original['input_tokens']:,} → {low['input_tokens']:,}；"
              f"输出 token：{original['output_tokens']:,} → {low['output_tokens']:,}。"
              "未换算成费用，缓存和计费规则未纳入。")
    cards, rows, md = [], [], ["# 三个原子任务首轮与低调用对照", "", "开发评测；轨迹固定选择 seed 0，不按成功结果筛选。", "", savings, "", profile, "", tokens, ""]
    for pair in pairs:
        task, a, b, c = pair["task"], pair["baseline"], pair["supervised"], pair["low_calls"]
        baselines = [r for r in records if r["task"] == task and r["mode"] == "baseline"]
        passed = sum(r["result"]["success"] for r in baselines)
        links = []
        for mode, title in (("low_calls", "查看低调用轨迹与工具流程"),
                            ("supervised", "查看原始监督轨迹"), ("baseline", "查看纯 VLA 对照")):
            selection = next((s for s in selected if s["task"] == task and s["mode"] == mode and s["available"]), None)
            if selection:
                links.append(f'<a href="{selection["replay"]}/index.html">{title} →</a>')
        preview = next((s for s in reversed(selected) if s["task"] == task and s["available"]), None)
        image = f'<img src="{preview["replay"]}/poster.jpg" alt="{html.escape(task)} 三视角预览">' if preview else ""
        equal = (pair["initial_observations_identical"] and pair["low_calls_initial_observations_identical"])
        match = "逐项完全一致" if equal is True else "有数值/像素差异，详见证据" if equal is False else "待核验"
        calls = f"{b['result']['astra_calls'] if b else '—'} → {c['result']['astra_calls'] if c else '—'}"
        cards.append(f'<article>{image}<div class="body"><div class="kicker">{task}</div><h2>{TASKS[task]}</h2>'
                     f'<p>低调用 seed 0：{result(c)}</p><p>Astra 调用：<strong>{calls}</strong></p>'
                     f'<p>纯 VLA 全部 seed：{passed}/{len(baselines)} 成功</p>'
                     f'<div class="links">{"".join(links)}</div></div></article>')
        rows.append(f'<tr><td>{task}</td><td>{result(a)}</td><td>{result(b)}</td><td>{result(c)}</td><td>{calls}</td><td>{match}</td></tr>')
        md += [f"## {task}", "", f"- 基线全部 seed：{passed}/{len(baselines)} 成功。",
               f"- 基线 seed 0：{result(a)}。", f"- 原始监督 seed 0：{result(b)}。",
               f"- 低调用 seed 0：{result(c)}；Astra {calls}。",
               f"- [低调用回放]({task}_low_calls/index.html) · [原始监督]({task}_supervised/index.html) · [纯 VLA]({task}_baseline/index.html)",
               f"- 初始状态与三相机 RGB 内容指纹：{match}。", ""]
        if equal is False:
            md += ["初始数组差异（保留严格哈希不相等的结果，不视为完全相同初始观测）：", "",
                   "```json", json.dumps(pair["initial_differences"], ensure_ascii=False, indent=2), "```", ""]
    detail_rows = []
    for r in records:
        v = r["result"]
        detail_rows.append(f'<tr><td>{r["task"]}</td><td>{r["mode"]}</td><td>{r["seed"]}</td>'
                           f'<td>{"成功" if v["success"] else "未完成"}</td><td>{v["control_steps"]}</td>'
                           f'<td>{v["decision_checks"]}</td><td>{v["astra_calls"]}</td>'
                           f'<td>{html.escape(v["reason"])}</td></tr>')
    shadow_text = (f"已生成 {report['decision_shadow']['count']} 个历史窗口的 Decision 预测，独立标签尚未填写；"
                   "这些预测不作为模型准确率或已校准阈值的证据。" if report["decision_shadow"] else "Decision 离线判断正在生成。")
    if report["threshold_sensitivity"]:
        raw = report["threshold_sensitivity"]["raw_argmax"]
        gated = report["decision_shadow"]["labels"]
        shadow_text += (f" 原始最高分类别为 KEEP 的窗口有 {raw.get('KEEP', 0)} 个；"
                        f"经过原始配置阈值后，KEEP 为 {gated.get('KEEP', 0)} 个，ESCALATE 为 {gated.get('ESCALATE', 0)} 个。"
                        "需区分模型原始选择与运行时阈值路由。")
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>三个原子任务 · VLA Agent 回放</title><style>
*{box-sizing:border-box}body{margin:0;background:#0d1521;color:#eaf2fc;font:15px/1.65 system-ui,sans-serif}main{max-width:1400px;margin:auto;padding:36px 28px}.kicker{font:12px ui-monospace,monospace;color:#72dace}h1{font-size:32px;margin:8px 0}h2{font-size:21px;margin:10px 0}p{color:#aebfd3}a{color:#85d3ef}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin:28px 0}article{border:1px solid #2b405b;background:#152235;border-radius:10px;overflow:hidden}article img{display:block;width:100%;aspect-ratio:1.6}.body{padding:20px}.links a{display:block;padding:6px 0}.panel{border:1px solid #2b405b;border-radius:9px;padding:22px;margin:22px 0;background:#111e2e;overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;border-bottom:1px solid #2b405b;padding:12px 9px;vertical-align:top}th{color:#8fa8c5}.note{border-left:3px solid #f4c56a;padding-left:14px}small{color:#a2b8d1}@media(max-width:950px){.cards{grid-template-columns:1fr}main{padding:20px}h1{font-size:25px}}</style>
<main><div class="kicker">VLA AGENT · ATOMIC PILOT · 2026-10-28</div><h1>三个原子任务，逐阶段查看执行过程</h1>
<p>三视角画面与事件日志同步。查看谁在推理、哪个工具被调用、控制权何时交接，以及环境最终如何判定。</p>
<p class="note">固定选择每个任务的 seed 0 展示，所有尝试保留在下表。当前监督实验启用语言和计划调整，Code Policy 未在本轮执行。小样本结果不能推断整体提升。</p>
<div class="panel"><h2>减少 Astra 调用的实测对照</h2><p>__SAVINGS__</p><p>__PROFILE__</p><p>__TOKENS__</p></div>
<div class="cards">__CARDS__</div><div class="panel"><h2>相同 seed 的对照</h2><p>初始状态与 RGB 内容逐项计算 SHA256，差异保留在 comparison.json；相同 seed 不保证初始观测逐位相同。策略使用同一服务器采样流，不保证随机动作逐位一致。</p>
<table><thead><tr><th>任务</th><th>纯 VLA</th><th>原始监督</th><th>低调用监督</th><th>Astra 次数</th><th>初始观测</th></tr></thead><tbody>__PAIRS__</tbody></table></div>
<div class="panel"><h2>所有运行记录</h2><table><thead><tr><th>任务</th><th>模式</th><th>seed</th><th>结果</th><th>控制步</th><th>Decision</th><th>Astra turns</th><th>停止原因</th></tr></thead><tbody>__TRIALS__</tbody></table></div>
<div class="panel"><h2>Decision 数据与验证边界</h2><p>__SHADOW__</p><p>真实环境目前只有整体任务成功判据。回放中的“阶段”是日志中的执行阶段，不能视作独立验证的抓取、放置或细分子任务标签。推理等待在视频中压缩，实际墙钟时间单独显示。</p></div>
<p><a href="trials.csv">全部结果 CSV</a> · <a href="comparison.json">配置与对照证据</a> · <a href="selection.json">轨迹选择规则</a> · <a href="REPORT.md">文字报告</a></p><small>原始 Episode 保留在同批次 baseline / supervised / low_calls 目录，回放只读取这些记录，不调用模型或改变物理状态。</small></main></html>'''
    page = page.replace("__CARDS__", "".join(cards)).replace("__PAIRS__", "".join(rows)).replace("__TRIALS__", "".join(detail_rows)).replace("__SHADOW__", shadow_text)
    page = page.replace("__SAVINGS__", savings).replace("__PROFILE__", profile).replace("__TOKENS__", tokens)
    (output / "index.html").write_text(page)
    md += ["## 当前边界", "", shadow_text, "", *["- " + line for line in report["limits"]]]
    (output / "REPORT.md").write_text("\n".join(md) + "\n")
    print(json.dumps(dict(trials=len(records), replays=sum(s["available"] for s in selected), index=str(output / "index.html"))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    build(args.root, args.render)
