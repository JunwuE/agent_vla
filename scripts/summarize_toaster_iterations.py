"""Summarize every retained trial and code attempt, including unsuccessful infrastructure trials."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from vla_agent.data import read_events


def summarize(root):
    records = []
    for result_path in sorted(root.glob("round*/seed*/result.json"),
                              key=lambda p: (int(re.match(r"round(\d+)", p.parent.parent.name)[1]),
                                             int(p.parent.name.removeprefix("seed")))):
        episode = result_path.parent
        result = json.loads(result_path.read_text())
        rows = read_events(episode)
        source_counts = Counter(r["source"] for r in rows if r["kind"] == "action_executed")
        owner, step, takeovers = None, 0, 0
        for event in rows:
            if event["kind"] == "observation":
                step = event["step"]
            elif event["kind"] == "control_claim":
                takeovers += int(owner == "vla" and event["owner"] != "vla" and step > 0)
                owner = event["owner"]
        versions = {p.parent.name: json.loads(p.read_text()) | {"path": str(p.relative_to(episode))}
                    for p in episode.glob("workspaces/*/versions/*/manifest.json")}
        attempts = []
        for r in rows:
            if r["kind"] != "recovery_result":
                continue
            notes = [n for n in rows if n["kind"] == "recovery_note" and n["intervention_id"] == r["intervention_id"]]
            attempts.append({k: r.get(k) for k in ("program_version", "sha256", "result", "verified", "executed_steps", "error", "failed_stage", "seconds", "artifacts")}
                            | {"program_notes": [{k: n.get(k) for k in ("facts", "hypotheses")} for n in notes],
                               "stages": r.get("stages", []), "evidence": r.get("evidence"),
                               "version_metadata": versions.get(r.get("program_version"), {})})
        replay = [r for r in rows if r["kind"] == "vla_replay"]
        records.append(dict(path=str(episode.relative_to(root)), result=result,
                            success_within_native_horizon=result["success"] and result["control_steps"] <= 450,
                            source_steps=dict(source_counts), code_attempts=attempts,
                            unexecuted_versions=[v for k, v in versions.items()
                                                 if k not in {a["program_version"] for a in attempts}],
                            vla_takeovers=takeovers,
                            attempt_interrupt_records=sum(r["kind"] == "attempt_interrupted" for r in rows),
                            expired_reviews=sum(r["kind"] == "review_discarded" for r in rows),
                            refreshed_reviews=sum(r["kind"] == "review_refresh" for r in rows),
                            replay_max_state_error=max((r["state_error"] for r in replay), default=None),
                            model_outcomes=[r["result"] for r in rows if r["kind"] == "flora_result"],
                            skill_invocations=[r for r in rows if r["kind"] == "skill_invocation"]))
    (root / "all-results.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
    lines = ["# CloseToasterOvenDoor 多轮开发实验", "", "所有运行保留；下表不等同于冻结条件下的 SR 对照。", "",
             "| 运行 | 成功 | 总步数 | VLA / Code 步数 | Astra turns | 代码尝试 | 450 步内成功 |", "|---|---|---:|---:|---:|---:|---|"]
    for r in records:
        v = r["result"]
        lines.append(f"| [{r['path']}]({r['path']}/result.json) | {v['success']} | {v['control_steps']} | "
                     f"{r['source_steps'].get('vla', 0)} / {r['source_steps'].get('code_policy', 0)} | "
                     f"{v['astra_calls']} | {len(r['code_attempts'])} | {r['success_within_native_horizon']} |")
    lines += ["", "VLA 步数在 replay 实验中包含明示的历史命令重放。初始化 replan 不计为物理纠正。",
              "原生 horizon 为 450；扩展开发使用 2000/4000 host 步数、无 wall timeout，不能与原 48% 直接比较。",
              "旧运行使用共享采样流；固定采样协议和失败重放分别记录。Astra calls 为 Codex turn 数。", "",
              "## 每版操作程序", ""]
    for r in records:
        if not r["code_attempts"]:
            continue
        lines += [f"### {r['path']}", ""]
        for a in r["code_attempts"]:
            source = r["path"] + "/" + a["artifacts"]["source"]
            lines.append(f"- [{a['program_version']}]({source})：{a['result']}，{a['executed_steps']} 步；"
                         f"错误={a['error']}；失败阶段={a['failed_stage']}。")
            metadata = a["version_metadata"]
            lines.append("  修改假设（未必得到证实）：" + metadata.get("hypothesis", "未记录"))
            lines.append("  预期效果：" + metadata.get("expected_effect", "未记录"))
            for stage in a["stages"]:
                if stage.get("error"):
                    lines.append(f"  阶段错误：{stage['name']} — {stage['error']}")
            for note in a["program_notes"]:
                lines.append("  程序报告：" + "; ".join(note.get("facts") or []))
            if a["evidence"]:
                lines.append("  独立验证：" + json.dumps(a["evidence"], ensure_ascii=False))
        lines.append("")
    lines += ["## 生成但未执行", "", "以下源码不能作为已执行动作或成功示范。", ""]
    for r in records:
        for version in r["unexecuted_versions"]:
            source = r["path"] + "/" + str(Path(version["path"]).with_name("policy.py"))
            lines.append(f"- [{r['path']} / {version['version']}]({source})：未执行。")
    (root / "results.md").write_text("\n".join(lines) + "\n")
    return dict(episodes=len(records), code_attempts=sum(len(r["code_attempts"]) for r in records))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    print(json.dumps(summarize(p.parse_args().root)))
