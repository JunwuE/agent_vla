"""Write the final development findings and skill inventory from measured sweep artifacts."""

import argparse
import json
from pathlib import Path

import numpy as np
from build_seed_sweep_report import statistics


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    records = [read(p) for p in sorted((root/'analysis').glob('seed*.json'))]
    aggregate = dict(all=statistics(records),
                     without_code=statistics([r for r in records if not r['program_attempts']]),
                     with_code=statistics([r for r in records if r['program_attempts']]))
    audit = read(root/'execution-audit.json')
    total = aggregate['all']
    times = np.array([r['result']['wall_seconds'] for r in records])
    components = {key:sum(r['wall_components_seconds'].get(key,0) for r in records)
                  for key in ('vla_inference','vla_actions','code_execution','astra_paused','other_and_unattributed')}
    lines = ['# CloseToasterOvenDoor：50 seeds 开发结果与耗时诊断', '',
             f'主批已完成 {len(records)}/50，原生任务成功 {total["successes"]}/{len(records)}；'
             f'450 步内成功 {total["within_450"]}/{len(records)}。', '',
             '## 实验口径', '',
             '固定 OmniSF π0.5 checkpoint 59999，scene seed 0–49，策略采样 seed=42+scene seed；'
             '每条 2000 动作步，无 Episode 墙钟超时。Flora 每 50 VLA 步监督，低末端平移可提前请求后台审核；'
             '只有 Astra 审核并重新检查当前状态后才能交给 Code Policy。直接局部动作通道关闭。', '',
             '这是允许跨 Episode 积累 Memory、且包含框架修订的开发批次。每条使用启动时冻结的软件和技能快照，'
             '最多 3 条主批并发，共享模型服务；诊断基线先单条执行，收尾随主批退出增至最多三条并发，'
             '暂停问题定位时还有临时定向复测。'
             '因此各段并发负载不同。失败结果保留，定向复测单列。'
             '模型服务已预热，单条 runtime 计时包含仿真初始化、规划、执行与清理，不包含一次性模型服务加载。'
             '不能把这里的成功率直接解释为冻结系统的独立泛化成功率，也不能与旧 48% 作同条件比较。', '',
             '## 总体速度与调用', '',
             f'- 单条墙钟耗时：均值 {times.mean()/60:.2f} 分钟，中位数 {np.median(times)/60:.2f} 分钟，'
             f'P90 {np.quantile(times,.9)/60:.2f} 分钟。包含初始化和清理。',
             f'- 成功轨迹单独计算：均值 {(total["successful_mean_seconds"] or 0)/60:.2f} 分钟，'
             f'中位数 {(total["successful_median_seconds"] or 0)/60:.2f} 分钟。失败提前停止不视为能力提速，'
             '成功子集也有选择偏差，需连同全部结果阅读。',
             f'- Astra 共 {total["astra_turns"]} turns，包含初始化、后台审核、暂停确认和代码修订；'
             f'代码尝试 {total["code_attempts"]} 次。',
             f'- Flora 共 {total["flora_calls"]} 次；Decision 路由类别 {total["decision_labels"]}；'
             f'后台审核触发来源 {total["review_triggers"]}。本批 Decision 使用 Flora 评分路由。',
             f'- 代码程序累计执行 {total["code_program_seconds"]/60:.2f} 分钟；代码修订推理累计 '
             f'{total["astra_revision_seconds"]/60:.2f} 分钟；零动作退出尝试 {total["zero_action_attempts"]} 次；'
             f'Astra 直接局部动作 {total["direct_action_steps"]} 步。',
             f'- 旧程序导入 {total["imported_programs"]} 次，直接技能调用 {total["direct_skill_calls"]} 次。'
             '导入、直接调用和验证成功是三个不同指标。',
             f'- 按累计 Episode 主链路耗时分解：Astra 暂停规划/修订 '
             f'{components["astra_paused"]/times.sum():.1%}，VLA 动作确认 '
             f'{components["vla_actions"]/times.sum():.1%}，代码执行 '
             f'{components["code_execution"]/times.sum():.1%}，VLA 推理请求 '
             f'{components["vla_inference"]/times.sum():.1%}。动作确认含仿真、传输和记录；'
             '这是各 Episode 累计分解，不是并发批次墙钟。当前主要优化空间在接管与修订链路。', '',
             '| 执行类别 | 条数 | 成功 | 均值分钟 | 中位分钟 | 代码尝试 |',
             '|---|---:|---:|---:|---:|---:|']
    for key,label in [('without_code','未执行代码'),('with_code','执行代码')]:
        group=aggregate[key]
        lines.append(f'| {label} | {group["episodes"]} | {group["successes"]} | '
                     f'{(group["mean_seconds"] or 0)/60:.2f} | {(group["median_seconds"] or 0)/60:.2f} | '
                     f'{group["code_attempts"]} |')
    lines += ['', 'Flora、后台审核与 VLA 异步重叠，不把其耗时重复加到主链路。'
              '日志中的 input tokens 含多次请求与缓存输入计数，不能直接等同收费 token 或美元成本。', '',
              '“未执行代码”仍可能包含暂停复核后继续 VLA，例如 seed 28；旧版暂停会丢弃待执行 chunk，'
              '不能据动作来源全部为 VLA 就认定它与纯 VLA 的连续轨迹等价。每条报告保留暂停后方案。', '',
              '## 是否越跑越快', '',
              '| seed 区间 | 已完成 | 成功 | 均值分钟 | 中位分钟 | 代码尝试 |',
              '|---|---:|---:|---:|---:|---:|']
    for start in range(0,50,10):
        group=[r for r in records if start<=r['seed']<start+10]
        if group:
            t=[r['result']['wall_seconds'] for r in group]
            lines.append(f'| {start}–{start+9} | {len(group)} | {sum(r["result"]["success"] for r in group)} | '
                         f'{np.mean(t)/60:.2f} | {np.median(t)/60:.2f} | {sum(r["program_attempts"] for r in group)} |')
    lines += ['', '本批没有出现随经验积累持续加速的结果，最后十条的平均耗时反而上升。'
              '这些是描述统计：seed 难度、接管比例、软件版本、Memory 快照与并发排队同时变化，'
              '不能把组间差异直接归因于技能学习；并发下 seed 顺序也不等于完成或经验合并顺序。', '',
              '## 同 seed 诊断对照', '',
              '对发生代码接管或曾暂停后继续的 seed 运行或复用匹配的纯 VLA 对照；'
              '该子集不能估计完整纯 VLA 成功率。'
              '下面同时检查配置、初始观测数组及首次暂停前的每条动作和实测状态。'
              '暂停后即使继续 VLA，也已进入处理效应范围；另外记录整个 VLA 段的首个差异。', '',
              '| seed | 首次暂停前匹配 | 暂停步数 | Agent 成功/步数/分钟 | 纯 VLA 成功/步数/分钟 | 结论 |',
              '|---:|---|---:|---|---|---|']
    labels={'agent_success_baseline_budget_failure':'对照预算内失败，Agent 恢复成功',
            'baseline_already_succeeds':'纯 VLA 已能完成，不能认定需要接管',
            'both_fail':'双方失败','pairing_incomplete':'前缀或配置未完全匹配，谨慎比较'}
    for p in audit['matched_controls']:
        a,b=p['agent_result'],p['baseline_result']
        lines.append(f'| {p["seed"]} | {p["matched_initial_conditions_and_pre_pause_prefix"]} | {p["first_pause_step"]} | '
                     f'{a["success"]}/{a["control_steps"]}/{a["wall_seconds"]/60:.2f} | '
                     f'{b["success"]}/{b["control_steps"]}/{b["wall_seconds"]/60:.2f} | {labels[p["conclusion"]]} |')
    matched=[p for p in audit['matched_controls'] if p['matched_initial_conditions_and_pre_pause_prefix']]
    gained=[p['seed'] for p in matched if p['agent_result']['success'] and not p['baseline_result']['success']]
    lost=[p['seed'] for p in matched if not p['agent_result']['success'] and p['baseline_result']['success']]
    both=[p['seed'] for p in matched if p['agent_result']['success'] and p['baseline_result']['success']]
    lines += ['', f'在前缀完全匹配的诊断案例中，恢复收益案例为 {gained}；纯 VLA 成功而 Agent 失败的案例为 {lost}；'
              f'双方都成功的案例为 {both}。双方都成功时继续比较步数和墙钟时间，不把代码执行成功自动当作净收益。'
              '这些是所选案例的对照结果，不能替代完整同预算基线。', '',
              '本批的后续优先项是校准接管价值：同时检查当前失败证据、已有恢复程序的物理前置条件及其历史效果。'
              '把程序写得更快，并不能抵消对本可成功的 VLA 轨迹造成的损害。'
              '反复出现的可见面/目标对应失败需要稳定的感知与跟踪支持；只读几何拟合的数值正确性不代表目标识别正确。'
              '这些限制尚未解决，不能把候选技能库的增长直接解释为稳定能力增长。']
    lines += ['', '## 定向修改及验证', '',
              '1. **批量修改同一程序版本**：`patch_program_batch` 一次校验并应用多个互不重叠修改，'
              '减少多次工具往返；任一修改不匹配则全部不写入。',
              '2. **减少重复历史上下文**：同一线程已发送的历史经验改为引用，支持 `read_experience` 按需展开；'
              '当前观测和失败反馈保留。旧轨迹离线回放中重复经验上下文从 55,425 字符降到 19,387 字符，'
              '这不是等比例的端到端延迟提升承诺。',
              '3. **提供机器人手指垫几何**：读取已知机器人构型和当前关节变换，减少反复从遮挡图像估计手指位置；'
              '不提供目标真值、接触真值或抓取成功。',
              '4. **修正经验检索偏向恢复案例的问题**：相关经验中保留普通 VLA 成功和失败对照；'
              'Astra 区分接近/对齐与实际操控。seed 5 定向 v4 复测 358 步、198.57 秒成功，'
              '全部动作与纯 VLA 一致；原始失败 836 步、1912.62 秒仍计入主批。',
              '5. **数值几何拟合工具**：`fit_geometry` 支持只读平面/圆拟合和逐点残差、退化、弧跨度诊断，'
              '避免只为测试算式而启动零动作恢复轮次。不放宽阈值、不替代视觉对应或原生任务检查。', '',
              '6. **技能检索保留同任务成功证据**：避免关键词丰富的失败变体占满前五个结果；'
              '保留一份同任务有成功证据的程序，继续保留失败阶段匹配项，不改变适用性判断或执行权限。', '',
              '7. **技能搜索只返回运行摘要**：实际查询回放减少约 22%–37% 返回字符，'
              '原生验证、失败原因、阶段与参数信息保留；完整历史数值仍可用 read_skill 展开。', '',
              '8. **支持嵌套恢复阶段**：修复子阶段提前关闭外层记录而导致的协议错误；'
              '父子阶段分别检查完成条件，预算耗尽时全部关闭，不增加动作额度。', '',
              '9. **普通复核等待动作 Chunk 边界**：seed 35 在第 532 步暂停并丢弃 3 个动作，'
              '虽然 Astra 随后选择 continue，第 533 步起便与纯 VLA 不同。普通后台审核请求现在等待'
              '当前 Chunk 结束后重新校验时效并暂停；危险判断仍立即处理。推理期间到达的请求也不丢弃'
              '已采样 Chunk。仍由 Harness 在真实接管时清理队列，不恢复旧控制权。', '',
              'seed 44 复用了 seed 25 的候选，直接执行因观测参考检查失败；一次批量补丁后，'
              '17 个代码动作通过原生验证。该次修订推理约 34.52 秒；匹配的纯 VLA 在 2000 步内失败，'
              '这条提供了适配旧代码后的恢复收益证据。', '',
              '完整发布时间和测试见 [changes.md](changes.md)；最新核心回归 216 tests passed，'
              '另有机器人几何零动作仿真检查。', '',
              '## 直接技能调用的实际结果', '',
              '| seed | 技能 | 验证 | 执行步数 | 程序秒数 |', '|---:|---|---|---:|---:|']
    for call in audit['direct_invocations']:
        outcome=call.get('following_recovery') or {}
        skill=f'{call["skill_id"]}/{call["version"]}'
        lines.append(f'| {call["seed"]} | [{skill}](shared/skills/{skill}/policy.py) | '
                     f'{outcome.get("verified")} | {outcome.get("executed_steps")} | '
                     f'{outcome.get("seconds",0):.2f} |')
    direct_success = [c for c in audit['direct_invocations'] if (c.get('following_recovery') or {}).get('verified')]
    lines += ['', f'直接按技能 ID 调用共 {len(audit["direct_invocations"])} 次，原生验证成功 '
              f'{len(direct_success)} 次；导入后修改的成功另计，不能混入原样复用成功率。']
    if [c['seed'] for c in direct_success] == [9]:
        lines += ['', '唯一原样调用成功的是 seed 9 的近竖直门收尾程序，它来自此前同 seed 的探索。'
                  '这是再次执行成功，尚不能证明未见布局下的原样迁移。seed 44 则提供了导入逻辑经局部修改后跨 seed 恢复的案例。']
    lines += ['', '技能仍按实际状态前提使用。接近竖直门的外侧压合代码可用于收尾，'
              '不表示它能从任意初始状态完成全部抓取与关门。候选技能不会因为单次成功自动提升为通用验证技能。', '',
              '`check_skill` 的兼容性结果主要核对机器人、接口和参数契约；没有声明物理前置条件时，'
              '它不能证明当前门角度、接触状态或观测区域适合直接执行。实际适用性仍依赖新观测和程序入口检查。'
              'seed 25 的候选在 seed 35、44 原样执行都曾失败，随后才修改，这也是尚未形成稳定直接复用的具体限制。', '',
              '## 每条轨迹与保留产物', '',
              '[50 条结果表](report.md) · [交互耗时/调用表](dashboard.html) · [逐条 CSV](per-seed.csv) · '
              '[技能目录](learned-skills.md) · [执行与配对审计](execution-audit.json)', '',
              f'已检查 {audit["episodes"]} 条主轨迹，命令/回执、连续观测、原生结果和冻结源码检查全部通过：'
              f'{audit["all_execution_audits_pass"]}。每条原始 RGB-D、动作命令、实测轨迹、Astra 工具调用和程序版本保留。', '',
              '可视化：[seed 0 恢复](videos/seed00/index.html)、[seed 9 恢复](videos/seed09/index.html)、'
              '[seed 5 首次失败](videos/seed05-first/index.html)、[seed 5 v4 复测](videos/seed05-v4-recheck/index.html)。'
              '另有 [seed 25 Agent](videos/seed25-agent/index.html) 与 [seed 25 纯 VLA](videos/seed25-baseline/index.html) 对照，'
              '以及 [seed 44 技能复用](videos/seed44-agent/index.html)。'
              '视频动作回放时长不等于包含模型思考等待的 Episode 墙钟时长。']
    diagnostic = root/'rechecks/readonly-pause-comparison.json'
    if diagnostic.exists():
        comparison = read(diagnostic)
        lines += ['', '## 只读暂停的定向实证', '',
                  '该实验使用真实 RoboCasa 和同一 π0.5 权重，但用脚本注入一次复核及 continue，'
                  '隔离动作 Chunk 管理的影响；不调用 Astra，不计入 50 条主批成功率，也不计为真实失败标签。', '',
                  f'对比详情： [readonly-pause-comparison.json](rechecks/readonly-pause-comparison.json)。'
                  f' v9 完整动作/实测状态匹配：{comparison["fixed"]["complete_prefix_matched"]}；'
                  f'步数 {comparison["fixed"]["agent_result"]["control_steps"]}；'
                  f'原生成功 {comparison["fixed"]["agent_result"]["success"]}。']
        old = comparison.get('old_runtime_diagnostic')
        if isinstance(old,dict):
            result = old['agent_result']
            lines += ['', f'相同注入在旧 v8 中从第 {old["first_vla_divergence_step"]} 步开始偏离；'
                      f'原生成功 {result["success"]}，执行 {result["control_steps"]} 步，'
                      f'耗时 {result["wall_seconds"]:.2f} 秒。新版 633 步成功；并发负载不同，'
                      '核心验证依据是固定采样下的完整动作/实测状态与原生结果，而不是孤立的墙钟比值。']
    (root/'findings.md').write_text('\n'.join(lines)+'\n')
    skill_lines=['# 保留的技能与实际调用', '',
                 '本目录包含原有技能及本次新增候选，代码与版本保留；原生验证通过只证明对应已测状态。'
                 '“本批成功记录”按主批 Episode ID 归属，去重后计数，不把历史成功当成本批验证。', '',
                 '| 技能/版本 | 来源 | 状态 | 本批执行/成功 | 代码及适用描述 |', '|---|---|---|---:|---|']
    episode_ids=set()
    for record in records:
        with (root/'episodes'/f'seed{record["seed"]:02d}'/'events.jsonl').open() as stream:
            for line in stream:
                event=json.loads(line)
                if event['kind']=='observation':
                    episode_ids.add(event['episode_id'])
                    break
    for path in sorted((root/'shared/skills').glob('*/*/manifest.json')):
        manifest=read(path)
        relative=path.parent.relative_to(root/'shared/skills')
        # Seed 1 launched before seed 0 completed and made no code attempts.
        # Its private library retains the original 47-entry snapshot; root/skills was seed 0's writable library.
        prior=(root/'inputs/seed01/skills'/relative/'manifest.json').exists()
        evidence=list(manifest.get('evidence',[]))
        runs=path.parent/'runs.jsonl'
        if runs.exists():
            evidence += [json.loads(line)['evidence'] for line in runs.read_text().splitlines()]
        evidence={json.dumps(e,sort_keys=True):e for e in evidence}.values()
        current=[e for e in evidence if e.get('episode_id') in episode_ids]
        description=manifest.get('description','').replace('|','/').replace('\n',' ')
        skill_lines.append(f'| {relative} | {"沿用" if prior else "本批新增"} | {manifest["status"]} | '
                           f'{len(current)}/{sum(e["report"]["verified"] for e in current)} | '
                           f'[policy.py](shared/skills/{relative}/policy.py) · {description} |')
    (root/'learned-skills.md').write_text('\n'.join(skill_lines)+'\n')
    print(json.dumps(dict(episodes=len(records),findings=str(root/'findings.md')),ensure_ascii=False))


if __name__ == '__main__':
    main()
