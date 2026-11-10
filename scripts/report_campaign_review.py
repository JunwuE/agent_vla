"""Combine completed paired trials, routing/service audits and independent Flora calibration."""

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main(root):
    comparison = read(root / 'comparison.json')
    groups = comparison['groups']
    lines = ['# OpenDrawer 20 seeds：迭代结果、路由、成本与失败分析', '',
        '本报告统计 seeds 0–19 的完整配对轮次。各组独立使用同一初始技能/经验库、checkpoint 59999、'
        '原生 OpenDrawer 指令和 policy sampling seed=42+seed。没有用不同轮次的最佳 seed 拼接成功率。', '',
        '## 1. 完整结果', '',
        '| 版本 | 750 步成功 | 最终成功 | 全部 episode 平均分钟 | 上报 token | Astra turns |',
        '|---|---:|---:|---:|---:|---:|']
    for name, group in groups.items():
        lines.append(f"| {name} | {group['native_successes']}/20 | {group['extended_successes']}/20 | "
            f"{group['mean_seconds']/60:.2f} | {group['reported_tokens']:,} | {group['astra_turns']} |")
    lines += ['', '750 步是原生 horizon，1500 步是本次统一延长预算。原生/延长表现不能混报。'
        '服务错误、预算耗尽和仿真失败均保留在 20 条分母内；提前失败的短耗时不等于任务提速。', '']
    baseline = next(iter(groups))
    for name in list(groups)[1:]:
        pair = comparison['paired_comparisons'][f'{baseline}_to_{name}']
        lines += [f"- {name} 相对纯 VLA：新增成功 {pair['gains']}；失去成功 {pair['losses']}；"
            f"仍失败 {pair['still_failed']}。共同成功轨迹平均 "
            f"{pair['common_success_mean_seconds_a']/60:.2f} → {pair['common_success_mean_seconds_b']/60:.2f} 分钟。"]
    lines += ['', '[逐 seed 完整对照](comparison.md) · [逐条时间与调用 CSV](per-episode.csv) · '
        '[逐条代码恢复与版本](code-recovery-evidence.md)。', '',
        '## 2. 每轮改了什么', '',
        '- VLA 对照：关闭初始 Astra、监督、Flora 和代码恢复。',
        '- R1：完整工具审核，冻结当时实现。它保住了纯 VLA 的 12 条成功，并用已有持握拉抽屉代码救回 seed 5；'
        '但 seeds 4/6/7 在实际代码执行前耗尽 token。',
        '- R2：运行中审核限制为 4 次只读工具调用，暂停规划保留完整 24 次工具权限；'
        '对过期的停滞证据重新观测；可选审核预算耗尽后允许 VLA 继续；已通过宿主校验的具体方案不会因末尾 token 通知被丢弃。'
        '它通过旧代码救回 seeds 5/19，但 seed 11 在初始化时被远端过载阻断，不能视为物理策略退化。',
        '- R3：保留 R2 机器人策略与预算，增加全局 3 个 Astra 请求槽位、共享退避、服务错误分类和最多一次有界重试。'
        '重试只用于初始/暂停规划，计入原有 8-turn/1.5M-token 预算；只读审核不重试旧观测。'
        '槽位等待计入 episode 时间。', '',
        '三轮都保持 5 个仿真 worker、Astra high、每 episode 8 turns、暂停规划每 turn 24 tools、'
        '累计上报 token 1.5M 停止阈值。没有 episode 墙钟上限，但单次 planner 保留 300 秒上限。'
        '最后在途 token 通知可能超出阈值，这不代表额外获得了额度。'
        '未执行的 agent-r2 目录只是准备快照，不计一轮实验。', '',
        '## 3. 当前路由到底如何工作', '',
        '```mermaid', 'flowchart TD',
        '  A[原生 atomic instruction] --> B[Astra 初始化阶段计划]',
        '  B --> C[VLA 连续生成动作]', '  C --> D[Harness 唯一动作入口]',
        '  D --> E[当前 RGB-D 与实测机器人状态]',
        '  E --> C', '  E --> F[Flora 每 50 步 + 运动停滞事件]',
        '  F --> G[异步 Astra 只读审核]',
        '  G -->|continue| C', '  G -->|请求干预| H[时效与状态复核]',
        '  H -->|仍有必要| I[暂停 VLA、清空旧动作、读取新状态]',
        '  I --> J[Astra 检索技能或修改程序]',
        '  J --> K[Code Policy 分段执行整个已分配子任务]',
        '  K --> D', '  K --> L[独立完成条件验证]',
        '  L -->|失败且有预算| J', '  L -->|成功| M[记录证据与候选技能]', '```', '',
        '当前配置 decision.kind=flora；配置里保留的 d3-flash endpoint 不代表 Decision3 同时在跑。'
        '五类判定接口仍保留，只有 KEEP 直接继续，其余均进入 Astra。'
        'Flora 是观察证据/审核触发器，不拥有动作权限；Astra 也没有在本实验直接输出 EEF 修正动作。'
        '实际动作来源及控制 lease 由 routing-audit 检查。', '',
        'OpenDrawer 仅安装 task_success 作为独立验证器，因此保留一个可执行 atomic 子任务：'
        '“Open the left/right drawer.”。Astra 把 approach、engage、pull 等细阶段记入 phase plan，'
        '没有将它们改写成 VLA 的细粒度动作语言。Code Policy 一旦接管，必须完成整个已分配子任务或返回明确失败；'
        '夹爪闭合和 EEF 移动都不是成功证明。', '',
        '## 4. 服务、时间与成本', '',
        '| 版本 | 过载通知 | 断流通知 | 初始规划均秒* | 只读审核均秒* | 已记录的暂停规划总秒 |',
        '|---|---:|---:|---:|---:|---:|']
    for name, group in groups.items():
        audit = read(root / name / 'agent-runtime-audit.json')
        service_path = root / name / 'planner-service-audit.json'
        service = read(service_path) if service_path.exists() else {}
        errors = service.get('error_notifications', {})
        initial = audit['intents'].get('initialize_plan', {}).get('metrics', {}).get('seconds', {}).get('mean')
        review = audit['intents'].get('review_running', {}).get('metrics', {}).get('seconds', {}).get('mean')
        lines.append(f"| {name} | {errors.get('serverOverloaded', 0)} | {errors.get('responseStreamDisconnected', 0)} | "
            f"{initial:.1f} | {review:.1f} | {audit['wall_components_seconds'].get('astra_paused', 0):.1f} |"
            if initial is not None and review is not None else f"| {name} | 0 | 0 | — | — | 0 |")
    lines += ['', '*模型 turn 延迟只对有结束事件的调用统计；不将被取消、断流或超时调用当成 0 秒。'
        '暂停规划总秒是运行时分析中可归属的已结束区间，不等于表中模型均秒之和；未归属部分单独保存。'
        '后台审核/Flora 与物理执行重叠，不能直接相加成 episode 时间。', '',
        'R2 的 12 次 serverOverloaded 和 46 次断流通知不是 58 次独立计费请求。'
        'seeds 11/15 在首次动作前终止，其 final-audit 保持失败：缺少 instruction/checkpoint 的实际执行证据。'
        '因此较低 token 和较短失败耗时不能全部归因于优化。', '',
        '上报 token 是重复输入上下文与输出的累计，不是账号剩余额度，也不是货币账单。'
        '没有可靠的缓存计费拆分，不能伪造金额。账号快照和原始错误的核对见 '
        '[额度/过载诊断](usage-limit-diagnosis.md)。', '',
        '每组 action-performance.md 将动作拆为仿真（含内部渲染）、几何快照、通信/序列化/调度残差、'
        '原始观测压缩存盘。VLA 推理约几十毫秒一 chunk，通常不是当前分钟级延迟的主因。'
        '初始化推理、暂停后的技能检索/几何确认、未及时接管的无效物理步和仿真/存盘共同决定总时间。', '',
        '## 5. Flora 是否适合作为失败判断', '',
        '独立核验使用 20 条纯 VLA 轨迹，每条早期/中期/完整视频各一次，共 60 次。'
        '完整轨迹分数 ≥4 时准确率 70%，8 条原生失败中 5 条被判成功；阈值升到 5 时准确率 75%，'
        '仍把失败 seeds 7/15 判为满分，并漏掉三条原生成功。', '',
        '40 个历史窗口中 28 个评分 ≤2，其中 16 个来自后来纯 VLA 成功的轨迹。'
        '这说明低分不能直接等同失败，但它不是严格的在线误报率：事后成功不能代替当时是否应干预的独立标签。'
        '完整 atomic 目标的离线矩阵也不能替代在线细阶段判断的准确率。', '',
        '本次客户端平均 2.864 秒、服务端平均 0.955 秒。Flora Outcome 适合作辅助证据，'
        '当前数据不支持让它单独决定失败、接管或成功。Dense 模式未测试，也没有 Decision3 的同窗口配对评测。'
        '[全部分数、协议与局限](flora-evaluation/report.md)。', '',
        '## 6. Astra 与已有 Code Policy 的价值和瓶颈', '',
        'R1 seed 5 和 R2 seeds 5/19 的恢复均复用了初始库已有候选技能，分别是 '
        '`drawer-pull-from-handle-hold@95bf87d8c3414ad4` 与 '
        '`drawer-regrasp-and-open@0ed2fc3427424179`。每次用当前 RGB-D 重算区域/几何参数；'
        '没有把此前绝对坐标直接重放。R2 两次代码尝试全部成功，说明复用在适合的入口条件下确有收益。', '',
        'R1 seeds 8/15/16 在 align_and_seat、establish_reacquisition_envelope、clear_counter_before_regrasp '
        '阶段未达到程序的位姿收敛要求。最后若干短段仍存在约厘米级位置残差及姿态残差；'
        '这只是观察事实，不能确定其物理原因。程序通过 metadata 契约检查也不等于当前抓取入口可执行。', '',
        'Astra 的合理职责是审核语义进展、检查技能适用条件、根据失败证据修改局部程序。'
        '当前实验支持这条链路偶尔救回 VLA 失败；不支持把“选用 Astra”本身称为已验证最优。'
        '没有替代 reasoner 的同条件对照，而且初始化、工具往返和服务可用性成本明显。', '',
        '各轮 retained/OpenDrawer 保存新增候选代码及经验，原始 workspace 还保存未执行的编辑。'
        '候选未因单次成功自动晋升；本批每 seed 使用相同初始库，不共享新经验。'
        '因此本报告衡量固定库复用与路由，不是跨 seed 自进化加速曲线。', '',
        '## 7. 下一步最有根据的改进', '',
        '1. 减少低价值审核：Outcome 低分必须结合任务相关进展/重复失败证据；为真正暂停后的恢复预留调用预算。'
        '在相同因果窗口标注审核需求后，再对照 Decision3 或 Flora Dense，不能仅按参数量或终局分数选型。',
        '2. 缓存经 Astra 审核的 atomic 阶段模板，按当前场景检查适用性；减少每个 seed 重建相同原生任务计划。',
        '3. 让技能前置检查返回可诊断的当前几何/遮挡/位姿条件，优先参数修订和已有程序复用；'
        '对不收敛问题做可复现控制诊断，不放松碰撞、几何或完成判据来获得表面成功率。',
        '4. 将服务排队与机器人决策分离；只读审核拿不到模型槽位时保留最新待审证据，避免消耗恢复容量或使用过期状态。', '',
        '这些是本轮证据支持的后续方向，不是声称已实现或已通过评测的改进。'
        '版本配置、源码哈希、初始 Memory 哈希、完整事件、模型请求/回复和验证证据均随每组保存。', '',
        '代码入口：`src/vla_agent/runtime.py`（主循环）、`supervision.py`（异步审核）、'
        '`codex_planner.py`（模型/工具/预算）、`planner_service.py`（限并发与退避）、'
        '`harness.py`（唯一动作执行）、`recovery.py`（代码恢复）；'
        '本轮配置 `configs/open_drawer_resilient.yaml`。', '',
        '测试：63 项相关回归通过；后续锁实现优化的 3 项测试和服务终局错误的 2 项测试通过；'
        '跨进程并发、取消释放、进程死亡释放、预算内重试、额度错误不重试均有覆盖。']
    if 'agent-r3-resilient' in groups:
        lines += ['', '## 8. R3 的具体新发现', '',
            'R3 为 14/20（70%），比纯 VLA 多成功 seeds 5/7，并保留原有 12 条成功。'
            '平均 8.14 分钟，较 R2 的 8.21 分钟没有明显改善；不能把服务恢复正常等同端到端提速。'
            'R3 上报 9,001,903 token，高于 R2 的 7,180,308；R2 有两条零动作早退，比较时必须保留这个区别。', '',
            'R3 远端过载通知为 0、可重连断流通知为 2，没有账号额度错误。全局同时最多 3 个 lease，'
            '等待均值约 4.50 秒、最长 75.02 秒；初始规划平均 65.61 秒，完成的后台审核平均 57.42 秒。'
            '一次服务时段的对比不能把这些变化全部归因于限并发。', '',
            'R3 82 次审核请求中 22 次本地槽位超时，27 次被丢弃，7 次被取消，仅 26 次形成被接受的审核；'
            '后者 22 次继续、4 次请求干预。59 个真实审核 turn 与 82 个审核请求是不同计数。'
            'seed 6 因等待被误计入 max_reviews，实际只有初始化一次模型 turn，随后跑到 1500 步失败。', '',
            'seeds 15/16 多次请求在 step/时间窗口或位移复核处被丢弃。5 cm 位移复核的适用范围也需要收紧：'
            '移动可以反驳“持续静止”，却不能单独反驳“持续任务相关错位”。这解释了为什么有些请求仍会循环审核。'
            '该实验选项未据此声称是已验证的默认最优策略。', '',
            'seed 4 先因 handle 遮挡/有效表面不足在零动作阶段失败，更换观测参数后执行 118 步，'
            '在 clear_and_align 未收敛，随后耗尽 token。seed 19 在第 1435 步才接管，'
            '同样未通过 handle 几何入口，执行 0 个代码动作；再次修订前已用完 8 turns，虽然 token 仅 742,954。'
            '几何入口失败、太晚接管、turn 预算分别是独立事实。', '',
            'R3 seed 5 直接复用持握拉动技能，代码 220 步成功；seed 7 直接复用 '
            '`drawer-regrasp-and-open@7766677d945ef0ca`，代码 276 步成功。'
            '成功来自已有程序与当前参数，不需要为相似任务重新生成一套代码。']
    followup = root / 'agent-r4-admission-followup'
    if (followup / 'planner-service-audit.json').exists() and (followup / 'atomic-metrics.json').exists():
        metrics, service = read(followup / 'atomic-metrics.json'), read(followup / 'planner-service-audit.json')
        summary = metrics['summary']
        lines += ['', '## 9. R4：5 条针对性复测', '',
            '预选 seeds 4/6/7/8/16，修正本地槽位超时消耗审核次数的问题。'
            '只有“未发出模型请求”的 localAdmissionTimeout 退还审核次数；真实模型调用和 token 不退还，'
            '请求编号保持唯一，后续请求读取新状态。35 项相关测试通过。', '',
            f"本小组成功 {summary['extended_successes']}/5，平均 {summary['mean_seconds']/60:.2f} 分钟，"
            f"Astra {summary['astra_turns']} turns，上报 {summary['reported_tokens']:,} token。"
            f"实际发生 {service['no_request_budget_refunds']} 次无模型请求的预算退还。", '',
            '| seed | 成功 | 终止原因 | 步数 | 分钟 | Astra turns | token |',
            '|---:|---|---|---:|---:|---:|---:|']
        for row in metrics['rows']:
            lines.append(f"| {row['seed']} | {row['success']} | {row['reason']} | {row['steps']} | "
                f"{row['wall_seconds']/60:.2f} | {row['astra']} | {row['reported_tokens']:,} |")
        lines += ['', '这 5 条针对已知失败和已恢复案例选取，不能当成新 20-seed 成功率，也不并入 R3。'
            '异步调度、干预时刻和模型回复也会变化，不能将每条成功/失败差异单独归因于计数修复。', '',
            'seed 6 首次救回：第 840 步接管，复用旧 0ed2fc3427424179，并增加有条件的张开夹爪入口；'
            '代码 332 步、120.05 秒完成，全 episode 第 1172 步成功。保留候选版本 5ada71444e905c90，'
            '只在本次一个布局通过验证，尚未跨布局晋升。seed 7 本轮未进入代码恢复，再次失败；'
            '因此这组复测没有证明系统成功率稳定提升。', '',
            'seed 4 的第二版避免重试时重复累加 16 cm 外移/12 cm 上移：验证当前位姿接近先前测得的 clearance 目标后，'
            '继续完成旋转。它保留原有容差与几何检查，成功进入抓取和拉动阶段，但在第 1500 步耗尽预算，'
            '仍然算失败。候选 83f6bbe4e358fc9c 只保存失败与阶段进展证据。', '',
            '[两个候选的适用条件与证据](agent-r4-admission-followup/retained/OpenDrawer/DEVELOPMENT_NOTES.md) · '
            '[版本索引](agent-r4-admission-followup/retained/OpenDrawer/INDEX.md)。', '',
            '[R4 完整结果](agent-r4-admission-followup/atomic-results.md) · '
            '[请求/实际调用/退还核验](agent-r4-admission-followup/planner-service-audit.json)。']
    lines += ['', '## 回放', '',
        '[本次完整 20 条](agent-r3-resilient/report/README.md) · '
        '[5 条针对性复测](agent-r4-admission-followup/report/README.md) · [统一回放索引](replays.md)。', '',
        '视频保持全部物理帧和阶段，概览展示关键工具，包括程序导入、编辑、执行和验证。'
        '推理等待在视频中压缩，真实耗时以结果表为准；全部工具调用保存在 events.jsonl。']
    (root / 'experiment-review.md').write_text('\n'.join(lines) + '\n')
    print(root / 'experiment-review.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root.resolve())
