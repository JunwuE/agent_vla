# Flora 每 50 步监督（2026-10-30）

CloseToasterOvenDoor 的默认开发配置改为 Flora Outcome 监督。此变更在前十轮实验结束后完成，不能用前十轮的成功结果宣称新监督方案有效。

## 运行方式

- `configs/close_toaster_dev.yaml`：原生 450 步预算。
- `configs/close_toaster_explore.yaml`：扩展开发预算 4000 步。
- `configs/close_toaster_d3_legacy.yaml`：修改前的 D3 扩展配置，供对照。
- 通用 `configs/robocasa.yaml` 及历史运行 manifest 不改写。

在 VLA 执行阶段，第 50、100、150……个物理执行步产生评估时刻。每次使用当前阶段内最近最多 200 步的历史视频；没有阶段计划时评估整个当前子任务。输入保持 left/wrist 两视角，使用官方 Outcome 采样方式，最多 64 个时间采样帧。50 是检查周期，不是视频帧数。

Flora 自动监督强制 `with_rationale=false`。它不读取未来观测、不发送动作数组或环境 success 标签。Astra 仍能通过 `evaluate_subtask(with_rationale=true)` 请求解释；自动评分和诊断工具拥有独立次数预算，共用同一服务与设备请求锁。

评分与 Astra 只读审核解耦：审核期间继续采样；新评分记入 Working Memory，已开始的审核仍使用冻结快照。已有审核时合并触发，不创建多个 Astra 请求。审核结束不会重置 Flora 的 50 步时钟。默认关闭每 150 步无条件调用 Astra 的旧周期审核。

每个时刻至多一个自动评分请求在途。若服务耗时超过检查间隔，记录 `monitor_tick_skipped` 并跳过该次请求，不积压、不暂停 VLA；下个 50 步时刻再取新快照。因而保证的是评估时刻和跳过记录，不能在不阻塞控制的同时保证任意慢服务都完成每个请求。

## 分数如何触发 Astra

以下是初版可配置规则，尚未通过机器人标签数据校准。它们只请求审核，不声明失败、不自动修改语言或控制权。

| 评分情况 | 默认处理 |
|---|---|
| 单次低分（1–2） | KEEP，继续收集证据 |
| 同一阶段连续两次有效评分 ≤2 | ESCALATE，请 Astra 判断是否正常未完成或实际异常 |
| 连续三次有效评分完全相同（如 3、3、3） | ESCALATE，检查是否需要进一步判断；不能据此宣称物体停滞 |
| 相邻有效评分下降至少 2 分 | ESCALATE，检查是否发生退化或观察不一致 |
| 评分 ≥4 | ESCALATE，请 Astra 检查阶段／子任务是否完成；不直接 SWITCH |
| 其余情况 | KEEP |
| 请求超时、无效评分或预算不足 | ESCALATE；没有分数不伪造 KEEP |

低分、相同分数不是物理失败证据。评分是离散 Outcome 判断，不是等距进度百分比。阶段或尝试改变时重置评分历史；缺失评分会打断连续评分计数。请求返回时检查 Episode、指令／子任务版本、控制权、尝试、阶段定义和时效，过期结果丢弃，不能触发干预或污染分数历史。

默认两次评分触发的 Astra 审核至少间隔 150 步；已有审核合并同类触发。过期 Astra 问题需要重新取证的流程保留。代码接管后由程序阶段检查和 Astra 修订闭环负责，不在暂停状态反复调用“VLA 是否正常”的评分器。

五类 Decision 契约继续兼容。Flora 路由当前只输出 KEEP / ESCALATE；不会把 1–5 分硬映射成五种语义。REFINE、REPLAN、切换及代码恢复由 Astra 审核与独立验证决定。

## 末端低运动的独立审核入口

2026-10-30 后续修正：初版 Flora 分支只把数值停滞信号附在上下文，没有独立调度审核。
现在每个 VLA 执行步检查历史状态，满足规则后直接请求 Astra 只读审核，不等待 Flora 返回或改变评分，
不改变 Flora 的 50 步时钟。仍只有 Astra 的有效干预请求能进入暂停后的确认和 Code Policy 接管。

CloseToaster 的两个配置采用 `stagnation_scope: eef_translation`：最近 2 秒仿真时间内，
末端与底盘平移范围各不超过 0.01 米，底盘转角不超过 0.05 弧度。20 Hz 下是 40 个控制步、
41 条连续状态观测；`stagnation_steps: 10` 只是最低样本数。末端转腕、夹爪变化保留在证据里，
不再否决低平移审核，因为它们既可能是正常对齐，也可能是持续无效调整。范围使用窗口内各坐标的
最大最小值计算，不仅比较首尾位置，因此明显往返运动不会因净位移为零被判为静止。

该规则仍有独立 4 秒仿真时间（80 步）冷却，已有 Astra 审核时合并请求；不受评分审核 150 步冷却限制。
完整动作历史、同一尝试和阶段、有效状态、VLA 控制来源的要求保留；代码执行、缺测和停机不算 VLA 停滞。
通用配置默认 `whole_robot`，保留原先 4 秒、2 mm 及姿态和夹爪都低运动的条件。

这些都是审核候选，不是失败标签。正常的原地对齐也可能命中，Astra 仍应结合视频、Flora 和阶段判断。
仿真时间不等于墙钟时间，40 步窗口也不保证 2 秒内得到 Astra 回复。

## 配置与启动

服务配置位于 `flora`；调度、评分阈值和自动监督次数预算位于 `decision` / `decision.flora`。Astra 的诊断工具次数预算仍使用 `flora.max_calls`。扩展配置自动评分上限 100 次，原生预算配置上限 20 次。

```bash
# 分别在独立终端启动；Flora 与 pi05 的显存和实际共卡延迟需实测。
.venv/bin/python scripts/serve_from_config.py vla --config configs/close_toaster_explore.yaml
.venv/bin/python scripts/serve_from_config.py flora --config configs/close_toaster_explore.yaml

# Flora 模式不需要启动 D3 服务。
.venv/bin/vla-agent doctor --config configs/close_toaster_explore.yaml
.venv/bin/vla-agent run --config configs/close_toaster_explore.yaml --output runs/close-toaster-flora50-seed0
```

兼容旧启动入口时，`serve_from_config.py decision` 在 `decision.kind=flora` 下启动 Flora。不要再与 `flora` 命令同时启动同一端口。

## 记录和验证边界

自动监督输出 `flora_monitor_result`、`decision`、`review_coalesced`、`monitor_tick_skipped` 等事件。`flora_calls` 汇总自动与诊断请求，另外分别提供 `flora_monitor_calls`、`flora_tool_calls`。Decision 窗口导出带有实际 Flora 请求文件、视频范围、评分与路由，不会把仅评分结果伪装成五类模型概率。

单独 BF16 Flora 热调用记录约 0.58–0.98 秒服务内耗时，HTTP 约 0.71–1.28 秒；这不包含全部新视频准备成本，也不保证共卡、量化或首请求延迟。当前扩展配置保留 int8，新链路的实际在线延迟和判别质量尚需重新测量。

本次软件验证覆盖：低分/高分/回落/相同分数规则、阶段切换、因果视频窗口、并行审核期间 50 步采样、审核合并和冷却、慢服务跳过、过期结果、诊断锁超时、独立预算及数据导出。前十轮仍是 D3 监督实验，不更改其原始结果。

2026-10-30 首次接入评分监督时，全量测试结果为 `176 passed in 14.89s`。HTTP 集成使用模拟服务，没有新增任务成功率结果。

同日补齐独立低运动审核后：`181 passed in 17.69s`，Ruff 通过。新增测试覆盖 40 步/41 条状态边界、
转腕和夹爪信息保留、底盘运动排除，以及 Flora 在途时独立触发、请求合并、不暂停 VLA、50 步评分时钟不变。

旧日志的因果数值回放如下；这是候选触发时刻，未执行新 Astra 请求或仿真，不代表新的恢复结果。

| 旧运行 | 旧规则首次命中 | 新规则首次命中 |
|---|---|---|
| baseline / 第 1 轮 seed0 | 无 | 第 126 步 |
| supervised / 第 2 轮 seed0（最终 VLA 成功） | 无 | 第 112 步 |
| 第 9 轮 seed0 的 VLA 前缀 | 无 | 第 96 步 |
| 第 10 轮 seed0 的纯 VLA 失败轨迹 | 无 | 第 96 步 |

原地调整的成功轨迹也会触发，说明规则适合请求审核，不能自动贴失败标签。
复现脚本：[replay_low_motion_rules.py](../scripts/replay_low_motion_rules.py)；
详细数值：[低运动回放结果](../runs/toaster-cross-seed-20261030/low-motion-current-replay.json)。
