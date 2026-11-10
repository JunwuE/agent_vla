# Decision 数值历史精简

本文的速度/15 任务结果描述历史 v1 配置。当前 v2 的十五图、六个动作样本、详细判据和规则证据见 [Decision 监督设置](decision-supervision.md)，不应混用两批测试的延迟数字。

原输入把 Working Memory 的最近 30 步 `command[12]`、`measured_state[16]` 和完整浮点精度
逐步序列化，同时重复发送多个观测中的控制器配置、状态及任务语言。OpenDrawer 第 50 步的
21,126 个输入 token 中，30 步动作记录占 17,137 个；图像数量不是该请求的主要 token 来源。

`decision.context_mode: compact` 现在为默认模式，生成 `vla_agent.decision_context.v1`：

- 保留主任务、当前子任务、当前有效指令及版本、首次尝试/重试状态。
- 当前实测 16 维状态保留三位小数；命令、位移统计同样舍入到三位小数。
- 最近 30 步、当前尝试内的历史转为实测末端净位移/路径长度、底盘净位移、夹爪关节变化，
  以及命令绝对值均值/峰值、夹爪与控制模式变化次数、动作来源。
- 仅附最近三步命令和对应末端/夹爪实测值，仍分开表达意图与实际运动。
- 保留最近三条事实/假设笔记、两次审核、两次干预和两条带观测步数的奖励模型评分。
- 五个主视角历史时刻、图片顺序、周期/事件触发以及 Astra 审核流程不因压缩而改变。

摘要不生成“停滞”“失败”或“完成”标签。末端位置使用随机器人移动的策略参考系；相对位移
不是物体距离或任务进度。夹爪命令/闭合不等于抓取成功。小于舍入精度的变化不应被当作严格
静止，精细控制继续使用原始状态和工具。

只改变发送给 Decision 的表示；Memory、Recorder、VLA/Code Policy 控制数据保持原始精度。
`decision_window` 保存完整审核输入，`decision_request` 保存实际发送的摘要、schema 和字符数；
`context_mode: full` 可恢复原表示用于消融。辅助说明过长会明确标注截断，任务指令不截断。

## 同图对照

同一 H100、同一 d3-flash 版本与加速算子，对三个固定历史请求分别预热两次、计时三次。
两组使用完全相同的九张图和五类问题，只替换数值上下文，不把五图配置变化混入本次比较。

| 窗口 | token：原始→摘要 | 服务端热调用中位数：原始→摘要 | 原始分类变化 |
|---|---:|---:|---|
| OpenDrawer:50 | 21,126 → 2,509 | 1.304 → 0.150 秒 | REFINE → KEEP |
| PickPlaceCounterToStove:50 | 20,941 → 2,610 | 1.282 → 0.155 秒 | REFINE → KEEP |
| TurnOnSinkFaucet:50 | 20,996 → 2,476 | 1.293 → 0.148 秒 | KEEP → KEEP |

输入减少约 88%，速度约提升 8.6 倍。时间包括服务内预处理与评分，不包含客户端、HTTP、
加载及预热。上下文精简是有损变换，不保证预测相同；分类改变不是准确率提高的证据。
原始请求和逐次测量在 `runs/context-suite15-20261029/benchmark/`。

## 15 个任务的当前架构测试

配置 `configs/suite15.yaml`，入口 `scripts/run_task_suite.py`；三个官方划分各五个任务，
任务名、seed 0 和原生 horizon 在开始前写入 plan.json，所有结果保留，不按成功替换任务。
采用 target 场景，三个 worker，共享固定 π₀.₅ checkpoint 59999 与 Decision 服务。

周期检查每 100 步，保留事件与疑似停滞触发；非 KEEP 交 Astra 后台审核。使用低调用配置的
KEEP 门槛 0、概率间隔 0.05，阈值仍未准确率校准。每 Episode 最多 12 次 Astra turn（包含
初始规划/审核/暂停复核）、8 次干预、1,800 秒墙钟预算；控制步数使用各任务原生 horizon。
Code Policy 工具可用，直接 local/eef/edit 关闭；首次尝试不因监督检查而暂停。

本轮 Flora 未启用，单卡三模型共驻尚未验证；现有真实验证器只有 task_success，因此细分
任务步骤由 Astra 记录为推理笔记，不能将本轮称为已验证逐子任务 SWITCH 的完整长序系统。
Memory/技能检索使用 frozen 模式，不晋级候选技能。Seen/Unseen 指官方任务划分，基座模型的
训练暴露尚未完整审计。每任务一条轨迹属于开发测试，不足以估计稳定成功率或 Decision 漏判率。

## 复现入口

在项目目录分别启动模型服务，然后运行评测。输出目录必须是新的，脚本拒绝覆盖已有 Episode：

```bash
.venv/bin/python scripts/serve_from_config.py decision --config configs/suite15.yaml
.venv/bin/python scripts/serve_from_config.py vla --config configs/suite15.yaml
.venv/bin/python scripts/run_task_suite.py --config configs/suite15.yaml --output runs/suite15-new --workers 3 --seed 0 --render
.venv/bin/python scripts/build_suite_report.py runs/suite15-new
```

整批结束后可对每个任务最后一次实际 Decision 检查做同图重放。保留 Decision 服务，运行：

```bash
.venv/bin/python scripts/compare_recorded_contexts.py runs/suite15-new --config configs/suite15.yaml --output runs/suite15-new/context-comparison
.venv/bin/python scripts/build_suite_report.py runs/suite15-new
```

脚本检查观测归属、时间边界和文件校验和，并要求重建摘要与在线发送的摘要完全一致。
最终 Episode 结果仅进入对照报告，不作为模型输入；此比较检查表示敏感性，不提供干预正确性标签。

## 已完成实测与质量限制

15 条轨迹全部完成且均生成带工具时间线的回放。Atomic 为 4/5，Composite-Seen 为 0/5，
Composite-Unseen 为 0/5。373 次 Decision 全部为 KEEP；Astra 各一次初始化，执行中审核和接管均为零。
在线请求中位数 2,172 tokens，客户端总延迟中位数 0.234 秒、P95 0.315 秒。

末次检查同图重放的 15 对输入中，完整上下文原始类别为 13 KEEP、1 REFINE、1 ESCALATE，
应用固定阈值后有 5 次 ESCALATE；精简上下文全部 KEEP。压缩改变了监督行为，速度提升不能视为
判别质量已通过验收。完整上下文也不提供正确性标签，不能把五次差异全部算作漏判。
后续在 CloseToasterOvenDoor 上优先增加多时间尺度和多视角证据，并标注、校准审核触发。
详见 [单任务开发方案](close-toaster-development.md) 与 [完整报告](../runs/context-suite15-20261029/README.md)。
