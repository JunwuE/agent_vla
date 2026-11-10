# 三个原子任务与低调用监督实验

本文是 2026-10-28 旧实现的历史结果。2026-10-29 已收紧语言指令并加入后台只读审核，当前配置不再复现旧版同步暂停/自由改写的行为；旧 Episode 的 manifest、日志与视频保持原样。这里的成功率和调用次数不代表新版效果。

2026-10-28 在真实 RoboCasa365 仿真中完成 15 条 Episode：三个任务各 3 条纯 VLA 基线、1 条原始监督、1 条低调用监督。模型使用 OmniSF π₀.₅ checkpoint 59999、固定版本 d3-flash、Codex gpt-6-astra / xhigh。不是 Composite-Seen/Unseen 正式评测。

所有运行位于 `runs/atomic-pilot-20261028`。默认展示每个任务的 seed 0，在看到结果前固定，不按成功挑选。初始机器人状态与三相机 RGB 内容按数组计算 SHA256 核对；策略服务器共享随机采样流，不保证动作逐位可复现。单任务 horizon 使用 RoboCasa 原生预算，控制步长 20 Hz，每个 VLA chunk 执行前 5 步。

抽屉、搬运三种模式的初始数组完全一致。水龙头低调用运行存在初始差异：状态一项相差约 6.82×10⁻¹³，左右 RGB 分别有 851 / 1220 个通道值不同（各占图像通道值的 0.43% / 0.62%），腕部 RGB 完全一致。严格哈希比较为不相等，完整差异在报告 JSON 中，不宣称该条完全相同初始观测。

| 任务 | 纯 VLA seeds 0/1/2 | 原始监督 seed 0 | 低调用监督 seed 0 | Astra turns 原始→低调用 |
| --- | --- | --- | --- | --- |
| OpenDrawer | 成功 / 未完成 / 成功 | 未完成，550 步主动停止 | 未完成，750 步 horizon | 11 → 2 |
| PickPlaceCounterToStove | 全部成功 | 未完成，385 步主动停止 | 成功，268 步 | 8 → 2 |
| TurnOnSinkFaucet | 全部成功 | 成功，572 步 | 成功，274 步 | 11 → 1 |

每个 turn 包含一次主机发起的 Astra 审核，可有多个动态工具调用；不等于 Codex 内部的模型请求数。次数包含开场规划。原始监督动态工具共 58 次：`record_note` 23 次、`submit_intervention` 30 次、`read_guide` 3 次、`capabilities` 2 次。低调用共 10 次：`record_note` 与 `submit_intervention` 各 5 次。读取指南和能力列表没有执行机器人动作。

三任务合计 Astra turns 从 30 降为 5（减少 83.3%），Codex 报告的输入 token 从 4,506,837 降为 153,813，输出从 18,857 降为 1,352。单 Episode 耗时之和从 1662.0 秒降为 630.8 秒；三任务并行运行，该和不是整个批次墙钟耗时。没有换算为 API 费用。这些是三个样本的观察，不能归因于某一项修改，也不能推断总体成功率提升。

## 为什么旧配置调用多

对 9 条基线抽取 180 个历史窗口，Decision 原始最高分类别为 KEEP 的有 151 个，但原始 `keep_threshold: 0.8`、`min_margin: 0.1` 将它们全部转为 ESCALATE。最终路由为 171 ESCALATE、9 SWITCH、0 KEEP。这并不表示机器人失败了 171 次。

`decision-shadow/predictions.jsonl` 保留原始概率、运行时路由、当时可见窗口及待填写的独立标签。事后 Episode 结果单独保存，不进入在线输入。`threshold-sensitivity.json` 只比较路由敏感性，不是准确率评估。尚无独立人工标签，不能报告漏判率和误判率。

## 低调用配置

独立配置是 [`configs/atomic_low_calls.yaml`](../configs/atomic_low_calls.yaml)，原始 `robocasa.yaml` 保留。试验同时调整以下项目：

- 周期检查从 50 改为 100 控制步；事件触发和异步观测过期校验保留。
- `keep_threshold: 0.0` 取消未经校准的绝对 KEEP 门槛；`min_margin: 0.05` 仍在最高两类接近时转为 ESCALATE。模型输出不合法或超时仍转为 ESCALATE。
- `max_calls / max_interventions: 4` 限制每条轨迹最多四次主机审核；遇到新的非 KEEP 而预算用尽，记录 `astra_budget` 并停止，不擅自放行给 VLA。这三条实际都未触发预算停止。
- `preserve_initial_instruction: true`：只有 `task_success` 验证器时，Astra 开场保留一个整体可执行子任务，指令必须与环境原生任务语言完全一致。细分推理保存在笔记，必要时才调整后续指导。
- `compact_context: true`：模型输入保留最近 6 条动作、最多 2 个历史观测时刻、6 条笔记、4 条干预；附带较长窗口的实测末端位移摘要。完整观测、命令、实测轨迹和失败证据仍在原始日志，可通过工具读取。这不是对 Codex 会话历史的语义压缩。

`PickPlaceCounterToStove` 在 seed 0 的真实指令是 **Pick the cream cheese stick from the plate and place it in the pan.**，不能从类名推断成移动整个锅。Astra 负责计划，保持训练时任务语言是执行接口约束。

抽屉低调用轨迹在第 100 步触发一次审核，Astra 建议继续；后续六次 Decision 为 KEEP，最终 horizon 时任务仍未完成。当前证据能确认没有及时升级，不能仅凭终局失败就把所有早期 KEEP 标为错误。下一步应标注失败前的可见证据，补充任务相关的停滞检测和完成验证，再校准路由。不能靠持续提高 Astra 调用次数替代这个过程。

## 运行与可视化

先按 README 启动 VLA 与 Decision 服务，再运行批次。输出模式目录要求不存在，以免覆盖原始数据。

```bash
.venv/bin/python scripts/run_atomic_pilot.py --output runs/new-pilot --mode baseline --seeds 0 1 2
.venv/bin/python scripts/run_atomic_pilot.py --output runs/new-pilot --mode supervised --seeds 0
.venv/bin/python scripts/run_atomic_pilot.py --config configs/atomic_low_calls.yaml --output runs/new-pilot --mode low_calls --seeds 0
.venv/bin/python scripts/build_atomic_report.py runs/new-pilot --render
```

已有报告：[`report/index.html`](../runs/atomic-pilot-20261028/report/index.html)。每个任务有低调用、原始监督、纯 VLA 三条 seed 0 回放，共 9 条，失败轨迹同样展示。HTML 可搜索 tool、点击事件跳转、查看返回参数；MP4 叠加当前模块、接口/工具、阶段、步数、实际耗时、控制权和 Decision。需要在本机浏览时，可直接打开下载后的 HTML，或启动静态服务。

```bash
python -m http.server 8030 --bind 127.0.0.1 --directory runs/atomic-pilot-20261028/report
```

渲染参考 `gpt_as_policy/hybrid_rollout/robocasa365/robocasa_server/session.py` 的逐步三视角记录，并采用相邻 robodojo 回放脚本的冻结记录、来源哈希做法。它只读取已录制证据，不重跑模型和物理环境。动作按 20 Hz 展示，推理等待压缩，原始墙钟时间另列。

回放区分 `Pi05.infer`、`Decision3.decide`、Codex turn、动态工具请求/返回、Harness 控制权与执行、完成验证。系统接口与 Astra 调用的工具分别展示；提案接受不等于动作执行。阶段是运行日志里的阶段，不是人工标注的“抓取成功”等语义标签。

这两轮监督只启用语言/计划调整，没有运行真实 Code Policy 恢复或直接动作纠正。几何恢复、细粒度子任务验证、Decision 标注校准仍需后续完成。
