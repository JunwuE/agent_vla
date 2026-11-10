# 历史 Memory 的索引与按需展开

2026-11-01。本次改变 Astra 输入的历史记录呈现方式，保留原始经验、代码、图像、深度和证据文件。

参考 [RPent Memory Design](https://github.com/RLinf/RPent/blob/main/docs/source-en/rst_source/development/memory.rst)
的 global / task-family / task-specific 分层、按需读取与评测冻结，以及本地
[RAX experience-memory](../../rax/agent_skills/task/experience-memory/SKILL.md)
的 INDEX → 匹配技能 → 必要时读取证据。本次先落实索引和读取边界，没有把任务实例自动晋升为通用技能，
也没有迁移全部库目录或新增未经验证的跨任务规律。

## 实际变化

- `planner.historical_memory_mode: summary` 为自动输入生成抽取式历史卡片；`full` 在自动输入中保留完整历史。
  两种模式的 `search_experiences` 均返回索引，`read_experience` 提供完整记录。
- `historical_memory_chars: 10000` 限制自动注入的历史字段总字符数，不是 token 上限，也不限制当前视觉/状态或完整工具结果。
- 每条经验保留精确 Episode ID、证据哈希、任务结果、代码尝试计数和可展开的技能引用。
  只纳入长度容许的完整事实/假设条目，二者分栏；不截断句子，不生成新解释。
- Task Memory 自动输入为任务简介、阶段索引、规则索引。条件和禁用情形不在索引中，明确要求采纳恢复规则前用 `read_task_memory` 展开。
- `search_experiences` 返回简短卡片；`read_experience`、`read_task_memory` 和 `read_skill` 保留完整读取能力。
- 原有同线程去重继续生效。新线程或上下文压缩后重新提供索引，而非自动灌入全部历史。
- 当前任务指令、动作/观测、尝试状态、阶段信念、控制权、成功验证和新失败证据均不因历史压缩而改写。
- 没有新增总结模型调用。索引只用于发现历史材料；不能据历史成功跳过当前几何与入口检查。

本次不压缩显式请求的完整证据；因此 Astra 若主动读取大量历史，仍会增加上下文和用时。
后续可按阶段继续细分读取接口，但需要验证不会丢失前置条件和反例。

## 验证

对上一轮 50 条首次规划输入执行同输入离线变换：

| 字符统计均值 | 原输入 | 新输入 |
|---|---:|---:|
| 自动注入的历史字段 | 52,619 | 5,937 |
| 整条用户上下文 | 57,668.56 | 11,304.56 |

图像不变；工具定义和固定指令不计入此表。字符数不等于 token 数，离线压缩不证明模型延迟或任务成功率改善。
详见 [对照 JSON](../runs/composite-memory-20261101/memory-context-comparison.json)。

新增测试覆盖：原始记录不变、成功/失败和假设身份保留、长度上限、整条省略而非截断、Codex 实际输入与完整读取链路。

## 本次长程评测

预先固定 SteamInMicrowave（Composite-Seen）和 WaffleReheat（Composite-Unseen），各 seed 0–19。
使用同一 OmniSF atomic checkpoint 59999；官方类别不是该 checkpoint 训练覆盖的独立审计。
官方动作预算分别为 2100 / 4050，不设 Episode 墙钟超时。
Flora 每 50 个 VLA 步监督，Astra high，原有 VLA/Code Policy 控制权规则保留；
为长程任务统一采用 40 次后台审核、100 次 Astra turn 上限。

每条使用相同起始 Memory 和冻结源码，新记录留在私有库，批次结束前不传给其他 seed。
两个新任务在现有库中没有专属经验。只提供 task_success 原生验证器，细阶段仍是 Astra 信念；
初始化保留原生整任务 instruction，并不假装已经实现独立验证的原子指令切换。
这是当前系统在两个复合任务上的评测，不是 Memory 压缩效果或 Agent 相对纯 VLA 增益的因果消融。

结果入口：[本批目录](../runs/composite-memory-20261101/results.md)。
