# 当前架构与实现范围

截至 2026-11-10，代码的当前入口说明。历史实验配置保持原样；具体行为由每轮冻结配置决定。

## 执行闭环

1. Environment Adapter 初始化任务，返回原生指令、RGB-D、相机标定和机器人实测状态。
2. 监督配置可启用一次 Astra 初始规划；纯 VLA 对照不调用规划器。
3. π₀.₅ 生成 Action Chunk；Harness 检查动作和控制权，执行配置指定的前缀。OpenDrawer 实验为每轮 5 步。
4. Flora、规则检查和 Astra 只读审核在后台进行，VLA 继续执行。
5. 有效干预请求到达 Chunk 边界后暂停、刷新观测，由 Astra 重新确认。
6. 需要代码恢复时，Astra 检索技能、检查契约、导入工作区并按反馈修改。程序使用唯一 Harness 动作入口。
7. 程序自报结果与独立验证分开；完整当前子任务通过核验后才释放 Code Policy 分配。

后台请求绑定 Episode、子任务/语言版本、attempt、控制 epoch 和观测时刻。
过期请求不能覆盖新状态。异常停止独立于模型回复；执行不确定时不盲目重发动作。

## 主要模块

| 模块 | 文件 | 职责 |
|---|---|---|
| 契约与配置 | `contracts.py`, `config.py` | 统一输入输出、预算和配置校验 |
| Runtime | `runtime.py`, `code_session.py` | 生命周期、执行器分配、干预与交还 |
| Harness | `harness.py` | 单一动作控制权、Chunk 队列和独立验证 |
| 监督 | `supervision.py`, `supervision_rules.py`, `flora_supervision.py` | 只读审核、低运动证据、Flora 路由 |
| Astra | `codex_planner.py`, `codex_transport.py`, `planner_service.py` | Codex 会话、预算、进程间并发门控和服务错误分类 |
| 工具 | `toolkit.py`, `code_tools.py` | schema、参数验证、程序编辑和证据读取 |
| Code Policy | `recovery.py`, `recovery_worker.py`, `workspace.py`, `skill_packages.py`, `stages.py` | 冻结程序与依赖、隔离执行、阶段证据和版本迭代 |
| Memory | `memory.py`, `task_memory.py`, `experience.py`, `memory_context.py`, `working_context.py` | 当前状态、任务经验、历史检索与上下文压缩 |
| 感知 | `perception.py`, `grounding.py`, `geometry_fitting.py` | 对齐 RGB-D、三维测量、当前目标区域和几何拟合 |
| 数据 | `data.py`, `visualization.py`, `scripts/` | 原始记录、轨迹导出、审计、指标和回放 |

文件位于 `src/vla_agent/`，脚本位于仓库根 `scripts/`。

## 模型与判断边界

- VLA：π₀.₅，最近实验使用 OmniSF atomic 权重。语言保持注册原子指令，详细阶段和失败解释存在 Memory。
- 进展评估：最近 OpenDrawer 配置使用 Flora Outcome，每 50 控制步触发；Decision 3.0 为兼容可选后端。
- Astra：负责规划、审核与代码开发，实验配置采用 `high`。没有与其他 reasoning 模型的独立配对比较。
- 独立验证：OpenDrawer 最终结果来自环境 `task_success`。部分复合任务配置有预先安装的局部仿真验证器，不能称为纯视觉判断。
- 阶段证据：`assess_phase` 记录阶段信念，Flora `evaluate_subtask` 提供模型评分，均不是最终成功证明。

Flora 对 20 条纯 VLA 完整轨迹的固定阈值分类准确率为 70%–75%。这是该任务、输入和阈值下的终局分类结果，
不是在线失败检测准确率，也不能证明该模型普遍不适合失败判断。

## Memory 与技能

Working Memory 保存当前任务、阶段、语言、尝试、工具结果和控制来源；Task Memory 保存精确任务范围内的经验。
跨 Episode 经验与版本化技能按需检索，原始日志不持续堆入模型上下文。
OpenDrawer 配置采用历史摘要 4000 字符、工作摘要 8000 字符，线程每 4 个 turn 可重建；调用与 token 总预算不重置。

技能以任务语义名检索，以版本哈希定位源码与契约。`main(ctx, **params)` 组合固定版本依赖；旧 `run(ctx)` 接口兼容。
同一程序换参数不会生成新源码版本。新程序先保留为 candidate，失败记录也保留。

R3 增加共享 Astra 请求门控与服务错误分类；R4 修复未获得模型槽位的审核消耗审核次数的问题。
只有确认没有实际发起模型请求的 admission timeout 才退还审核预算，真实模型调用不退款。

## 尚未实现或尚未验证的内容

- Astra 自动生成、注册并持续迭代任务视觉判断程序，目前仍是设计方向。
- 未训练 Code Agent、Decision 或 VLA；当前积累的是程序、任务经验和候选动作数据。
- 没有证明当前 Agent 提速：R3 平均耗时高于纯 VLA；成功提升仅出现在扩展步数预算。
- 当前 atomic 权重的实验不能用于声称复合长程任务能力已经得到验证。
- 模板、候选技能和单次布局成功不能作为通用抓取/规划能力的保证。
- 宿主隔离与动作限幅不等于具有 IK、碰撞规划或真实机器人安全认证。

后续任务判断程序应单独版本化、读取只读观测、返回证据及 UNKNOWN，并保持正式任务成功标准固定。
在旧轨迹旁路校验及新 seed 验证后，再考虑赋予其在线路由权限。
