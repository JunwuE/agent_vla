# 发布前 README 历史快照

本页保留旧文档，不代表最新默认配置和实验状态。当前入口见 [项目 README](../../README.md)。

# VLA Agent v0.1

默认由 **π₀.₅** 生成动作，**Decision 3.0** 监督进展，**Astra** 按需调整语言、计划或开发/调用恢复程序。默认关闭 Astra 直接 local/eef/edit 干预，细粒度控制由 Code Policy 使用 Harness 工具完成。原始观测、控制命令、实测轨迹和验证证据分别记录。

已接入共享的 cjy RoboCasa365 仿真、用户 OmniSF 原子任务 checkpoint `59999`、本地 `d3-flash` 和已登录的 Codex/Astra。当前处于真实链路联调阶段；短步数集成检查不代表完整任务成功率。

本轮子任务级接管、阶段工具、RGB-D、双层 Memory 和有限自修改的实现说明见 [框架更新](../../docs/framework-v2.md)。

技能复用入口见 [操作目录](../../skills/INDEX.md)：语义名称、不可变版本、运行记录分别管理。
同一操作换当前测量参数直接重跑；修改逻辑才增加版本。历史现场修订稿保留但不占默认检索位置。

现已接入 [RAX 风格的多文件 Code Policy](../../docs/rax-code-policy.md)：模块通过固定版本依赖 import 复用，
整个程序和依赖冻结后在同一 Harness 下执行。两个已有关门程序已迁移，旧日志和源码保留。

[Task Memory](../../docs/task-memory-and-primitives.md) 按机器人、环境和任务名保存阶段指南、条件到技能的建议与证据。
CloseToasterOvenDoor 已有首版指南；运行时固定快照，Astra 可提出后续整合的记忆候选。

Astra 历史输入现默认使用[有长度上限的 Memory 索引](../../docs/memory-compaction.md)，完整经验按需展开。

Decision 的当前十五图输入、详细判据、低运动复核规则及实测见 [Decision 监督设置](../../docs/decision-supervision.md)。

模块关系、执行顺序和后续接入顺序见 [架构说明](../../docs/architecture.md)。
新增操作接口、参数化技能与示例见 [Tools 与 Skills](../../docs/tools-and-skills.md)。
参考 RAX 的程序工作区、修订与阶段反馈见 [Code Policy 探索](../../docs/code-exploration.md)。
当前机器的环境、模型版本、启动命令与验证边界见 [部署说明](../../docs/deployment.md)。
Decision 的数值历史压缩、同图速度对照与 15 任务测试条件见 [上下文精简说明](../../docs/decision-context.md)。
CloseToasterOvenDoor 的单任务开发顺序、输入补充与恢复验证方案见 [单任务开发方案](../../docs/close-toaster-development.md)。
2026-10-29 起的真实多轮实验见 [逐轮结果](../../runs/toaster-iteration-20261029/README.md)；
扩展预算、失败程序积累、固定采样对照和复现方式见 [迭代实验说明](../../docs/iterative-development.md)。
原子任务阶段拆分、Grounding/RGB-D、双层 Memory 与自主代码修改边界见 [阶段探索与自改进方案](../../docs/stage-exploration-and-improvement.md)。

## 已有能力与当前边界

| 能力 | 状态 |
|---|---|
| KEEP / REFINE / SWITCH / REPLAN / ESCALATE | 已实现；非 KEEP 统一交给 Astra |
| VLA 快速循环 + 异步 Decision | 已实现周期、夹爪事件、疑似停滞触发及未审核步数上限 |
| 单一控制权、旧 Chunk 失效、语言边界、回复有效期 | 已实现，有行为测试 |
| π₀.₅ WebSocket 接口 | OmniSF checkpoint 59999 已加载并完成真实推理；检查 checkpoint、统计文件哈希和动作契约 |
| Decision 3.0 本地服务接口 | 独立环境与固定版本权重已安装；真实多视角历史窗口已返回五类概率，阈值尚待校准 |
| Astra 工具循环 | Codex app-server 已完成真实规划和工具调用；Responses API 作为备选保留 |
| Code Policy | Python 子进程、工具代理、预算与独立验证；CloseToasterOvenDoor 已完成一次十版迭代后的真实恢复，属于扩展预算下的单例成功 |
| Code Agent 工作区 | 冻结源码版本、精确补丁、静态检查、阶段失败证据和连续修订；不修改模型参数 |
| Memory / Skills | Episode 状态、运行时冻结技能快照、参数/能力/前置条件校验、候选技能与证据积累 |
| Task Memory | 任务范围精确匹配、带证据的指南、版本归档、上下文去重、仅保存候选的更新工具 |
| EEF / Edit 工具 | Code Policy 可用末端/夹爪/底盘工具；直接 Astra 动作修正需显式开启实验配置 |
| 证据工具 | 原始历史帧/裁图、逐步命令与实测状态、操作说明按需读取、事实/假设笔记 |
| Flora 子任务评估工具 | Astra 按需调用 `evaluate_subtask`，使用当前尝试的历史视频；模型评分不直接触发干预或替代完成验证 |
| 数据导出 | Decision 历史窗口、KEEP 抽样审核、有效恢复动作候选索引 |
| RoboCasa Adapter | 使用 cjy 的独立 Python，已验证初始化、三视角 RGB、实测控制器与单步执行 |
| RGB-D 几何 | 真实 RGB-D 对齐、运动相机外参、OSC 基座变换已做仿真检查；接触精度仍待验证 |
| 真实子任务验证 | 当前 RoboCasa 只接原生 task_success；其他条件返回 unknown，需要后续接视觉或任务专用验证器 |

单次程序成功只产生候选技能。默认只检索 `verified` 技能，首版不会自动晋级技能，也不会训练 Code Agent、Decision 或 VLA。
库内提供三个 `template`（eef_offset / gripper_hold / mobile_nudge），需要显式检索；尚未经真实仿真验证。

## 立即运行离线闭环

本机已安装轻量 Agent 环境 `.venv`：

```bash
cd /home/junwue/code/vla_agent
source .venv/bin/activate
vla-agent run --config configs/demo.yaml --output runs/demo-001
vla-agent inspect runs/demo-001
vla-agent export-decisions runs/demo-001 --output runs/demo-001/decision-windows.jsonl
vla-agent export-candidates runs/demo-001 --output runs/demo-001/action-candidates.jsonl
vla-agent summarize runs
vla-agent skills --include-templates
vla-agent tools
pytest -q
```

输出目录必须尚不存在。Demo 使用明确标记的 mock 感知、策略和 Planner，不调用任何外部模型。场景包含空抓 → REFINE → Code Policy 对齐/抓取/提起 → 观察验证 → SWITCH → 放置。Demo 成功仅表示软件链路通过，不是机器人 benchmark 成绩。

将配置替换为 `configs/demo_exploration.yaml` 可演示“第一版阶段检查失败 → 读取证据 → 补丁修改程序 → 第二版恢复成功”，全部使用真实工作区/执行器与 mock 场景。

重新安装轻量环境：`bash scripts/setup_agent.sh`。所有脚本可通过 `bash` 或相应 Python 运行。

## 真实模型与仿真启动

在三个终端分别执行，工作目录均为 `/home/junwue/code/vla_agent`：

```bash
# 终端 1：OmniSF π₀.₅，自动预热后监听 8010
.venv/bin/python scripts/serve_from_config.py vla

# 终端 2：Decision 3.0，加载固定版本后监听 8020
.venv/bin/python scripts/serve_from_config.py decision

# 终端 3：最多 20 控制步、600 秒、4 次 Codex turn 的真实集成检查
.venv/bin/python scripts/smoke_stack.py --output runs/real-smoke-001
```

π₀.₅ 解释器是 `/home/junwue/envs/pi05/bin/python`，服务进程从 `/home/junwue/code/omnisf` 导入模型和 transforms。原环境中的官方 RoboCasa OpenPI 安装保留；不需要把训练使用的 Pinocchio 等无关依赖装进推理环境。当前服务采用 JAX、quantile normalization、base-first 16D 输入/12D 输出，加载 checkpoint 59999 内配套的归一化统计。

Decision 解释器是 `/home/junwue/envs/decision3/bin/python`，Transformers 5.17.0 与 π₀.₅ 的 4.53.2 隔离。权重 `vllm-sr/d3-flash` 固定到提交 `13dc1cc8ef23caefa765907e1d764751e5629f7d`，缓存位于共享 Hugging Face 目录。复建使用 `bash scripts/setup_decision3.sh`，安装快照见 `environments/decision3-freeze.txt`。

Astra 使用 `/home/junwue/tools/bin/codex app-server --stdio`，复用当前 Codex 登录，不要求额外设置 `OPENAI_API_KEY`。模型为 `gpt-6-astra`，默认与现有模型运行配置的 reasoning effort 统一为 `high`；启动时核对服务返回的模型和推理档位。会话在 Episode 内复用，启用 `thread_max_turns` 的配置定期从当前状态重建线程。初始化先规划；执行中的非 KEEP 默认触发后台只读审核，VLA 继续执行。只有 Astra 根据 Memory 与观察证据请求干预时才暂停，再依据新状态复核。工具参数错误返回给 Astra 修正；过期审核丢弃，未完成审核在任务结束时取消。当前架构、逐类调用耗时和优化顺序见 [运行审计](../../docs/agent-runtime-audit-20261102.md)。

`refine` 只选择注册的原子任务指令，不再允许自由生成详细动作教程。任务的 `subtasks` 与可选 `language_catalog` 定义可用指令；主任务、尝试次数、失败解释和计划留在 Memory，VLA 仍只收到当前有效指令。首次尝试默认继续正常接近/对齐，具体失败由 Astra 判断；全局预算和独立停止仍有效。详见[审核与语言边界](../../docs/architecture.md)。

RoboCasa/robosuite 源码及解释器已经填入 `configs/robocasa.yaml`，仅作为独立仿真进程加载，未修改共享环境。默认读取环境的 `annotation.human.task_description`：当前 `PickPlaceToasterToCounter` 的实际描述是 **Place the toasted item on a plate.**，不能按类名推断目标。

`task_group`（atomic_seen/composite_seen/composite_unseen）与 `scene_split`（pretrain/target）分别校验。当前选用 atomic_seen 进行接入检查，尚未执行 Composite-Seen/Unseen 正式评测。检查和完整 Episode 命令：

```bash
.venv/bin/vla-agent doctor --config configs/robocasa.yaml
.venv/bin/vla-agent run --config configs/robocasa.yaml --output runs/robocasa-001
```

`doctor` 只检查静态配置。当前真实环境只有 `task_success` 验证器；Astra 可制定细分步骤，但执行计划先保留一个可验证的整体子任务。细粒度 SWITCH 和局部恢复成功判断需要补充验证器。RGB-D 几何恢复还需验证真实相机外参、朝向和米制深度；默认保持关闭。

## 动作和控制权

核心采用 EEF-first canonical 格式：

```text
state16 = eef_pos(3), eef_quat_xyzw(4), base_pos(3), base_quat_xyzw(4), gripper_qpos(2)
action12 = eef_delta(3), rotation_delta(3), gripper_close(1), base_torso(4), control_mode(1)
```

动作是原生归一化控制量。OmniSF 适配显式重排 base-first，并将原生夹爪/mode 的 `-1/+1` 转成 Harness 的 `0/1`。当前配置启用与原控制器范围一致的限幅，同时保存限幅前输出与转换后命令。Harness 继续拒绝运动通道超出 `[-1,1]`、夹爪/mode 超出 `[0,1]` 的输入。

Code Policy 的 move_eef 接受 osc_arm_base 中的米制绝对目标，基于独立的 eef_control_pose 和控制器缩放，每步重新转换为 native action。不能使用策略 state16 的末端四元数代替控制 site 位姿。可选实验中的直接 eef/edit 干预遵守同样约束；edit 必须关联当前预览前缀，不能作为真实未来轨迹。默认工具 schema 隐藏直接动作干预及其预览工具，运行时也检查配置。

实际控制权接管、恢复交还和停止会更换 epoch 并清空队列；后台只读审核及其 continue 不触发交接。普通干预请求等待当前 Chunk 完成后重新校验并暂停，危险请求立即处理。暂停确认后若选择 continue，则使用新租约从正常推理边界继续，避免仅观察就丢弃动作或已采样 Chunk。VLA 结果校验 Episode、子任务/语言版本、控制权和观测步。过期 Decision 转为 ESCALATE 并请求审核，不自动暂停；审核再校验 attempt ID、版本及证据时效。语言在下一次 VLA 推理边界生效。

## 恢复、技能和数据

Code Policy 实现 `run(ctx)`，每个版本作为 Python 文件归档。进程通过工具代理访问观测和动作，有独立时间、工具次数、动作步数预算。返回 RECOVERED/SUBTASK_DONE 后还需要 Harness 新观测验证。两种返回值均必须通过完整当前子任务判据，局部恢复不再自动交还 VLA。

Astra 先通过短技能索引与接口/契约视图复用已有程序；需要修改时，使用 `read_program` 按文件或函数读取，`edit_program` 批量编辑，写入回执自动包含语法、入口和固定依赖检查。新过程使用 `write_skill_package`，然后提交不可变 `program_version`；已有技能直接提交 `skill_id/skill_version/parameters`。旧单文件工具保留兼容。checked stage 支持前置条件、完成条件和阶段错误定位；程序自报测量与独立验证分开。`ctx.read_observation` 读取当前或过去的 RGB、米制深度、状态和标定，原始数组通过带哈希的 NPZ 提供。详见 [代码工具与技能复用](../../docs/code-tools-v2.md)。

recovery_steps / recovery_seconds 限制单次运行；recovery_total_steps / recovery_total_seconds 限制同一恢复会话所有尝试，后者包含恢复开始后的模型修订等待。首次选择策略的 Astra 调用由 Planner 和 Episode 总预算约束。

生成代码默认经 Landlock + seccomp 隔离，禁止宿主文件写入、网络和进程创建；隔离失败不执行程序。worker 不继承 API 凭据；超时先撤销动作权限，再清理进程。`trusted` 模式不提供 OS 隔离，详情及测试边界见 [框架更新](../../docs/framework-v2.md)。

`memory_mode: frozen` 固定 Episode 开始时的技能快照；`accumulate` 将实际执行过的成功及失败程序和证据加入候选库。当前没有 verified 技能。失败候选只供显式检索、分析和修订，不是成功动作示范。快照同时记录代码与 manifest 哈希；技能组织格式见 [skills/README.md](../../skills/README.md)，逐阶段事实与原因假设分开记录。

每个 Episode 目录包含：

```text
manifest.json                 配置及实现源文件哈希
events.jsonl                  观测、提议、判断、干预、命令、执行、验证、用量
observations/0000000.npz       RGB、原始米制深度、机器人状态（无 pickle）
observations/0000000.json      时间、相机参数、文件哈希、来源
recovery/<attempt>/policy.py   每次实际运行的程序
recovery/<attempt>/stderr.log  程序诊断
recovery/<attempt>/observations/  程序按需读取的原始数组副本及元数据
workspaces/<session>/versions/   冻结代码、父版本、修改假设、预期效果与 diff
workspaces/<session>/attempts/   逐次运行的阶段检查、错误、预算与结果
image_previews/<sha256>.png    实际发送给 Astra 的历史图像或裁图
result.json                   成功、终止原因、步数、调用次数、耗时
```

`export-decisions` 导出当时可见窗口、原始预测、待标注字段，按确定性随机种子抽样审核 KEEP。事后结果不会进入在线输入。`export-candidates` 仅索引独立验证通过的干预动作；失败尝试、未执行代码和纯文字建议不成为候选。候选仍标记 `training_ready=false`，需人工/离线片段筛选、动作频率和目标 VLA 表示转换。

`summarize` 按环境、任务组、场景 split、Memory 模式、baseline/supervised 及配置/技能快照签名分组。`supervision_enabled: false` 可跑纯 VLA；`allowed_interventions` 可做语言监督和动作/代码恢复的消融。正式评测还需建立固定任务清单、初始场景快照和 Decision 人工标签集。

## 三个原子任务与执行回放

首轮已完成 15 条真实 RoboCasa Episode，保留所有成功与失败记录，并为三个任务各生成低调用、原始监督、纯 VLA 的 seed 0 回放。页面可按实际工具/接口事件跳转，查看三视角画面、控制来源、Decision 和工具返回。

- [交互报告与视频](../../runs/atomic-pilot-20261028/report/index.html)
- [实验结果、低调用配置与复现命令](../../docs/atomic-pilot.md)
- [低调用实验配置](../../configs/atomic_low_calls.yaml)

低调用对照将三条轨迹的 Astra turns 从 30 降到 5，搬运和水龙头成功，抽屉仍未完成。该配置尚未经过 Decision 准确率校准，保留为独立实验，不替换原始配置。

## 参考来源

- [原有 gpt_as_policy](../gpt_as_policy/hybrid_rollout/robocasa365/README.md)：独立模拟器、局部纠正与逐步记录。
- [Harness-VLA / RPent](https://rpent.readthedocs.io/en/latest/rst_source/development/architecture.html)：模块边界和统一工具执行。
- [RAX](https://github.com/robot-agent-exploration/rax)、[技能库](https://github.com/robot-agent-exploration/skills-lib)：程序、前置条件和验证证据一起版本化。
- [RoboCasa OpenPI](https://github.com/robocasa-benchmark/openpi)：模型和 WebSocket 协议。
- [Decision 3.0](https://huggingface.co/vllm-sr/d3-flash)：多模态概率判别接口。
- [Codex App Server](https://developers.openai.com/codex/app-server)：会话、动态工具和事件协议。
- [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling)：Astra Responses 工具循环。
