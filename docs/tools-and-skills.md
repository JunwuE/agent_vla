# Tools 与 Skills

这次参考旧框架的 ControllerAdapter、渐进动作编辑和操作/记录说明，将接口接入本项目的
按需 Astra、统一 Harness 和独立验证流程。不会要求 Astra 审核每个 VLA Chunk。

## Astra 工具

工具参数来自 `toolkit.py` 的 Pydantic 契约，运行 `vla-agent tools` 可查看 JSON Schema。
这些工具读取证据、计算、记笔记或编辑程序；运行中的只读审核通过 submit_review 返回判断，暂停后的干预才通过 submit_intervention 提交给 Runtime。
默认干预仅开放 continue/refine/switch/replan/code/observe/stop。直接 local/eef/edit 及预览工具
需要显式实验配置；Code Policy 内部的末端、夹爪和底盘工具仍可用。

| 工具 | 用途 |
|---|---|
| read_guide | 按需读取 control / recovery / evidence 操作说明 |
| read_task_memory / propose_task_memory | 读取当前任务的固定指南，或保存有证据的新规则候选；不自动改变当前策略 |
| search_experiences / read_experience | 在当前机器人、环境、任务名范围内检索或展开历史 Episode |
| write_skill_package / patch_skill_package | 新建多文件技能包，或只修订指定模块；声明固定版本 uses 依赖 |
| write_program / read_program / patch_program | 在同一恢复会话中保存、读取与修订不可变程序版本 |
| check_program / read_recovery_trace | 不执行代码的静态检查；读取先前运行的阶段证据与错误 |
| observe / capabilities | 当前状态、独立 OSC site 位姿、实际控制器缩放和已安装验证器 |
| view_image | 读取本 Episode 的原图或裁图；验证数组哈希，归档实际发送的 PNG |
| read_execution | 读取 `(start_step,end_step]` 至多 20 步命令、实测状态、跟踪/编辑诊断 |
| evaluate_subtask | 调用本地 Flora，对当前子任务本次尝试的历史视频评估 Outcome；只提供证据 |
| record_note | 分开记录 facts、hypotheses、next_check |
| search_skills / list_skill_versions / read_skill / check_skill | 按语义技能检索，展开版本与起始条件，读取源码/证据，检查声明的调用契约 |
| geometry | 根据同一时刻的米制深度与已验证标定，计算可见表面几何 |
| preview_vla | 暂停期间按需推理，保存最多 5 步当前动作前缀和绑定 ID，不执行 |
| preview_edit | 预览总位姿偏移和夹爪覆盖，报告限幅，不执行 |
| submit_review | 只读审核：继续，或附带观察证据请求干预；本身不执行动作/语言修改 |
| submit_intervention | 暂停后提交干预；refine 选择注册 instruction_id，code 支持冻结 program_version |

`review_running` 模式允许读取指南、冻结观测/图像/执行记录、能力、几何、技能、调用 Flora，以及记录审核笔记和不生效的记忆/策略候选。禁止 VLA 预览、程序写入和干预提交，避免只读审核改变采样状态或动作队列。工具中的“当前”指审核快照时刻；不得读取后续帧。笔记和模型评分先写入快照，只有审核仍有效才合入 Working Memory。

任务记忆的保存、精确匹配和控制 primitive 的分层建议见 [Task Memory 与 primitives](task-memory-and-primitives.md)。

### Flora 子任务评估

`evaluate_subtask({})` 使用 Memory 中的当前子任务和尝试边界，无需 Astra 自行选帧。
从本次尝试开始到审核快照为止，按仿真时间均匀采样至多 64 帧，默认拼接 left / wrist
两个视角；给模型的输入是视频和子任务 instruction，不包含动作数组、仿真成功标签或未来帧。
使用官方 Outcome 提示词，关闭 rationale，返回 1–5 分及片段边界、模型版本、请求 ID。

它适用于 Decision 提示 SWITCH 或 Astra 对完成状态不确定时的补充判断，不是每个 Chunk 的
固定步骤。低分也可能表示任务尚在正常进行；高分不等于已校准的成功概率。
评分不直接暂停 VLA、切换子任务或成为正向训练标签，仍需 Astra 结合其他证据和已有验证器。
真实细分子任务的成功验证器尚需单独接入。

同一快照重复调用命中缓存，每 Episode 默认最多 6 次，单次含准备视频的预算为 30 秒。
缺帧、相机缺失、非均匀时间戳、片段超出 120 秒、服务异常或无效分数返回 unknown/unavailable，
不将其解释为失败，也不静默截掉片段开头。原始请求和响应保存在 `flora/`，事件日志及可视化
标注 `evaluate_subtask`，Working Memory 仅保留最近 8 条评分摘要。

`refine` 示例：`{"kind":"refine","instruction_id":"pick","reason":"恢复原有原子任务指令"}`。`pick` 必须已注册在 Memory 的 language_catalog；也可提供完全匹配的 `instruction` 文本。改目标/验证器需 replan，所有新计划仍使用注册指令。更详细的运动说明会被拒绝并反馈给 Astra。

无效参数、缺少能力、未知完成条件或过期提议会返回错误，Astra 可在工具轮次预算内修正。
no_execution=true 表示这个被拒绝的 Planner 调用没有执行动作。机器人执行回执不确定时仍然
停止，不会用这个机制重放物理动作。请求参数、工具输出和发送图像的文件/哈希可追溯。

## EEF 和 Edit

本节的直接干预是可选实验能力，默认关闭。Code Policy 调用底层 EEF 工具遵守相同坐标约定。

EEF 目标的坐标系是 osc_arm_base，位置单位米，旋转为 EEF site 的 xyzw 单位四元数。
策略 state16 中的姿态不能替代该控制位姿。RoboCasa Adapter 直接读机器人控制 site 和机械臂
参考系，不读物体真值；控制器结构不匹配时不开放 OSC 工具。

EEF 干预字段为 kind=eef、reason、steps、completion_condition 和 target。target 包含：

```json
{
  "frame": "osc_arm_base",
  "position": [0.01, 0.0, 0.1],
  "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
  "gripper_closed": false,
  "base_motion": [0.0, 0.0, 0.0, 0.0],
  "control_mode": "arm"
}
```

坐标只演示格式，实际目标必须由当前观测计算；完成条件必须已有独立验证器。
默认目标位移范数上限 5 cm、旋转上限 0.35 rad、移动输入幅度上限 0.25。每步根据实际位置/
旋转重新计算误差、反向缩放为 native action，记录限幅与跟踪误差。Arm 模式禁止移动通道；
Base 模式复制当前 EEF 位姿，维持控制器 desired arm goal。

Edit 先调用 preview_vla，提交其 proposal_id 和不超过已预览长度的 steps。位置总偏移均分到
各步；旋转按 `E_i D_i E_(i-1)^-1` 组合，不能把旋转向量直接相加。底盘、躯干和 mode 保留，
夹爪可 keep/open/closed。零偏移/keep 保留学生命令，不能理解为让机器人静止。
控制权、观测步、语言或子任务版本改变后，旧 proposal_id 失效。

## Code Policy 工具

ctx.act(action12) 执行一个原生步。新增动作工具均为 1..5 步，实际每一步都经过 Harness，
计入恢复时间/动作预算及 Episode 总预算：

```python
ctx.move_eef(target, steps=2)
ctx.move_relative(delta_position, gripper_closed, delta_rotation_vector=[0, 0, 0], steps=2)
ctx.set_gripper(closed, steps=1)
ctx.move_base(motion, gripper_closed, steps=1)
```

move_relative 从当前实测 OSC 位姿构造目标，不累加过时的目标。set_gripper 保持实测末端位姿；
闭合只表示发送夹爪命令。move_base 的 [x,y,yaw,torso] 是归一化控制量，不是米制位移。
这些工具没有实现 IK、碰撞检测或自动抓取点生成。

读和记录工具为 observe、capabilities、geometry、check、stage、note；stop 立即撤销执行权并
结束本次恢复。所有动作通过 Harness，默认 worker 使用 Landlock + seccomp 限制宿主访问。
新增阶段、Grounding、Flora 和策略候选接口见 [框架更新](framework-v2.md)。

程序还可以使用 ctx.read_observation(camera, step=None) 读取真实 RGB/深度数组和标定元数据，
并用 with ctx.stage(...) 声明独立的阶段验证。开发工具、版本和阶段示例见 [Code Policy 探索](code-exploration.md)。

## 技能调用

实际操作目录见 [skills/INDEX.md](../skills/INDEX.md)。优先选择适用版本并传入当前测量参数。
新开发使用 [RAX 技能包](rax-code-policy.md)：`main(ctx, **params)` 与可组合 Python 模块。
`read_skill` / `read_program` 的 `path` 参数用于只读取一个文件；`import_skill_program` 保留整个包和依赖。
参数变化只产生运行记录，不产生新程序；真正的逻辑修改沿用同一技能名生成新版本。
`write_program` 必填 `skill_name`、`when_to_use`、`reuse_reason`，已有名称需直接调用或导入后修改。
`check_skill` 的空前置条件不证明场景适用，程序仍需检查当前几何与状态。

```json
{
  "kind": "code",
  "reason": "试验短程末端微调",
  "skill_id": "eef_offset",
  "skill_version": "v1",
  "parameters": {
    "delta_position": [0.005, 0.0, 0.0],
    "delta_rotation_vector": [0.0, 0.0, 0.0],
    "gripper_closed": false,
    "steps": 2
  },
  "completion_condition": "完整当前子任务的已安装验证条件"
}
```

完成条件的占位文本需要替换。三个内置技能是 template。要看到模板，显式设置
search_skills(include_templates=true)。模板执行被标记为实验，不能算作已验证技能复用。
单次成功后只记录 candidate，保留参数契约、调用参数和验证证据。详见 [技能格式](../skills/README.md)。

## 当前验证范围

离线测试覆盖实测控制坐标系、短程反馈、旋转组合、限幅、旧提议失效、参数/能力检查、历史
图像和执行证据、恢复子进程预算、独立成功验证，以及模拟 HTTP 的 Astra 参数修正和图片工具。
configs/demo.yaml 的空抓恢复已使用新 Code Policy 动作工具，仍是明确标记的 mock 场景。

真实 RoboCasa 已接入 cjy 环境，并完成控制器、RGB-D 对齐和隔离 Code Policy 两步动作检查。
当前只提供 task_success，未知条件不算成功；阶段判断与完整子任务核验分离。
新测试证据与边界见 [框架更新](framework-v2.md)，短程接口检查不代表任务成功率。
