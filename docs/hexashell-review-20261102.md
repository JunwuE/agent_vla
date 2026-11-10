# 指令、完成判断与经验开销：HexaShell 对照设计

日期：2026-11-02。本文是设计建议与源码核查记录，尚未改变正在运行的冻结实验。
HexaShell 已克隆至 `/home/junwue/code/HexaShell`，核查 commit：`d1a2a24dc4c528a652b359fd689398530ef941d8`。

## 建议

1. VLA 默认使用贴近 RoboCasa365 原子任务的目标指令；主任务继续提供给 Astra 和完成判别器。主任务与子任务拼接作为独立消融，不直接成为默认。
2. 独立定义 CompletionChecker，只判断当前目标已完成、未完成或证据不足。Decision 3.0 和 Flora 都可以作为候选后端，先对照再选择；无需默认双模型同时常驻、同时判断同一件事。
3. 在线 Astra 默认选择工具和填写当前观测参数，只读取接口与简短任务经验。源码和完整失败历史主要给修订模式读取。
4. 将代码开发与正式执行区分：开发允许修订，固定版本评测以调用技能为主；相同模型可承担两种角色，无需训练 Code Agent。

## 原子语言与进入状态

本地 OmniSF 配置 `src/openpi/training/misc/robocasa365_config.py` 使用 `prompt_from_task=True`；
`data_loader.py` 将数据集 task 文本转成 prompt。没有在这条训练管线发现主任务与当前子任务的自动拼接。
这是训练管线证据，不代表已经审计 checkpoint 的全部预训练语料。

| 本地原子任务 | 原生指令模板/本任务实例化建议 | 当前情况 |
|---|---|---|
| OpenMicrowave | `Open the microwave door.` | 当前已经对齐 |
| PickPlaceCounterToMicrowave | 原生模板 `Pick the {obj_lang} from the counter and place it in the microwave.`；本任务可实例化为 `Pick the bowl from the counter and place it in the microwave.` | 当前是 `Place the bowl with waffle inside the microwave.`，建议测试原子句式 |
| CloseMicrowave | `Close the microwave door.` | 当前已经对齐 |
| TurnOnMicrowave | `Press the start button on the microwave.` | 当前已经对齐 |

来源为本地 RoboCasa365 的 `atomic/kitchen_doors.py`、`atomic/kitchen_pick_place.py` 和 `atomic/kitchen_microwave.py`。
多碗场景应保留必要的目标区分；华夫饼必须仍在所选碗中，不能将目标改成只拿华夫饼。上述碗指令是按原生模板实例化，不声称训练集中存在完全相同的实例。

特别需要保留的差异：原子 CounterToMicrowave 默认操作 food，初始场景预先打开微波炉；WaffleReheat 要搬带有食物的碗，而且上一步可能让夹爪留在把手附近。句式对齐不能替代几何状态、夹持对象和交接状态的对齐。

拼接实验建议比较：A = 原子子任务；B = 简短主目标 + 相同原子子任务。固定 seed、起始物理状态、动作预算、模型和完成条件，记录子任务成功、提前执行其他步骤、交接回退和总动作数。不要在 B 中同时修改执行器或完成条件。

## CompletionChecker 与 Flora

CompletionChecker 的职责建议限定为：

```text
输入：当前目标、目标身份、明确完成条件、截至当前的短视频/多视角图像
输出：DONE / NOT_DONE / UNCERTAIN，以及对应选项分数和观测绑定
```

- NOT_DONE 只表示尚未完成，不自动要求恢复代码。
- DONE 是完成候选，先交 Astra 审核，并使用新观测检查下一执行器的进入条件。
- UNCERTAIN 可以补观测；不能把每次不确定都变成打断或无限审核循环。
- 回复绑定 Episode、语义目标、目标版本和观测步；过期或目标变化的回复丢弃。
- 完成与交接就绪分开：门已开，并不证明夹爪已脱离把手、适合开始搬碗。

Decision 3.0 支持多模态选项分类和概率输出，适合实现这个窄接口；它不是已验证的机器人完成检测器，概率也不能直接视为本任务上校准过的置信度。
[Decision 3.0 模型说明](https://huggingface.co/vllm-sr/d3-flash)

Flora 同时提供 Outcome、Dense 和 Rubric 任务；当前项目服务只接入 Outcome。Outcome 为 1–5 分，5 表示模型判断完全成功，因此 Flora 本身也能成为完成检测候选；不能把它仅当作动作质量模型。Rubric 要另行接入、验证，不能视为现已支持。
[Flora 模型说明](https://huggingface.co/nvidia/Flora-Rewarder)

原五 seed 的已记录 subtask 评分中，25 次 Outcome=5 对应同一快照 `microwave_open=false`，全部是开门目标。当前模型收到的主要目标文本是“打开门”，与严格 native 门状态阈值并不完全等价。这是判据不一致的证据，不是一个无偏误判率，也不能推出 Decision 一定更准。

离线比较应使用相同观测、相同目标和明确条件，对照 Decision 三类输出与 Flora Outcome/Rubric。包含完成、接近完成、完成后回退、遮挡和正常未完成样本；按 Episode/场景划分，不把重叠窗口随机拆进训练和测试。评估误报完成、漏报完成、检测延迟、额外 Astra 调用与实际耗时。

仿真 native checks 可作单独评测标签，不得混入被评测模型的图像摘要/JSON；否则无法测量视觉完成判断。正式 Agent 如继续使用 native verifier，仍应标明 verifier-assisted。

当前 H100 80GB 的检查结果为已用 70,208 MiB、空闲 10,799 MiB。恢复额外判别模型前需要安排模型驻留，不能假定 π0.5、Flora 与额外 BF16 Decision 可以无代价同时加载；离线对照可错峰进行。

## 减少每轮经验读写

以下是建议的初始工程预算，需要根据信息遗漏和性能评估调整，不是已验证最优阈值：

| 内容 | 默认读取 | 更新时机 |
|---|---|---|
| Task card | 当前阶段、目标身份、进入/完成/交接条件、未解决问题；约 800–1,200 token | 任务开始、阶段切换或证据改变 |
| 技能目录 | 当前目标相关的最多 3 项，每项短接口、适用条件、版本与证据引用 | 目标改变或当前技能不适用 |
| 近期失败 | 最近相关的 1–2 条，按语义目标/失败类型/技能版本去重 | 新失败类型或结论改变 |
| 工具反馈 | 本次阶段、实际误差、是否执行、停止原因、证据引用 | 每次调用，重复字段省略 |
| 长期经验 | 执行期间只读固定快照 | Episode 后归纳、验证，再发布新快照 |
| 原始 RGB-D、动作、源码、trace | 自动归档，按需定位读取 | 全量证据继续记录，不反复塞入模型 |

工具层要实际改变返回结构：

- `search_skills` 默认只返回简短适用性索引；完整契约在选定后读取。
- `list_skill_versions` 分页/过滤，只给版本差异和验证摘要；旧版本保留在归档。
- `read_program(path)` 返回所选模块、版本和 hash；文档、契约和参数 schema 用独立引用。契约变化时必须重新展开，不能用旧参数约束执行新代码。
- 当前工作状态用稳定 goal_id 保存，重规划不会抹去同一目标的交接失败。
- `record_note` 写事实/假设的增量，不反复复制历史；同一问题更新一条活跃记录，证据仍可追溯。
- 一次成功只增加运行证据，不自动生成新源代码版本；代码/契约/依赖未改变时复用原版本。

本轮审计显示 seed 3 的 91 次 `read_program` 中 90 次已经选了模块，但重复文档仍占约 170 万字符。单靠提示模型少读代码无法解决这一接口问题。

## HexaShell 能借鉴什么

它的工具契约明确将 `interface.md` 给执行 Agent，而不展示 `tool.py`。工具通过 `run(api, command, args)` 调用测量和运动接口，按任务启用相关工具。我们可以借鉴这种读取边界，并保留现有 Harness 的控制权、观测绑定与日志。
[EpisodeAPI](https://github.com/HexaFuture/HexaShell/blob/main/interface/EpisodeAPI.md)

例如 `guarded_grasp` 将接近、闭合、提起或放下、释放、撤离组织成带参数和误差检查的操作。它区分运动执行与物体真实抓取，返回的 `grasp_verified` 不把夹爪闭合当作抓取证明。我们应借鉴完整操作和清晰失败阶段，不能直接移植其坐标系、机器人尺寸或参数值。
[guarded_grasp](https://github.com/HexaFuture/HexaShell/blob/main/library/guarded_grasp/interface.md)

论文的开发循环在 Episode 之间修订工具、测试后重试同一布局，最终冻结评测。它使用 cuRobo 逐点 IK、路径插值与时间化；并没有完整场景避障模型。RGB 运行在此前工具基础上开发，每任务预算 12 小时。论文报告复测 Episode 中位数 739 秒，对应机器人运动 26 秒，说明它也受模型延迟影响。对我们最有用的启示是提升测量和执行接口，并将常规执行与代码开发分开。
[论文，§3–4](https://arxiv.org/pdf/2610.23479)

官方冻结系统总体成功率约 46.9%，Long-Horizon 31.5%、Memory 81.0%；不同机器人、任务、布局和计分方式不能直接与我们的 RoboCasa WaffleReheat 相比。总体是任务类型平均，不代表每个任务都高成功。
[官方结果表（项目发布）](https://github.com/HexaFuture/HexaShell/blob/main/results/official_robodojo.md)

仓库发布工具、接口和结果，没有发布完整 server、executors、agent container 和 evolution loop；不能仅克隆仓库就复现整套运行系统。我们已有带尺度 RGB-D，可继续使用，不应因为它采用 RGB 三角测量而删除深度。
[仓库范围](https://github.com/HexaFuture/HexaShell)

## 建议实施顺序

1. 固定原子指令目录和语义目标 ID；保存可复现的开门/搬碗入口，比较子任务单独输入与拼接输入。
2. 先改工具返回和经验增量更新，保持物理控制与成功条件不变，测上下文与耗时。
3. 离线比较完成判别器，再决定在线使用 Decision、Flora 或单模型多种评估模式。
4. 整理开门交接、抓碗和搬运的参数化工具；失败由优化角色针对局部代码修订并回归。
5. 用固定工具版本与相同预算再做长程对照；开发探索成本和冻结执行成本分别报告。
