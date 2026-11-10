# WaffleReheat：可执行子任务与当前上下文压缩

2026-11-02。针对上一轮两个长程任务 0/20 的结果，修改三个执行边界。

## 已实现

1. Adapter 提供注册的原子语言与前置/完成条件；Astra 选择剩余计划，Harness 安装与核验。
   `set_phase_plan` 继续表示诊断阶段，`replan` 才会改变实际执行子任务。
2. VLA 每次只接收当前原子指令。Chunk 边界发现局部完成后交给 Astra 审核，`switch` 再核验并推进。
   Code Policy 负责当前已安装的完整局部目标，完成后由 Astra 为下一目标选择执行器；旧动作不得继续。
3. 当前 notes、reward、reviews、attempts、干预与恢复反馈使用抽取式索引，完整快照在 `context_archive`。
   `read_context(reference, path, start, limit)` 可分页展开，校验哈希、Episode 和历史观测边界。
   新旧程序身份保留，最新契约完整传入，旧契约通过 `read_program` 或归档展开。
   每 12 turn 重建 Codex 线程，保留权威任务状态、工作区、未解决问题和按需证据。

历史经验仍使用前一版索引。记录预算不是整个上下文 token 上限：控制器、当前任务、视觉、工具定义、
完整最新契约与显式读取结果不被任意截断。原始失败证据不会删除，也不新增总结模型。

## Waffle 可用的局部契约

| 任务级指令 | 进入条件 | 完成条件 |
|---|---|---|
| Open the microwave door. | 无额外条件 | 原生 microwave.is_open，归一化门关节位置 ≥ 0.90 |
| Place the bowl with waffle inside the microwave. | 门已打开、华夫饼在碗中 | 碗在炉内、华夫饼在碗中、夹爪离碗超过原生默认 0.25 m |
| Close the microwave door. | 上述放置条件成立 | 放置条件仍成立且原生 microwave.is_closed |
| Press the start button on the microwave. | 上述关门条件成立 | 原生整任务 task_success |

这些是可选能力，不是写死的执行顺序；Astra 必须提供计划，并在实际进入前核对条件。
任意局部计划的末项保留 `task_success`，本轮最终成功仍按 RoboCasa 原生信息记录。
进入条件不当作全程不变量，例如持碗过程中不要求夹爪一直远离碗。

**信息边界：本轮局部验证器使用仿真真值布尔状态，不是纯视觉判别。** 不向 VLA/代码提供物体真值坐标、
物体关节角或未观测抓取位姿；机器人自身状态和运动学仍可读取。不修改共享 RoboCasa 源码。
该设置验证执行架构是否接通；纯视觉/Flora 条件需另测。开门阈值不是 90 度，也没有证明是放碗所需的最小开度。

旧整任务程序可通过 `import_skill_program → adapt_program_goal → patch_skill_package` 改为局部目标。
目标适配形成新候选版本，保留原技能与失败条件；程序终止检查仍须显式修改，适配本身不等于成功或预算重置。

## 本轮测试协议

配置：[waffle_subtasks.yaml](../configs/waffle_subtasks.yaml)。WaffleReheat seed 0–4，固定源码和配置，5 路并行。
同一 OmniSF atomic checkpoint 59999、π0.5、Flora 每 50 个 VLA 步监督、Astra high；无 Episode 墙钟超时。

| 预算 | 上轮 | 本轮 |
|---|---:|---:|
| Episode 动作步数 | 4050 | 8100 |
| 每个代码子任务最多尝试 | 12 | 32 |
| 单次程序步数 / 秒数 | 400 / 240 | 600 / 360 |
| 每个代码子任务总步数 / 秒数 | 2400 / 5400 | 6000 / 10800 |
| Astra turns / 干预上限 | 100 / 16 | 200 / 64 |

单次程序隔离、动作检查和独立停止保持。停止后记录确切预算或判断原因，不为凑时长盲目继续。

每个 seed 从上一轮 Waffle 归档的同一冻结库开始，包括失败候选；新记录私有积累。本次是同任务开发回测，
Memory 已见过这些 seed 的旧轨迹，不是未见场景评测。局部验证器、计划、预算及起始经验同时变化，
不能把成功率或时延差异单独归因于 Memory 压缩；没有新增纯 VLA 配对对照。

结果目录：[waffle-subtasks-20261102](../runs/waffle-subtasks-20261102)。本轮将保留全部 5 条回放和阶段切换证据。
