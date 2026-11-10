# 子任务级 Code Policy：已实现的框架更新

2026-10-29。入口配置：[close_toaster_dev.yaml](../configs/close_toaster_dev.yaml)。
本版修改运行框架、工具和数据契约；尚未得到 CloseToasterOvenDoor 的新成功率。
用户的 48% 使用同一 OmniSF/pi05 权重，正式对照还需固定场景、seed、采样随机性与预算。

## 执行方式

1. Astra 初始化可执行任务和细执行阶段。RoboCasa 当前只有 `task_success` 核验器，因此关门保留一个可执行子任务，内部阶段由 `set_phase_plan` / `assess_phase` 跟踪。
2. VLA 接收简洁原生指令。阶段改变不改写指令、不暂停动作。Decision 使用历史信息；非 KEEP 交 Astra 只读审核，首次尝试也允许正常推进。
3. 可配置的周期审核补充连续 KEEP 的盲区：开发配置每 150 步提供一次有预算的后台 Astra 审核机会；有其他审核时去重。它不标记失败、不直接抢占控制。设 `audit_interval_steps: 0` 可关闭。
4. Astra 提出有证据的干预后，Harness 暂停、清旧 Chunk、重新观察，Astra 再确认。
5. Code Policy 接管后负责**完整当前子任务**。阶段成功和 `RECOVERED` 返回值都不能让未完成子任务交回 VLA。失败后修改代码并重试，共享同一个会话预算。
6. 只有固定子任务核验通过才能结束 code 分配。还有后续子任务时，由 Astra 明确选择其执行器；最后一个任务完成则结束。预算耗尽只允许停止或明确改变剩余计划，同目标改名不能刷新预算。

开发配置：原生 horizon / Episode 总动作上限 450 步；代码每版最多 150 步、90 秒，累计最多 450 步、300 秒、4 次尝试。实际可用步数还要扣除 VLA 已执行部分。Astra 初始化/修订与整个 Episode 也各有预算。

## 工具和 Memory

| 接口 | 实际能力与边界 |
|---|---|
| `set_phase_plan`, `assess_phase` | 有版本的阶段目标、进展/完成/失败信号和观测证据；属于 Astra 判断，不是独立物理核验 |
| `ground_target` | Astra 或程序在指定当前原图上选择 bbox，保存观测步、图像哈希、坐标和自报置信度；返回叠框图。尚未接独立 grounding 模型或 mask 分割 |
| `query_geometry` | 对齐的米制深度反投影到 OSC arm base；返回可见点中位位置、范围、有效深度比例、深度四分位差和末端相对向量。动作后旧定位失效；不声称得到物体中心、接触点或无碰撞轨迹 |
| `evaluate_subtask` | 整个当前尝试，或 `phase_id` 指定的当前阶段视频；`with_rationale` 可取 Outcome 评分和短解释。按请求缓存、按 Episode 限额。评分不代替成功判据，解释作为待验证假设 |
| `search_experiences` | 按机器人、任务/阶段文本检索最多三条成功或失败经验，附工具使用、程序版本、日志路径及哈希；是历史建议，不会自动执行 |
| 程序工作区工具 | 冻结代码、读代码、精确补丁、静态检查、执行反馈；程序经隔离子进程和 Harness 执行 |
| `propose_policy` | 保存有限调度参数的候选版本及证据，不改变当前 Episode，不直接晋级 |

短期 Memory 保存阶段、尝试、执行器、定位、Flora 记录和最近进展；原始图像、深度、命令和实测状态独立存档。
长期经验在 Episode 结束后由宿主写入内容寻址文件，成功失败都保留。`frozen` 只检索启动快照；`accumulate` 追加候选经验和已有技能机制的候选程序。单次成功不会自动得到 verified 技能。

Decision 当前为五个历史时刻（距当前 10、5、2、0.5、0 秒）各取 left/right/wrist 三视角，最多十五张；初期去重，不跨尝试。
数值上下文保持三位小数、六个近期动作样本，并增加近 2 秒/10 秒实测运动摘要、阶段和历史奖励证据；完整精度数据不变。
五类判据、持续低运动规则及 7/15 图实测见 [Decision 监督设置](decision-supervision.md)。
上下文 schema 已升至 `vla_agent.decision_context.v2`。旧测试报告描述 v1，不能把旧速度结果当成 v2 实测。

## 有限自修改

当前可在线修改的是恢复程序及参数；它必须通过观测/控制工具执行。调度策略第一版使用声明式候选参数，支持 Decision 周期、阈值、历史时刻、只读审核间隔和 Flora 是否输出解释。
模型权重、Harness、动作/停止校验、权限、成功判据、硬预算、日志和评测代码不在修改面内。

候选可由宿主生成独立实验配置：

```bash
.venv/bin/python scripts/build_policy_experiment.py \
  --config configs/close_toaster_dev.yaml \
  --candidate runs/EPISODE/policy_candidates/HASH.json \
  --output configs/candidate_experiment.yaml
```

这不是自动晋级。**跨 Episode 的自动配对评测、晋级和回滚控制器尚未实现**；须先按固定开发/回归种子比较 SR、救回/破坏数和成本，再由外部流程选择版本。
当前已经能积累证据和候选，不宣称实现了经验证的完整 RSI。

生成程序默认采用 Linux Landlock + seccomp 白名单：只读运行库和当前尝试导出的证据，禁止宿主文件写入、网络、进程创建和执行。只经 stdio 代理调用宿主工具。
隔离初始化失败会终止尝试。`doctor` 会实测隔离是否可用。显式 `trusted` 模式仅保留资源预算，没有这些 OS 约束。
本机 Bubblewrap 的挂载命名空间操作被禁止，因此默认使用已测试的 Landlock 模式。
相关机制：[Linux Landlock 文档](https://docs.kernel.org/userspace-api/landlock.html)。本实现仍需按实际部署环境做安全审查，不是安全认证。

## 启动与验证

Agent：`/home/junwue/code/vla_agent/.venv`；pi05：`/home/junwue/envs/pi05`。
模型地址、checkpoint、仿真解释器均在开发配置中。先检查：

```bash
.venv/bin/python -m vla_agent.cli doctor --config configs/close_toaster_dev.yaml
```

模型分别启动：

```bash
.venv/bin/python scripts/serve_from_config.py vla --config configs/close_toaster_dev.yaml
.venv/bin/python scripts/serve_from_config.py decision --config configs/close_toaster_dev.yaml
# Flora 必须使用额外 GPU/独立服务，或先验证其内存方案；不要直接在同一 80GB 卡上启动全部 BF16 模型。
.venv/bin/python scripts/serve_from_config.py flora --config configs/close_toaster_dev.yaml
```

当前单张 H100 80GB 上，Flora BF16 单独约 59GB，pi05 + Decision 约 27GB，三者常驻的方案未验证。
可先单独重放 Flora；完整联调需提供独立 Flora 服务或验证量化/卸载。Flora 不可用会记录 unknown，不会当作失败证据。

```bash
.venv/bin/python -m vla_agent.cli run --config configs/close_toaster_dev.yaml --output runs/close-toaster-dev-001
.venv/bin/python scripts/render_episode.py runs/close-toaster-dev-001 --output runs/close-toaster-dev-001/replay
```

当前验证与产物见 [验证记录](../runs/framework-v2-20261029/README.md)。
RGB-D 检查验证了包装器方向、原始数值转换、数值重投影和运动相机更新，尚未验证目标跨视角特征匹配及实际接触精度。
相机标定读取机器人和相机参数；在线工具不读取门板/物体真值来替代视觉定位。
