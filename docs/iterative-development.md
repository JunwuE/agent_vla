# 单任务迭代实验

2026-10-29 起的 CloseToasterOvenDoor 实验位于
[toaster-iteration-20261029](../runs/toaster-iteration-20261029/)。
原生预算、扩展预算和失败重放必须分别报告；每个目录记录配置、原始观测、动作命令、实测状态和模型调用。

## 可配置的探索预算

- `environment.ignore_native_horizon: true` 只修改本进程创建的 RoboCasa 实例；不改共享仿真源码。
- `runtime.max_wall_seconds: null` 关闭整个 Episode 的墙钟限时。
- `runtime.max_steps` 保留实验动作总预算，包含 VLA 和 Code Policy；超原生 450 步的成功单独标记。
- `recovery_attempts` 是完整子任务代码会话内的最大尝试数，开发配置使用 12。
- `recovery_steps/seconds/tool_calls` 是每个程序的预算；`recovery_total_steps/seconds` 是整个会话预算。
- `recovery_program_versions` 控制不可变程序版本数，开发配置使用 64。生成新版本不会重置当前物理状态。

Astra 仍决定接管。Code Policy 一旦接管，整个当前可执行子任务继续归代码所有，直到独立验证完成或明确停止。
规则不会因为经过某个步数而直接选定 Code Policy。`stalled_progress` 是 Astra 内部审核评估，
要求至少两个时刻的观察证据；Decision 仍然只有原来的五类。

过期的干预请求不执行，也不提交其阶段状态。若 Episode、任务版本和控制来源仍一致，
运行时立即发起新的只读复核，让 VLA 继续执行；新审核通过后仍需在暂停后的新观测上确认。

## 程序与经验积累

所有执行过的程序，包括失败尝试，保存为 candidate，并保留版本、源代码、运行证据、参数和原始轨迹位置。
默认技能检索仍只返回 verified；Astra 必须用 `include_candidates: true` 明确检索实验程序。
历史候选可能绑定旧图像区域或起始 step，必须重新测量、修改适用部分，不能直接当成通用技能。
单次成功不会自动晋级 verified。单个 Episode 的技能快照保持固定，新的候选供下一 Episode 检索。
同等文本匹配的候选优先返回有独立成功运行证据的版本，但它仍是 candidate，不能据此绕过当前状态和适用条件检查。

Code Policy 报告包含逐阶段实测记录和 `program_notes`；程序声称的事实、原因假设与独立成功判据分开。
机器人自身几何通过 `controller.gripper_kinematics` 提供。`jaw_separation_axis_eef` 包含手指 body
相对末端控制 site 的旋转，不应把关节定义的局部 Y 轴直接当成末端 Y 轴；它不提供目标物姿态或抓取成功真值。

## 配对与复现

`vla.sampling_seed` 可选。服务声明 `vla-agent.seed-step.v1` 后，客户端根据 seed 和当前控制步
生成请求种子，服务只在该次推理使用它并回传确认；其他请求不会改变这个采样结果。
默认 `null` 保持旧的共享随机流。该功能不保证物理求解器跨平台逐位一致。

`run_toaster_experiment.py --policy-seed N` 使用 N + 仿真 seed，并保留每条配置。
`replay_failure_experiment.py` 用新仿真实际执行已有基线的命令；每次分块边界检查实测状态，
初始图像也必须匹配。它属于恢复开发工况，不是新 VLA SR。进入 Code Policy 后不再回放旧动作。

## 单卡 Flora

服务支持实验性的 `flora.quantization: int8`，使用本轮固定的 bitsandbytes 0.48.1。
默认仍为 `none`。服务响应、请求缓存身份和配置均记录量化模式，不允许静默混用。
BF16 CPU 卸载可设置 `flora.gpu_memory_gib`；它限制权重放置，不是整个进程显存上限。
实际并存、延迟和少量评分结果见本轮报告。Flora 的理由是模型解释，不是接触或原因真值。

Astra 的 `compact_numeric_history` 仅把 30 条数值历史均匀采样为 6 条、保留三位小数，
不减少图像、不修改当前控制位姿，也不修改原始日志。完整历史通过 `read_execution` 获取。

## 继续实验

`configs/close_toaster_explore.yaml` 保存本轮扩展开发设置：4000 个动作总步数、最多 12 次程序执行、
固定采样 seed 42、Flora int8 和积累模式。原 `close_toaster_dev.yaml` 的原生预算配置保留。
候选程序及经验分别位于项目 `skills/` 和 `memory/close_toaster/experiences/`；默认不检索未经验证的候选。

```bash
# 可选量化依赖，安装到独立 Flora 环境。
uv pip install --python /home/junwue/envs/flora/bin/python -r environments/flora-int8-requirements.txt
# 以下两个服务各在一个终端启动；Flora 模式无需 D3 服务。
.venv/bin/python scripts/serve_from_config.py vla --config configs/close_toaster_explore.yaml
.venv/bin/python scripts/serve_from_config.py flora --config configs/close_toaster_explore.yaml
# 使用尚不存在的输出目录。
.venv/bin/python scripts/run_toaster_experiment.py --config configs/close_toaster_explore.yaml --output runs/toaster-next --seeds 0 --policy-seed 42
```

组件调用仍有独立 deadline。`component_timeout` 与 Episode 的 `wall_time_budget` 分开记录；
关闭总限时不表示一个没有返回的模型请求可以永久占用执行权。第七轮的旧日志把组件超时误标为总限时，
原始文件保留，勘误记录在实验报告中。
