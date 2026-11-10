# 本机部署与验证

更新日期：2026-10-29。当前配置是接入检查配置，尚未做 Composite-Seen/Unseen 成功率评测。

## 进程和环境

| 组件 | Python / 入口 | 加载内容 |
|---|---|---|
| Agent、Harness、工具与日志 | `/home/junwue/code/vla_agent/.venv/bin/python` | 本项目 |
| π₀.₅ 推理 | `/home/junwue/envs/pi05/bin/python` | `/home/junwue/code/omnisf` 的模型和 transforms，JAX 路径 |
| Decision 3.0 | `/home/junwue/envs/decision3/bin/python` | Transformers 5.17.0、PyTorch 2.7.1、`vllm-sr/d3-flash` |
| Flora（可选） | `/home/junwue/envs/flora/bin/python` | 同版独立 Torch/Transformers 环境、`nvidia/Flora-Rewarder` |
| RoboCasa 仿真 | `/share/project/robot-rfm/robot-exp/usr/cjy/envs/robocasa365/bin/python` | cjy 的 robocasa365、robosuite365 和 simulation_tools |
| Astra | `/home/junwue/tools/bin/codex app-server --stdio` | 已登录 Codex，`gpt-6-astra`、`xhigh` |

仿真源码路径：

```text
/share/project/robot-rfm/robot-exp/usr/cjy/simulation/robocasa365
/share/project/robot-rfm/robot-exp/usr/cjy/simulation/robosuite365
```

用户既有评测实现 `/share/project/robot-rfm/robot-exp/usr/ejw/posttrain/environments/robocasa365` 用于核对模型转换、启动变量和 checkpoint。当前 Agent 使用自己的逐步执行循环和独立 worker，没有套用原来的并行 `launch.py`，也没有修改这些共享源码或环境。

## 模型与动作契约

当前 checkpoint：

```text
/share/project/robot-rfm/robot-exp/usr/ejw/posttrain/outputs/omnisf_atomic/checkpoints/robocasa365_target_atomic/robocasa365_atomic_noalign_4gpu_20261025T081627Z/59999
```

统计文件使用其下的 `assets/robocasa365_target_atomic/norm_stats.json`，SHA256 为 `2d4c74ff56377ea3e79223dcf7b223913126f0bc332f8f6936a09550687fccaa`。服务检查 checkpoint 已提交，客户端检查服务声明的 checkpoint、统计文件哈希、排列和编码。

OmniSF recipe 固定为 `robocasa365_target_atomic`：π₀.₅、action horizon 50、max token length 200、quantile normalization。传入 256×256 三视角，模型 transforms 内缩放至 224×224。训练输入/输出是 base-first；Harness 使用 EEF-first。夹爪和 mode 从模型原生 `-1/+1` 映射至 `0/1`。运动限幅和编码转换均记录限幅前后数组。

`serve_pi05.py` 可保留官方 RoboCasa recipe，但当前部署必须通过 `serve_from_config.py vla` 选中 OmniSF recipe，不能将另一套 checkpoint 的 normalization 或排列混用。推理直接构造模型/Policy/transforms，避免导入 OmniSF 所有训练配置及不相关的机器人依赖。

Decision 固定版本：`vllm-sr/d3-flash@13dc1cc8ef23caefa765907e1d764751e5629f7d`。权重位于 `/share/project/robot-rfm/cache/huggingface`。安装命令是 `bash scripts/setup_decision3.sh`；当前已安装并下载，包快照在 `environments/decision3-freeze.txt`。服务使用仓库提供的 `system_one` 接口，对五类计算概率；不要求模型用自由文本猜测 JSON 分类。

H100 的 Decision 环境使用 `flash-linear-attention==0.6.0` / `fla-core==0.6.0` 和 `causal-conv1d==1.7.0` 加速 Gated DeltaNet 与因果卷积。依赖固定在 `environments/decision3-acceleration-requirements.txt`，使用上游提供的 CPython 3.11 / Linux x86_64 / CUDA 12 / Torch 2.7 / CXX11 ABI=true wheel 及 SHA256；其他平台需要选择匹配的 wheel。保持 PyTorch 2.7.1、Triton 3.3.1、Transformers 5.17.0，不修改 π₀.₅ 环境。

启动器在 `DECISION_READY` 前完成预热。`/health` 返回实际设备和算子绑定；`torch_chunk_gated_delta_rule` 应指向 `fla.ops`，`causal_conv1d_fn` 应指向 `causal_conv1d`，不应为 `torch-reference`。响应的 `meta.system_one_seconds` 是服务端单次调用耗时，包含模型侧输入处理和评分，不包含模型加载、客户端图像编码及 HTTP 往返，也不等同于纯 GPU 前向耗时。

Decision 的一次调用对应一个历史观测窗口的监督检查，不是每个控制步或每次 VLA 推理。默认后台检查不会要求 VLA 等待；周期、事件触发、已有在途请求和 Astra 审核共同决定实际调用频率。H100 上安装前后的固定输入比较见 `runs/decision-acceleration-20261029/`。可在 Decision 环境使用 `scripts/benchmark_decision3.py` 重放该目录的 `requests.json`；只读历史轨迹，不执行仿真或机器人动作。

当前通用/CloseToaster 配置为 `window_frames: 5`、`primary_camera: null`、
`history_cameras: [left, right, wrist]`，即五个历史时刻的三个视角，最多十五张。
旧九图加速报告不代表当前十五图延迟；新的同窗口实测见 [Decision 监督设置](decision-supervision.md)。
Astra 使用独立的 `planner.window_frames: 3` 多视角窗口；离线 `shadow_decision.py` 也遵守所选 Decision 配置。

## 可选的 Flora 服务

Flora 固定到 `nvidia/Flora-Rewarder@031299e46e452fc0cedb4e0975eb9fa95fc37783`，
独立环境通过 `bash scripts/setup_flora.sh` 安装，权重写入共享 Hugging Face 缓存。
启动命令：

```bash
.venv/bin/python scripts/serve_from_config.py flora --config configs/robocasa.yaml
curl -fsS http://127.0.0.1:8030/health
```

等待 `FLORA_READY`，再在 Agent 使用的配置中设置 `flora.enabled: true`。
当前真实配置默认关闭该可选服务：本进程仅看到一张 H100 80GB，Flora 的 BF16 权重本身约
55.6 GB，尚未验证与 π₀.₅、Decision 同卡运行。部署时先分配独立 GPU 或配置另一台机器的
`flora.endpoint`；不能把三模型的独立推理检查当成同时运行已经通过。

服务使用 Transformers 后端，复用固定模型版本中的官方 Outcome 提示词、解析器及采样约定；
不是官方 vLLM 批量评测服务。关闭 thinking；无理由模式最多生成 12 token，有理由模式最多 160 token；默认 2–64 个偶数帧、
left/wrist 拼图，输入上限 16K token。只提供 Outcome，不开放 Dense 或 Rubric。
`FLORA_READY` 前预热，返回 `seconds` 表示服务内预处理与推理耗时；工具结果的 `seconds`
还包含读历史、图像编码与 HTTP，均不包含模型加载。

只读重放已有 OpenDrawer 纯 VLA 轨迹：

```bash
.venv/bin/python scripts/replay_flora.py \
  --episode runs/atomic-pilot-20261028/baseline/OpenDrawer_seed0 \
  --steps 50 402 --output runs/flora-replay-001
```

加 `--prepare-only` 可以在不加载模型的情况下检查历史片段和模型输入；这种结果不是实际评分。
脚本要求未发生干预的单子任务基线，避免错用计划/重试边界。原日志不修改，新目录保存请求、
响应、缓存验证与计时，不启动机器人或仿真。少量样本只用于接口检查，不能证明在线子任务
判别准确率；完整轨迹 Outcome 模型对执行中片段的适用性仍需标注验证。

本机实际回放见 [Flora 验证报告](../runs/flora-tool-20261029/README.md)：早期 24 帧得到 2/5，
原生成功后的 64 帧得到 4/5。服务内热调用约 0.58–0.64 秒 / 0.98 秒，包含 HTTP 约
0.71–0.77 秒 / 1.28 秒；首次 64 帧工具调用含历史读取与编码用了 6.13 秒。
这些是模型常驻、Flora 单独占卡的少量测量，不包含加载时间。回放后显存快照为 58,915 MiB；
测试服务已停止。细节和原始请求/响应均在报告目录。

## 启动与短流程检查

以下命令均从项目根目录运行，服务分别占用一个终端：

```bash
cd /home/junwue/code/vla_agent
.venv/bin/python scripts/serve_from_config.py vla
```

等待 `POLICY_READY` 和 8010 监听日志，再启动另一个终端：

```bash
cd /home/junwue/code/vla_agent
.venv/bin/python scripts/serve_from_config.py decision
```

等待 `DECISION_READY`，可以检查 `curl -fsS http://127.0.0.1:8020/health`。第三个终端运行：

```bash
cd /home/junwue/code/vla_agent
.venv/bin/vla-agent doctor --config configs/robocasa.yaml
.venv/bin/python scripts/smoke_stack.py --output runs/real-smoke-001
```

输出目录必须不存在。短流程最多 20 步、600 秒、4 次 Codex turn，正常到达步数上限会报告 `reason=step_budget`，不要求 `success=true`。正式完整 Episode 使用：

```bash
.venv/bin/vla-agent run --config configs/robocasa.yaml --output runs/robocasa-001
```

启动的是本地服务，不自动下载替代 VLA 权重。检查结束后可以在模型服务终端按 Ctrl-C 释放 GPU；Agent 会自动关闭它拥有的模拟器、恢复程序和 Codex 子进程。

## Codex 推理与工具

调用方式参考 `gpt_as_policy/hybrid_rollout/robocasa365/skill/transport.py`，并对照本机 Codex 0.164.0 生成的 JSON schema 验证动态工具协议。使用现有登录，未复制凭据；不需要为该后端设置 `OPENAI_API_KEY`。

每个 Episode 建立一个 ephemeral Codex thread，各次干预复用该会话。`initial_planning: true` 时先根据真实任务描述规划，再让 VLA 执行。所有非 KEEP 判断进入 Astra；Dynamic Tools 提供证据读取、技能检索、程序编写/补丁/静态检查和干预提交。

原生环境访问关闭：`environments=[]`，shell/apps/web search 关闭。Astra 通过主机工具获取当前信息，工具调用按序处理。参数错误返回给模型修正；重复/过期调用、预算超限和模型失败会终止本次干预。`submit_intervention` 只接受候选提案，必须等待当前 turn 正常完成，随后才由 Runtime 执行。Code Policy 默认采用 Landlock + seccomp 子进程约束；当前支持范围和验证见 [框架更新](framework-v2.md)。

`astra_calls` 在此后端表示 Codex turn 数，`astra_call_unit=codex_turns`；不能将其理解为底层网络请求次数。输入/输出 token 从 app-server 的累计 usage 记录，未推算费用。

Working Memory 仍是有界历史和结构化笔记，不做语义摘要压缩。Codex 会话自身的历史维护/自动压缩由 Codex 管理，若收到 contextCompaction 事件则记录；本次没有压测长会话压缩效果。程序版本、失败证据、原始观测保存在独立文件中。

## 已验证与剩余工作

详细结果见 `environments/integration-verification.json`。真实四组件短流程保存在 `runs/integration-20261028-environment-instruction`：20 个实际仿真步、2 次 Decision 检查、3 次 Codex turn（含启动规划），无运行错误，按步数预算结束。该检查没有完成整体任务。

九张真实历史图的离线重放中，Decision 预热后两次推理约 0.65 / 0.63 秒；真实联动中的两次检查约 0.80 / 1.22 秒。样本很少，不能作为稳定延迟或分类准确率报告。默认 KEEP 阈值 0.8 未校准，这几次低置信结果触发了 Astra；成本目标仍需专门评测。

2026-10-29 的加速组件验证使用另外三份固定历史窗口，每份 9 张图、约 2.1 万输入 token。
同机同输入服务端热调用中位耗时由 2.40 秒降到 1.30 秒（1.85 倍）；原始分类及原阈值路由均未变，
最大类别概率差约 0.0058。真实本地 HTTP 热调用约 1.32 秒，预热后首个真实窗口仍用了 4.42 秒。
详见 `runs/decision-acceleration-20261029/README.md`。这些不同输入和计时范围的结果不能直接混比，
本次也没有与 π₀.₅ 共卡运行或重跑机器人任务。

Codex 的程序生成、工作区冻结和执行另用 mock 环境检查；不会作为真实机器人恢复证据。所有程序保持候选状态，未晋级 verified 技能。

剩余关键工作：

1. 跑完整原子任务基线，核对任务成功判据、动作限幅比例和模型表现，再选定长序复合任务集合。
2. 为真实子任务和局部恢复补充验证器。当前只有 `task_success`，Astra 因此保留一个可验证整体子任务，细分步骤写入笔记。
3. 验证 RGB-D 对齐、相机外参与几何工具，再做基于几何的真实代码恢复。当前没有接入 IK/路径规划器。
4. 标注 Decision 的漏判/误判并校准阈值、检查频率和上下文规模；固定测试条件后再报告恢复成功率、调用量与成本。

参考：[Decision 模型](https://huggingface.co/vllm-sr/d3-flash)、[Codex App Server](https://developers.openai.com/codex/app-server)。
