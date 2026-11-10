# Code Policy 探索与修改

本轮参考 RAX 的 robodojo-behavior/native 分支，提交为
`16e75db0d0258a4870935d0c4b2402ab618db11d`。本地只读参考代码位于
`/home/junwue/deps/rax-reference`，文件哈希见 [rax-reference.json](rax-reference.json)。

主要参考：

- [autonomous-control](https://github.com/robot-agent-exploration/rax/blob/16e75db0d0258a4870935d0c4b2402ab618db11d/agent_skills/auto-research/autonomous-control/SKILL.md)：从探索形成程序、分阶段验证、当前状态重观测、跨布局检验。
- [experience-memory](https://github.com/robot-agent-exploration/rax/blob/16e75db0d0258a4870935d0c4b2402ab618db11d/agent_skills/task/experience-memory/SKILL.md)：保留适用条件、效果与证据，局部观察和整个任务成功分开。
- [storage.py](https://github.com/robot-agent-exploration/rax/blob/16e75db0d0258a4870935d0c4b2402ab618db11d/services/controller/storage.py)：冻结运行版本并检查哈希。

完整子任务的执行器分配已由 `code_session.py` 管理。内部阶段完成不交还 VLA；代码版本之间共享预算，最终完成通过固定核验后才选择下一个执行器。新增接口与验证见 [框架更新](framework-v2.md)。

## 默认路由

VLA 仍是主执行策略。非 KEEP 交给 Astra，由 Astra 调整语言/计划，或者进入代码恢复。
默认允许 continue/refine/switch/replan/code/observe/stop；local/eef/edit 已关闭，其预览工具
也从默认 schema 隐藏。需要对照实验时可在 allowed_interventions 中显式加入。

Code Policy 内的 move_eef、move_relative、set_gripper、move_base、act 仍然可用，都经过 Harness。
这些是程序实现闭环控制的基础接口。Astra 不需要为每个小动作进行一次模型调用。

## 开发与执行循环

1. 先检索语义技能和版本；支持当前输入时直接传参数重跑。只有缺少操作程序时才用 write_program，
   提供 skill_name、when_to_use、reuse_reason，并保存 run(ctx) 源码、假设、预期效果和静态检查。
2. read_program / patch_program 读取和修改指定版本。补丁必须精确匹配一处文本；每次修改生成
   新版本，保留父版本、源码哈希和差异。
3. check_program 编译语法并检查同步 run(ctx) 入口，不执行顶层代码，也不验证机器人行为。
4. submit_intervention(kind=code, program_version=..., completion_condition=...) 请求运行冻结版本。
5. 执行失败后，read_recovery_trace 提供阶段、实际步数、检查证据、异常堆栈与文件索引。
6. 在同一个工作区修订，再从当前物理状态执行。成功则独立验证并交还 VLA；预算耗尽则交回外层决策。

首次提交原始 code 或现有 skill 也会登记到工作区。程序版本、原始源码和技能引用三选一。
当前开发版本是一个可包含多个函数的 Python 文件，参数放在显式常量或当前观测计算中；
持久化技能仍可使用既有 parameter_schema。尚未实现多文件工程与进程重启后的自动恢复。

```text
workspaces/<intervention>/
  task.json
  index.json
  versions/<version>/
    policy.py
    manifest.json        父版本、哈希、修改假设、预期效果、静态检查
    change.diff
  attempts/001.json       实际运行结果与阶段证据
recovery/<attempt>/
  policy.py              实际执行副本
  stderr.log
  observations/          本次程序读取的原始观测副本
```

版本文件以哈希检查防止意外修改；仍沿用受信任代码的子进程模型，不具备 RAX 容器级隔离。

## 阶段检查

```python
def run(ctx):
    with ctx.stage("locate") as stage:
        frame = ctx.read_observation("left")
        # 使用 frame["rgb"]、可选 frame["depth_m"] 和 metadata 中的标定进行计算。
        stage.record(depth_available="depth_m" in frame)

    with ctx.stage("recover", preconditions=["已安装的前置条件"],
                   completion_condition="已安装的恢复条件"):
        # 根据当前观测调用闭环控制工具。
        pass
    return "RECOVERED"
```

示例条件必须替换为实际已安装的验证器。前置条件为 false/unknown 时，该阶段不会执行。
完成检查失败会使上下文管理器抛出异常；主程序未处理的异常返回给 Code Agent，附带失败阶段
和堆栈。程序可以自行处理已知异常，但最终恢复成功仍由 Harness 独立核验。

stage.record 是程序自报测量，不能批准成功。没有 completion_condition 的阶段只标记 unchecked。
旧式 ctx.stage(name) 继续作为未验证的阶段标记使用。框架不会凭旧阶段编号自动跳过动作；
新程序必须重查当前状态，选择满足前置条件的步骤。停止和回执不确定仍立即撤销动作权限。

## 原始观测与预算

ctx.read_observation(camera, step=None) 返回 NumPy 数组 rgb、state、可选 depth_m，以及 metadata。
数据来自本 Episode 已记录的观测，只允许当前及历史步；相机标定保留原来的时刻、坐标系和
alignment_verified 标记。深度保留米制数值，不转换成可视化图片。

返回的 NPZ 副本带哈希，读取时检查；单次恢复的数组导出总量上限为 64 MiB。没有深度时不生成
替代数据，没有验证标定时不宣称几何可靠。用户代码可以用 NumPy/Pillow/SciPy 做局部计算。

recovery_steps / recovery_seconds 限制单次运行；recovery_total_steps / recovery_total_seconds
限制整个恢复会话（包括后续模型修订等待），重试不会重置这些累计预算。初次选择策略的模型
调用由 Planner 和 Episode 总预算控制。修改恢复目标时退出当前恢复循环，重新进行外层决策。

## 离线演示与仍待接入的部分

```bash
cd /home/junwue/code/vla_agent
.venv/bin/vla-agent run --config configs/demo_exploration.yaml --output runs/exploration-001
.venv/bin/vla-agent inspect runs/exploration-001
```

MockExplorer 通过同一套开发工具写程序：第一版提起距离不足，独立阶段检查失败；读取失败
证据后修改提起参数，第二版从当前状态继续，验证通过后交还 VLA。演示不调用真实模型，
也不证明模型能够自主提出有效修订。

目前没有实现 RAX 的完整场景恢复、多布局 rehearsal、正式 held-out 批量评测或自动技能晋级。
真实 RoboCasa、局部验证器和 RGB-D 对齐仍待联调；下一步应在固定失败场景与新布局上验证
修订收益，再将候选程序整理为可复用技能。
