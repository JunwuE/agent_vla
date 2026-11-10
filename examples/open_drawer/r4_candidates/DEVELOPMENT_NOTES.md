# 2026-11-06：当前任务的候选技能证据

这些记录属于 OpenDrawer 场景 Memory；不改变通用 primitives 或默认版本。完整参数、原始 RGB-D、动作与校验在对应 episode。

## 已在本次 seed 6 成功的局部修订

- 家族：drawer-regrasp-and-open；保留版本：5ada71444e905c90；状态仍为 candidate。
- 来源：旧 0ed2fc3427424179，导入后只修改进入条件与对应契约，执行版本 v002-14d0f01073fc。
- 原问题：step 840 夹爪间距约 47 mm，未满足旧程序要求的开放入口；视觉与测量显示夹爪位于把手下方。
- 新阶段 establish_open_missed_grasp：只有测得 pad 中心至少低于把手 60 mm、距离至少 80 mm 时，允许有界打开探测。
- 打开过程中限制腕部位移 15 mm，最多 10 段 × 2 步；必须测得间距至少 60 mm，再重新测量把手。
- 后续对齐、夹取、表面耦合跟踪与 task_success 检查复用旧程序。
- 本次 332 个代码动作、约 120.05 秒；全 episode 在第 1172 步成功，独立原生成功信号为真。
- 这是一个布局的一次验证，不是跨布局晋升。原 SKILL.md 的探索假设和原始代码不被事后重写。

[成功候选契约](skills/drawer-regrasp-and-open/5ada71444e905c90/SKILL.md) · [原始验证证据](skills/drawer-regrasp-and-open/5ada71444e905c90/manifest.json)

## 未成功的 seed 4 修订

候选 83f6bbe4e358fc9c（执行版本 v002-3206bfd29064）保留作失败证据。
第一次旧技能执行 120 步未收敛；修订后又执行 215 步，进入 pull 阶段但用尽 episode 剩余动作预算，未通过 task_success。
不能标为成功示范或默认版本。总体失败也不能证明该修改在充足预算下必然失败，需独立再验证。

[失败候选契约](skills/drawer-regrasp-and-open/83f6bbe4e358fc9c/SKILL.md) · [原始证据](skills/drawer-regrasp-and-open/83f6bbe4e358fc9c/manifest.json)
