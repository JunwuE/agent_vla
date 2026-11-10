# OpenDrawer：初始经验与候选程序

本目录发布程序、契约及精简经验，不包含原始 RGB-D、视频或完整模型会话。

| 目录 | 含义 | 使用方式 |
|---|---|---|
| `initial_memory/` | R3 配对实验开始前的 skills、tasks、experiences | 复现固定初始经验时每条 Episode 独立复制；不掺入后续候选 |
| `r4_candidates/` | R4 保留的固定依赖、两个候选修订及成功/失败说明 | 用于分析后续改进；不是原实验初始库，也不是已验证默认库 |
| `provenance.json` | 每个发布文件的原始相对路径与 SHA-256 | 检查复制完整性和代码/契约版本 |

两个 R4 候选沿用 `drawer-regrasp-and-open` 家族，不为每个 seed 新建技能名。
`5ada71444e905c90` 在 seed 6 一次成功；`83f6bbe4e358fc9c` 在 seed 4 未完成任务。
它们仍是 candidate，均未晋升为跨布局 verified 技能。

先读 [R4 开发证据](r4_candidates/DEVELOPMENT_NOTES.md)，再按目录中的 SKILL.md 检查前置条件与固定依赖。
`rgbd-surface-geometry` 和 `eef-feedback-control` 提供共享感知/控制逻辑。
执行时必须重新读取当前观测、定位目标和计算参数；旧记录的 observation step、图像框及位姿不能直接用于新场景。

快照保留原文件字节，以维持哈希、版本身份和实际验证记录。
其中的 `/home/...` 或 `runs/...` 路径是原机器证据引用，通常不会在克隆机器上存在；
不能把未提供的轨迹当成已在本地复核。移植代码时通过新版本或显式配置改路径，避免原地修改冻结契约。

首次运行参考 [绝对路径配置模板](../../configs/examples/open_drawer.yaml)。该示例使用 frozen Memory；
积累实验应先复制独立可写库，并保留实验初始哈希与后续修改。
