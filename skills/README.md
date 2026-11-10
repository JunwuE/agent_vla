# 恢复技能库

新 Code Policy 使用 RAX 风格包：版本目录包含 `SKILL.md`、`code/run.py` 和按技能名命名的 Python 包。
`main(ctx, **params)` 通过 `uses` 指定的固定版本依赖 import 已有函数，返回 `{ok, measured, why}`。
感知、控制和参数分开；执行快照记录所有依赖与运行时哈希。旧 `policy.py:run(ctx)` 继续兼容。
工具与约束详见 [包开发说明](../docs/rax-code-policy.md)。存储中的 candidate 对应未晋级的 draft。

当前可调用操作目录见 [INDEX.md](INDEX.md)。语义名称按“操作 + 起始状态”命名，
版本哈希仅用于不可变源码/契约定位。`catalog.json` 给已检查的旧版本建立语义别名，
不复制源码、不改旧 manifest；历史 ID 继续兼容。未整理的旧恢复稿需显式 `include_archived=true` 检索。

`search_skills` 每项返回一个技能；`list_skill_versions(skill_id)` 查看其适用状态、参数和历史版本。
换 seed、当前帧、图像框或支持的相机参数时，直接再次提交同一 `skill_id/skill_version/parameters`。
代码和契约不变就只追加 runs；逻辑改变沿用技能名新增版本，并保存 `revision_of`。
候选版本不会自动晋级；推荐版本也不等于当前场景适用。

新建程序工具必须说明 `skill_name`、`when_to_use`、`reuse_reason`；已有名称需直接调用或导入后修订。
可在命令行查看：`vla-agent skills --include-candidates --versions close-toaster-door-from-upright`。

每个版本位于 `<id>/<version>/`，包含 `manifest.json`、不可变的 `policy.py` 和可选 `runs.jsonl`。
通过 `vla-agent skills --include-templates` 查看内置模板；`vla-agent skills` 默认只列 verified 技能。

| 状态 | 含义 | 默认检索 |
|---|---|---|
| template | 可执行的参数化起点，尚无真实任务验证 | 否；显式 include_templates |
| candidate | 积累的实验程序与运行证据，可能全部失败 | 否；显式 include_candidates 检索，按确切 id/version 读取和试验 |
| verified | 经离线验证、适用条件明确的版本 | 是；按机器人和实际能力过滤 |

内置三个模板：`eef_offset/v1`（相对末端微调）、`gripper_hold/v1`（保持位姿并开闭夹爪）、
`mobile_nudge/v1`（底盘/躯干短程控制）。它们均不宣称已经具备可靠的抓取、撤回或避障能力。

Manifest 必须包含 id、version、代码 sha256、robot、status、description。其他字段：

- `parameter_schema`：本地 JSON Schema；拒绝未知参数、类型/形状错误和外部引用。
- `required_capabilities`：例如 osc_eef、mobile_base、gripper、rgbd、native_action12。
- `preconditions`：必须有已安装的验证器，且运行前检查明确通过。
- `completion_condition`：固定成功条件；null 表示调用方必须提供一个已支持的条件。
- `parameter_sources`、`known_failures`：参数的观测来源和适用限制。
- `evidence`：验证证据；verified 版本不能为空。是否满足重复执行、不同布局和回归要求仍需离线审核。

Astra 先 search/read/check，再提交 kind=code、skill_id、skill_version、parameters 和
completion_condition。程序通过 `ctx.parameters` 读取参数，通过 `ctx.completion_condition`
读取完成条件。运行时再次校验能力与前置条件，所有动作经过 Harness，报告成功后仍独立验证。

frozen 不写库；accumulate 只写 candidate 及独立运行记录，不自动晋级。参数化程序积累时
保留参数 schema、能力和前置条件，实际调用参数保存在运行证据中。失败尝试同时保留为候选程序和 Episode 日志。
Episode 开始后检索使用固定快照；新写入的版本要到下一个 Episode 才可按确切版本读取。

修改源码或技能契约应生成新版本。快照同时保存源码哈希和完整 manifest 哈希，避免只比较代码、
漏掉能力要求或成功条件变化。SkillLibrary 与参考 gpt_as_policy 没有运行时导入依赖。

CloseToasterOvenDoor 开发候选见 [实验记录](../runs/toaster-iteration-20261029/README.md)。
部分历史程序绑定当时的 step、图像区域和接触状态，保留在归档检索中。
优先选择目录中已有参数契约的版本，读取运行证据并重新 grounding；只有现有逻辑/契约不支持时才修订，不能靠删除保护强行复用。
