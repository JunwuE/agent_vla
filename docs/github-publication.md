# GitHub 发布与后续维护

目标仓库：https://github.com/JunwuE/agent_vla

## 首版内容

发布 `src/`、`tests/`、`scripts/`、`configs/`、`docs/`、`skills/`、`examples/`、依赖声明与第三方归属。
保留历史代码的版本和状态；根技能库的 candidate 不是通用可靠技能声明。
本机已有实验配置保留为历史参考，跨机器使用 `configs/examples/open_drawer.yaml`。

不发布 `runs/`、本机可写 `memory/`、虚拟环境、模型权重、checkpoint、登录凭据或账户用量快照。
原始视频和数据如需共享，应作为独立数据产物提供；正式代码仓库保存精简指标、来源哈希和获取说明。

## 本次整理

- README 改为当前状态、代码入口、离线示例和实验结果。
- 原 README 保存在 `docs/history/README-before-publication-20261110.md`。
- `docs/current-architecture.md` 明确实现与计划边界，任务判断程序标记为尚未实现。
- `docs/experiments/` 提供 OpenDrawer 汇总及全部 80 条配对结果；R4 另行报告。
- `examples/open_drawer/` 提供原始初始 Memory 与 R4 候选代码，附逐文件来源哈希。
- `.gitignore` 增加环境变体、凭据、模型与本地配置排除项。
- 没有更改机器人运行逻辑、已有原始实验结果或冻结技能内容。

## 后续更新命令

本机项目目录是 `/home/junwue/code/vla_agent`。提交路径使用绝对路径：

```bash
git -C /home/junwue/code/vla_agent status --short
git -C /home/junwue/code/vla_agent diff
git -C /home/junwue/code/vla_agent add /home/junwue/code/vla_agent/src /home/junwue/code/vla_agent/tests /home/junwue/code/vla_agent/docs /home/junwue/code/vla_agent/README.md
git -C /home/junwue/code/vla_agent diff --cached --stat
git -C /home/junwue/code/vla_agent commit -m "Describe the concrete change"
git -C /home/junwue/code/vla_agent push origin main
```

根据实际修改补充显式路径。提交前确认新文件没有包含凭据、原始观测或巨量模型文件。
首次发布不增加未经验证的成功率声明；后续 release 应对应固定源码、配置、Memory 与独立实验结果。

## 依赖与文档范围

`pyproject.toml` 是轻量 Agent 的安装入口；完整通信测试另需固定版本 OpenPI client。
`environments/*-freeze.txt` 保留原机器快照，可能含本地 editable 路径，不能当成可移植安装命令直接使用。
仿真、π₀.₅、Flora/Decision 与 Agent 采用独立环境，模型和资源不随代码分发。

历史文档可能链接本机未上传的 `runs/`。最新 README、架构页、实验页与示例目录是首版导航入口。
本项目尚未选择项目级 LICENSE，第三方来源与原有许可证见 `THIRD_PARTY_NOTICES.md`。

GitHub 操作依据：[上传已有本地代码](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)。
普通 Git 仓库单文件限制及大文件方案见 [GitHub 大文件说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。
