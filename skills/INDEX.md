# 操作技能目录

技能名称描述操作与起始状态；哈希仅标识不可变版本。旧 ID 仍可读取，原代码、manifest 和运行证据不迁移。

以下均为 candidate，成功记录不等于跨场景验证通过。当前没有经过验证的通用抓取技能。

调用：`search_skills → list_skill_versions → read_skill → check_skill → submit_intervention`。
换场景先更新 `parameters`；参数重试不导入、不 patch，不产生新技能或代码版本。

| 技能 | 已收录版本 | 用途 |
|---|---:|---|
| `close-toaster-door-from-upright` | 3 个历史版本 + 2 个模块化版本 | Close the toaster oven door from a measured nearly upright state: release, clear, approach and seat. |
| `close-toaster-door-from-open-clearance` | 1 | Close the toaster oven door after open fingers have already cleared the raised door. |
| `close-toaster-door-from-closed-clearance` | 3 | Close the toaster oven door from measured closed-finger exterior clearance and stalled alignment. |
| `seat-toaster-door-from-exterior` | 2 | Finish closing the toaster oven door from an already achieved exterior approach/contact pose. |

## close-toaster-door-from-upright

- 模块化 `b22b65f09b8cc992`：[SKILL.md](close-toaster-door-from-upright/b22b65f09b8cc992/SKILL.md)，来源 `6e0569a16adc2f86`。共享几何与 EEF 模块；seed 9 保存回执的 131 步路径完全匹配，尚无新物理验证。
- 模块化 `90a28a7feeb1d020`：[SKILL.md](close-toaster-door-from-upright/90a28a7feeb1d020/SKILL.md)，来源 `2ff9c7e5cdd39598`。共享相同依赖；seed 44 保存回执的 17 步路径完全匹配，尚无新物理验证。
- `6e0569a16adc2f86` — Nearly upright door; downward wrist; visible right-camera rim, roof and fixed lip. Arc-reference tracking; all image boxes and observation_step are fresh parameters. Full seating observed on seed 9; this does not cover grasping a lowered door. [源码](recovery-6e0569a16adc/6e0569a16adc2f86/policy.py) · [原始证据](recovery-6e0569a16adc/6e0569a16adc2f86/manifest.json)
- `bd071b31b22de36f` — Nearly upright door; current fingerpad geometry; right-camera roof and rim. Cartesian tracking supports a tilted wrist. Seed 25 completed during release/clearance; later seating branches were not exercised by that success. Subsequent failures remain recorded. [源码](recovery-bd071b31b22d/bd071b31b22de36f/policy.py) · [原始证据](recovery-bd071b31b22d/bd071b31b22de36f/manifest.json)
- `2ff9c7e5cdd39598` — Same nearly upright procedure with independently selectable oven_camera. Use a fresh unobscured roof region when the right roof view is blocked. Seed 44 completed during clearance; no evidence of all subsequent branches working. [源码](recovery-2ff9c7e5cdd3/2ff9c7e5cdd39598/policy.py) · [原始证据](recovery-2ff9c7e5cdd3/2ff9c7e5cdd39598/manifest.json)

## close-toaster-door-from-open-clearance

- `727ef0dcbde555fa` — Open fingers with measured upper clearance already achieved; retain height, withdraw, close and seat. Fresh right-camera rim, roof and fixed-lip parameters. Do not restart a blocked upward clearance stage. [源码](recovery-727ef0dcbde5/727ef0dcbde555fa/policy.py) · [原始证据](recovery-727ef0dcbde5/727ef0dcbde555fa/manifest.json)

## close-toaster-door-from-closed-clearance

- `8c814dfd1409ce7a` — High exterior clearance and closed fingers achieved. Bounded integral position feedback and latched angular targets during alignment. camera, arc_reference and resume flags are parameters; resume flags require current measured prerequisites. [源码](recovery-8c814dfd1409/8c814dfd1409ce7a/policy.py) · [原始证据](recovery-8c814dfd1409/8c814dfd1409ce7a/manifest.json)
- `e0ddab6a9e8b6fac` — Closed fingers at high exterior clearance; stalled wrist alignment. Measured outward relocation before alignment; current camera, fixed lip and alignment_withdrawal_m inputs. Checks require actual clearance and movement. [源码](recovery-e0ddab6a9e8b/e0ddab6a9e8b6fac/policy.py) · [原始证据](recovery-e0ddab6a9e8b/e0ddab6a9e8b6fac/manifest.json)
- `1d523dff27787f79` — Closed exterior pose; height restoration blocked before alignment. Bounded outward relocation precedes height restoration. Cartesian tracking, current camera, clearance_above_roof_m and alignment_withdrawal_m; do not assume contact from resume flags. [源码](recovery-1d523dff2778/1d523dff27787f79/policy.py) · [原始证据](recovery-1d523dff2778/1d523dff27787f79/manifest.json)

## seat-toaster-door-from-exterior

- `40d558b2ca66e37c` — Current downward exterior alignment and approach already achieved; fresh paired upper-rim measurements and current lateral pose. Final seating with independent tracking. No initial grasp or clearance procedure is established by this variant. [源码](recovery-40d558b2ca66/40d558b2ca66e37c/policy.py) · [原始证据](recovery-40d558b2ca66/40d558b2ca66e37c/manifest.json)
- `34bebce885e2e89d` — Achieved exterior contact approach after withdrawal/lowering. Paired thin upper-lip tracking; current camera, resume and tracking flags, bounded clearance radius. Select flags only when current checks establish the corresponding entry state. [源码](recovery-34bebce885e2/34bebce885e2e89d/policy.py) · [原始证据](recovery-34bebce885e2/34bebce885e2e89d/manifest.json)

## 共享模块

| 包 | 固定版本 | 公开函数 |
|---|---|---|
| [rgbd-surface-geometry](rgbd-surface-geometry/5aa70d58296eb1b4/SKILL.md) | `5aa70d58296eb1b4` | `decode`、`project`、`point_patch` |
| [eef-feedback-control](eef-feedback-control/6be2c1eb747b8105/SKILL.md) | `6be2c1eb747b8105` | `pose`、`move_bounded`，使用相同 Context 和预算 |

两者都是 candidate 辅助包，不能因为被关门程序导入就视为独立验证的完整子任务。组合方式和校验规则见 [RAX Code Policy](../docs/rax-code-policy.md)。

149 个未整理的历史恢复版本保留在原目录，通过旧 ID 或 `search_skills(include_candidates=true, include_archived=true)` 查阅；不占默认检索位置。
模板 `eef_offset`、`gripper_hold`、`mobile_nudge` 仍需显式 include_templates。
