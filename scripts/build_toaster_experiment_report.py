"""Build a local, evidence-linked report from the frozen ten-round development logs.

Reads generated programs with ast.parse only. It never executes them or contacts models.
"""

import ast
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/toaster-iteration-20261029'
OUTPUT = ROOT / 'docs/close-toaster-ten-rounds-detailed.md'


def cell(value):
    if value is None or value == '':
        return '—'
    return str(value).replace('|', '\\|').replace('\n', '<br>')


def link(path, title=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return f'[{title or path.name}]({os.path.relpath(path, OUTPUT.parent)})'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(cell(v) for v in row) + ' |' for row in rows]) + '\n'


def events(path):
    stats = dict(tools=Counter(), worker=Counter(), worker_by_attempt=defaultdict(Counter),
                 intents=Counter(), instructions=set(), decision_scores=[], threads=set(),
                 direct_skills=[], reads=[], decision_windows=[], review_timeline=[])
    for line in (path / 'events.jsonl').open():
        e = json.loads(line)
        kind = e['kind']
        if kind == 'codex_tool':
            stats['tools'][e['name']] += 1
            if e['name'] == 'read_skill':
                result = e.get('result', {})
                stats['reads'].append((result.get('id'), result.get('version'), result.get('status')))
        elif kind == 'recovery_tool_request':
            stats['worker'][e['operation']] += 1
            stats['worker_by_attempt'][e['intervention_id']][e['operation']] += 1
        elif kind == 'codex_turn_start':
            stats['intents'][e['context']['intent']] += 1
            stats['threads'].add(e['thread_id'])
        elif kind == 'action_command':
            stats['instructions'].add(e['instruction'])
        elif kind == 'decision':
            d = e['decision']
            stats['decision_scores'].append((d['binding']['observation_step'], d['label'],
                                             round(d.get('probabilities', {}).get('KEEP', 0), 4),
                                             round(e.get('seconds', 0), 3)))
        elif kind == 'decision_request':
            stats['decision_windows'].append((e['binding']['observation_step'], e.get('image_count')))
        elif kind == 'skill_invocation':
            stats['direct_skills'].append(e)
        elif kind in {'review_begin', 'review_end', 'review_discarded', 'review_refresh'}:
            stats['review_timeline'].append((kind, e.get('binding', {}).get('observation_step'),
                                             e.get('current_step'), e.get('result', {}).get('assessment')))
    return stats


def source_info(path):
    text = path.read_text()
    tree = ast.parse(text)
    functions = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    api = sorted({ast.unparse(n.func) for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and ast.unparse(n.func).startswith('ctx.')})
    return dict(lines=len(text.splitlines()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                functions=functions, api=api)


def counter_text(counter):
    return '；'.join(f'`{k}` × {v}' for k, v in sorted(counter.items())) or '无'


def build():
    runs = json.loads((BASE / 'all-results.json').read_text())
    rounds = json.loads((BASE / 'rounds.json').read_text())
    stats = {row['path']: events(BASE / row['path']) for row in runs}
    draft = (ROOT / 'docs/updates/2026-10-30-close-toaster/update.md').read_text()
    draft = draft.replace('# 架构与实验更新：CloseToasterOvenDoor 闭环恢复（2026-10-30）',
                          '# CloseToasterOvenDoor：十轮实验、改进过程与操作代码详细报告')
    draft = draft.replace('以及当前 `configs/close_toaster_explore.yaml` 的实际运行配置',
                          '以及各 Episode manifest 中冻结的历史配置')
    draft = draft.replace('](architecture.svg)', '](updates/2026-10-30-close-toaster/architecture.svg)')
    draft = draft.replace('当前实验中 Decision 和 Astra 都使用', '前十轮实验中 Decision 和 Astra 都使用')
    draft = draft.replace('## 一、现在的系统如何工作', '## 一、前十轮实际运行的系统如何工作')
    draft = draft.replace('![当前系统架构：', '![前十轮系统架构：')
    draft = draft.replace('当前开发配置另外每 150 步', '前十轮开发配置另外每 150 步')
    draft = draft.replace('| 当前开发配置 | configs/close_toaster_explore.yaml |',
                          '| 历史 D3 对照配置 | configs/close_toaster_d3_legacy.yaml；各 Episode manifest.json |')
    notice = ('\n> 范围：正式编号第 1—10 轮，加第 0 轮基础设施故障，共 18 条 Episode 记录。'
              '轮号是实验配置编号，部分实验并行，不能理解为每轮都在上一轮结束后学习。'
              '正文一至九节说明历史 D3 系统；十节说明 2026-10-30 新增的 Flora 50 步监督，'
              '后者尚无新的真实机器人任务成功率结果。十一至十五节为逐运行、逐程序与逐工具审计。\n')
    first, rest = draft.split('\n', 1)
    out = [first, notice, rest]
    out += ['\n## 十、本次落地的 Flora 50 步监督\n',
            'CloseToaster 的两个默认配置已改用 Flora。每 50 个 VLA 执行步取一次历史快照，'
            '最多读取当前阶段最近 200 步、left/wrist 两视角，最多 64 个采样帧。'
            '正常监督只评分；Astra 可单独请求理由。D3 扩展配置作为 legacy 保留。\n',
            table(['规则', '含义'], [
                ['连续两次评分 ≤2', '请求 Astra 判断低分是否属于正常未完成'],
                ['连续三次相同评分', '请求 Astra 复核；不直接证明物体停滞'],
                ['相邻评分下降 ≥2', '请求检查退化或观测不一致'],
                ['评分 ≥4', '请求检查阶段/子任务可能完成；不直接 SWITCH'],
                ['单次低分或其余情况', '继续 VLA，保存评分'],
                ['无效、超时、预算不足', 'ESCALATE；过期阶段/控制权结果直接丢弃'],
            ]),
            '所有数值阈值是待校准的开发配置。路由目前只输出 KEEP / ESCALATE，不把 1–5 分'
            '解释为五类语义或等距进度。默认两次评分触发审核至少间隔 150 步。'
            'Astra 审核期间仍采样，重复请求合并；服务慢时记录跳过时刻，不阻塞 VLA、不堆积请求。\n',
            '自动监控与 Astra 诊断分开计数，共用设备请求锁；总计和分类调用数分别记录。'
            '代码接管后的阶段检查继续由程序与 Astra 完成。新阈值、新调度尚不能归因于历史两次恢复成功。\n',
            '```mermaid\nflowchart TD\n'
            '  V["VLA 连续执行 → Harness → 仿真"] --> O["每 50 步冻结历史窗口"]\n'
            '  O --> F["Flora 评分：不暂停 VLA"]\n'
            '  F --> R{"评分规则"}\n'
            '  R -->|继续观察| M["记录到短期 Memory"]\n'
            '  R -->|需要审核| A["Astra + Memory + 诊断工具"]\n'
            '  A -->|保持执行| M\n'
            '  A -->|确认需要恢复| C["Harness 接管 → Code Policy 完整子任务"]\n'
            '  C --> I["新观测 → 修改代码 → 再执行"]\n'
            '  I -->|未完成且有预算| C\n'
            '  I -->|独立验证完成| P["Astra 选择下一子任务及执行器"]\n'
            '```\n',
            '2026-10-30 软件验证：全量 pytest **176 passed in 14.89s**；'
            '包含真实 HTTP 客户端的模拟服务集成测试，未启动 Flora 权重或仿真任务。'
            '该结果验证调度、请求和状态隔离，不代表新监督方案的成功率或实际模型延迟。\n',
            f'完整设置与启动说明：{link(ROOT / "docs/flora-supervision.md")}。\n',
            '\n## 十一、逐运行配置、调用与判断记录\n',
            '下表从原始 manifest、result、events 读取。控制步数包含代码分段内部的物理执行步；'
            'Astra turns、工具调用、程序版本、物理接管是不同计数。重复出现的 seed 0 不等于相同采样轨迹。\n']
    for round_ in rounds:
        number = round_['round']
        out.append(f'\n### 第 {number} 轮：{round_["path"]}\n')
        out.append('配置变化原始记录：' + round_['change'] + '\n')
        selected = [r for r in runs if r['path'].split('/')[0] == round_['path']]
        out.append(table(['运行', '结局', '物理步', '墙钟分钟', 'D3', 'Astra turns / 审核',
                          'Flora', '代码执行 / 源码版本', 'VLA 接管'], [
            [r['path'].split('/')[-1], '成功' if r['result']['success'] else r['result']['reason'],
             r['result']['control_steps'], f'{r["result"]["wall_seconds"]/60:.2f}',
             r['result']['decision_checks'], f'{r["result"]["astra_calls"]} / {r["result"]["astra_reviews"]}',
             r['result']['flora_calls'], f'{len(r["code_attempts"])} / {len(r["code_attempts"])+len(r["unexecuted_versions"])}',
             r['vla_takeovers']] for r in selected]))
        for row in selected:
            path = BASE / row['path']
            c = json.loads((path / 'manifest.json').read_text())['config']
            s = stats[row['path']]
            out.append(f'\n**{row["path"]}**：{link(path / "manifest.json", "冻结配置与源码哈希")}；'
                       f'{link(path / "result.json", "原始结果")}；{link(path / "events.jsonl", "完整事件")}。\n')
            out.append(f'场景 seed={c["environment"]["seed"]}；policy seed={c["vla"].get("sampling_seed")}；'
                       f'监督={c["runtime"]["supervision_enabled"]}；总步数上限={c["runtime"]["max_steps"]}；'
                       f'忽略原生 horizon={c["environment"].get("ignore_native_horizon", False)}；'
                       f'Episode 墙钟上限={c["runtime"]["max_wall_seconds"]}。\n')
            if s['instructions']:
                out.append('实际生效的 VLA 语言：' + '；'.join(f'`{x}`' for x in sorted(s['instructions'])) + '。\n')
            out.append('Astra turn 类型：' + counter_text(s['intents']) + '。\n')
            out.append('Astra 工具调用：' + counter_text(s['tools']) + '。\n')
            out.append('程序侧工具请求：' + counter_text(s['worker']) + '。\n')
            if s['decision_scores']:
                out.append(table(['观测步', 'D3 输出', 'KEEP 概率', '客户端耗时秒'], s['decision_scores']))
                out.append('实际图像数：' + '；'.join(f'第 {step} 步 {n} 张' for step, n in s['decision_windows']) +
                           '。启动历史不足时没有复制图片凑满 15 张。\n')
            if row['result'].get('error'):
                out.append('运行错误原文：`' + row['result']['error'] + '`。分类以正文说明和 annotations.json 为准。\n')
            if row['model_outcomes']:
                out.append(table(['快照步', '评价目标', '分数', '耗时秒', '解释（模型假设）'], [
                    [x['binding']['observation_step'], x['task'], x['score'], f'{x["seconds"]:.2f}', x.get('rationale')]
                    for x in row['model_outcomes']]))

    out += ['\n## 十二、全部 30 个生成程序：修改依据、阶段与实际工具调用\n',
            '下面列出所有源码版本，其中 27 个实际执行、3 个未执行。每版的修改假设和预期效果保留'
            '模型原文，便于审阅源码；这些文字不是已确认原因。阶段状态 unchecked 表示该阶段'
            '没有独立完成判据，不能当作完成证明。阶段表可能不覆盖程序在阶段之外执行的动作，'
            '完整动作数以 executed_steps 为准。\n']
    inventory = []
    for row in runs:
        path = BASE / row['path']
        versions = sorted(path.glob('workspaces/*/versions/*/manifest.json'))
        if not versions:
            continue
        out.append(f'\n### {row["path"]}\n')
        attempts = {a['program_version']: a for a in row['code_attempts']}
        for manifest in versions:
            meta = json.loads(manifest.read_text())
            version = meta['version']
            source = manifest.parent / 'policy.py'
            info = source_info(source)
            if info['sha256'] != meta['sha256']:
                raise ValueError(f'Hash mismatch: {source}')
            a = attempts.get(version)
            inventory.append(dict(run=row['path'], version=version, source=str(source.relative_to(ROOT)),
                                  executed=a is not None, **info))
            out.append(f'\n#### {version} — ' + (f'{a["result"]}，执行 {a["executed_steps"]} 步' if a else '未执行') + '\n')
            out.append(f'{link(source, "完整源码")}；{link(manifest, "版本元数据")}；'
                       f'{link(manifest.parent / "change.diff", "代码差异")}。'
                       f'源码 {info["lines"]} 行；SHA-256 `{info["sha256"]}`。\n')
            out.append('修改假设（原文，待验证）：\n\n> ' + str(meta.get('hypothesis', '未记录')).replace('\n', '\n> ') + '\n')
            out.append('预期效果（原文，不等于结果）：\n\n> ' + str(meta.get('expected_effect', '未记录')).replace('\n', '\n> ') + '\n')
            out.append('程序定义的函数：' + '、'.join(f'`{x}`' for x in info['functions']) + '。\n')
            out.append('源码引用的 ctx 接口：' + '、'.join(f'`{x}`' for x in info['api']) +
                       '。这是静态引用清单，不是实际调用次数。\n')
            if a is None:
                out.append('此版本没有对应 recovery_result，不计物理尝试或成功示范。'
                           '第 6、9 轮的此类版本在执行前继续改写；第 7 轮因 Codex turn 超时未进入执行。\n')
                continue
            attempt_id = Path(a['artifacts']['source']).parts[1]
            operations = stats[row['path']]['worker_by_attempt'][attempt_id]
            out.append(f'实际运行耗时约 {a["seconds"]:.2f} 秒；独立验证={a["verified"]}；'
                       f'执行器错误={a["error"]}；失败阶段={a["failed_stage"]}。\n')
            out.append('实际工具请求：' + counter_text(operations) + '。\n')
            out.append(table(['阶段', '起始 → 结束步', '阶段状态', '错误／停止原因'], [
                [s['name'], f'{s["start_step"]} → {s.get("end_step")}', s['status'], s.get('error')]
                for s in a['stages']]))
            for note in a.get('program_notes', []):
                for fact in note.get('facts', []):
                    out.append('程序报告的事实：' + fact + '\n')
                for hypothesis in note.get('hypotheses', []):
                    out.append('程序附带假设：' + hypothesis + '\n')
            if a.get('evidence'):
                out.append('独立验证回执：\n\n```json\n' + json.dumps(a['evidence'], ensure_ascii=False, indent=2) + '\n```\n')

    out += ['\n## 十三、最终保留的工具与操作代码分别是什么\n',
            '“工具”有两层：框架提供的观测、控制、验证 API，以及 Astra 通过这些 API 编写的操作程序。'
            'Astra 的工具调用会消耗模型交互时间；程序内部多次调用 ctx 并不意味着每次都调用 Astra。\n',
            table(['层次', '代码位置', '作用'], [
                ['Astra 工具目录', link(ROOT / 'src/vla_agent/toolkit.py'), '参数校验、经验/技能检索、Flora、grounding、代码编写检查和提交'],
                ['程序 SDK', link(ROOT / 'src/vla_agent/recovery_worker.py'), 'ctx 代理，不向程序暴露仿真器对象'],
                ['执行代理和版本反馈', link(ROOT / 'src/vla_agent/recovery.py'), '冻结源码、启动隔离进程、处理工具请求与验证结果'],
                ['实测控制', link(ROOT / 'src/vla_agent/control.py'), '以当前 OSC 位姿构造短程末端动作；不是 IK/路径规划器'],
                ['唯一动作入口', link(ROOT / 'src/vla_agent/harness.py'), '控制权、步数、动作检查、执行回执'],
                ['感知与坐标', link(ROOT / 'src/vla_agent/perception.py'), '深度反投影、坐标变换和可信度检查'],
                ['候选与检索', link(ROOT / 'src/vla_agent/skills.py'), '版本、前置条件、证据、快照与候选排序'],
                ['Flora 新监督', link(ROOT / 'src/vla_agent/flora_supervision.py'), '50 步评分结果的审核路由；不直接执行动作'],
            ]),
            '\n### 两个有独立成功回执的候选\n']
    success_skills = []
    for manifest in sorted((ROOT / 'skills').glob('*/*/manifest.json')):
        skill = json.loads(manifest.read_text())
        if skill['status'] != 'candidate' or not any(e.get('report', {}).get('verified') is True for e in skill['evidence']):
            continue
        success_skills.append(skill)
        source = manifest.parent / 'policy.py'
        info = source_info(source)
        out.append(f'\n**{skill["id"]} / {skill["version"]}**\n')
        out.append(f'{link(source, "长期目录中的操作代码")}；{link(manifest, "契约与验证证据")}。\n')
        out.append('状态：candidate；参数 schema：`' + json.dumps(skill['parameter_schema'], ensure_ascii=False) + '`。\n')
        out.append('函数：' + '、'.join(f'`{f}`' for f in info['functions']) + '。\n')
        out.append('接口：' + '、'.join(f'`{f}`' for f in info['api']) + '。\n')
        guards = [f'{i}: {line.strip()}' for i, line in enumerate(source.read_text().splitlines(), 1)
                  if '1335' in line or '1326' in line]
        out.append('特定起始状态限制：`' + '；'.join(guards) + '`。'
                   '还包含固定图像区域和当前布局假设。新 seed 不保证适用；不要直接删除这些保护，'
                   '应替换为重新定位和可观测的前置条件，再独立复测。\n')
    out += [
        '\n第 5 轮最后一版重点是：以左视角跟踪把手和炉顶参考，计算闭合方向和接触高度，'
        '以短程末端反馈完成最后压合。第 9 轮最后一版重点是：用 RGB-D 重新定位近直立门，'
        '释放夹爪、抬升避让、退到外侧、调整夹爪并接近压合，途中检查原生 task_success。'
        '第 9 轮成功发生在最后的外侧接近阶段，不能把源码中尚未运行的后续分支算作验证通过。\n',
        '框架原有 eef_offset、gripper_hold、mobile_nudge 是 template，不是本轮学到的 verified 技能。'
        '30 个工作区版本与 27 个长期候选不是两批独立训练样本；后者是执行过版本的归档。'
        '27 个候选中有两个版本拥有成功回执，全部仍为 candidate；通用抓取技能数不能据此报告为 2。\n',
        '\n## 十四、实际调用了哪些工具\n',
        'Astra 工具统计来自 codex_tool 事件；程序统计来自 recovery_tool_request。'
        '下表为发起请求次数，包含可能被拒绝或失败的请求，不等于成功动作数。'
        '四列分别对应各轮 seed 0 的代码探索运行；其他 Episode 的调用列于第十一节。'
        'move_eef 一次可执行多个物理步；result、stage_end 是程序协议消息。\n']
    prominent = ['round5-explore-int8/seed0', 'round6-replay-recovery/seed0',
                 'round7-replay-refresh/seed0', 'round9-seeded-agent/seed0']
    for key, title in [('tools', 'Astra 发起的工具'), ('worker', 'Code Policy 程序侧请求')]:
        names = sorted(set().union(*(stats[p][key] for p in prominent)))
        out.append(f'\n### {title}\n')
        out.append(table(['工具', '第5轮 seed0', '第6轮 seed0', '第7轮 seed0', '第9轮 seed0'],
                         [[f'`{n}`'] + [stats[p][key][n] for p in prominent] for n in names]))
    out += [table(['接口', '用途与边界'], [
        ['observe / read_observation', '读取当前实测状态或绑定时刻 RGB-D；读取本身不执行物理动作'],
        ['ground_target / query_geometry', '绑定目标区域、相机和时刻，计算带尺度几何；模型置信度不等于几何可靠性'],
        ['move_eef / move_relative / set_gripper', '程序通过 Harness 发出的动作；不能误算成 Astra 直接局部操控'],
        ['check(task_success)', '独立原生验证；源码返回 SUBTASK_DONE 仍需宿主核验'],
        ['stage / stage_end / note', '记录阶段、测量与解释；unchecked 或自报事实不等于独立成功'],
        ['search_skills / read_skill', '检索/阅读参考代码，不等于直接执行该技能'],
        ['write_program / patch_program / check_program', '编写、修改和静态检查；静态检查通过不等于物理成功'],
        ['evaluate_subtask', 'Flora 轨迹评价，理由属于模型假设'],
        ['submit_review / submit_intervention', '前者只读建议；后者受调度、控制权和验证约束'],
    ]),
        '第 9 轮 search_skills=14、read_skill=6，读的是两个不同的失败候选；'
        'skill_invocation=0。第 5 轮最终成功程序没有出现在第 9 轮启动时的快照，'
        '因此不能将第 9 轮解释为“复用已成功技能后更快”。所有实际物理动作源只有 VLA 和 Code Policy，'
        'Astra 直接局部动作数为 0。\n',
        '\n## 十五、如何检查这份报告与下一轮验收\n',
        f'本报告由 {link(Path(__file__), "构建脚本")} 读取冻结日志和源码 AST 生成，'
        '没有运行这些操作程序。每版源码做 SHA-256 核对；所有源码、元数据、差异和证据链接必须存在。\n',
        '下一轮应区分两个问题：Flora 是否更及时、准确地请求审核，以及已有程序是否减少探索。'
        '先整理完整参数化技能并在已知失败状态复测，再在相同新 seed、相同 VLA 采样、预算和监督配置下，'
        '比较禁用技能与冻结技能。报告成功率、全部 Episode 耗时、成功样本耗时、超时率、Astra turns、'
        '代码修订次数和直接技能复用比例。失败不能从耗时样本中消失；模型与技能版本必须记录。\n',
        '当前两次扩展预算成功只能证明恢复闭环可行，不能宣布超过用户原生预算下的 48% 基线。'
        '本次 Flora 路由变更的软件测试与实际新 seed 机器人评测分开报告。\n']
    assert len(runs) == 18 and len(inventory) == 30
    assert sum(x['executed'] for x in inventory) == 27 and len(success_skills) == 2
    OUTPUT.write_text('\n'.join(out))
    audit = dict(episodes=len(runs), generated_versions=len(inventory), executed_versions=27,
                 successful_candidates=len(success_skills), programs=inventory,
                 report=str(OUTPUT.relative_to(ROOT)), report_sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest())
    (BASE / 'detailed-report-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(report=str(OUTPUT), bytes=OUTPUT.stat().st_size,
                          lines=len(OUTPUT.read_text().splitlines()), programs=len(inventory)), ensure_ascii=False))


if __name__ == '__main__':
    build()
