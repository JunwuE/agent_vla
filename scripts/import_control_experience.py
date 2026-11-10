"""After the sweep only, retain matched baseline outcomes as future task experience."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from vla_agent.contracts import Task
from vla_agent.data import read_events
from vla_agent.experience import ExperienceStore
from vla_agent.memory import WorkingMemory


def export_control(store, pair):
    path = Path(pair['baseline'])
    rows = read_events(path)
    first = next(r for r in rows if r['kind']=='observation')
    last = next(r for r in reversed(rows) if r['kind']=='observation')
    result = json.loads((path/'result.json').read_text())
    config = json.loads((path/'manifest.json').read_text())['config']
    if config['runtime']['supervision_enabled'] or result['astra_calls'] or result['flora_calls']:
        raise ValueError('Control must have no supervision/model review calls')
    if result['success'] != last['success'] or result['control_steps'] != last['step']:
        raise ValueError('Control result does not match its terminal recorded observation')
    task = Task.model_validate(next(r['task'] for r in rows if r['kind']=='task_initialized'))
    memory = WorkingMemory(task, first['episode_id'])
    # ExperienceStore.record reads only terminal camera/controller metadata from this field.
    # No synthetic RGB or estimated trajectory is created, and no control loop is entered.
    memory.observations.append(SimpleNamespace(rgb=dict.fromkeys(last['cameras']),
        depth_m=dict.fromkeys(last['depth_available']), controller=last['controller']))
    memory.notes.append(dict(source='post_sweep_matched_control_import',
        facts=[f"Native baseline result: success={result['success']}, steps={result['control_steps']}, "
               f"wall_seconds={result['wall_seconds']:.3f}, no Astra/Flora or code intervention.",
               f"Matched pre-first-pause prefix={pair['matched_initial_conditions_and_pre_pause_prefix']}; "
               f"pause_step={pair['first_pause_step']}; full_VLA_segment_match={pair['complete_prefix_matched']}.",
               'The outcome is retrospective episode evidence, not a label available at early observation times.'],
        hypotheses=[], paired_agent_result=pair['agent_result'], paired_agent_evidence_root=pair['agent'],
        diagnosis=pair['conclusion'], imported_after_sweep=True))
    receipt = store.record(memory, result, SimpleNamespace(root=path), rows)
    return dict(episode_id=first['episode_id'], baseline=str(path), **receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--target', type=Path, default=Path('memory/close_toaster/experiences'))
    args = parser.parse_args()
    root = args.root.resolve()
    status = json.loads((root/'status.json').read_text())
    controls = json.loads((root/'controls/index.json').read_text())
    audit = json.loads((root/'execution-audit.json').read_text())
    if (status['state'] != 'completed' or controls['state'] != 'completed' or audit['episodes'] != 50
            or not audit['all_execution_audits_pass']
            or len(audit['matched_controls']) != len(controls['targets'])):
        raise ValueError('Finish and audit all 50 main episodes and all selected controls before importing')
    store = ExperienceStore(args.target, 'accumulate')
    seen = {r['episode_id'] for r in store.index}
    imported, skipped = [], []
    for pair in audit['matched_controls']:
        with (Path(pair['baseline'])/'events.jsonl').open() as stream:
            episode_id = next(json.loads(line)['episode_id'] for line in stream
                              if json.loads(line)['kind']=='observation')
        if episode_id in seen:
            skipped.append(episode_id)
            continue
        receipt = export_control(store, pair)
        imported.append(receipt)
        seen.add(episode_id)
    checked = ExperienceStore(args.target)
    output = dict(target=str(args.target.resolve()), imported=imported, skipped_episode_ids=skipped,
                  total_experiences=len(checked.index),
                  note='Post-sweep import only. Main sweep snapshots/shared library are unchanged; no online evaluation leakage.')
    (root/'post-sweep-control-memory.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(imported=len(imported), skipped=len(skipped), total=len(checked.index))))


if __name__ == '__main__':
    main()
