"""Compare causal Outcome windows and whole-episode scores on pure VLA controls."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from vla_agent.config import load_config
from vla_agent.contracts import Task
from vla_agent.data import Recorder, read_events
from vla_agent.flora import FloraEvaluator
from vla_agent.memory import WorkingMemory


async def main(args):
    root, output = args.baseline.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(root / 'config.yaml')
    config.flora.enabled = True
    config.flora.max_calls = 4
    config.flora.with_rationale = False
    plan = json.loads((root / 'plan.json').read_text())
    protocol = dict(episode_seeds=plan['seeds'], whole_episode_thresholds=[4, 5],
        causal_samples=['step 100', 'halfway to episode end'], causal_window_steps=200,
        note='Halfway sampling uses hindsight for offline selection only; each model input contains no frames after its endpoint. No causal failure ground truth is inferred from eventual outcome.')
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    rows = []
    for case in plan['cases']:
        source = root / 'episodes' / case['key']
        result = json.loads((source / 'result.json').read_text())
        events = read_events(source)
        assert not any(e['kind'] in {'intervention_plan', 'codex_turn_start', 'attempt_interrupted'} for e in events)
        observations = [e for e in events if e['kind'] == 'observation']
        task = next(e['task'] for e in events if e['kind'] == 'task_initialized')
        log = Recorder(output / case['key'], config)
        client = FloraEvaluator(config.flora, log)
        try:
            for observation in observations:
                log.observations[f"{observation['episode_id']}:{observation['step']}"] = os.path.relpath(
                    source / observation['path'], log.root)
            memory = WorkingMemory(Task.model_validate(task), observations[0]['episode_id'])
            end = result['control_steps']
            samples = [('early_window', min(100, end - 1), 200),
                       ('middle_window', max(1, end // 2), 200), ('whole_episode', end, None)]
            for label, step, window in samples:
                obs = log.read_observation(step, memory.episode_id, step)
                memory.add(obs)
                score = await client.evaluate(memory, memory.binding(0), window_steps=window)
                row = dict(seed=case['seed'], sample=label, step=step, native_success_at_endpoint=obs.success,
                    hindsight_episode_success=result['success'], native_instruction=task['instruction'],
                    status=score['status'], score=score.get('score'), seconds=score.get('seconds'),
                    evidence=str(log.root), model_request_id=score.get('request_id'))
                rows.append(row)
                with (output / 'scores.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
                print(json.dumps(row), flush=True)
            log.finish(dict(source=str(source), model_calls=client.calls, robot_actions_executed=0))
        finally:
            await client.close()
            log.close()
    full = [r for r in rows if r['sample'] == 'whole_episode' and r['status'] == 'scored']
    confusion = {}
    for threshold in (4, 5):
        counts = dict(tp=0, fp=0, tn=0, fn=0)
        for row in full:
            prediction, actual = row['score'] >= threshold, row['native_success_at_endpoint']
            counts['tp' if prediction and actual else 'fp' if prediction else 'fn' if actual else 'tn'] += 1
        confusion[str(threshold)] = counts
    partial = [r for r in rows if r['sample'] != 'whole_episode' and r['status'] == 'scored']
    summary = dict(whole_episode_evaluated=len(full), whole_episode_confusion=confusion,
        partial_windows=len(partial), low_partial_windows=sum(r['score'] <= 2 for r in partial),
        low_partial_windows_later_vla_success=sum(r['score'] <= 2 and r['hindsight_episode_success'] for r in partial),
        unavailable=[r for r in rows if r['status'] != 'scored'], model='Flora Outcome',
        revision=config.flora.revision, native_task_instruction_used=True,
        limitations='Outcome is an ordinal completion score. Later success is not a ground-truth label for whether intervention was necessary at an earlier time. Dense mode and alternative reasoning models are not evaluated by this script.')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
