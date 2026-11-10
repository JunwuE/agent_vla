"""Compare legacy and packaged controllers against recorded RPC receipts, without a simulator."""

import argparse
import asyncio
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path

from vla_agent.sandbox import recovery_command
from vla_agent.skill_packages import execution_bundle, export_bundle
from vla_agent.skills import SkillLibrary


def read_trace(episode, source_sha):
    rows = [json.loads(line) for line in (episode / 'events.jsonl').open()]
    candidates = [r for r in rows if r['kind'] == 'code_version' and r['sha256'] == source_sha]
    for version in candidates:
        iid = version['intervention_id']
        outcome = next((r for r in rows if r['kind'] == 'recovery_result' and r['intervention_id'] == iid), None)
        if not outcome or outcome['verified'] is not True:
            continue
        replies = {r['request_event_id']: r['result'] for r in rows
                   if r['kind'] == 'recovery_tool_result' and r['intervention_id'] == iid}
        requests = [r for r in rows if r['kind'] == 'recovery_tool_request' and r['intervention_id'] == iid
                    and r['operation'] not in {'result', 'failure'}]
        return [(r, replies[r['id']]) for r in requests], outcome, version
    raise ValueError('No verified recorded execution of the requested exact source')


async def replay(directory, skill, trace, version, bundle=None):
    directory.mkdir(parents=True, exist_ok=False)
    source = directory / 'policy.py'
    source.write_text(skill['code'])
    if bundle:
        export_bundle(directory, bundle)
    constants = {r['operation']: value for r, value in trace
                 if r['operation'] in {'parameters', 'completion_condition'}}
    expected = [(r, value) for r, value in trace if r['operation'] not in constants]
    runtime = Path(__file__).resolve().parents[1] / 'src/vla_agent/recovery_worker.py'
    environment = {k: os.environ[k] for k in ('PATH', 'LANG', 'LD_LIBRARY_PATH') if k in os.environ}
    environment.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    index, actions, steps, final_reads = 0, 0, 0, 0
    last_observation = None
    stderr_path = directory / 'stderr.log'
    with stderr_path.open('wb') as stderr:
        process = await asyncio.create_subprocess_exec(*recovery_command(runtime, source, directory, 'landlock'),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=stderr, env=environment)
        try:
            async with asyncio.timeout(120):
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        raise RuntimeError('Replay worker stopped: ' + stderr_path.read_text()[-1500:])
                    message = json.loads(line)
                    operation, args = message['op'], message.get('args', {})
                    if operation == 'failure':
                        raise RuntimeError(message)
                    if operation == 'result':
                        if index != len(expected):
                            raise AssertionError(f'Program ended before consuming receipts: {index}/{len(expected)}')
                        result = message['result']
                        if bundle:
                            assert result['ok'] is True, result
                        else:
                            assert result == 'SUBTASK_DONE', result
                        break
                    if operation in constants:
                        value = copy.deepcopy(constants[operation])
                    elif index == len(expected) and operation == 'observe' and last_observation:
                        # The new structured return includes a final read of the same actual state,
                        # already supplied by the recorded final move's observation receipt.
                        value = copy.deepcopy(last_observation)
                        final_reads += 1
                    else:
                        if index >= len(expected):
                            raise AssertionError('Unexpected operation after the recorded execution: ' + operation)
                        old, value = expected[index]
                        expected_args = dict(old.get('arguments', {}))
                        if operation == 'stage':
                            expected_args.setdefault('parent_stage_id', None)
                        if operation != old['operation'] or args != expected_args:
                            raise AssertionError(dict(index=index, expected_operation=old['operation'], operation=operation,
                                                      expected=expected_args, actual=args))
                        index += 1
                        value = copy.deepcopy(value)
                        if operation in {'move_eef', 'move_relative', 'set_gripper', 'move_base', 'act'}:
                            actions += 1
                            steps += value.get('executed_steps', 1)
                        if operation == 'observe':
                            last_observation = copy.deepcopy(value)
                        elif isinstance(value, dict) and 'observation' in value:
                            last_observation = copy.deepcopy(value['observation'])
                    if operation == 'read_observation':
                        old_path = Path(value['path'])
                        if hashlib.sha256(old_path.read_bytes()).hexdigest() != value['sha256']:
                            raise ValueError('Recorded observation checksum mismatch')
                        target = directory / 'observations' / (value['sha256'] + '.npz')
                        target.parent.mkdir(exist_ok=True)
                        if not target.exists():
                            shutil.copyfile(old_path, target)
                        value['path'] = str(target.resolve())
                    process.stdin.write((json.dumps(dict(result=value), allow_nan=False) + '\n').encode())
                    await process.stdin.drain()
                await process.wait()
                if process.returncode:
                    raise RuntimeError('Replay worker exited unsuccessfully')
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
    return dict(exact_requests_matched=index, action_calls=actions, recorded_executed_steps=steps,
                final_read_only_observations=final_reads, result=result,
                source_sha256=skill['sha256'], bundle_sha256=bundle['sha256'] if bundle else None,
                recorded_parameters=version['parameters'])


async def main(args):
    migrations = json.loads(args.migration.read_text())
    library = SkillLibrary(args.library)
    receipts = []
    for row, seed in zip(migrations['procedures'], (9, 44), strict=True):
        episode = args.episodes / f'seed{seed:02d}'
        trace, outcome, version = read_trace(episode, row['source']['sha256'])
        old = library.get(row['source']['id'], row['source']['version'], 'PandaOmron')
        new = library.get(row['migrated']['skill_id'], row['migrated']['version'], 'PandaOmron')
        bundle = execution_bundle(new['package'], new['package_files'], library, 'PandaOmron')
        before = await replay(args.output / f'seed{seed:02d}/legacy', old, trace, version)
        after = await replay(args.output / f'seed{seed:02d}/package', new, trace, version, bundle)
        assert before['action_calls'] == after['action_calls']
        assert before['recorded_executed_steps'] == after['recorded_executed_steps'] == outcome['executed_steps']
        receipt = dict(seed=seed, original_episode=str(episode.resolve()), original_attempt=version['intervention_id'],
            legacy=before, package=after, exact_action_arguments=True, new_simulation=False,
            note='Counterfactual software replay of saved observations/results; not a new physical success.')
        (args.output / f'seed{seed:02d}/receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        receipts.append(receipt)
        print(json.dumps(dict(seed=seed, exact_action_arguments=True, recorded_steps=outcome['executed_steps'])), flush=True)
    (args.output / 'summary.json').write_text(json.dumps(receipts, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, default=Path('skills'))
    parser.add_argument('--migration', type=Path, default=Path('runs/rax-code-policy-20261031/migration.json'))
    parser.add_argument('--episodes', type=Path, default=Path('runs/toaster-seeds50-20261030/episodes'))
    parser.add_argument('--output', type=Path, default=Path('runs/rax-code-policy-20261031/replay'))
    asyncio.run(main(parser.parse_args()))
