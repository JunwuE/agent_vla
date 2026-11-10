"""Check retained execution evidence and compare selected matched VLA controls, without changing runs."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from vla_agent.data import read_events
from vla_agent.visualization import episode_summary


def read(path):
    return json.loads(path.read_text())


def compare(agent, baseline, agent_rows):
    baseline_rows = read_events(baseline)
    ac, bc = (read(p/'manifest.json')['config'] for p in (agent, baseline))
    contract = {}
    for section, keys in {'environment': ['task_name', 'seed', 'scene_split', 'task_group',
                                         'camera_size', 'ignore_native_horizon'],
                          'vla': ['checkpoint', 'norm_stats', 'sampling_seed', 'action_layout',
                                  'state_layout', 'normalization', 'binary_encoding'],
                          'runtime': ['max_steps', 'chunk_steps']}.items():
        for key in keys:
            contract[f'{section}.{key}'] = ac[section].get(key) == bc[section].get(key)
    prefixes = []
    firsts = []
    for rows in (agent_rows, baseline_rows):
        prefix = []
        for row in rows:
            if row['kind'] == 'action_executed':
                if row['source'] != 'vla':
                    break
                prefix.append(row)
        prefixes.append(prefix)
        firsts.append(next(r for r in rows if r['kind'] == 'observation'))
    a, b = prefixes
    count = min(len(a), len(b))
    # A read-only pause is an intervention too: an old runtime could discard
    # queued actions on a subsequent 'continue'. Pair the unmodified prefix up
    # to the FIRST pause, and separately retain the longer VLA-only comparison.
    pauses = [r['context_binding']['observation_step'] for r in agent_rows
              if r['kind'] == 'intervention_begin' and r['context_binding']['observation_step'] > 0]
    first_pause = min(pauses) if pauses else len(a)
    before_pause = min(first_pause, count)
    with np.load(agent/firsts[0]['path'], allow_pickle=False) as x, \
            np.load(baseline/firsts[1]['path'], allow_pickle=False) as y:
        initial = set(x.files) == set(y.files) and all(np.array_equal(x[k], y[k]) for k in x.files)
    commands = all(x['action'] == y['action'] for x, y in zip(a[:count], b[:count]))
    states = all(x['measured_state'] == y['measured_state'] for x, y in zip(a[:count], b[:count]))
    matched = all(contract.values()) and initial and commands and states and len(b) >= len(a)
    before_pause_identical = all(x['action'] == y['action'] and x['measured_state'] == y['measured_state']
                                for x,y in zip(a[:before_pause], b[:before_pause]))
    paired = all(contract.values()) and initial and before_pause_identical and count >= first_pause
    divergence = next((i+1 for i,(x,y) in enumerate(zip(a[:count], b[:count]))
                       if x['action'] != y['action'] or x['measured_state'] != y['measured_state']), None)
    ar, br = read(agent/'result.json'), read(baseline/'result.json')
    conclusion = ('agent_success_baseline_budget_failure' if ar['success'] and not br['success'] else
                  'baseline_already_succeeds' if br['success'] else 'both_fail') if paired else 'pairing_incomplete'
    return dict(agent=str(agent.resolve()), baseline=str(baseline.resolve()), contract=contract,
                initial_arrays_identical=initial, compared_prefix_steps=count, agent_vla_prefix_steps=len(a),
                commands_identical=commands, measured_states_identical=states,
                first_pause_step=first_pause, pre_pause_commands_and_states_identical=before_pause_identical,
                matched_initial_conditions_and_pre_pause_prefix=paired, first_vla_divergence_step=divergence,
                complete_prefix_matched=matched, agent_result=ar, baseline_result=br, conclusion=conclusion)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    folder = root/'audits'
    folder.mkdir(exist_ok=True)
    controls = read(root/'controls/index.json')['completed']
    summaries, pairs, candidates, invocations = [], [], [], []
    for path in sorted((root/'episodes').glob('seed[0-9][0-9]/result.json')):
        episode = path.parent
        rows = read_events(episode)
        summary = episode_summary(episode, rows)
        checks = {k:v for k,v in summary['audit'].items() if isinstance(v, bool)}
        commands = {r['id']:r for r in rows if r['kind'] == 'action_command'}
        receipts = [r for r in rows if r['kind'] == 'action_executed']
        checks['command_receipt_payloads_match'] = all(
            all(r[k] == commands[r['command_id']][k] for k in ('action', 'before', 'source', 'instruction'))
            for r in receipts if r['command_id'] in commands)
        checks['observation_artifacts_present'] = all((episode/r['path']).is_file()
                                                      for r in rows if r['kind']=='observation')
        checks['only_vla_or_code_actions'] = set(r['source'] for r in receipts) <= {'vla','code_policy'}
        launch = read(episode.with_name(episode.name+'-launch.json'))
        source = Path(launch['source'])
        hashes = read(source.parent/'source-hashes.json')
        hashed_paths = {source.parent/name if name.startswith('src/') else source/name: digest
                        for name, digest in hashes.items()}
        checks['pinned_release_unchanged'] = all(
            path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest
            for path,digest in hashed_paths.items())
        summary.update(checks=checks, passed=all(checks.values()), release=launch['release'])
        (folder/f'{episode.name}.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        summaries.append(dict(seed=summary['seed'], passed=summary['passed'], checks=checks))
        for row in rows:
            if row['kind'] == 'skill_candidate_recorded':
                candidates.append(dict(seed=summary['seed'], **row))
            elif row['kind'] == 'skill_invocation':
                outcome = next((r for r in rows if r['id'] > row['id'] and r['kind']=='recovery_result'), None)
                invocations.append(dict(seed=summary['seed'], **row,
                    following_recovery={k:outcome.get(k) for k in ('verified','executed_steps','seconds','program_version')}
                                       if outcome else None))
        if str(summary['seed']) in controls:
            pair = dict(seed=summary['seed'], **compare(episode,Path(controls[str(summary['seed'])]),rows))
            pairs.append(pair)
    result = dict(episodes=len(summaries), all_execution_audits_pass=all(r['passed'] for r in summaries),
                  summaries=summaries, matched_controls=pairs, skill_candidates=candidates,
                  direct_invocations=invocations,
                  note='Controls are selected after code handoffs or a diagnostic read-only pause, not a full baseline SR estimate. '
                       'Candidate status remains candidate; an invocation or import alone is not validated reuse.')
    (root/'execution-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(episodes=len(summaries), all_passed=result['all_execution_audits_pass'],
                         controls=Counter(p['conclusion'] for p in pairs),
                         direct_invocations=len(invocations)),ensure_ascii=False))


if __name__ == '__main__':
    main()
