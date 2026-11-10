"""Check historical/current initial arrays and first VLA proposals without invoking models."""

import argparse
import json
from pathlib import Path

import numpy as np


def first_inputs(episode):
    observation = None
    with (episode / 'events.jsonl').open() as stream:
        for line in stream:
            event = json.loads(line)
            if event['kind'] == 'observation' and observation is None:
                observation = event
            elif event['kind'] == 'vla_proposal':
                return observation, event
    raise ValueError(f'No initial VLA proposal in {episode}')


def main(root):
    root = root.resolve()
    previous = Path(json.loads((root / 'plan.json').read_text())['baseline'])
    rows = []
    contract_keys = {
        'environment': ['task_name', 'seed', 'scene_split', 'task_group', 'camera_size', 'ignore_native_horizon'],
        'vla': ['checkpoint', 'norm_stats', 'sampling_seed', 'action_layout', 'state_layout',
                'normalization', 'binary_encoding'],
        'runtime': ['max_steps', 'chunk_steps'],
    }
    for analysis in sorted((root / 'analysis').glob('seed*.json')):
        seed = int(analysis.stem[4:])
        a, b = (parent / 'episodes' / f'seed{seed:02d}' for parent in [previous, root])
        ao, ap = first_inputs(a)
        bo, bp = first_inputs(b)
        ac, bc = [json.loads((p / 'manifest.json').read_text())['config'] for p in [a, b]]
        contract = {f'{section}.{key}': ac[section].get(key) == bc[section].get(key)
                    for section, keys in contract_keys.items() for key in keys}
        with np.load(a / ao['path'], allow_pickle=False) as x, np.load(b / bo['path'], allow_pickle=False) as y:
            same_keys = set(x.files) == set(y.files)
            differences = [k for k in set(x.files) & set(y.files) if not np.array_equal(x[k], y[k])]
        aa, ba = np.asarray(ap['actions']), np.asarray(bp['actions'])
        same_shape = aa.shape == ba.shape
        rows.append(dict(seed=seed, config_contract=contract, initial_arrays_identical=same_keys and not differences,
                         differing_arrays=sorted(differences), initial_array_keys_equal=same_keys,
                         initial_native_instruction_identical=ao['facts']['task_instruction'] == bo['facts']['task_instruction'],
                         proposal_shapes_identical=same_shape, proposals_identical=same_shape and np.array_equal(aa, ba),
                         max_abs_first_chunk_difference=float(np.max(np.abs(aa - ba))) if same_shape else None,
                         max_abs_first_action_difference=float(np.max(np.abs(aa[0] - ba[0]))) if same_shape else None))
    result = dict(episodes=len(rows), initial_arrays_identical_count=sum(r['initial_arrays_identical'] for r in rows),
                  config_contract_pass_count=sum(all(r['config_contract'].values()) for r in rows),
                  first_proposals_identical_count=sum(r['proposals_identical'] for r in rows), rows=rows,
                  note='Proposal differences are in the canonical mixed-unit action representation, not a metric distance. '
                       'Equal initial data does not imply identical historical model inference or a shared action prefix. '
                       'No model calls or environment actions were made by this audit; root cause of numeric differences is unknown.')
    (root / 'historical-state-audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['rows', 'note']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
