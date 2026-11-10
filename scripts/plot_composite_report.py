"""Export measured episode timing and calls; no causal or isolated-latency claims."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(root):
    plan = json.loads((root/'plan.json').read_text())
    records = [json.loads(p.read_text()) for p in sorted((root/'analysis').glob('*.json'))]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout='constrained')
    fields = [('astra_paused', 'Astra while paused', '#7551b0'),
              ('code_execution', 'Code execution', '#e89136'),
              ('vla_actions', 'VLA action execution', '#3284b8'),
              ('vla_inference', 'VLA inference', '#48a89b'),
              ('other_and_unattributed', 'Other / unattributed', '#a4abb5')]
    for axis, task in zip(axes, plan['tasks']):
        selected = [r for r in records if r['task'] == task['task']]
        seeds = np.array([r['seed'] for r in selected])
        bottom = np.zeros(len(seeds))
        for key, label, color in fields:
            values = np.array([r.get('wall_components_seconds', {}).get(key, 0)/60 for r in selected])
            axis.bar(seeds, values, bottom=bottom, label=label, color=color, width=.75)
            bottom += values
        successes = sum(r['result']['success'] for r in selected)
        axis.set(title=f"{task['task']} ({task['group']}) | native success {successes}/{len(selected)}",
                 ylabel='Episode wall time (minutes)', xlabel='Seed', xticks=plan['seeds'], xlim=(-.7, 19.7))
        axis.set_axisbelow(True)
        axis.grid(axis='y', alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, fontsize=9)
    fig.suptitle('Evaluation with identical initial memory; asynchronous reviews / Flora overlap execution\n'
                 'Shared services; timings are not an isolated hardware benchmark', fontsize=11)
    fig.savefig(root/'timing.png', dpi=160)
    fig.savefig(root/'timing.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), layout='constrained')
    for axis, task in zip(axes, plan['tasks']):
        selected = [r for r in records if r['task'] == task['task']]
        seeds = np.array([r['seed'] for r in selected])
        axis.bar(seeds-.2, [r['result']['astra_calls'] for r in selected], .38,
                 label='Astra turns', color='#7551b0')
        axis.bar(seeds+.2, [r['program_attempts'] for r in selected], .38,
                 label='Code attempts (includes checkpoints)', color='#e89136')
        axis.set(title=task['task'], xlabel='Seed', ylabel='Count', xticks=plan['seeds'], xlim=(-.7, 19.7))
        axis.set_axisbelow(True)
        axis.grid(axis='y', alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2, fontsize=9)
    fig.suptitle('Astra turns can contain multiple tool/model calls; attempts are not all physical failures', fontsize=11)
    fig.savefig(root/'calls.png', dpi=160)
    fig.savefig(root/'calls.pdf')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root.resolve())
