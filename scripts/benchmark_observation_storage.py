"""Benchmark lossless NPZ levels on recorded RGB-D arrays; never alters source episodes."""

import argparse
import io
import json
import statistics
import time
from pathlib import Path

import numpy as np

from vla_agent.data import save_observation_arrays


def main(args):
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    samples = []
    for episode in sorted((root / 'episodes').glob('*/result.json')):
        files = sorted((episode.parent / 'observations').glob('*.npz'))
        for index in sorted({0, len(files) // 2, len(files) - 1}):
            samples.append(files[index])
    rows = []
    for path in samples:
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        for level in (6, 1):
            times, sizes = [], []
            for _ in range(args.repeats):
                buffer = io.BytesIO()
                start = time.perf_counter()
                save_observation_arrays(buffer, arrays, level)
                times.append(time.perf_counter() - start)
                sizes.append(buffer.tell())
                buffer.seek(0)
                with np.load(buffer, allow_pickle=False) as recovered:
                    assert set(recovered.files) == set(arrays)
                    assert all(recovered[k].dtype == a.dtype and np.array_equal(recovered[k], a, equal_nan=True)
                               for k, a in arrays.items())
            rows.append(dict(source=str(path.relative_to(root)), level=level,
                             mean_seconds=statistics.mean(times), bytes=sizes[0], lossless=True))
    summary = {str(level): dict(mean_seconds=statistics.mean(r['mean_seconds'] for r in rows if r['level'] == level),
                               mean_bytes=statistics.mean(r['bytes'] for r in rows if r['level'] == level))
               for level in (6, 1)}
    report = dict(summary=summary, samples=len(samples), rows=rows, repeats=args.repeats,
                  scope='In-memory compression only, no disk writes or simulator. Not an end-to-end speedup.')
    (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    main(parser.parse_args())
