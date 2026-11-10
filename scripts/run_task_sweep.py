"""Run an atomic or composite task sweep using the shared, resumable scheduler."""

import argparse
from pathlib import Path

from run_composite_sweep import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
