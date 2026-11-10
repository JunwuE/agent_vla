import argparse
import json
from pathlib import Path

from vla_agent.visualization import render_episode

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Render a recorded episode without model calls")
    p.add_argument("episode", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--overview", action="store_true", help="Keep every physical frame and stage; omit repetitive tool detail")
    p.add_argument("--label", help="Display label; original configured mode is retained in metadata")
    args = p.parse_args()
    print(json.dumps(render_episode(args.episode, args.output, overview=args.overview, label=args.label), ensure_ascii=False, indent=2))
