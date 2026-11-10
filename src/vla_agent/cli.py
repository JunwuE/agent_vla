import argparse
import asyncio
import json
import shutil
from pathlib import Path

from .adapters.mock import MockDecision, MockEnvironment, MockExplorer, MockPlanner, MockVLA
from .adapters.pi05 import Pi05
from .adapters.robocasa import RoboCasaEnvironment
from .codex_planner import CodexPlanner
from .config import load_config
from .data import Recorder, export_candidates, export_decision_windows, read_events, summarize_episodes
from .decision import Decision3
from .planner import AstraPlanner
from .runtime import AgentRuntime
from .skills import SkillLibrary
from .toolkit import planner_tools


def doctor(config):
    issues = []
    isolation = None
    if 'code' in config.runtime.allowed_interventions:
        from .sandbox import probe

        try:
            isolation = probe(config.runtime.recovery_sandbox)
        except Exception as error:
            issues.append(f'recovery_sandbox: {error}')
    if config.environment.kind == "robocasa":
        for key in ("source_path", "python"):
            value = getattr(config.environment, key)
            if not value or not Path(value).exists():
                issues.append(f"environment.{key}: not configured or missing")
        for key in ("robosuite_path", "library_path"):
            value = getattr(config.environment, key)
            if value and not Path(value).is_dir():
                issues.append(f"environment.{key}: missing")
        for value in config.environment.extra_python_paths:
            if not Path(value).is_dir():
                issues.append(f"environment.extra_python_paths: missing {value}")
    if config.vla.kind == "pi05":
        if not Path(config.vla.python).exists():
            issues.append("vla.python: missing")
        if not config.vla.checkpoint:
            issues.append("vla.checkpoint: not configured (an external running server may supply it)")
        if not config.vla.norm_stats:
            issues.append("vla.norm_stats: not configured (must match the served checkpoint)")
        elif not Path(config.vla.norm_stats).is_file():
            issues.append("vla.norm_stats: missing")
        if config.vla.checkpoint and not (Path(config.vla.checkpoint) / "params").is_dir():
            issues.append("vla.checkpoint: params directory missing")
        if not Path(config.vla.openpi_path).is_dir():
            issues.append("vla.openpi_path: missing")
    if config.planner.kind == "codex" and not shutil.which(config.planner.codex_command):
        issues.append("planner.codex_command: executable missing")
    if config.flora.enabled and not Path(config.flora.python).is_file():
        issues.append("flora.python: missing; run scripts/setup_flora.sh")
    return dict(
        environment=config.environment.kind,
        issues=issues,
        ready=not issues,
        recovery_isolation=isolation,
        note="Static configuration check; endpoints/credentials/weights not exercised",
    )


async def run(config, output):
    # Validate absent simulator configuration before creating an episode or connecting to a model.
    env = (
        MockEnvironment(config.environment.seed)
        if config.environment.kind == "mock"
        else RoboCasaEnvironment(config.environment, output)
    )
    log = Recorder(output, config)
    try:
        vla = MockVLA() if config.vla.kind == "mock" else Pi05(config.vla, log)
        decision = (MockDecision() if config.decision.kind == "mock" else
                    Decision3(config.decision, log) if config.decision.kind == "decision3" else None)
        planner = (
            MockPlanner()
            if config.planner.kind == "mock"
            else MockExplorer()
            if config.planner.kind == "mock_explorer"
            else CodexPlanner(config.planner, log, config.runtime.allowed_interventions)
            if config.planner.kind == "codex"
            else AstraPlanner(config.planner, log)
        )
        return await AgentRuntime(config, env, vla, decision, planner, log).run()
    finally:
        log.close()


def main():
    p = argparse.ArgumentParser(description="Supervised pi0.5 robot agent")
    commands = p.add_subparsers(dest="command", required=True)
    launch = commands.add_parser("run")
    launch.add_argument("--config", default="configs/demo.yaml")
    launch.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("doctor")
    check.add_argument("--config", default="configs/robocasa.yaml")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("episode", type=Path)
    export = commands.add_parser("export-candidates")
    export.add_argument("episode", type=Path)
    export.add_argument("--output", type=Path, required=True)
    decisions = commands.add_parser("export-decisions")
    decisions.add_argument("episode", type=Path)
    decisions.add_argument("--output", type=Path, required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("root", type=Path)
    commands.add_parser("tools", help="Print Astra tool schemas")
    skills = commands.add_parser("skills", help="Validate and list compatible versioned skills")
    skills.add_argument("--library", type=Path, default=Path("skills"))
    skills.add_argument("--robot", default="PandaOmron")
    skills.add_argument("--include-templates", action="store_true")
    skills.add_argument('--include-candidates', action='store_true')
    skills.add_argument('--include-archived', action='store_true')
    skills.add_argument('--query', default='')
    skills.add_argument('--versions', help='List variants of one semantic skill name')
    args = p.parse_args()
    try:
        if args.command == "tools":
            result = planner_tools(include_review=True)
        elif args.command == "skills":
            library = SkillLibrary(args.library)
            result = (library.versions(args.versions, args.robot) if args.versions else
                      library.search(args.robot, args.query, include_templates=args.include_templates,
                                     include_candidates=args.include_candidates, include_archived=args.include_archived))
        elif args.command == "doctor":
            result = doctor(load_config(args.config))
        elif args.command == "run":
            result = asyncio.run(run(load_config(args.config), args.output.resolve()))
        elif args.command == "inspect":
            events = read_events(args.episode)
            result = dict(
                result=json.loads((args.episode / "result.json").read_text()),
                events=len(events),
                decisions=[r["decision"]["label"] for r in events if r["kind"] == "decision"],
            )
        elif args.command == "export-candidates":
            result = dict(candidate_actions=export_candidates(args.episode, args.output))
        elif args.command == "export-decisions":
            result = dict(decision_windows=export_decision_windows(args.episode, args.output))
        else:
            result = summarize_episodes(args.root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.command == "run" and result.get("error"):
            raise SystemExit(1)
    except (ValueError, OSError) as error:
        p.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
